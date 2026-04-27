package com.haptictrack.tracking

import android.content.Context
import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import android.graphics.Rect
import android.graphics.RectF
import android.graphics.Typeface
import android.util.Log
import java.io.File
import java.io.FileOutputStream
import java.time.LocalTime
import java.time.format.DateTimeFormatter
import java.util.concurrent.Executor
import java.util.concurrent.RejectedExecutionException

/**
 * Audit instrumentation (#92): captures the actual crops fed to each embedder
 * so we can eyeball input quality before reaching for a bigger model (#69).
 *
 * On LOCK and every [SAMPLE_INTERVAL_FRAMES] confirmed frames, builds a composite
 * JPEG showing:
 *   - full frame with the locked bbox drawn
 *   - raw bbox crop (what OSNet/face see — no segmenter)
 *   - segmenter masked crop (what MNV3 sees when segmentation succeeds)
 *   - MNV3 input (224×224 stretched, what MediaPipe ImageEmbedder receives)
 *   - OSNet input (256×128 stretched, what PersonReIdEmbedder feeds in)
 *   - person crop with BlazeFace box + 6 keypoints overlaid
 *   - face crop fed to MobileFaceNet (112×112 stretched)
 *
 * Files land under the active DebugFrameCapture session at:
 *   session_TIMESTAMP/crops/NNN_event.jpg
 *
 * Off the processing thread — bitmap snapshot taken synchronously, the rest
 * (resize, canvas draw, JPEG encode, file I/O) runs on a single low-priority
 * worker. Capped at [MAX_CAPTURES_PER_SESSION].
 *
 * Disable by flipping [AUDIT_ENABLED] to false. The flag is checked on the
 * processing thread so a disabled audit costs one boolean read per call site.
 */
class CropDebugCapture(
    private val appearanceEmbedder: AppearanceEmbedder,
    private val personReId: PersonReIdEmbedder,
    private val faceEmbedder: FaceEmbedder,
    /**
     * Worker that runs composite-write tasks. Shared with [ObjectTracker.auditExecutor]
     * so all audit work — stability sampling and composite writing — serializes on
     * one thread. Avoids a separate `captureExecutor` adding a third concurrent
     * caller to the `@Synchronized` segmenter / BlazeFace monitors.
     */
    private val executor: Executor,
) {

    companion object {
        private const val TAG = "CropAudit"
        private const val DIR_NAME = "crops"
        const val SAMPLE_INTERVAL_FRAMES = 30  // every ~1s at 30fps during VT-confirmed
        private const val MAX_CAPTURES_PER_SESSION = 20
        private const val TILE_WIDTH = 320
        private const val HEADER_HEIGHT = 28
        private const val PADDING = 6

        /**
         * Audit enabled at compile time. **Off by default** because the audit
         * thread shares `@Synchronized` embedder monitors with the production
         * pipeline; on multi-test suite runs the contention slows search-mode
         * embedding work enough to perturb tracking-rate and reacquire-identity
         * assertions (see #96). The audit's own *output* is unaffected by
         * this contention — embeddings are deterministic functions of (frame,
         * box) — so flipping this to true is the right one-shot move when
         * collecting a baseline. Then flip back to false before merging.
         */
        const val AUDIT_ENABLED = false
    }

    private val frameTimeFormat = DateTimeFormatter.ofPattern("HHmmss_SSS")

    @Volatile private var sessionDir: File? = null
    @Volatile private var capturesThisSession: Int = 0

    private val labelPaint = Paint().apply {
        color = Color.WHITE; textSize = 18f; isAntiAlias = true
        typeface = Typeface.DEFAULT_BOLD
    }
    private val headerBgPaint = Paint().apply { color = Color.rgb(40, 40, 40) }
    private val frameBgPaint = Paint().apply { color = Color.rgb(20, 20, 20) }
    private val boxPaint = Paint().apply {
        color = Color.GREEN; style = Paint.Style.STROKE; strokeWidth = 3f
    }
    private val faceBoxPaint = Paint().apply {
        color = Color.YELLOW; style = Paint.Style.STROKE; strokeWidth = 2f
    }
    private val keypointPaint = Paint().apply {
        color = Color.MAGENTA; style = Paint.Style.FILL
    }
    private val keypointStrokePaint = Paint().apply {
        color = Color.BLACK; style = Paint.Style.STROKE; strokeWidth = 1f
    }
    private val titlePaint = Paint().apply {
        color = Color.YELLOW; textSize = 22f; isAntiAlias = true
        typeface = Typeface.DEFAULT_BOLD
    }

    /**
     * Start a new audit session pegged to the given debug-capture session
     * directory. Subsequent [capture] calls write under `<sessionDir>/crops/`.
     */
    fun startSession(parentSessionDir: File?) {
        sessionDir = parentSessionDir?.let { File(it, DIR_NAME).apply { mkdirs() } }
        capturesThisSession = 0
    }

    fun endSession() {
        sessionDir = null
        capturesThisSession = 0
    }

    /**
     * Capture a per-embedder composite. Synchronous bitmap snapshot, async
     * resize + encode + write. Caller's bitmap is not retained beyond return.
     *
     * [bitmap] is the rotated-image bitmap fed to the embedders (display
     * orientation during the lock-pending VT path; either way, the same
     * bitmap the embedders see). [normalizedBox] is the locked object's
     * bbox in that bitmap's coordinate space.
     */
    fun capture(
        eventTag: String,
        frameIndex: Int,
        bitmap: Bitmap,
        normalizedBox: RectF,
        isPerson: Boolean,
        label: String?,
    ) {
        if (!AUDIT_ENABLED) return
        val dir = sessionDir ?: return
        if (capturesThisSession >= MAX_CAPTURES_PER_SESSION) return

        val snapshot = try {
            bitmap.copy(Bitmap.Config.ARGB_8888, false) ?: return
        } catch (e: Throwable) { return }
        capturesThisSession++  // count optimistically; cheaper than reverting on failure
        val ts = LocalTime.now().format(frameTimeFormat)
        val box = RectF(normalizedBox)

        try {
            executor.execute {
                try {
                    writeComposite(dir, eventTag, frameIndex, ts, snapshot, box, isPerson, label)
                } catch (e: Throwable) {
                    Log.w(TAG, "Composite write failed: ${e.message}")
                } finally {
                    snapshot.recycle()
                }
            }
        } catch (e: RejectedExecutionException) {
            // Shared executor shut down concurrently — drop the snapshot.
            snapshot.recycle()
        }
    }

    fun shutdown() {
        endSession()
    }

    // ------------------------------------------------------------------
    // Composite layout
    // ------------------------------------------------------------------

    private data class Tile(
        val label: String,
        val bitmap: Bitmap,
        /** When non-null, drawn ONTO the tile after letterbox-fit. */
        val overlay: ((Canvas, RectF /* drawDest in tile coords */) -> Unit)? = null,
        /** When true, recycle [bitmap] after drawing. */
        val recycleAfter: Boolean = false,
    )

    private fun writeComposite(
        dir: File,
        eventTag: String,
        frameIndex: Int,
        timestamp: String,
        frame: Bitmap,
        normalizedBox: RectF,
        isPerson: Boolean,
        label: String?,
    ) {
        val tiles = mutableListOf<Tile>()

        // 1. Full frame (with bbox drawn)
        tiles += Tile(
            label = "full frame  ${frame.width}×${frame.height}",
            bitmap = frame,
            overlay = { canvas, dest ->
                val x = dest.left + normalizedBox.left * dest.width()
                val y = dest.top + normalizedBox.top * dest.height()
                val r = dest.left + normalizedBox.right * dest.width()
                val b = dest.top + normalizedBox.bottom * dest.height()
                canvas.drawRect(x, y, r, b, boxPaint)
            }
        )

        // 2. Raw bbox crop (what OSNet/face/MNV3-fallback receive pre-segmenter)
        cropNormalized(frame, normalizedBox)?.let { raw ->
            tiles += Tile(
                label = "raw bbox crop  ${raw.width}×${raw.height}",
                bitmap = raw,
                recycleAfter = true,
            )
        }

        // 3. Segmenter masked crop (what MNV3 actually sees on success)
        val masked = appearanceEmbedder.debugMaskedCrop(frame, normalizedBox)
        if (masked != null) {
            tiles += Tile(
                label = "masked crop  ${masked.width}×${masked.height} (segmenter)",
                bitmap = masked,
                recycleAfter = true,
            )
        } else {
            // Render a placeholder tile noting segmentation was rejected
            tiles += placeholderTile("masked crop", "segmentation null (large/empty)")
        }

        // 4. OSNet 256×128 input (current behavior: pure stretch, no aspect preserve)
        val osnetInput = personReId.debugInput(frame, normalizedBox)
        if (osnetInput != null) {
            tiles += Tile(
                label = "OSNet input  ${osnetInput.width}×${osnetInput.height} (stretched)",
                bitmap = osnetInput,
                recycleAfter = true,
            )
        }

        // 5+6. Face: only when locked is person AND BlazeFace finds a face
        if (isPerson) {
            val faceDbg = faceEmbedder.debugFaceCrop(frame, normalizedBox)
            if (faceDbg != null) {
                val faceBox = faceDbg.faceBoxOnPerson
                val kps = faceDbg.keypoints
                tiles += Tile(
                    label = "person crop + BlazeFace bbox+keypoints",
                    bitmap = faceDbg.personCrop,
                    recycleAfter = true,
                    overlay = if (faceBox != null) ({ canvas, dest ->
                        val sx = dest.width() / faceDbg.personCrop.width
                        val sy = dest.height() / faceDbg.personCrop.height
                        canvas.drawRect(
                            dest.left + faceBox.left * sx,
                            dest.top + faceBox.top * sy,
                            dest.left + faceBox.right * sx,
                            dest.top + faceBox.bottom * sy,
                            faceBoxPaint
                        )
                        for (kp in kps) {
                            val cx = dest.left + kp.x * sx
                            val cy = dest.top + kp.y * sy
                            canvas.drawCircle(cx, cy, 4f, keypointPaint)
                            canvas.drawCircle(cx, cy, 4f, keypointStrokePaint)
                        }
                    }) else null
                )
                if (faceDbg.faceCrop != null) {
                    tiles += Tile(
                        label = "face input  ${faceDbg.faceCrop.width}×${faceDbg.faceCrop.height} (stretched, NO alignment)",
                        bitmap = faceDbg.faceCrop,
                        recycleAfter = true,
                    )
                } else {
                    tiles += placeholderTile("face input", "BlazeFace found no face")
                }
            }
        }

        // Build composite
        val titleHeight = 32
        // Compute actual tile heights (preserve aspect) and total composite height
        val tileHeights = tiles.map { tile ->
            val aspectH = (TILE_WIDTH.toFloat() / tile.bitmap.width * tile.bitmap.height).toInt()
            HEADER_HEIGHT + aspectH.coerceAtMost(TILE_WIDTH) + PADDING
        }
        val composite = Bitmap.createBitmap(
            TILE_WIDTH + PADDING * 2,
            titleHeight + tileHeights.sum() + PADDING,
            Bitmap.Config.ARGB_8888
        )
        val canvas = Canvas(composite)
        canvas.drawColor(frameBgPaint.color)

        // Title
        val titleText = buildString {
            append(eventTag)
            append("  f=").append(frameIndex)
            if (label != null) append("  ").append(label)
            append("  box=[")
            append("%.2f".format(normalizedBox.left)).append(",")
            append("%.2f".format(normalizedBox.top)).append(",")
            append("%.2f".format(normalizedBox.right)).append(",")
            append("%.2f".format(normalizedBox.bottom)).append("]")
        }
        canvas.drawText(titleText, PADDING.toFloat(), 22f, titlePaint)

        // Tiles
        var y = titleHeight + PADDING
        for ((tile, h) in tiles.zip(tileHeights)) {
            // Header strip
            canvas.drawRect(
                PADDING.toFloat(), y.toFloat(),
                (PADDING + TILE_WIDTH).toFloat(), (y + HEADER_HEIGHT).toFloat(),
                headerBgPaint
            )
            canvas.drawText(tile.label, (PADDING + 4).toFloat(), (y + HEADER_HEIGHT - 8).toFloat(), labelPaint)

            val imgTop = y + HEADER_HEIGHT
            val drawW = TILE_WIDTH
            val drawH = (h - HEADER_HEIGHT - PADDING).coerceAtLeast(1)
            val dest = RectF(
                PADDING.toFloat(), imgTop.toFloat(),
                (PADDING + drawW).toFloat(), (imgTop + drawH).toFloat()
            )
            // Letterbox-fit the tile bitmap into [dest]
            val src = Rect(0, 0, tile.bitmap.width, tile.bitmap.height)
            canvas.drawBitmap(tile.bitmap, src, dest, null)
            tile.overlay?.invoke(canvas, dest)

            if (tile.recycleAfter && !tile.bitmap.isRecycled) tile.bitmap.recycle()
            y += h
        }

        // Write JPEG
        val filename = "%03d_%s_%s.jpg".format(capturesThisSession, eventTag, timestamp)
        try {
            FileOutputStream(File(dir, filename)).use { out ->
                composite.compress(Bitmap.CompressFormat.JPEG, 85, out)
            }
        } catch (e: Exception) {
            Log.w(TAG, "Failed to write composite: ${e.message}")
        } finally {
            composite.recycle()
        }
    }

    private fun placeholderTile(label: String, message: String): Tile {
        val bm = Bitmap.createBitmap(TILE_WIDTH, 80, Bitmap.Config.ARGB_8888)
        val c = Canvas(bm)
        c.drawColor(Color.rgb(60, 30, 30))
        val p = Paint().apply { color = Color.WHITE; textSize = 18f; isAntiAlias = true }
        c.drawText(message, 8f, 48f, p)
        return Tile(label = label, bitmap = bm, recycleAfter = true)
    }
}
