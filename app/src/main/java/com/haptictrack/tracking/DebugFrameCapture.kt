package com.haptictrack.tracking

import android.content.Context
import android.graphics.*
import android.util.Log
import java.io.File
import java.io.FileOutputStream
import java.io.FileWriter
import java.io.PrintWriter
import java.time.LocalDateTime
import java.time.LocalTime
import java.time.format.DateTimeFormatter

/**
 * Saves camera frames with bounding box overlays on tracking events.
 *
 * Each tracking session (lock → clear/timeout) gets its own timestamped folder:
 *   /sdcard/Android/data/com.haptictrack/files/debug_frames/session_20260415_143052/
 *     ├── 143052_100_LOCK.png
 *     ├── 143055_200_LOST.png
 *     ├── 143056_300_REACQUIRE.png
 *     └── session.log          ← copy of all log messages for this session
 *
 * Pull with: adb pull /sdcard/Android/data/com.haptictrack/files/debug_frames/
 */
enum class DebugEvent {
    LOCK, LOST, SEARCH, REACQUIRE, TIMEOUT
}

class DebugFrameCapture(context: Context) {

    companion object {
        private const val TAG = "DebugCapture"
        private const val DIR_NAME = "debug_frames"
        private const val MAX_SESSIONS = 10
    }

    private val baseDir: File? = context.getExternalFilesDir(null)?.let {
        File(it, DIR_NAME).apply { mkdirs() }
    }

    private val frameTimeFormat = DateTimeFormatter.ofPattern("HHmmss_SSS")
    private val sessionTimeFormat = DateTimeFormatter.ofPattern("yyyyMMdd_HHmmss")

    /** Current session folder and log writer — created on LOCK, closed on CLEAR/TIMEOUT. */
    private var sessionDir: File? = null
    private var sessionLog: PrintWriter? = null

    // Paint objects (pre-allocated)
    private val lockedPaint = Paint().apply {
        color = Color.GREEN; style = Paint.Style.STROKE; strokeWidth = 4f
    }
    private val candidatePaint = Paint().apply {
        color = Color.WHITE; style = Paint.Style.STROKE; strokeWidth = 2f; alpha = 180
    }
    private val reacquiredPaint = Paint().apply {
        color = Color.CYAN; style = Paint.Style.STROKE; strokeWidth = 4f
    }
    private val lostPaint = Paint().apply {
        color = Color.RED; style = Paint.Style.STROKE; strokeWidth = 3f
        pathEffect = DashPathEffect(floatArrayOf(10f, 10f), 0f)
    }
    private val labelPaint = Paint().apply {
        color = Color.WHITE; textSize = 28f; isAntiAlias = true
    }
    private val labelBgPaint = Paint().apply {
        color = Color.BLACK; alpha = 160
    }
    private val bannerBgPaint = Paint().apply {
        color = Color.BLACK; alpha = 200
    }
    private val bannerTextPaint = Paint().apply {
        color = Color.YELLOW; textSize = 36f; isAntiAlias = true
        typeface = Typeface.DEFAULT_BOLD
    }

    /**
     * Start a new tracking session. Creates a timestamped folder and log file.
     */
    fun startSession(label: String?, trackingId: Int) {
        endSession()

        val base = baseDir ?: return
        val timestamp = LocalDateTime.now().format(sessionTimeFormat)
        val safeLabel = label?.replace(Regex("[^a-zA-Z0-9]"), "_") ?: "unknown"
        sessionDir = File(base, "session_${timestamp}_${safeLabel}").apply { mkdirs() }

        try {
            val logFile = File(sessionDir, "session.log")
            sessionLog = PrintWriter(FileWriter(logFile, true), true)
            log("SESSION START: id=$trackingId label=$label time=$timestamp")
        } catch (e: Exception) {
            Log.w(TAG, "Failed to create session log: ${e.message}")
        }

        pruneOldSessions(base)
        Log.i(TAG, "Session started: ${sessionDir?.name}")
    }

    /**
     * End the current session.
     */
    fun endSession() {
        if (sessionLog != null) {
            log("SESSION END")
            sessionLog?.close()
            sessionLog = null
        }
        sessionDir = null
    }

    /**
     * Write a log message to both Android logcat and the session log file.
     */
    fun log(message: String) {
        val timestamp = LocalTime.now().format(frameTimeFormat)
        val line = "$timestamp $message"
        Log.d(TAG, line)
        sessionLog?.println(line)
    }

    /**
     * Capture a frame on a tracking event.
     * Saves both a raw (unannotated) frame and an annotated frame with bounding boxes.
     */
    fun capture(
        event: DebugEvent,
        bitmap: Bitmap,
        detections: List<TrackedObject>,
        lockedObject: TrackedObject? = null,
        lastKnownBox: RectF? = null,
        extraInfo: String? = null
    ) {
        val dir = sessionDir ?: baseDir ?: return

        val timestamp = LocalTime.now().format(frameTimeFormat)

        // Save raw frame first (before any annotations)
        val rawFilename = "${timestamp}_${event}_raw.png"
        try {
            FileOutputStream(File(dir, rawFilename)).use { out ->
                bitmap.compress(Bitmap.CompressFormat.PNG, 90, out)
            }
        } catch (e: Exception) {
            Log.w(TAG, "Failed to save raw frame: ${e.message}")
        }

        // Create annotated copy
        val annotated = bitmap.copy(Bitmap.Config.ARGB_8888, true) ?: return
        val canvas = Canvas(annotated)
        val w = bitmap.width.toFloat()
        val h = bitmap.height.toFloat()

        for (obj in detections) {
            val paint = when {
                lockedObject != null && obj.id == lockedObject.id && event == DebugEvent.REACQUIRE -> reacquiredPaint
                lockedObject != null && obj.id == lockedObject.id -> lockedPaint
                else -> candidatePaint
            }
            val screenBox = toPixelRect(obj.boundingBox, w, h)
            canvas.drawRect(screenBox, paint)
            drawLabel(canvas, obj, screenBox, paint.color)
        }

        if (lastKnownBox != null && (event == DebugEvent.LOST || event == DebugEvent.SEARCH)) {
            canvas.drawRect(toPixelRect(lastKnownBox, w, h), lostPaint)
        }

        val eventLabel = "$event${if (extraInfo != null) " | $extraInfo" else ""}"
        drawEventBanner(canvas, eventLabel, w)

        val filename = "${timestamp}_${event}.png"
        try {
            FileOutputStream(File(dir, filename)).use { out ->
                annotated.compress(Bitmap.CompressFormat.PNG, 90, out)
            }
            log("FRAME $event: $filename + $rawFilename (${detections.size} detections) ${extraInfo ?: ""}")
        } catch (e: Exception) {
            Log.w(TAG, "Failed to save annotated frame: ${e.message}")
        } finally {
            annotated.recycle()
        }
    }

    private fun toPixelRect(normalized: RectF, w: Float, h: Float): RectF {
        return RectF(normalized.left * w, normalized.top * h, normalized.right * w, normalized.bottom * h)
    }

    private fun drawLabel(canvas: Canvas, obj: TrackedObject, box: RectF, color: Int) {
        val text = buildString {
            append("#${obj.id}")
            if (obj.label != null) append(" ${obj.label}")
            append(" ${(obj.confidence * 100).toInt()}%")
        }
        val textBounds = Rect()
        labelPaint.getTextBounds(text, 0, text.length, textBounds)
        val bgRect = RectF(box.left, box.top - textBounds.height() - 12f,
            box.left + textBounds.width() + 16f, box.top)
        labelBgPaint.color = color; labelBgPaint.alpha = 160
        canvas.drawRect(bgRect, labelBgPaint)
        canvas.drawText(text, box.left + 4f, box.top - 6f, labelPaint)
    }

    private fun drawEventBanner(canvas: Canvas, text: String, frameWidth: Float) {
        val textBounds = Rect()
        bannerTextPaint.getTextBounds(text, 0, text.length, textBounds)
        canvas.drawRect(0f, 0f, frameWidth, textBounds.height() + 24f, bannerBgPaint)
        canvas.drawText(text, 12f, textBounds.height() + 8f, bannerTextPaint)
    }

    private fun pruneOldSessions(base: File) {
        val sessions = base.listFiles { f -> f.isDirectory && f.name.startsWith("session_") }
            ?.sortedBy { it.lastModified() } ?: return
        if (sessions.size > MAX_SESSIONS) {
            sessions.take(sessions.size - MAX_SESSIONS).forEach { it.deleteRecursively() }
        }
    }

    fun clearAll() {
        endSession()
        baseDir?.listFiles()?.forEach { it.deleteRecursively() }
        Log.d(TAG, "Cleared all debug sessions")
    }
}
