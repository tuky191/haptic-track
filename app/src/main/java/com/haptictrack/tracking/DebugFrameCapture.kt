package com.haptictrack.tracking

import android.content.Context
import android.graphics.*
import android.util.Log
import java.io.File
import java.io.FileOutputStream
import java.time.LocalTime
import java.time.format.DateTimeFormatter

/**
 * Saves camera frames with bounding box overlays on tracking events.
 *
 * Dumps annotated PNGs to app-specific external storage:
 *   /sdcard/Android/data/com.haptictrack/files/debug_frames/
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
        private const val MAX_FILES = 200
    }

    private val outputDir: File? = context.getExternalFilesDir(null)?.let {
        File(it, DIR_NAME).apply { mkdirs() }
    }

    private val dateFormat = DateTimeFormatter.ofPattern("HHmmss_SSS")

    private val lockedPaint = Paint().apply {
        color = Color.GREEN
        style = Paint.Style.STROKE
        strokeWidth = 4f
    }

    private val candidatePaint = Paint().apply {
        color = Color.WHITE
        style = Paint.Style.STROKE
        strokeWidth = 2f
        alpha = 180
    }

    private val reacquiredPaint = Paint().apply {
        color = Color.CYAN
        style = Paint.Style.STROKE
        strokeWidth = 4f
    }

    private val lostPaint = Paint().apply {
        color = Color.RED
        style = Paint.Style.STROKE
        strokeWidth = 3f
        pathEffect = DashPathEffect(floatArrayOf(10f, 10f), 0f)
    }

    private val labelPaint = Paint().apply {
        color = Color.WHITE
        textSize = 28f
        isAntiAlias = true
    }

    private val labelBgPaint = Paint().apply {
        color = Color.BLACK
        alpha = 160
    }

    private val bannerBgPaint = Paint().apply {
        color = Color.BLACK
        alpha = 200
    }

    private val bannerTextPaint = Paint().apply {
        color = Color.YELLOW
        textSize = 36f
        isAntiAlias = true
        typeface = Typeface.DEFAULT_BOLD
    }

    /**
     * Capture a frame on a tracking event.
     *
     * @param event Event name: LOCK, LOST, REACQUIRE, SEARCH, TIMEOUT
     * @param bitmap The raw camera frame
     * @param detections All detected objects in this frame
     * @param lockedObject The currently locked/re-acquired object, if any
     * @param lastKnownBox The last known box of the lost object, if searching
     * @param extraInfo Additional text to overlay (e.g. similarity scores)
     */
    fun capture(
        event: DebugEvent,
        bitmap: Bitmap,
        detections: List<TrackedObject>,
        lockedObject: TrackedObject? = null,
        lastKnownBox: RectF? = null,
        extraInfo: String? = null
    ) {
        val dir = outputDir ?: return

        // Draw annotations onto a mutable copy. The caller (ObjectTracker)
        // has already saved lastFrameBitmap, so we don't need a third bitmap.
        val annotated = if (bitmap.isMutable) bitmap
            else bitmap.copy(Bitmap.Config.ARGB_8888, true) ?: return
        val canvas = Canvas(annotated)
        val w = bitmap.width.toFloat()
        val h = bitmap.height.toFloat()

        // Draw all detections
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

        // Draw last known box if searching
        if (lastKnownBox != null && (event == DebugEvent.LOST || event == DebugEvent.SEARCH)) {
            canvas.drawRect(toPixelRect(lastKnownBox, w, h), lostPaint)
        }

        // Draw event label top-left
        val eventLabel = "$event${if (extraInfo != null) " | $extraInfo" else ""}"
        drawEventBanner(canvas, eventLabel, w)

        // Save
        val timestamp = LocalTime.now().format(dateFormat)
        val filename = "${timestamp}_${event}.png"
        val file = File(dir, filename)

        try {
            FileOutputStream(file).use { out ->
                annotated.compress(Bitmap.CompressFormat.PNG, 90, out)
            }
            Log.d(TAG, "Saved $filename (${detections.size} detections)")
        } catch (e: Exception) {
            Log.w(TAG, "Failed to save debug frame: ${e.message}")
        } finally {
            // Only recycle if we allocated the copy ourselves
            if (annotated !== bitmap) annotated.recycle()
        }

        pruneOldFiles(dir)
    }

    private fun toPixelRect(normalized: RectF, w: Float, h: Float): RectF {
        return RectF(
            normalized.left * w,
            normalized.top * h,
            normalized.right * w,
            normalized.bottom * h
        )
    }

    private fun drawLabel(canvas: Canvas, obj: TrackedObject, box: RectF, color: Int) {
        val text = buildString {
            append("#${obj.id}")
            if (obj.label != null) append(" ${obj.label}")
            append(" ${(obj.confidence * 100).toInt()}%")
        }

        val textBounds = Rect()
        labelPaint.getTextBounds(text, 0, text.length, textBounds)

        val bgRect = RectF(
            box.left,
            box.top - textBounds.height() - 12f,
            box.left + textBounds.width() + 16f,
            box.top
        )
        labelBgPaint.color = color
        labelBgPaint.alpha = 160
        canvas.drawRect(bgRect, labelBgPaint)
        canvas.drawText(text, box.left + 4f, box.top - 6f, labelPaint)
    }

    private fun drawEventBanner(canvas: Canvas, text: String, frameWidth: Float) {
        val textBounds = Rect()
        bannerTextPaint.getTextBounds(text, 0, text.length, textBounds)
        canvas.drawRect(0f, 0f, frameWidth, textBounds.height() + 24f, bannerBgPaint)
        canvas.drawText(text, 12f, textBounds.height() + 8f, bannerTextPaint)
    }

    private fun pruneOldFiles(dir: File) {
        val files = dir.listFiles()?.sortedBy { it.lastModified() } ?: return
        if (files.size > MAX_FILES) {
            files.take(files.size - MAX_FILES).forEach { it.delete() }
        }
    }

    fun clearAll() {
        outputDir?.listFiles()?.forEach { it.delete() }
        Log.d(TAG, "Cleared all debug frames")
    }
}
