package com.haptictrack.tracking

import android.content.Context
import android.graphics.Bitmap
import android.graphics.Color
import android.graphics.PointF
import android.graphics.RectF
import android.util.Log
import org.opencv.core.CvType
import org.opencv.core.Mat
import org.opencv.core.MatOfPoint
import org.opencv.core.MatOfPoint2f
import org.opencv.imgproc.Imgproc
import org.tensorflow.lite.Interpreter
import java.nio.ByteBuffer
import java.nio.ByteOrder

/**
 * Segments an object from its background using the magic_touch model via raw TFLite.
 *
 * Given a [CanonicalCrop] at [MODEL_SIZE]² (already aspect-preserved-letterboxed),
 * produces a binary foreground mask in canonical pixel space and applies it to
 * yield a masked bitmap. Pre-#100 callers used [segmentAndCrop] which built its
 * own stretch-to-square crop; that wrapper now delegates to the canonical path
 * so the model sees uniformly-shaped input regardless of bbox aspect.
 *
 * Model: magic_touch.tflite (~6MB).
 * Input:  [1, 512, 512, 4] float32 — RGB (0-1) + keypoint channel (1.0 at ROI center)
 * Output: [1, 512, 512, 1] float32 — confidence mask (0-1)
 *
 * Runs via raw TFLite GPU delegate (bypasses MediaPipe's broken GPU mask readback).
 */
class ObjectSegmenter(
    context: Context,
    private val cropper: CanonicalCropper = CanonicalCropper(),
) {

    companion object {
        private const val TAG = "ObjSegmenter"
        private const val MODEL_ASSET = "magic_touch.tflite"
        const val MODEL_SIZE = 512
        /** Confidence threshold: pixels below this are considered background. */
        private const val MASK_THRESHOLD = 0.8f
        /** Skip segmentation if the bbox covers more than this fraction of the frame. */
        private const val MAX_BBOX_FRACTION = 0.5f
        /** Minimum / maximum acceptable foreground percentage of a useful mask. */
        private const val MIN_USEFUL_FG_PCT = 5
        private const val MAX_USEFUL_FG_PCT = 95
        /** Radius (in model pixels) for the keypoint indicator blob. */
        private const val KEYPOINT_RADIUS = 8
    }

    private val gpu: GpuInterpreter
    private val interpreter: Interpreter get() = gpu.interpreter

    // Pre-allocated buffers
    private val inputBuffer: ByteBuffer = ByteBuffer.allocateDirect(4 * MODEL_SIZE * MODEL_SIZE * 4).apply {
        order(ByteOrder.nativeOrder())
    }
    private val outputBuffer: ByteBuffer = ByteBuffer.allocateDirect(4 * MODEL_SIZE * MODEL_SIZE * 1).apply {
        order(ByteOrder.nativeOrder())
    }

    init {
        val model = loadTfliteModel(context, MODEL_ASSET)
        gpu = createGpuInterpreter(model, modelName = "magic_touch-seg")
        Log.i(TAG, "Loaded magic_touch segmenter (${MODEL_SIZE}x${MODEL_SIZE}, raw TFLite)")
    }

    /**
     * Segment the object inside [canonical] (must be [MODEL_SIZE]²) and apply
     * the mask. Returns a freshly-allocated [MODEL_SIZE]² bitmap with non-foreground
     * pixels zeroed, or null when the bbox is too large, when the model is
     * unhappy with the mask quality (foreground % out of range), or on error.
     *
     * Caller owns [canonical] (no recycle here) and the returned bitmap.
     */
    @Synchronized
    fun segmentCanonical(canonical: CanonicalCrop): Bitmap? {
        if (canonical.targetWidth != MODEL_SIZE || canonical.targetHeight != MODEL_SIZE) {
            Log.w(TAG, "Canonical dims ${canonical.targetWidth}×${canonical.targetHeight} != expected ${MODEL_SIZE}²")
            return null
        }
        // Skip segmentation for crops covering most of the frame — same guard
        // as the legacy path, since a >50% bbox usually means we're already
        // pointing at the subject directly.
        val srcBox = canonical.sourceBoxNormalized
        val frac = (srcBox.right - srcBox.left) * (srcBox.bottom - srcBox.top)
        if (frac > MAX_BBOX_FRACTION) return null

        return try {
            fillInputBuffer(canonical.bitmap)

            val t0 = android.os.SystemClock.elapsedRealtimeNanos()
            outputBuffer.rewind()
            interpreter.run(inputBuffer, outputBuffer)
            val segMs = (android.os.SystemClock.elapsedRealtimeNanos() - t0) / 1_000_000f

            outputBuffer.rewind()
            val mask = FloatArray(MODEL_SIZE * MODEL_SIZE)
            outputBuffer.asFloatBuffer().get(mask)

            // Apply mask in canonical space — no remap needed (bitmap is at MODEL_SIZE²).
            val totalPixels = MODEL_SIZE * MODEL_SIZE
            val srcPixels = IntArray(totalPixels)
            canonical.bitmap.getPixels(srcPixels, 0, MODEL_SIZE, 0, 0, MODEL_SIZE, MODEL_SIZE)

            var fgCount = 0
            val black = Color.BLACK
            for (i in 0 until totalPixels) {
                if (mask[i] >= MASK_THRESHOLD) {
                    fgCount++
                } else {
                    srcPixels[i] = black
                }
            }

            val fgPct = if (totalPixels > 0) fgCount * 100 / totalPixels else 0
            Log.d(TAG, "Mask: ${fgCount}/${totalPixels} fg pixels (${fgPct}%) ${String.format("%.0f", segMs)}ms")

            if (fgPct < MIN_USEFUL_FG_PCT || fgPct > MAX_USEFUL_FG_PCT) {
                Log.d(TAG, "Mask not useful (${fgPct}%), falling back to raw crop")
                return null
            }

            val masked = Bitmap.createBitmap(MODEL_SIZE, MODEL_SIZE, Bitmap.Config.ARGB_8888)
            masked.setPixels(srcPixels, 0, MODEL_SIZE, 0, 0, MODEL_SIZE, MODEL_SIZE)
            masked
        } catch (e: Exception) {
            Log.w(TAG, "Segmentation failed: ${e.message}")
            null
        }
    }

    /**
     * Legacy entrypoint — builds a canonical, segments, and crops the masked
     * canonical back to source-bbox aspect (matching the historical contract
     * that callers expect for color-histogram + MNV3 input). Returned bitmap's
     * dimensions are the crop's *post-pad* source pixel size; pixels are
     * masked with [Color.BLACK] outside the foreground.
     */
    fun segmentAndCrop(bitmap: Bitmap, normalizedBox: RectF): Bitmap? {
        val canonical = cropper.prepare(
            bitmap, normalizedBox,
            targetWidth = MODEL_SIZE, targetHeight = MODEL_SIZE,
        ) ?: return null
        val pad = canonical.padding
        val drawW = canonical.drawWidth
        val drawH = canonical.drawHeight
        val masked = try {
            segmentCanonical(canonical)
        } finally {
            canonical.bitmap.recycle()
        } ?: return null

        return try {
            if (drawW <= 0 || drawH <= 0) null
            // createBitmap on a sub-region copies pixels into a new buffer
            // (Android API contract), so recycling `masked` in the finally
            // block doesn't invalidate the returned bitmap.
            else Bitmap.createBitmap(masked, pad.left, pad.top, drawW, drawH)
        } catch (e: Exception) {
            Log.w(TAG, "Crop after mask failed: ${e.message}")
            null
        } finally {
            masked.recycle()
        }
    }

    /**
     * Fill the input buffer with RGB [0,1] + keypoint channel.
     * Channel 4 is 1.0 in a small circle at the center, 0.0 elsewhere.
     */
    private fun fillInputBuffer(bitmap: Bitmap) {
        inputBuffer.rewind()
        val pixels = IntArray(MODEL_SIZE * MODEL_SIZE)
        bitmap.getPixels(pixels, 0, MODEL_SIZE, 0, 0, MODEL_SIZE, MODEL_SIZE)

        val cx = MODEL_SIZE / 2
        val cy = MODEL_SIZE / 2
        val radiusSq = KEYPOINT_RADIUS * KEYPOINT_RADIUS

        for (i in pixels.indices) {
            val y = i / MODEL_SIZE
            val x = i % MODEL_SIZE
            val pixel = pixels[i]

            inputBuffer.putFloat(Color.red(pixel) / 255f)
            inputBuffer.putFloat(Color.green(pixel) / 255f)
            inputBuffer.putFloat(Color.blue(pixel) / 255f)

            val dx = x - cx
            val dy = y - cy
            inputBuffer.putFloat(if (dx * dx + dy * dy <= radiusSq) 1f else 0f)
        }
        inputBuffer.rewind()
    }

    /**
     * Extract the object contour as normalized [0,1] points in the full source
     * frame's coordinate space. Uses the segmentation mask + OpenCV
     * findContours + approxPolyDP for a smooth outline.
     */
    @Synchronized
    fun extractContour(bitmap: Bitmap, normalizedBox: RectF): List<PointF> {
        val canonical = cropper.prepare(
            bitmap, normalizedBox,
            targetWidth = MODEL_SIZE, targetHeight = MODEL_SIZE,
        ) ?: return emptyList()
        return try {
            extractContourFromCanonical(canonical, bitmap.width, bitmap.height)
        } finally {
            canonical.bitmap.recycle()
        }
    }

    private fun extractContourFromCanonical(
        canonical: CanonicalCrop,
        fullW: Int,
        fullH: Int,
    ): List<PointF> {
        return try {
            fillInputBuffer(canonical.bitmap)
            outputBuffer.rewind()
            interpreter.run(inputBuffer, outputBuffer)

            outputBuffer.rewind()
            val mask = FloatArray(MODEL_SIZE * MODEL_SIZE)
            outputBuffer.asFloatBuffer().get(mask)

            val maskMat = Mat(MODEL_SIZE, MODEL_SIZE, CvType.CV_8UC1)
            val maskBytes = ByteArray(MODEL_SIZE * MODEL_SIZE)
            for (i in mask.indices) {
                maskBytes[i] = if (mask[i] >= MASK_THRESHOLD) 255.toByte() else 0
            }
            maskMat.put(0, 0, maskBytes)

            val kernel = Imgproc.getStructuringElement(Imgproc.MORPH_ELLIPSE, org.opencv.core.Size(15.0, 15.0))
            Imgproc.erode(maskMat, maskMat, kernel)
            kernel.release()

            val contours = mutableListOf<MatOfPoint>()
            val hierarchy = Mat()
            Imgproc.findContours(maskMat, contours, hierarchy, Imgproc.RETR_EXTERNAL, Imgproc.CHAIN_APPROX_SIMPLE)
            maskMat.release()
            hierarchy.release()

            if (contours.isEmpty()) return emptyList()

            val largest = contours.maxByOrNull { Imgproc.contourArea(it) } ?: return emptyList()

            val contour2f = MatOfPoint2f(*largest.toArray())
            val epsilon = Imgproc.arcLength(contour2f, true) * 0.003
            val approx = MatOfPoint2f()
            Imgproc.approxPolyDP(contour2f, approx, epsilon, true)

            val pad = canonical.padding
            val drawW = canonical.drawWidth.toFloat()
            val drawH = canonical.drawHeight.toFloat()
            val sc = canonical.sourceCropPx

            val points = approx.toArray().map { pt ->
                // Mask coord == canonical coord (bitmap is exactly MODEL_SIZE²).
                val cx = pt.x.toFloat()
                val cy = pt.y.toFloat()
                // Within the rendered source region (drawDest in target pixels)?
                val dx = ((cx - pad.left) / drawW).coerceIn(0f, 1f)
                val dy = ((cy - pad.top) / drawH).coerceIn(0f, 1f)
                // Source-pixel coords inside the cropped (post-pad) region.
                val sxPx = sc.left + dx * sc.width()
                val syPx = sc.top + dy * sc.height()
                PointF(sxPx / fullW, syPx / fullH)
            }

            contours.forEach { it.release() }
            contour2f.release()
            approx.release()

            Log.d(TAG, "Contour: ${points.size} points")
            points
        } catch (e: Exception) {
            Log.w(TAG, "Contour extraction failed: ${e.message}")
            emptyList()
        }
    }

    fun shutdown() {
        gpu.close()
    }
}
