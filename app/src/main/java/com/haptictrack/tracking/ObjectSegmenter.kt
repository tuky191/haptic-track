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
 * Given a full frame and a bounding box, returns a cropped bitmap where background
 * pixels are zeroed out (transparent black). This ensures embeddings encode only the
 * object's appearance, not the surrounding context.
 *
 * Model: magic_touch.tflite (~6MB).
 * Input:  [1, 512, 512, 4] float32 — RGB (0-1) + keypoint channel (1.0 at ROI center)
 * Output: [1, 512, 512, 1] float32 — confidence mask (0-1)
 *
 * Runs via raw TFLite GPU delegate (bypasses MediaPipe's broken GPU mask readback).
 */
class ObjectSegmenter(context: Context) {

    companion object {
        private const val TAG = "ObjSegmenter"
        private const val MODEL_ASSET = "magic_touch.tflite"
        private const val MODEL_SIZE = 512
        /** Confidence threshold: pixels below this are considered background. */
        private const val MASK_THRESHOLD = 0.8f
        /** Max pixels for segmenter input. Large crops downscaled for speed. */
        private const val MAX_SEGMENT_PIXELS = 90_000  // ~300x300
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
     * Segment the object at [normalizedBox] from the full [bitmap], then crop.
     *
     * Returns a cropped bitmap where background pixels are zeroed, or null on failure.
     * The crop region matches [normalizedBox] (same as AppearanceEmbedder.cropBitmap).
     */
    @Synchronized
    fun segmentAndCrop(bitmap: Bitmap, normalizedBox: RectF): Bitmap? {
        var cropBitmap: Bitmap? = null
        return try {
            val imgW = bitmap.width
            val imgH = bitmap.height

            // Crop to bounding box with padding FIRST, then segment the crop.
            val pad = 0.05f
            val cl = ((normalizedBox.left - pad) * imgW).toInt().coerceIn(0, imgW - 1)
            val ct = ((normalizedBox.top - pad) * imgH).toInt().coerceIn(0, imgH - 1)
            val cr = ((normalizedBox.right + pad) * imgW).toInt().coerceIn(cl + 1, imgW)
            val cb = ((normalizedBox.bottom + pad) * imgH).toInt().coerceIn(ct + 1, imgH)
            val cw = cr - cl
            val ch = cb - ct
            if (cw < 10 || ch < 10) return null

            // Skip segmentation for crops that cover most of the frame
            val cropFraction = (cw.toFloat() * ch) / (imgW.toFloat() * imgH)
            if (cropFraction > 0.5f) return null

            cropBitmap = Bitmap.createBitmap(bitmap, cl, ct, cw, ch)
            val inputCrop = if (cropBitmap!!.config != Bitmap.Config.ARGB_8888) {
                val c = cropBitmap!!.copy(Bitmap.Config.ARGB_8888, false)
                cropBitmap!!.recycle()
                c.also { cropBitmap = it }
            } else cropBitmap!!

            // Resize to model input size
            val resized = Bitmap.createScaledBitmap(inputCrop, MODEL_SIZE, MODEL_SIZE, true)

            // Fill input buffer: RGB normalized to [0,1] + keypoint channel
            fillInputBuffer(resized)
            if (resized !== inputCrop) resized.recycle()

            val t0 = android.os.SystemClock.elapsedRealtimeNanos()
            outputBuffer.rewind()
            interpreter.run(inputBuffer, outputBuffer)
            val segMs = (android.os.SystemClock.elapsedRealtimeNanos() - t0) / 1_000_000f

            // Read mask from output buffer
            outputBuffer.rewind()
            val mask = FloatArray(MODEL_SIZE * MODEL_SIZE)
            outputBuffer.asFloatBuffer().get(mask)

            // Apply mask to the crop pixels (map model coords back to crop coords)
            val srcPixels = IntArray(cw * ch)
            inputCrop.getPixels(srcPixels, 0, cw, 0, 0, cw, ch)

            var fgCount = 0
            val totalPixels = cw * ch
            val black = Color.BLACK
            for (y in 0 until ch) {
                for (x in 0 until cw) {
                    val maskX = (x.toFloat() / cw * MODEL_SIZE).toInt().coerceIn(0, MODEL_SIZE - 1)
                    val maskY = (y.toFloat() / ch * MODEL_SIZE).toInt().coerceIn(0, MODEL_SIZE - 1)
                    if (mask[maskY * MODEL_SIZE + maskX] >= MASK_THRESHOLD) {
                        fgCount++
                    } else {
                        srcPixels[y * cw + x] = black
                    }
                }
            }

            val fgPct = if (totalPixels > 0) fgCount * 100 / totalPixels else 0
            Log.d(TAG, "Mask: ${fgCount}/${totalPixels} fg pixels (${fgPct}%) crop=${cw}x${ch} ${String.format("%.0f", segMs)}ms")

            if (fgPct < 5 || fgPct > 95) {
                Log.d(TAG, "Mask not useful (${fgPct}%), falling back to raw crop")
                return null
            }

            val maskedCrop = Bitmap.createBitmap(cw, ch, Bitmap.Config.ARGB_8888)
            maskedCrop.setPixels(srcPixels, 0, cw, 0, 0, cw, ch)
            maskedCrop
        } catch (e: Exception) {
            Log.w(TAG, "Segmentation failed: ${e.message}")
            null
        } finally {
            cropBitmap?.recycle()
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

            // Keypoint channel: 1.0 within radius of center
            val dx = x - cx
            val dy = y - cy
            inputBuffer.putFloat(if (dx * dx + dy * dy <= radiusSq) 1f else 0f)
        }
        inputBuffer.rewind()
    }

    /**
     * Extract the contour of the object at [normalizedBox] as normalized [0,1] points.
     * Uses the segmentation mask + OpenCV findContours + approxPolyDP for a smooth outline.
     * Returns an empty list on failure.
     */
    @Synchronized
    fun extractContour(bitmap: Bitmap, normalizedBox: RectF): List<PointF> {
        var crop: Bitmap? = null
        return try {
            val imgW = bitmap.width
            val imgH = bitmap.height
            val pad = 0.05f
            val cropLeft = ((normalizedBox.left - pad) * imgW).toInt().coerceIn(0, imgW - 1)
            val cropTop = ((normalizedBox.top - pad) * imgH).toInt().coerceIn(0, imgH - 1)
            val cropRight = ((normalizedBox.right + pad) * imgW).toInt().coerceIn(cropLeft + 1, imgW)
            val cropBottom = ((normalizedBox.bottom + pad) * imgH).toInt().coerceIn(cropTop + 1, imgH)
            val cropW = cropRight - cropLeft
            val cropH = cropBottom - cropTop
            if (cropW < 10 || cropH < 10) return emptyList()

            crop = Bitmap.createBitmap(bitmap, cropLeft, cropTop, cropW, cropH)
            val inputCrop = if (crop.config != Bitmap.Config.ARGB_8888) {
                val c = crop.copy(Bitmap.Config.ARGB_8888, false)
                crop.recycle()
                c.also { crop = it }
            } else crop!!

            // Resize to model input size
            val resized = Bitmap.createScaledBitmap(inputCrop, MODEL_SIZE, MODEL_SIZE, true)
            fillInputBuffer(resized)
            if (resized !== inputCrop) resized.recycle()

            outputBuffer.rewind()
            interpreter.run(inputBuffer, outputBuffer)

            outputBuffer.rewind()
            val mask = FloatArray(MODEL_SIZE * MODEL_SIZE)
            outputBuffer.asFloatBuffer().get(mask)

            // Build binary mask
            val maskMat = Mat(MODEL_SIZE, MODEL_SIZE, CvType.CV_8UC1)
            val maskBytes = ByteArray(MODEL_SIZE * MODEL_SIZE)
            for (i in mask.indices) {
                maskBytes[i] = if (mask[i] >= MASK_THRESHOLD) 255.toByte() else 0
            }
            maskMat.put(0, 0, maskBytes)

            // Heavy erosion to shrink mask well inside the object boundary.
            val kernel = Imgproc.getStructuringElement(Imgproc.MORPH_ELLIPSE, org.opencv.core.Size(15.0, 15.0))
            Imgproc.erode(maskMat, maskMat, kernel)
            kernel.release()

            // Find contours
            val contours = mutableListOf<MatOfPoint>()
            val hierarchy = Mat()
            Imgproc.findContours(maskMat, contours, hierarchy, Imgproc.RETR_EXTERNAL, Imgproc.CHAIN_APPROX_SIMPLE)
            maskMat.release()
            hierarchy.release()

            if (contours.isEmpty()) return emptyList()

            val largest = contours.maxByOrNull { Imgproc.contourArea(it) } ?: return emptyList()

            // Smooth contour
            val contour2f = MatOfPoint2f(*largest.toArray())
            val epsilon = Imgproc.arcLength(contour2f, true) * 0.003
            val approx = MatOfPoint2f()
            Imgproc.approxPolyDP(contour2f, approx, epsilon, true)

            // Convert mask-space points → full image normalized [0,1] coordinates
            val points = approx.toArray().map { pt ->
                val pixX = cropLeft + pt.x.toFloat() / MODEL_SIZE * cropW
                val pixY = cropTop + pt.y.toFloat() / MODEL_SIZE * cropH
                PointF(pixX / imgW, pixY / imgH)
            }

            contours.forEach { it.release() }
            contour2f.release()
            approx.release()

            Log.d(TAG, "Contour: ${points.size} points")
            points
        } catch (e: Exception) {
            Log.w(TAG, "Contour extraction failed: ${e.message}")
            emptyList()
        } finally {
            crop?.recycle()
        }
    }

    fun shutdown() {
        gpu.close()
    }
}
