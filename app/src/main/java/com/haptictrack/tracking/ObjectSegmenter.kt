package com.haptictrack.tracking

import android.content.Context
import android.graphics.Bitmap
import android.graphics.Color
import android.graphics.PointF
import android.graphics.RectF
import android.util.Log
import com.google.mediapipe.framework.image.BitmapImageBuilder
import com.google.mediapipe.framework.image.ByteBufferExtractor
import com.google.mediapipe.tasks.components.containers.NormalizedKeypoint
import com.google.mediapipe.tasks.core.BaseOptions
import com.google.mediapipe.tasks.vision.interactivesegmenter.InteractiveSegmenter
import org.opencv.core.CvType
import org.opencv.core.Mat
import org.opencv.core.MatOfPoint
import org.opencv.core.MatOfPoint2f
import org.opencv.imgproc.Imgproc

/**
 * Segments an object from its background using MediaPipe Interactive Segmenter.
 *
 * Given a full frame and a bounding box, returns a cropped bitmap where background
 * pixels are zeroed out (transparent black). This ensures embeddings encode only the
 * object's appearance, not the surrounding context — preventing false matches when
 * a large detection (e.g. "dining table") happens to contain the target object.
 *
 * Model: magic_touch.tflite (~6MB). Uses a keypoint (box center) as the ROI.
 */
class ObjectSegmenter(context: Context) {

    companion object {
        private const val TAG = "ObjSegmenter"
        private const val MODEL_PATH = "magic_touch.tflite"
        /** Confidence threshold: pixels below this are considered background. */
        private const val MASK_THRESHOLD = 0.85f
    }

    private val segmenter: InteractiveSegmenter

    init {
        val baseOptions = BaseOptions.builder()
            .setModelAssetPath(MODEL_PATH)
            .build()

        val options = InteractiveSegmenter.InteractiveSegmenterOptions.builder()
            .setBaseOptions(baseOptions)
            .setOutputConfidenceMasks(true)
            .setOutputCategoryMask(true)
            .build()

        segmenter = InteractiveSegmenter.createFromOptions(context, options)
    }

    /**
     * Segment the object at [normalizedBox] from the full [bitmap], then crop.
     *
     * Returns a cropped bitmap where background pixels are zeroed, or null on failure.
     * The crop region matches [normalizedBox] (same as AppearanceEmbedder.cropBitmap).
     */
    fun segmentAndCrop(bitmap: Bitmap, normalizedBox: RectF): Bitmap? {
        var convertedBitmap: Bitmap? = null
        return try {
            // Ensure ARGB_8888 for MediaPipe
            val inputBitmap = if (bitmap.config != Bitmap.Config.ARGB_8888) {
                bitmap.copy(Bitmap.Config.ARGB_8888, false).also { convertedBitmap = it }
            } else bitmap

            val mpImage = BitmapImageBuilder(inputBitmap).build()

            // Use bounding box center as the keypoint (pixel coordinates)
            val centerX = (normalizedBox.left + normalizedBox.right) / 2f * inputBitmap.width
            val centerY = (normalizedBox.top + normalizedBox.bottom) / 2f * inputBitmap.height
            val roi = InteractiveSegmenter.RegionOfInterest.create(
                NormalizedKeypoint.create(centerX, centerY)
            )

            val result = segmenter.segment(mpImage, roi)

            // Try confidence masks first, fall back to category mask
            val maskPixels = extractConfidenceMask(result)
                ?: extractCategoryMask(result)

            if (maskPixels == null) {
                Log.w(TAG, "No masks returned (conf=${result.confidenceMasks().isPresent} cat=${result.categoryMask().isPresent})")
                return null
            }

            val maskWidth = maskPixels.first
            val maskHeight = maskPixels.second
            val mask = maskPixels.third

            val imgW = bitmap.width
            val imgH = bitmap.height
            val coords = cropCoordinates(normalizedBox, imgW, imgH) ?: return null
            val left = coords[0]; val top = coords[1]; val right = coords[2]; val bottom = coords[3]
            val cropW = right - left
            val cropH = bottom - top

            // Bulk read source pixels
            val srcPixels = IntArray(cropW * cropH)
            bitmap.getPixels(srcPixels, 0, cropW, left, top, cropW, cropH)

            // Apply mask in bulk
            var fgCount = 0
            val totalPixels = cropW * cropH
            val black = Color.BLACK
            for (y in 0 until cropH) {
                for (x in 0 until cropW) {
                    val maskX = ((left + x).toFloat() / imgW * maskWidth).toInt().coerceIn(0, maskWidth - 1)
                    val maskY = ((top + y).toFloat() / imgH * maskHeight).toInt().coerceIn(0, maskHeight - 1)
                    if (mask[maskY * maskWidth + maskX] >= MASK_THRESHOLD) {
                        fgCount++
                    } else {
                        srcPixels[y * cropW + x] = black
                    }
                }
            }

            val fgPct = if (totalPixels > 0) fgCount * 100 / totalPixels else 0
            Log.d(TAG, "Mask: ${fgCount}/${totalPixels} fg pixels (${fgPct}%) maskSize=${maskWidth}x${maskHeight} crop=${cropW}x${cropH}")

            if (fgPct < 5 || fgPct > 95) {
                Log.d(TAG, "Mask not useful (${fgPct}%), falling back to raw crop")
                return null
            }

            // Bulk write masked pixels
            val maskedCrop = Bitmap.createBitmap(cropW, cropH, Bitmap.Config.ARGB_8888)
            maskedCrop.setPixels(srcPixels, 0, cropW, 0, 0, cropW, cropH)
            maskedCrop
        } catch (e: Exception) {
            Log.w(TAG, "Segmentation failed: ${e.message}")
            null
        } finally {
            convertedBitmap?.recycle()
        }
    }

    /** Extract foreground mask from confidence masks (float 0..1). */
    private fun extractConfidenceMask(
        result: com.google.mediapipe.tasks.vision.imagesegmenter.ImageSegmenterResult
    ): Triple<Int, Int, FloatArray>? {
        val masks = result.confidenceMasks()
        if (!masks.isPresent || masks.get().isEmpty()) return null

        val maskList = masks.get()
        val foregroundMask = if (maskList.size >= 2) maskList[1] else maskList[0]
        val w = foregroundMask.width
        val h = foregroundMask.height

        return try {
            val byteBuffer = ByteBufferExtractor.extract(foregroundMask)
            val floatBuffer = byteBuffer.asFloatBuffer()
            val pixels = FloatArray(w * h)
            floatBuffer.get(pixels)
            Triple(w, h, pixels)
        } catch (e: Exception) {
            Log.d(TAG, "Confidence mask extraction failed: ${e.message}")
            null
        }
    }

    /** Extract foreground mask from category mask (uint8: 0=bg, >0=fg) → convert to float. */
    private fun extractCategoryMask(
        result: com.google.mediapipe.tasks.vision.imagesegmenter.ImageSegmenterResult
    ): Triple<Int, Int, FloatArray>? {
        val mask = result.categoryMask()
        if (!mask.isPresent) return null

        val catMask = mask.get()
        val w = catMask.width
        val h = catMask.height

        return try {
            val byteBuffer = ByteBufferExtractor.extract(catMask)
            val pixels = FloatArray(w * h)
            for (i in pixels.indices) {
                // Category 0 = background, anything else = foreground
                pixels[i] = if ((byteBuffer.get(i).toInt() and 0xFF) > 0) 1f else 0f
            }
            Triple(w, h, pixels)
        } catch (e: Exception) {
            Log.d(TAG, "Category mask extraction failed: ${e.message}")
            null
        }
    }

    /**
     * Extract the contour of the object at [normalizedBox] as normalized [0,1] points.
     * Uses the segmentation mask + OpenCV findContours + approxPolyDP for a smooth outline.
     * Returns an empty list on failure.
     */
    fun extractContour(bitmap: Bitmap, normalizedBox: RectF): List<PointF> {
        var crop: Bitmap? = null
        return try {
            // Crop to bounding box with padding, then segment the crop.
            // This gives the segmenter a close-up view → much tighter mask.
            val imgW = bitmap.width
            val imgH = bitmap.height
            val pad = 0.05f // 5% padding around the box
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

            val mpImage = BitmapImageBuilder(inputCrop).build()

            // Keypoint at center of the crop
            val roi = InteractiveSegmenter.RegionOfInterest.create(
                NormalizedKeypoint.create(inputCrop.width / 2f, inputCrop.height / 2f)
            )

            val result = segmenter.segment(mpImage, roi)
            val maskData = extractConfidenceMask(result) ?: extractCategoryMask(result) ?: return emptyList()
            val maskW = maskData.first
            val maskH = maskData.second
            val pixels = maskData.third

            // Build binary mask
            val maskMat = Mat(maskH, maskW, CvType.CV_8UC1)
            val maskBytes = ByteArray(maskW * maskH)
            for (i in pixels.indices) {
                maskBytes[i] = if (pixels[i] >= MASK_THRESHOLD) 255.toByte() else 0
            }
            maskMat.put(0, 0, maskBytes)

            // Heavy erosion to shrink mask well inside the object boundary.
            // This ensures the glow sits on the object, not on the background.
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
                val pixX = cropLeft + pt.x.toFloat() / maskW * cropW
                val pixY = cropTop + pt.y.toFloat() / maskH * cropH
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
        segmenter.close()
    }
}
