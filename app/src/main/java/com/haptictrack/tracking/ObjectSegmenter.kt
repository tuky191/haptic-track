package com.haptictrack.tracking

import android.content.Context
import android.graphics.Bitmap
import android.graphics.Color
import android.graphics.RectF
import android.util.Log
import com.google.mediapipe.framework.image.BitmapImageBuilder
import com.google.mediapipe.framework.image.ByteBufferExtractor
import com.google.mediapipe.tasks.components.containers.NormalizedKeypoint
import com.google.mediapipe.tasks.core.BaseOptions
import com.google.mediapipe.tasks.vision.interactivesegmenter.InteractiveSegmenter

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
        private const val MASK_THRESHOLD = 0.5f
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
        return try {
            // Ensure ARGB_8888 for MediaPipe
            val inputBitmap = if (bitmap.config != Bitmap.Config.ARGB_8888) {
                bitmap.copy(Bitmap.Config.ARGB_8888, false)
            } else bitmap

            val mpImage = BitmapImageBuilder(inputBitmap).build()

            // Use bounding box center as the keypoint (pixel coordinates)
            val centerX = (normalizedBox.left + normalizedBox.right) / 2f * inputBitmap.width
            val centerY = (normalizedBox.top + normalizedBox.bottom) / 2f * inputBitmap.height
            val roi = InteractiveSegmenter.RegionOfInterest.create(
                NormalizedKeypoint.create(centerX, centerY)
            )

            val result = segmenter.segment(mpImage, roi)

            if (inputBitmap !== bitmap) inputBitmap.recycle()

            // Try confidence masks first, fall back to category mask
            val maskPixels = extractConfidenceMask(result)
                ?: extractCategoryMask(result)

            if (maskPixels == null) {
                Log.w(TAG, "No masks returned (conf=${result.confidenceMasks().isPresent} cat=${result.categoryMask().isPresent})")
                return null
            }

            val maskWidth = maskPixels.first
            val maskHeight = maskPixels.second
            val pixels = maskPixels.third

            // Crop coordinates (same logic as AppearanceEmbedder.cropBitmap)
            val imgW = bitmap.width
            val imgH = bitmap.height
            val left = (normalizedBox.left * imgW).toInt().coerceIn(0, imgW - 1)
            val top = (normalizedBox.top * imgH).toInt().coerceIn(0, imgH - 1)
            val right = (normalizedBox.right * imgW).toInt().coerceIn(left + 1, imgW)
            val bottom = (normalizedBox.bottom * imgH).toInt().coerceIn(top + 1, imgH)
            val cropW = right - left
            val cropH = bottom - top
            if (cropW <= 0 || cropH <= 0) return null

            // Create masked crop: only foreground pixels survive
            val maskedCrop = Bitmap.createBitmap(cropW, cropH, Bitmap.Config.ARGB_8888)
            var fgCount = 0
            val totalPixels = cropW * cropH
            for (y in 0 until cropH) {
                for (x in 0 until cropW) {
                    val srcX = left + x
                    val srcY = top + y

                    // Map source pixel to mask coordinates (mask may differ from bitmap size)
                    val maskX = (srcX.toFloat() / imgW * maskWidth).toInt().coerceIn(0, maskWidth - 1)
                    val maskY = (srcY.toFloat() / imgH * maskHeight).toInt().coerceIn(0, maskHeight - 1)
                    val confidence = pixels[maskY * maskWidth + maskX]

                    if (confidence >= MASK_THRESHOLD) {
                        maskedCrop.setPixel(x, y, bitmap.getPixel(srcX, srcY))
                        fgCount++
                    } else {
                        maskedCrop.setPixel(x, y, Color.BLACK)
                    }
                }
            }

            val fgPct = if (totalPixels > 0) fgCount * 100 / totalPixels else 0
            Log.d(TAG, "Mask: ${fgCount}/${totalPixels} fg pixels (${fgPct}%) maskSize=${maskWidth}x${maskHeight} crop=${cropW}x${cropH}")

            // If mask is nearly empty or nearly full, it's not useful — return null to fall back to raw crop
            if (fgPct < 5 || fgPct > 95) {
                Log.d(TAG, "Mask not useful (${fgPct}%), falling back to raw crop")
                maskedCrop.recycle()
                return null
            }

            maskedCrop
        } catch (e: Exception) {
            Log.w(TAG, "Segmentation failed: ${e.message}")
            null
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

    fun shutdown() {
        segmenter.close()
    }
}
