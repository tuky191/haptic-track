package com.haptictrack.tracking

import android.content.Context
import android.graphics.Bitmap
import android.graphics.Matrix
import android.graphics.RectF
import android.util.Log
import com.google.mediapipe.framework.image.BitmapImageBuilder
import com.google.mediapipe.tasks.components.containers.Embedding
import com.google.mediapipe.tasks.core.BaseOptions
import com.google.mediapipe.tasks.vision.imageembedder.ImageEmbedder

/**
 * Wraps MediaPipe Image Embedder to extract visual feature vectors from object crops.
 *
 * Used to distinguish "which cup" vs "a cup" — at lock time we store the locked
 * object's embedding, then during re-acquisition we compare candidates by cosine
 * similarity against the stored reference.
 *
 * Model: MobileNetV3 Small (075, 224) — ~4MB, ~4ms per crop.
 */
class AppearanceEmbedder(context: Context) {

    companion object {
        private const val TAG = "AppearEmbed"
        private const val MODEL_PATH = "mobilenet_v3_large_embedder.tflite"
    }

    private val embedder: ImageEmbedder
    private val segmenter: ObjectSegmenter = ObjectSegmenter(context)

    init {
        val baseOptions = BaseOptions.builder()
            .setModelAssetPath(MODEL_PATH)
            .build()

        val options = ImageEmbedder.ImageEmbedderOptions.builder()
            .setBaseOptions(baseOptions)
            .setL2Normalize(true)
            .setQuantize(false)
            .build()

        embedder = ImageEmbedder.createFromOptions(context, options)
    }

    /**
     * Extract an embedding using segmentation masking. Returns null if segmentation
     * fails — caller should NOT match against unmasked embeddings since they include
     * too much background noise for reliable identity discrimination.
     */
    fun embed(bitmap: Bitmap, normalizedBox: RectF): FloatArray? {
        val crop = segmenter.segmentAndCrop(bitmap, normalizedBox) ?: return null
        return try {
            embedBitmap(crop)
        } finally {
            crop.recycle()
        }
    }

    /**
     * Extract an embedding with raw crop fallback. Used at lock time when we need
     * at least some embedding in the gallery even if segmentation fails.
     */
    fun embedWithFallback(bitmap: Bitmap, normalizedBox: RectF): FloatArray? {
        val crop = segmenter.segmentAndCrop(bitmap, normalizedBox)
            ?: cropNormalized(bitmap, normalizedBox)
            ?: return null
        return try {
            embedBitmap(crop)
        } finally {
            crop.recycle()
        }
    }

    /**
     * Embed the crop plus augmented variants (rotated 90°/180°/270°, flipped).
     * Uses the segmenter to mask out background pixels before embedding.
     * Returns a list of embeddings covering multiple orientations.
     */
    fun embedWithAugmentations(bitmap: Bitmap, normalizedBox: RectF): List<FloatArray> {
        val crop = segmenter.segmentAndCrop(bitmap, normalizedBox)
            ?: cropNormalized(bitmap, normalizedBox)
            ?: return emptyList()
        val results = mutableListOf<FloatArray>()

        try {
            // Original
            embedBitmap(crop)?.let { results.add(it) }

            // Rotated 90°, 180°, 270°
            for (degrees in listOf(90f, 180f, 270f)) {
                val rotated = rotateBitmap(crop, degrees)
                embedBitmap(rotated)?.let { results.add(it) }
                if (rotated !== crop) rotated.recycle()
            }

            // Horizontal flip
            val flipped = flipBitmap(crop)
            embedBitmap(flipped)?.let { results.add(it) }
            flipped.recycle()
        } catch (e: Exception) {
            Log.w(TAG, "Augmented embedding failed: ${e.message}")
        } finally {
            crop.recycle()
        }

        Log.d(TAG, "Generated ${results.size} augmented embeddings")
        return results
    }

    private fun embedBitmap(bitmap: Bitmap): FloatArray? {
        return try {
            val mpImage = BitmapImageBuilder(bitmap).build()
            val result = embedder.embed(mpImage)
            val embedding = result.embeddingResult().embeddings().firstOrNull()
            val raw = embedding?.floatEmbedding()
            if (raw != null && raw.isNotEmpty()) raw else null
        } catch (e: Exception) {
            Log.w(TAG, "Embed failed: ${e.message}")
            null
        }
    }

    private fun rotateBitmap(bitmap: Bitmap, degrees: Float): Bitmap {
        val matrix = Matrix().apply { postRotate(degrees) }
        return Bitmap.createBitmap(bitmap, 0, 0, bitmap.width, bitmap.height, matrix, true)
    }

    private fun flipBitmap(bitmap: Bitmap): Bitmap {
        val matrix = Matrix().apply { preScale(-1f, 1f) }
        return Bitmap.createBitmap(bitmap, 0, 0, bitmap.width, bitmap.height, matrix, true)
    }

    /**
     * Extract the object contour as normalized [0,1] points for UI overlay.
     */
    fun extractContour(bitmap: Bitmap, normalizedBox: RectF): List<android.graphics.PointF> {
        return segmenter.extractContour(bitmap, normalizedBox)
    }

    /**
     * Get the segmented (masked) crop for color histogram computation.
     * Returns null if segmentation fails. Caller must recycle the bitmap.
     */
    fun getMaskedCrop(bitmap: Bitmap, normalizedBox: RectF): Bitmap? {
        return segmenter.segmentAndCrop(bitmap, normalizedBox)
    }

    fun shutdown() {
        embedder.close()
        segmenter.shutdown()
    }
}
