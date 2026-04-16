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
 * Model: MobileNetV3 Large — ~10MB, ~8ms per crop.
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
     * Result of a single segmentation pass that provides both embedding and masked crop.
     * Avoids running the segmenter twice when both signals are needed.
     */
    data class EmbedResult(
        val embedding: FloatArray?,
        /** Masked crop bitmap — caller must recycle when done. Null if segmentation failed. */
        val maskedCrop: Bitmap?
    )

    /**
     * Extract an embedding using segmentation masking. Returns null if segmentation
     * fails — caller should NOT match against unmasked embeddings since they include
     * too much background noise for reliable identity discrimination.
     */
    fun embed(bitmap: Bitmap, normalizedBox: RectF): FloatArray? {
        return embedAndCrop(bitmap, normalizedBox, fallback = false).also { it.maskedCrop?.recycle() }.embedding
    }

    /**
     * Extract an embedding with raw crop fallback. Used at lock time or during
     * gallery accumulation when we need at least some embedding even if segmentation fails.
     */
    fun embedWithFallback(bitmap: Bitmap, normalizedBox: RectF): FloatArray? {
        return embedAndCrop(bitmap, normalizedBox, fallback = true).also { it.maskedCrop?.recycle() }.embedding
    }

    /**
     * Single segmentation pass that returns both the embedding and the masked crop.
     * When [fallback] is true, falls back to raw crop if segmentation fails (used at lock time).
     * Caller must recycle [EmbedResult.maskedCrop] when done.
     */
    fun embedAndCrop(bitmap: Bitmap, normalizedBox: RectF, fallback: Boolean = false): EmbedResult {
        val segmented: Bitmap? = segmenter.segmentAndCrop(bitmap, normalizedBox)
        val crop: Bitmap = segmented
            ?: (if (fallback) cropNormalized(bitmap, normalizedBox) else null)
            ?: return EmbedResult(null, null)
        val embedding = try {
            embedBitmap(crop)
        } catch (e: Exception) {
            Log.w(TAG, "Embed failed: ${e.message}")
            null
        }
        // Return the segmented crop for histogram use (not the fallback raw crop).
        // If segmentation failed and we used a raw crop fallback, recycle it and return null maskedCrop.
        if (segmented == null) crop.recycle()
        return EmbedResult(embedding, segmented)
    }

    /**
     * Result of augmented embedding: the embeddings plus the masked crop for histogram use.
     */
    data class AugmentedResult(
        val embeddings: List<FloatArray>,
        /** Masked crop bitmap — caller must recycle when done. Null if segmentation failed. */
        val maskedCrop: Bitmap?
    )

    /**
     * Embed the crop plus augmented variants (rotated 90°/180°/270°, flipped).
     * Uses the segmenter to mask out background pixels before embedding.
     * Returns embeddings covering multiple orientations plus the masked crop for histogram use.
     * Single segmentation pass — avoids calling the segmenter twice.
     * Caller must recycle [AugmentedResult.maskedCrop] when done.
     */
    fun embedWithAugmentations(bitmap: Bitmap, normalizedBox: RectF): AugmentedResult {
        val segmented = segmenter.segmentAndCrop(bitmap, normalizedBox)
        val crop = segmented ?: cropNormalized(bitmap, normalizedBox) ?: return AugmentedResult(emptyList(), null)
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
        }
        // Don't recycle crop here — if it's the segmented one, return it for histogram use
        if (segmented == null) crop.recycle()

        Log.d(TAG, "Generated ${results.size} augmented embeddings")
        return AugmentedResult(results, segmented)
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

    fun shutdown() {
        embedder.close()
        segmenter.shutdown()
    }
}
