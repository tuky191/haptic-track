package com.haptictrack.tracking

import android.content.Context
import android.graphics.Bitmap
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
        private const val MODEL_PATH = "mobilenet_v3_small_075_224_embedder.tflite"
    }

    private val embedder: ImageEmbedder

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
     * Extract an embedding from a crop of the given bitmap at the normalized bounding box.
     * Returns null if the crop is invalid.
     */
    fun embed(bitmap: Bitmap, normalizedBox: RectF): FloatArray? {
        val crop = cropBitmap(bitmap, normalizedBox) ?: return null
        return try {
            val mpImage = BitmapImageBuilder(crop).build()
            val result = embedder.embed(mpImage)
            val embedding = result.embeddingResult().embeddings().firstOrNull()
            val raw = embedding?.floatEmbedding()
            if (raw != null && raw.isNotEmpty()) raw else null
        } catch (e: Exception) {
            Log.w(TAG, "Embedding failed: ${e.message}")
            null
        } finally {
            crop.recycle()
        }
    }

    /**
     * Cosine similarity between two embeddings. Returns value in [-1, 1].
     * Since embeddings are L2-normalized, this is just the dot product.
     */
    fun cosineSimilarity(a: FloatArray, b: FloatArray): Float {
        if (a.size != b.size) return 0f
        var dot = 0f
        for (i in a.indices) {
            dot += a[i] * b[i]
        }
        return dot
    }

    fun shutdown() {
        embedder.close()
    }

    private fun cropBitmap(bitmap: Bitmap, normalizedBox: RectF): Bitmap? {
        val imgW = bitmap.width
        val imgH = bitmap.height

        val left = (normalizedBox.left * imgW).toInt().coerceIn(0, imgW - 1)
        val top = (normalizedBox.top * imgH).toInt().coerceIn(0, imgH - 1)
        val right = (normalizedBox.right * imgW).toInt().coerceIn(left + 1, imgW)
        val bottom = (normalizedBox.bottom * imgH).toInt().coerceIn(top + 1, imgH)

        val cropW = right - left
        val cropH = bottom - top
        if (cropW <= 0 || cropH <= 0) return null

        return try {
            Bitmap.createBitmap(bitmap, left, top, cropW, cropH)
        } catch (e: Exception) {
            Log.w(TAG, "Crop failed: ${e.message}")
            null
        }
    }
}
