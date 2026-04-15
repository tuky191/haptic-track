package com.haptictrack.tracking

import android.graphics.Bitmap
import android.graphics.RectF
import android.util.Log

/**
 * Shared utilities for embedding comparison and image cropping.
 */

/** Number of embeddings generated at lock time (original + 3 rotations + 1 flip). */
const val LOCK_AUGMENTATION_COUNT = 5

/**
 * Cosine similarity between two L2-normalized embeddings.
 * Since embeddings are L2-normalized, this is just the dot product.
 */
fun cosineSimilarity(a: FloatArray, b: FloatArray): Float {
    if (a.size != b.size) return 0f
    var dot = 0f
    for (i in a.indices) dot += a[i] * b[i]
    return dot
}

/**
 * Best cosine similarity between a candidate embedding and any embedding in a gallery.
 */
fun bestGallerySimilarity(candidate: FloatArray, gallery: List<FloatArray>): Float {
    if (gallery.isEmpty()) return 0f
    return gallery.maxOf { cosineSimilarity(it, candidate) }
}

/**
 * Crop a bitmap at a normalized bounding box. Returns null if the crop is invalid.
 */
fun cropNormalized(bitmap: Bitmap, normalizedBox: RectF): Bitmap? {
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
        Log.w("EmbeddingUtils", "Crop failed: ${e.message}")
        null
    }
}

/**
 * Compute pixel crop coordinates from a normalized box and image dimensions.
 * Returns (left, top, right, bottom) or null if invalid.
 */
fun cropCoordinates(normalizedBox: RectF, imgW: Int, imgH: Int): IntArray? {
    val left = (normalizedBox.left * imgW).toInt().coerceIn(0, imgW - 1)
    val top = (normalizedBox.top * imgH).toInt().coerceIn(0, imgH - 1)
    val right = (normalizedBox.right * imgW).toInt().coerceIn(left + 1, imgW)
    val bottom = (normalizedBox.bottom * imgH).toInt().coerceIn(top + 1, imgH)
    if (right - left <= 0 || bottom - top <= 0) return null
    return intArrayOf(left, top, right, bottom)
}
