package com.haptictrack.tracking

import android.content.Context
import android.graphics.Bitmap
import android.graphics.Color
import android.graphics.RectF
import android.util.Log
import java.io.FileInputStream
import java.nio.MappedByteBuffer
import java.nio.channels.FileChannel

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

// ---------------------------------------------------------------------------
// Color Histogram — lightweight color identity signal
// ---------------------------------------------------------------------------

/** Number of bins per channel in the HSV histogram. */
private const val H_BINS = 18
private const val S_BINS = 8
const val COLOR_HISTOGRAM_SIZE = H_BINS + S_BINS

/**
 * Compute an HSV color histogram from a bitmap crop at a normalized bounding box.
 * Returns a normalized float array of [COLOR_HISTOGRAM_SIZE] bins, or null if crop is invalid.
 *
 * Hue is the primary discriminator (red car vs dark car), saturation adds
 * texture information. Value (brightness) is excluded because it varies too much
 * with lighting conditions.
 */
fun computeColorHistogram(bitmap: Bitmap, normalizedBox: RectF): FloatArray? {
    val coords = cropCoordinates(normalizedBox, bitmap.width, bitmap.height) ?: return null
    val left = coords[0]; val top = coords[1]; val right = coords[2]; val bottom = coords[3]
    val w = right - left
    val h = bottom - top
    if (w < 4 || h < 4) return null

    val pixels = IntArray(w * h)
    bitmap.getPixels(pixels, 0, w, left, top, w, h)

    val hHist = FloatArray(H_BINS)
    val sHist = FloatArray(S_BINS)
    val hsv = FloatArray(3)

    for (pixel in pixels) {
        if (pixel == Color.BLACK) continue // skip masked-out pixels
        Color.colorToHSV(pixel, hsv)
        val hBin = ((hsv[0] / 360f) * H_BINS).toInt().coerceIn(0, H_BINS - 1)
        val sBin = (hsv[1] * S_BINS).toInt().coerceIn(0, S_BINS - 1)
        hHist[hBin]++
        sHist[sBin]++
    }

    // Combine and normalize
    val combined = FloatArray(COLOR_HISTOGRAM_SIZE)
    System.arraycopy(hHist, 0, combined, 0, H_BINS)
    System.arraycopy(sHist, 0, combined, H_BINS, S_BINS)

    val sum = combined.sum()
    if (sum <= 0f) return null
    for (i in combined.indices) combined[i] /= sum

    return combined
}

/**
 * Load a TFLite model from an asset file into a memory-mapped buffer.
 */
fun loadTfliteModel(context: Context, assetName: String): MappedByteBuffer {
    val fd = context.assets.openFd(assetName)
    return fd.use {
        FileInputStream(it.fileDescriptor).use { input ->
            input.channel.map(FileChannel.MapMode.READ_ONLY, it.startOffset, it.declaredLength)
        }
    }
}

/**
 * IoU (Intersection over Union) between two bounding boxes.
 */
fun computeIou(a: RectF, b: RectF): Float {
    val interLeft = maxOf(a.left, b.left)
    val interTop = maxOf(a.top, b.top)
    val interRight = minOf(a.right, b.right)
    val interBottom = minOf(a.bottom, b.bottom)

    if (interLeft >= interRight || interTop >= interBottom) return 0f

    val interArea = (interRight - interLeft) * (interBottom - interTop)
    val aArea = a.width() * a.height()
    val bArea = b.width() * b.height()
    val unionArea = aArea + bArea - interArea

    return if (unionArea > 0f) interArea / unionArea else 0f
}

/**
 * Histogram correlation — measures how similar two color distributions are.
 * Returns a value in [-1, 1] where 1 = identical distributions.
 */
fun histogramCorrelation(a: FloatArray, b: FloatArray): Float {
    if (a.size != b.size) return 0f
    val n = a.size
    var aMean = 0f; var bMean = 0f
    for (i in 0 until n) { aMean += a[i]; bMean += b[i] }
    aMean /= n; bMean /= n

    var num = 0f; var denA = 0f; var denB = 0f
    for (i in 0 until n) {
        val da = a[i] - aMean
        val db = b[i] - bMean
        num += da * db
        denA += da * da
        denB += db * db
    }

    val den = kotlin.math.sqrt(denA * denB)
    return if (den > 1e-8f) num / den else 0f
}
