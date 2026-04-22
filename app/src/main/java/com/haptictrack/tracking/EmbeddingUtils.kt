package com.haptictrack.tracking

import android.content.Context
import android.graphics.Bitmap
import android.graphics.Color
import android.graphics.RectF
import android.util.Log
import org.tensorflow.lite.Interpreter
import org.tensorflow.lite.gpu.GpuDelegate
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
 * Minimum pairwise cosine similarity within a gallery.
 * Measures how internally consistent the gallery is — used for adaptive thresholds.
 * Returns 1.0 for galleries with < 2 embeddings.
 */
fun minPairwiseSimilarity(gallery: List<FloatArray>): Float {
    if (gallery.size < 2) return 0f  // No self-similarity info → conservative (low floor)
    var minSim = 1f
    for (i in gallery.indices) {
        for (j in i + 1 until gallery.size) {
            val sim = cosineSimilarity(gallery[i], gallery[j])
            if (sim < minSim) minSim = sim
        }
    }
    return minSim
}

/**
 * Compute the centroid (mean) of a gallery of L2-normalized embeddings, then L2-normalize.
 * More stable identity representation than any single embedding.
 */
fun computeCentroid(gallery: List<FloatArray>): FloatArray? {
    if (gallery.isEmpty()) return null
    val dim = gallery[0].size
    val sum = FloatArray(dim)
    for (emb in gallery) {
        for (i in 0 until dim) sum[i] += emb[i]
    }
    // L2-normalize the mean
    var norm = 0f
    for (i in 0 until dim) norm += sum[i] * sum[i]
    norm = kotlin.math.sqrt(norm)
    if (norm < 1e-8f) return null
    for (i in 0 until dim) sum[i] /= norm
    return sum
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

data class GpuInterpreter(val interpreter: Interpreter, val gpuDelegate: GpuDelegate?) {
    fun close() {
        interpreter.close()
        gpuDelegate?.close()
    }
}

/**
 * Create a TFLite Interpreter with GPU delegate, falling back to CPU if GPU is unavailable.
 */
fun createGpuInterpreter(model: MappedByteBuffer, modelName: String = "unknown", cpuThreads: Int = 2): GpuInterpreter {
    return try {
        val gpuDelegate = GpuDelegate(GpuDelegate.Options())
        val options = Interpreter.Options().addDelegate(gpuDelegate)
        GpuInterpreter(Interpreter(model, options), gpuDelegate).also {
            Log.i("TFLiteGPU", "GPU delegate active for $modelName")
        }
    } catch (e: Throwable) {
        Log.w("TFLiteGPU", "GPU delegate unavailable for $modelName, using CPU ($cpuThreads threads): ${e.message}")
        GpuInterpreter(Interpreter(model, Interpreter.Options().apply { setNumThreads(cpuThreads) }), null)
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

// ---------------------------------------------------------------------------
// Online Logistic Regression — learns to separate locked object from scene
// ---------------------------------------------------------------------------

/**
 * Lightweight logistic regression trained on-device from gallery (positive)
 * and scene negative embeddings. Learns a hyperplane in embedding space that
 * separates THIS object from EVERYTHING ELSE in the scene.
 *
 * Training takes <1ms for typical gallery sizes (12 positives + 10 negatives).
 * Inference is a single dot product + sigmoid.
 */
class OnlineClassifier {
    private var weights: FloatArray? = null
    private var bias: Float = 0f
    /** True after successful training with enough data. */
    var isTrained: Boolean = false
        private set

    /**
     * Train on positive (gallery) and negative (scene) embeddings.
     * Uses mini-batch gradient descent with L2 regularization.
     */
    fun train(positives: List<FloatArray>, negatives: List<FloatArray>,
              iterations: Int = 80, learningRate: Float = 0.5f, l2Lambda: Float = 0.01f) {
        if (positives.size < 3 || negatives.size < 3) {
            isTrained = false
            return
        }
        val dim = positives[0].size
        val w = FloatArray(dim) // zero-initialized
        var b = 0f

        // Combine into training set with labels
        val data = positives.map { Pair(it, 1f) } + negatives.map { Pair(it, 0f) }

        for (iter in 0 until iterations) {
            var gradB = 0f
            val gradW = FloatArray(dim)
            for ((x, y) in data) {
                // Forward: sigmoid(w·x + b)
                var z = b
                for (i in 0 until dim) z += w[i] * x[i]
                val pred = 1f / (1f + kotlin.math.exp(-z.coerceIn(-10f, 10f)))

                // Gradient of binary cross-entropy
                val err = pred - y
                gradB += err
                for (i in 0 until dim) gradW[i] += err * x[i]
            }

            val n = data.size.toFloat()
            b -= learningRate * (gradB / n)
            for (i in 0 until dim) {
                w[i] -= learningRate * (gradW[i] / n + l2Lambda * w[i])
            }
        }

        weights = w
        bias = b
        isTrained = true
    }

    /**
     * Predict probability that the candidate is the locked object.
     * Returns value in [0, 1]. Returns 0.5 if not trained.
     */
    fun predict(candidate: FloatArray): Float {
        val w = weights ?: return 0.5f
        if (!isTrained || w.size != candidate.size) return 0.5f
        var z = bias
        for (i in w.indices) z += w[i] * candidate[i]
        return 1f / (1f + kotlin.math.exp(-z.coerceIn(-10f, 10f)))
    }

    fun clear() {
        weights = null
        bias = 0f
        isTrained = false
    }
}

/** L2 norm of a float array. */
fun l2Norm(arr: FloatArray): Float {
    var sum = 0f
    for (v in arr) sum += v * v
    return kotlin.math.sqrt(sum)
}

/** L2-normalize a float array in place. */
fun l2Normalize(arr: FloatArray) {
    val n = l2Norm(arr)
    if (n > 1e-6f) for (i in arr.indices) arr[i] /= n
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
