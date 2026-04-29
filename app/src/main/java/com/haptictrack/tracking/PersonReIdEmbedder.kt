package com.haptictrack.tracking

import android.content.Context
import android.graphics.Bitmap
import android.graphics.Color
import android.graphics.RectF
import android.util.Log
import org.tensorflow.lite.Interpreter
import java.nio.ByteBuffer
import java.nio.ByteOrder

/**
 * Person re-identification embedder using OSNet x1.0 (512-dim).
 *
 * Trained on Market-1501 for person re-ID across camera viewpoints.
 * Takes a full-body or waist-up person crop and outputs a 512-dim embedding
 * that encodes body shape, proportions, clothing, and appearance.
 *
 * Much more discriminative for person-vs-person than the generic MobileNetV3
 * embedder (which is trained for general visual similarity, not re-ID).
 *
 * Input: [1, 256, 128, 3] float32 (NHWC, ImageNet-normalized)
 * Output: [1, 512] float32
 *
 * Preprocessing: ImageNet normalization
 *   R: (pixel/255 - 0.485) / 0.229
 *   G: (pixel/255 - 0.456) / 0.224
 *   B: (pixel/255 - 0.406) / 0.225
 *
 * Input is supplied as a [CanonicalCrop] at exactly [INPUT_WIDTH] × [INPUT_HEIGHT];
 * the cropper handles aspect-preserving letterbox so OSNet sees uniformly
 * shaped people instead of arbitrarily stretched ones (#100).
 */
class PersonReIdEmbedder(
    context: Context,
    private val cropper: CanonicalCropper = CanonicalCropper(),
) {

    companion object {
        private const val TAG = "PersonReId"
        private const val MODEL_ASSET = "osnet_x1_0_market.tflite"
        const val INPUT_HEIGHT = 256
        const val INPUT_WIDTH = 128
        private const val EMBEDDING_DIM = 512
        /**
         * Below this raw bbox dim, OSNet output is unreliable. Lower than the
         * canonical default (28) — OSNet specifically copes better with low-res
         * persons than the more generic embedders, and tracking small subjects
         * is a real use case.
         */
        private const val MIN_SOURCE_PIXELS = 16

        // ImageNet normalization constants
        private const val MEAN_R = 0.485f
        private const val MEAN_G = 0.456f
        private const val MEAN_B = 0.406f
        private const val STD_R = 0.229f
        private const val STD_G = 0.224f
        private const val STD_B = 0.225f

        /**
         * Self-test threshold. Two distinct synthetic inputs (horizontal vs
         * vertical stripes) should never produce cosine > this on a working
         * re-ID model. The collapsed-output failure mode (#117) pins it to
         * 1.0 exactly. 0.99 leaves room for legitimately-similar real-world
         * crops while still catching the degenerate case.
         */
        private const val SELF_TEST_MAX_COSINE = 0.99f
    }

    private val gpu: GpuInterpreter
    private val interpreter: Interpreter get() = gpu.interpreter

    // Pre-allocated buffers
    private val inputBuffer: ByteBuffer = ByteBuffer.allocateDirect(4 * INPUT_HEIGHT * INPUT_WIDTH * 3).apply {
        order(ByteOrder.nativeOrder())
    }
    private val outputArray = Array(1) { FloatArray(EMBEDDING_DIM) }

    /**
     * Set to true if the init-time self-test detects that the model is producing
     * a collapsed/constant output (cosine ≈ 1 between two distinct synthetic
     * inputs). When true, [embed] returns null so the cascade falls back to
     * MNV3-only scoring rather than feeding garbage reId vectors that pin every
     * candidate to a fake same-identity match. Tracked in #117.
     */
    private var disabled: Boolean = false

    init {
        val model = loadTfliteModel(context, MODEL_ASSET)
        gpu = createGpuInterpreter(model, modelName = "OSNet-ReID", cpuThreads = 2)
        Log.i(TAG, "Loaded OSNet (${EMBEDDING_DIM}-dim, ${INPUT_HEIGHT}x${INPUT_WIDTH})")

        // Init-time self-test: embed three distinct real person crops (bundled
        // at assets/reid_selftest_{a,b,c}.jpg, each from a different scene)
        // and verify the maximum pairwise cosine is well below 1.0. Catches
        // silent GPU-delegate miscompilation (#117) before bad reId vectors
        // poison the cascade. Real crops are required because the IBN failure
        // mode is input-distribution-dependent — synthetic stripes pass but
        // real photos collapse to a near-constant vector.
        val maxCos = selfTestMaxPairwiseCosine(context)
        if (maxCos == null) {
            disabled = true
            Log.e(TAG, "OSNet self-test FAILED to run (asset missing or inference threw). Disabling embedder.")
        } else if (maxCos > SELF_TEST_MAX_COSINE) {
            disabled = true
            Log.e(TAG, "OSNet self-test FAILED: max pairwise cosine across 3 distinct-scene crops = $maxCos > $SELF_TEST_MAX_COSINE. " +
                "Model is producing collapsed/constant output (likely GPU delegate miscompile, see #117). " +
                "Disabling embedder — cascade will fall back to MNV3-only scoring.")
        } else {
            Log.i(TAG, "OSNet self-test passed: max pairwise cosine = $maxCos (< $SELF_TEST_MAX_COSINE)")
        }
    }

    /**
     * Compute a re-ID embedding for a person crop. Wrapper that builds the
     * OSNet canonical and invokes [embed]. Recycles the canonical bitmap
     * before returning. Returns null if the bbox is too small or invalid.
     */
    fun embed(bitmap: Bitmap, personBox: RectF): FloatArray? {
        if (disabled) return null
        val canonical = cropper.prepare(
            bitmap, personBox,
            targetWidth = INPUT_WIDTH, targetHeight = INPUT_HEIGHT,
            minSourcePixels = MIN_SOURCE_PIXELS,
        ) ?: return null
        return try {
            embed(canonical)
        } finally {
            canonical.bitmap.recycle()
        }
    }

    /**
     * Compute a re-ID embedding from a prepared [CanonicalCrop]. Caller owns
     * [canonical] and is responsible for recycling. Required dims: 128×256.
     */
    @Synchronized
    fun embed(canonical: CanonicalCrop): FloatArray? {
        if (disabled) return null
        if (canonical.targetWidth != INPUT_WIDTH || canonical.targetHeight != INPUT_HEIGHT) {
            Log.w(TAG, "Canonical dims ${canonical.targetWidth}×${canonical.targetHeight} != expected ${INPUT_WIDTH}×${INPUT_HEIGHT}")
            return null
        }
        return try {
            fillInputBuffer(canonical.bitmap)
            interpreter.run(inputBuffer, outputArray)

            val embedding = outputArray[0].copyOf()
            com.haptictrack.tracking.l2Normalize(embedding)

            Log.d(TAG, "Re-ID embedding computed (${EMBEDDING_DIM}-dim)")
            embedding
        } catch (e: Exception) {
            Log.w(TAG, "Re-ID embedding failed: ${e.message}")
            null
        }
    }

    /**
     * Embed three real person crops from distinct scenes and return the
     * maximum pairwise cosine. Returns null if any inference throws or any
     * asset is missing.
     *
     * Synthetic inputs (random pixels, stripes, gradients) are *not*
     * sufficient — observed in #117 that the broken IBN MSMT17 model on
     * Adreno GPU delegate gives varied output on stripes (cos=0.54) but
     * collapses to a near-constant vector on real photos. The bug is
     * input-distribution-dependent and the test inputs need to look like
     * production crops.
     *
     * Three crops are used (man_desk, boy_indoor_wife_swap, kid_to_wife) so
     * the test is sensitive to a partial collapse — even if one pair happens
     * to look distinct, a degenerate model produces high cosine across at
     * least one of the three pairs. Crops are bundled at
     * app/src/main/assets/reid_selftest_{a,b,c}.jpg (~3 KB each, 128×256
     * letterboxed). Three different scenes, three different identities — a
     * working re-ID model should give all pairwise cosines well below 0.99.
     */
    private fun selfTestMaxPairwiseCosine(context: Context): Float? {
        val crops = listOf("reid_selftest_a.jpg", "reid_selftest_b.jpg", "reid_selftest_c.jpg")
            .map { loadAssetBitmap(context, it) ?: return null }
        return try {
            val embs = crops.map { embedRawBitmap(it) ?: return null }
            var maxCos = -1f
            for (i in embs.indices) {
                for (j in i + 1 until embs.size) {
                    val c = cosineSimilarity(embs[i], embs[j])
                    if (c > maxCos) maxCos = c
                }
            }
            maxCos
        } finally {
            crops.forEach { it.recycle() }
        }
    }

    private fun embedRawBitmap(bitmap: Bitmap): FloatArray? {
        return try {
            fillInputBuffer(bitmap)
            interpreter.run(inputBuffer, outputArray)
            val emb = outputArray[0].copyOf()
            l2Normalize(emb)
            emb
        } catch (e: Throwable) {
            Log.w(TAG, "Self-test inference failed: ${e.message}")
            null
        }
    }

    private fun loadAssetBitmap(context: Context, name: String): Bitmap? {
        return try {
            context.assets.open(name).use { stream ->
                val raw = android.graphics.BitmapFactory.decodeStream(stream) ?: return null
                if (raw.width == INPUT_WIDTH && raw.height == INPUT_HEIGHT) {
                    raw
                } else {
                    val scaled = Bitmap.createScaledBitmap(raw, INPUT_WIDTH, INPUT_HEIGHT, true)
                    raw.recycle()
                    scaled
                }
            }
        } catch (e: Throwable) {
            Log.w(TAG, "Self-test asset load failed for $name: ${e.message}")
            null
        }
    }

    fun close() {
        gpu.close()
    }

    /**
     * Audit/debug only — returns the OSNet canonical input bitmap (post-letterbox)
     * without computing the embedding. Caller must recycle. Used by
     * [CropDebugCapture] to make the aspect-preserving behavior visible.
     */
    fun debugInput(bitmap: Bitmap, personBox: RectF): Bitmap? {
        val canonical = cropper.prepare(
            bitmap, personBox,
            targetWidth = INPUT_WIDTH, targetHeight = INPUT_HEIGHT,
            minSourcePixels = MIN_SOURCE_PIXELS,
        ) ?: return null
        return canonical.bitmap // caller recycles
    }

    /** ImageNet normalization: (pixel/255 - mean) / std per channel */
    private fun fillInputBuffer(bitmap: Bitmap) {
        inputBuffer.rewind()
        val pixels = IntArray(INPUT_HEIGHT * INPUT_WIDTH)
        bitmap.getPixels(pixels, 0, INPUT_WIDTH, 0, 0, INPUT_WIDTH, INPUT_HEIGHT)
        for (pixel in pixels) {
            inputBuffer.putFloat((Color.red(pixel) / 255f - MEAN_R) / STD_R)
            inputBuffer.putFloat((Color.green(pixel) / 255f - MEAN_G) / STD_G)
            inputBuffer.putFloat((Color.blue(pixel) / 255f - MEAN_B) / STD_B)
        }
        inputBuffer.rewind()
    }

}
