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
        private const val MODEL_ASSET = "osnet_ibn_x1_0_msmt17.tflite"
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
    }

    private val gpu: GpuInterpreter
    private val interpreter: Interpreter get() = gpu.interpreter

    // Pre-allocated buffers
    private val inputBuffer: ByteBuffer = ByteBuffer.allocateDirect(4 * INPUT_HEIGHT * INPUT_WIDTH * 3).apply {
        order(ByteOrder.nativeOrder())
    }
    private val outputArray = Array(1) { FloatArray(EMBEDDING_DIM) }

    init {
        val model = loadTfliteModel(context, MODEL_ASSET)
        gpu = createGpuInterpreter(model, modelName = "OSNet-ReID", cpuThreads = 2)
        Log.i(TAG, "Loaded OSNet-IBN x1.0 MSMT17 (${EMBEDDING_DIM}-dim, ${INPUT_HEIGHT}x${INPUT_WIDTH})")
    }

    /**
     * Compute a re-ID embedding for a person crop. Wrapper that builds the
     * OSNet canonical and invokes [embed]. Recycles the canonical bitmap
     * before returning. Returns null if the bbox is too small or invalid.
     */
    fun embed(bitmap: Bitmap, personBox: RectF): FloatArray? {
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
