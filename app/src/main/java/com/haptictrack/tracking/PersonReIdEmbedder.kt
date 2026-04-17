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
 */
class PersonReIdEmbedder(context: Context) {

    companion object {
        private const val TAG = "PersonReId"
        private const val MODEL_ASSET = "osnet_x1_0_market.tflite"
        private const val INPUT_HEIGHT = 256
        private const val INPUT_WIDTH = 128
        private const val EMBEDDING_DIM = 512

        // ImageNet normalization constants
        private const val MEAN_R = 0.485f
        private const val MEAN_G = 0.456f
        private const val MEAN_B = 0.406f
        private const val STD_R = 0.229f
        private const val STD_G = 0.224f
        private const val STD_B = 0.225f
    }

    private val interpreter: Interpreter

    // Pre-allocated buffers
    private val inputBuffer: ByteBuffer = ByteBuffer.allocateDirect(4 * INPUT_HEIGHT * INPUT_WIDTH * 3).apply {
        order(ByteOrder.nativeOrder())
    }
    private val outputArray = Array(1) { FloatArray(EMBEDDING_DIM) }

    init {
        val model = loadTfliteModel(context, MODEL_ASSET)
        interpreter = Interpreter(model, Interpreter.Options().apply { setNumThreads(2) })
        Log.i(TAG, "Loaded OSNet x1.0 Market-1501 (${EMBEDDING_DIM}-dim, ${INPUT_HEIGHT}x${INPUT_WIDTH})")
    }

    /**
     * Compute a re-ID embedding for a person crop.
     * [bitmap] is the full frame, [personBox] is the normalized bounding box.
     * Returns a 512-dim L2-normalized embedding, or null if the crop is invalid.
     */
    fun embed(bitmap: Bitmap, personBox: RectF): FloatArray? {
        val crop = cropNormalized(bitmap, personBox) ?: return null
        try {
            return embedFromCrop(crop)
        } finally {
            crop.recycle()
        }
    }

    /**
     * Compute a re-ID embedding from an already-cropped person bitmap.
     * Does NOT recycle [personCrop].
     */
    @Synchronized
    fun embedFromCrop(personCrop: Bitmap): FloatArray? {
        if (personCrop.width < 10 || personCrop.height < 20) return null
        try {
            val resized = Bitmap.createScaledBitmap(personCrop, INPUT_WIDTH, INPUT_HEIGHT, true)
            fillInputBuffer(resized)
            if (resized !== personCrop) resized.recycle()

            interpreter.run(inputBuffer, outputArray)

            val embedding = outputArray[0].copyOf()
            com.haptictrack.tracking.l2Normalize(embedding)

            Log.d(TAG, "Re-ID embedding computed (${EMBEDDING_DIM}-dim)")
            return embedding
        } catch (e: Exception) {
            Log.w(TAG, "Re-ID embedding failed: ${e.message}")
            return null
        }
    }

    fun close() {
        interpreter.close()
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
