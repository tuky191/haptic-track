package com.haptictrack.tracking

import android.content.Context
import android.graphics.Bitmap
import android.graphics.Color
import android.util.Log
import org.tensorflow.lite.Interpreter
import java.nio.ByteBuffer
import java.nio.ByteOrder
import kotlin.math.exp

/** Gender + age estimate for one face. [genderConfidence] is the |male − female| logit gap. */
data class FaceAttributes(
    val isMale: Boolean,
    /** Point-estimate age in years (probability-weighted bucket midpoint). */
    val age: Int,
    val genderConfidence: Float,
    /** FairFace age-bucket label, e.g. "30-39" / "70+". */
    val ageBucket: String = "",
) {
    val genderLabel: String get() = if (isMale) "M" else "F"
}

/**
 * Face gender/age classifier — FairFace ResNet34 (224×224 RGB, ImageNet-normalized,
 * outputs race(7) / gender(2:[M,F]) / age(9 buckets)). Chosen over InsightFace
 * genderage after an offline eval on 19 labeled faces (age MAE 4.6y, 100%
 * within-one-bucket, correct on the real test face) — and it's a CNN, so it
 * GPU-delegates on Adreno where MiVOLO (transformer) can't.
 *
 * Age is reported as a 9-bucket band plus a probability-weighted midpoint
 * point estimate (per-year age isn't reliable from any model).
 */
class FaceAttributeClassifier(context: Context) {

    companion object {
        private const val TAG = "FaceAttr"
        private const val MODEL_ASSET = "fairface.tflite"
        const val INPUT_SIZE = 224
        private val BUCKETS = arrayOf("0-2", "3-9", "10-19", "20-29", "30-39", "40-49", "50-59", "60-69", "70+")
        private val BUCKET_MID = floatArrayOf(1f, 6f, 14.5f, 24.5f, 34.5f, 44.5f, 54.5f, 64.5f, 80f)
        // ImageNet normalization (FairFace training convention).
        private val MEAN = floatArrayOf(0.485f, 0.456f, 0.406f)
        private val STD = floatArrayOf(0.229f, 0.224f, 0.225f)
    }

    private val gpu: GpuInterpreter
    private val interpreter: Interpreter get() = gpu.interpreter

    private val inputBuffer: ByteBuffer =
        ByteBuffer.allocateDirect(4 * INPUT_SIZE * INPUT_SIZE * 3).apply { order(ByteOrder.nativeOrder()) }

    // Output tensors, sized at init. Mapped by length: gender=2, age=9, race=7.
    private val outBuffers: Map<Int, Array<FloatArray>>
    private val genderOutIdx: Int
    private val ageOutIdx: Int

    init {
        gpu = createGpuInterpreter(loadTfliteModel(context, MODEL_ASSET), modelName = "FairFace", cpuThreads = 2)
        val n = interpreter.outputTensorCount
        val bufs = HashMap<Int, Array<FloatArray>>()
        var gIdx = -1; var aIdx = -1
        for (i in 0 until n) {
            val len = interpreter.getOutputTensor(i).shape().last()
            bufs[i] = arrayOf(FloatArray(len))
            when (len) { 2 -> gIdx = i; 9 -> aIdx = i }
        }
        outBuffers = bufs
        genderOutIdx = gIdx
        ageOutIdx = aIdx
        require(gIdx >= 0 && aIdx >= 0) { "FairFace outputs not found (gender=2, age=9)" }
        Log.i(TAG, "Loaded FairFace classifier (${n} outputs; gender@$gIdx age@$aIdx)")
    }

    /** Classify a face bitmap (rescaled to 224). Returns null on failure. Thread-safe. */
    @Synchronized
    fun classify(face: Bitmap): FaceAttributes? = try {
        val scaled = if (face.width == INPUT_SIZE && face.height == INPUT_SIZE) face
                     else Bitmap.createScaledBitmap(face, INPUT_SIZE, INPUT_SIZE, true)
        fillInput(scaled)
        if (scaled !== face) scaled.recycle()

        @Suppress("UNCHECKED_CAST")
        val outputs = outBuffers.mapValues { it.value as Any }
        interpreter.runForMultipleInputsOutputs(arrayOf(inputBuffer), HashMap(outputs))

        val gender = outBuffers[genderOutIdx]!![0]   // [male, female] logits
        val ageLogits = outBuffers[ageOutIdx]!![0]   // 9 bucket logits
        val isMale = gender[0] > gender[1]
        val probs = softmax(ageLogits)
        var pt = 0f
        for (i in probs.indices) pt += probs[i] * BUCKET_MID[i]
        var bidx = 0
        for (i in probs.indices) if (probs[i] > probs[bidx]) bidx = i

        FaceAttributes(
            isMale = isMale,
            age = pt.toInt().coerceIn(0, 120),
            genderConfidence = kotlin.math.abs(gender[0] - gender[1]),
            ageBucket = BUCKETS[bidx],
        )
    } catch (e: Exception) {
        Log.w(TAG, "classify failed: ${e.message}")
        null
    }

    private fun softmax(x: FloatArray): FloatArray {
        var max = Float.NEGATIVE_INFINITY
        for (v in x) if (v > max) max = v
        var sum = 0f
        val out = FloatArray(x.size)
        for (i in x.indices) { out[i] = exp(x[i] - max); sum += out[i] }
        for (i in out.indices) out[i] /= sum
        return out
    }

    private fun fillInput(bitmap: Bitmap) {
        inputBuffer.rewind()
        val pixels = IntArray(INPUT_SIZE * INPUT_SIZE)
        bitmap.getPixels(pixels, 0, INPUT_SIZE, 0, 0, INPUT_SIZE, INPUT_SIZE)
        for (p in pixels) {
            inputBuffer.putFloat((Color.red(p) / 255f - MEAN[0]) / STD[0])
            inputBuffer.putFloat((Color.green(p) / 255f - MEAN[1]) / STD[1])
            inputBuffer.putFloat((Color.blue(p) / 255f - MEAN[2]) / STD[2])
        }
        inputBuffer.rewind()
    }

    fun close() = gpu.close()
}
