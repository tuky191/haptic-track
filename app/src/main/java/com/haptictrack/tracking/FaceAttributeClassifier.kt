package com.haptictrack.tracking

import android.content.Context
import android.graphics.Bitmap
import android.graphics.Color
import android.util.Log
import org.tensorflow.lite.Interpreter
import java.nio.ByteBuffer
import java.nio.ByteOrder

/** Gender + age estimate for one face. [genderConfidence] is |male − female| logit gap. */
data class FaceAttributes(
    val isMale: Boolean,
    val age: Int,
    val genderConfidence: Float,
) {
    val genderLabel: String get() = if (isMale) "M" else "F"
}

/**
 * Face gender/age classifier — InsightFace genderage (96×96 RGB, output
 * [female, male, age×100]). Validated off-device (tools/sentry_genderage.py):
 * robust down to ~35px faces; below that face DETECTION fails, not this model.
 * Consumes the aligned face crop [FaceEmbedder] already produces (resized to 96).
 *
 * Preprocessing: raw RGB 0-255, NHWC (no normalization — matches the ONNX
 * blobFromImage(scale=1, mean=0, swapRB=true) the model was trained against).
 */
class FaceAttributeClassifier(context: Context) {

    companion object {
        private const val TAG = "FaceAttr"
        private const val MODEL_ASSET = "genderage.tflite"
        private const val INPUT_SIZE = 96
    }

    private val gpu: GpuInterpreter
    private val interpreter: Interpreter get() = gpu.interpreter

    private val inputBuffer: ByteBuffer =
        ByteBuffer.allocateDirect(4 * INPUT_SIZE * INPUT_SIZE * 3).apply { order(ByteOrder.nativeOrder()) }
    private val output = Array(1) { FloatArray(3) }

    init {
        gpu = createGpuInterpreter(loadTfliteModel(context, MODEL_ASSET), modelName = "GenderAge", cpuThreads = 2)
        Log.i(TAG, "Loaded genderage classifier")
    }

    /**
     * Classify a face bitmap (any size — rescaled to 96×96). Returns null on
     * failure. Thread-safe: serialized on the shared interpreter/buffer.
     */
    @Synchronized
    fun classify(face: Bitmap): FaceAttributes? = try {
        val scaled = if (face.width == INPUT_SIZE && face.height == INPUT_SIZE) face
                     else Bitmap.createScaledBitmap(face, INPUT_SIZE, INPUT_SIZE, true)
        fillInput(scaled)
        if (scaled !== face) scaled.recycle()
        interpreter.run(inputBuffer, output)
        val (female, male, ageRaw) = output[0]
        Log.i(TAG, "raw output: female=$female male=$male ageRaw=$ageRaw -> age=${(ageRaw*100).toInt()}")
        FaceAttributes(
            isMale = male > female,
            age = (ageRaw * 100f).toInt().coerceIn(0, 120),
            genderConfidence = kotlin.math.abs(male - female),
        )
    } catch (e: Exception) {
        Log.w(TAG, "classify failed: ${e.message}")
        null
    }

    private fun fillInput(bitmap: Bitmap) {
        inputBuffer.rewind()
        val pixels = IntArray(INPUT_SIZE * INPUT_SIZE)
        bitmap.getPixels(pixels, 0, INPUT_SIZE, 0, 0, INPUT_SIZE, INPUT_SIZE)
        for (p in pixels) {
            inputBuffer.putFloat(Color.red(p).toFloat())
            inputBuffer.putFloat(Color.green(p).toFloat())
            inputBuffer.putFloat(Color.blue(p).toFloat())
        }
        inputBuffer.rewind()
    }

    fun close() = gpu.close()
}
