package com.haptictrack.tracking

import android.content.Context
import android.graphics.Bitmap
import android.graphics.Color
import android.graphics.RectF
import android.util.Log
import com.google.mediapipe.framework.image.BitmapImageBuilder
import com.google.mediapipe.tasks.core.BaseOptions
import com.google.mediapipe.tasks.core.Delegate
import com.google.mediapipe.tasks.vision.facedetector.FaceDetector
import org.tensorflow.lite.Interpreter
import java.nio.ByteBuffer
import java.nio.ByteOrder

/**
 * Person classifier combining BlazeFace + age-gender-retail-0013 for
 * face-based gender and age from detected face within the person crop.
 * 95.8% gender accuracy, ~7 year age error.
 *
 * Crossroad-0230 body attributes were removed in #124 (Phase 1) after
 * ablation testing (#123) confirmed zero measurable signal.
 */
class PersonAttributeClassifier(context: Context) {

    companion object {
        private const val TAG = "PersonAttr"
        private const val AGE_GENDER_MODEL_ASSET = "age_gender_retail_0013.tflite"
        private const val FACE_MODEL_ASSET = "blaze_face_short_range.tflite"
        private const val FACE_SIZE = 62
        private const val INPUT_CHANNELS = 3
        private const val FACE_MIN_CONFIDENCE = 0.5f
    }

    private val ageGenderGpu: GpuInterpreter
    private val ageGenderInterpreter: Interpreter get() = ageGenderGpu.interpreter
    /** Exposed for sharing with [FaceEmbedder] to avoid duplicate model loading. */
    val faceDetector: FaceDetector

    init {
        val ageGenderModel = loadTfliteModel(context, AGE_GENDER_MODEL_ASSET)
        ageGenderGpu = createGpuInterpreter(ageGenderModel, modelName = "age-gender", cpuThreads = 2)

        faceDetector = try {
            FaceDetector.createFromOptions(context, FaceDetector.FaceDetectorOptions.builder()
                .setBaseOptions(BaseOptions.builder().setModelAssetPath(FACE_MODEL_ASSET).setDelegate(Delegate.GPU).build())
                .setMinDetectionConfidence(FACE_MIN_CONFIDENCE).build())
        } catch (e: Exception) {
            Log.w(TAG, "BlazeFace GPU failed, falling back to CPU: ${e.message}")
            FaceDetector.createFromOptions(context, FaceDetector.FaceDetectorOptions.builder()
                .setBaseOptions(BaseOptions.builder().setModelAssetPath(FACE_MODEL_ASSET).build())
                .setMinDetectionConfidence(FACE_MIN_CONFIDENCE).build())
        }

        Log.i(TAG, "Loaded: BlazeFace (face detect), age-gender-retail-0013 (face classify)")
    }

    /** Face-based gender/age result. */
    data class FaceGenderResult(val isMale: Boolean, val maleProb: Float, val age: Float)

    /**
     * Classify a person crop and return face-based gender + age.
     * [bitmap] is the full frame, [normalizedBox] is the person's bounding box in [0,1].
     * Returns null for non-person labels, if the crop is too small, or if no face is detected.
     */
    // TODO(#35): wire face-based gender/age into auto-lock criteria
    fun classify(bitmap: Bitmap, normalizedBox: RectF, label: String?): FaceGenderResult? {
        if (label != "person") return null

        val fullResCrop = cropFullRes(bitmap, normalizedBox) ?: return null
        try {
            return classifyFaceFromCrop(fullResCrop)
        } catch (e: Exception) {
            Log.w(TAG, "Classification failed: ${e.message}")
            return null
        } finally {
            fullResCrop.recycle()
        }
    }

    fun shutdown() {
        ageGenderGpu.close()
        faceDetector.close()
    }

    /**
     * Detect a face within the already-cropped person bitmap, then classify gender + age.
     * Returns null if no face is found. Does NOT recycle [personCrop].
     */
    private fun classifyFaceFromCrop(personCrop: Bitmap): FaceGenderResult? {
        if (personCrop.width < 30 || personCrop.height < 30) return null
        try {
            val mpImage = BitmapImageBuilder(personCrop).build()
            val faces = synchronized(faceDetector) { faceDetector.detect(mpImage) }

            if (faces.detections().isEmpty()) return null

            val face = faces.detections().maxByOrNull {
                it.boundingBox().width() * it.boundingBox().height()
            }!!
            val fb = face.boundingBox()
            val fx = fb.left.toInt().coerceIn(0, personCrop.width - 1)
            val fy = fb.top.toInt().coerceIn(0, personCrop.height - 1)
            val fw = fb.width().toInt().coerceIn(1, personCrop.width - fx)
            val fh = fb.height().toInt().coerceIn(1, personCrop.height - fy)

            val faceCrop = Bitmap.createBitmap(personCrop, fx, fy, fw, fh)
            val faceResized = Bitmap.createScaledBitmap(faceCrop, FACE_SIZE, FACE_SIZE, true)
            if (faceResized !== faceCrop) faceCrop.recycle()

            val buffer = ByteBuffer.allocateDirect(4 * FACE_SIZE * FACE_SIZE * INPUT_CHANNELS)
            buffer.order(ByteOrder.nativeOrder())
            val pixels = IntArray(FACE_SIZE * FACE_SIZE)
            faceResized.getPixels(pixels, 0, FACE_SIZE, 0, 0, FACE_SIZE, FACE_SIZE)
            for (pixel in pixels) {
                buffer.putFloat(Color.red(pixel).toFloat() / 255f)
                buffer.putFloat(Color.green(pixel).toFloat() / 255f)
                buffer.putFloat(Color.blue(pixel).toFloat() / 255f)
            }
            buffer.rewind()
            faceResized.recycle()

            val ageOut = Array(1) { Array(1) { Array(1) { FloatArray(1) } } }
            val genderOut = Array(1) { Array(1) { Array(1) { FloatArray(2) } } }
            val agOutputs = HashMap<Int, Any>()
            agOutputs[0] = ageOut
            agOutputs[1] = genderOut
            ageGenderInterpreter.runForMultipleInputsOutputs(arrayOf(buffer), agOutputs)

            val age = ageOut[0][0][0][0] * 100f
            val femaleProb = genderOut[0][0][0][0]
            val maleProb = genderOut[0][0][0][1]
            val isMale = maleProb > femaleProb

            Log.d(TAG, "Face gender: ${if (isMale) "male" else "female"} (${"%.2f".format(maxOf(maleProb, femaleProb))}), age: ${"%.0f".format(age)}")
            return FaceGenderResult(isMale, maleProb, age)
        } catch (e: Exception) {
            Log.w(TAG, "Face classification failed: ${e.message}")
            return null
        }
    }

    /** Crop person from full frame at original resolution. Caller must recycle. */
    private fun cropFullRes(bitmap: Bitmap, normalizedBox: RectF): Bitmap? {
        val imgW = bitmap.width
        val imgH = bitmap.height
        val left = (normalizedBox.left * imgW).toInt().coerceIn(0, imgW - 1)
        val top = (normalizedBox.top * imgH).toInt().coerceIn(0, imgH - 1)
        val right = (normalizedBox.right * imgW).toInt().coerceIn(left + 1, imgW)
        val bottom = (normalizedBox.bottom * imgH).toInt().coerceIn(top + 1, imgH)
        val w = right - left
        val h = bottom - top
        if (w < 10 || h < 20) return null

        return try {
            Bitmap.createBitmap(bitmap, left, top, w, h)
        } catch (e: Exception) {
            Log.w(TAG, "Crop failed: ${e.message}")
            null
        }
    }
}
