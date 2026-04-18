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
 * Person attribute classifier combining two models:
 *
 * 1. **Crossroad-0230** (2.8 MB) — body attributes from full person crop:
 *    clothing, accessories, hair length. Gender from this model is unreliable
 *    on close-up/non-standard crops.
 *
 * 2. **BlazeFace + age-gender-retail-0013** (224 KB + 4.1 MB) — face-based
 *    gender and age from detected face within the person crop.
 *    95.8% gender accuracy, ~7 year age error.
 *
 * When a face is detected, face-based gender overrides Crossroad-0230's gender.
 * When no face is visible, falls back to Crossroad-0230's body-based gender.
 */
class PersonAttributeClassifier(context: Context) {

    companion object {
        private const val TAG = "PersonAttr"
        private const val BODY_MODEL_ASSET = "person_attributes_crossroad_0230.tflite"
        private const val AGE_GENDER_MODEL_ASSET = "age_gender_retail_0013.tflite"
        private const val FACE_MODEL_ASSET = "blaze_face_short_range.tflite"
        private const val BODY_HEIGHT = 160
        private const val BODY_WIDTH = 80
        private const val FACE_SIZE = 62
        private const val INPUT_CHANNELS = 3
        private const val ATTR_THRESHOLD = 0.5f
        private const val FACE_MIN_CONFIDENCE = 0.5f
    }

    private val bodyInterpreter: Interpreter
    private val ageGenderInterpreter: Interpreter
    /** Exposed for sharing with [FaceEmbedder] to avoid duplicate model loading. */
    val faceDetector: FaceDetector

    init {
        val bodyModel = loadTfliteModel(context, BODY_MODEL_ASSET)
        bodyInterpreter = createGpuInterpreter(bodyModel, cpuThreads = 2)

        val ageGenderModel = loadTfliteModel(context, AGE_GENDER_MODEL_ASSET)
        ageGenderInterpreter = createGpuInterpreter(ageGenderModel, cpuThreads = 2)

        val faceOptions = FaceDetector.FaceDetectorOptions.builder()
            .setBaseOptions(BaseOptions.builder().setModelAssetPath(FACE_MODEL_ASSET).setDelegate(Delegate.GPU).build())
            .setMinDetectionConfidence(FACE_MIN_CONFIDENCE)
            .build()
        faceDetector = FaceDetector.createFromOptions(context, faceOptions)

        Log.i(TAG, "Loaded: Crossroad-0230 (body), BlazeFace (face detect), age-gender-retail-0013 (face classify)")
    }

    /**
     * Classify a person crop and return attributes.
     * [bitmap] is the full frame, [normalizedBox] is the person's bounding box in [0,1].
     * Returns null for non-person labels or if the crop is too small.
     */
    fun classify(bitmap: Bitmap, normalizedBox: RectF, label: String?): PersonAttributes? {
        if (label != "person") return null

        // Crop person at full resolution (used for face detection + color sampling)
        val fullResCrop = cropFullRes(bitmap, normalizedBox) ?: return null
        // Resize for body model
        val crop = Bitmap.createScaledBitmap(fullResCrop, BODY_WIDTH, BODY_HEIGHT, true)

        try {
            // --- Body attributes from Crossroad-0230 ---
            val inputBuffer = bitmapToByteBuffer(crop, BODY_WIDTH, BODY_HEIGHT)

            val colorPointTop = Array(1) { Array(1) { Array(1) { FloatArray(2) } } }
            val colorPointBottom = Array(1) { Array(1) { Array(1) { FloatArray(2) } } }
            val attributes = Array(1) { Array(1) { Array(1) { FloatArray(8) } } }

            val outputs = HashMap<Int, Any>()
            outputs[0] = colorPointTop
            outputs[1] = colorPointBottom
            outputs[2] = attributes

            bodyInterpreter.runForMultipleInputsOutputs(arrayOf(inputBuffer), outputs)

            val attrProbs = attributes[0][0][0]

            // Sample clothing colors from fixed body regions (use body-model-sized crop)
            val upperColor = dominantRegionColor(crop, 0.25f, 0.50f)
            val lowerColor = dominantRegionColor(crop, 0.58f, 0.85f)

            // --- Face-based gender + age (use full-res crop for better face detection) ---
            val faceResult = classifyFaceFromCrop(fullResCrop)
            val isMale = faceResult?.isMale ?: (attrProbs[0] > ATTR_THRESHOLD)
            val age = faceResult?.age

            // Override gender probability in raw array for soft scoring
            val adjustedProbs = attrProbs.copyOf()
            if (faceResult != null) {
                adjustedProbs[0] = faceResult.maleProb
            }

            val genderSource = if (faceResult != null) "face" else "body"

            return PersonAttributes(
                isMale = isMale,
                hasBag = attrProbs[1] > ATTR_THRESHOLD,
                hasBackpack = attrProbs[2] > ATTR_THRESHOLD,
                hasHat = attrProbs[3] > ATTR_THRESHOLD,
                hasLongSleeves = attrProbs[4] > ATTR_THRESHOLD,
                hasLongPants = attrProbs[5] > ATTR_THRESHOLD,
                hasLongHair = attrProbs[6] > ATTR_THRESHOLD,
                hasCoatJacket = attrProbs[7] > ATTR_THRESHOLD,
                upperColor = upperColor,
                lowerColor = lowerColor,
                rawProbabilities = adjustedProbs
            ).also {
                Log.d(TAG, "Classified: ${it.summary()} gender=$genderSource age=${age?.let { a -> "${a.toInt()}y" } ?: "n/a"} probs=[${adjustedProbs.joinToString { p -> "%.2f".format(p) }}]")
            }
        } catch (e: Exception) {
            Log.w(TAG, "Classification failed: ${e.message}")
            return null
        } finally {
            crop.recycle()
            fullResCrop.recycle()
        }
    }

    fun shutdown() {
        bodyInterpreter.close()
        ageGenderInterpreter.close()
        faceDetector.close()
    }

    /** Face-based gender/age result. */
    private data class FaceGenderResult(val isMale: Boolean, val maleProb: Float, val age: Float)

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

            // Use the largest face
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

            // Run age-gender model (input: 62x62 RGB normalized 0-1)
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

    private fun bitmapToByteBuffer(bitmap: Bitmap, width: Int, height: Int): ByteBuffer {
        val buffer = ByteBuffer.allocateDirect(4 * height * width * INPUT_CHANNELS)
        buffer.order(ByteOrder.nativeOrder())

        val pixels = IntArray(width * height)
        bitmap.getPixels(pixels, 0, width, 0, 0, width, height)

        for (pixel in pixels) {
            buffer.putFloat(Color.red(pixel).toFloat())
            buffer.putFloat(Color.green(pixel).toFloat())
            buffer.putFloat(Color.blue(pixel).toFloat())
        }
        buffer.rewind()
        return buffer
    }

    /**
     * Sample the dominant clothing color from a horizontal strip of the crop.
     * Uses per-pixel quantization then picks the most common color, preferring
     * chromatic colors over neutral (gray/black/white) which often dominate
     * in dim indoor lighting even on colored clothing.
     *
     * [yStartPct]/[yEndPct] define the vertical strip (0=top, 1=bottom).
     * The center 10% and outer 10% columns are excluded to skip hands/edges.
     */
    private fun dominantRegionColor(crop: Bitmap, yStartPct: Float, yEndPct: Float): String? {
        val w = crop.width
        val h = crop.height
        val y0 = (yStartPct * h).toInt().coerceIn(0, h - 1)
        val y1 = (yEndPct * h).toInt().coerceIn(y0 + 1, h)
        val xMargin = (0.10f * w).toInt()
        val x0 = xMargin.coerceIn(0, w - 1)
        val x1 = (w - xMargin).coerceIn(x0 + 1, w)

        val regionW = x1 - x0
        val regionH = y1 - y0
        if (regionW < 4 || regionH < 4) return null

        val pixels = IntArray(regionW * regionH)
        crop.getPixels(pixels, 0, regionW, x0, y0, regionW, regionH)

        val votes = HashMap<String, Int>()
        val hsv = FloatArray(3)
        for (pixel in pixels) {
            Color.colorToHSV(pixel, hsv)
            val name = quantizeColor(hsv[0], hsv[1], hsv[2])
            votes[name] = (votes[name] ?: 0) + 1
        }

        val total = pixels.size
        if (total == 0) return null

        // Sort by vote count
        val sorted = votes.entries.sortedByDescending { it.value }
        val dominant = sorted.first().key

        // If dominant is neutral (gray/black/white), check for a chromatic runner-up
        // with at least 12% of pixels — indoor lighting desaturates real colors heavily
        val neutrals = setOf("gray", "black", "white")
        if (dominant in neutrals) {
            val chromatic = sorted.firstOrNull { it.key !in neutrals && it.value > total * 0.12 }
            if (chromatic != null) {
                Log.d(TAG, "Color override: $dominant (${sorted.first().value * 100 / total}%) → ${chromatic.key} (${chromatic.value * 100 / total}%)")
                return chromatic.key
            }
        }

        return dominant
    }

}

/**
 * Quantize HSV values to a named color string.
 * Covers the 11 basic color terms from Berlin & Kay.
 */
fun quantizeColor(h: Float, s: Float, v: Float): String {
    // Achromatic colors — thresholds tuned for indoor lighting where white
    // fabric often appears dim (v=0.4-0.6) but still very desaturated (s<0.08)
    if (v < 0.15f) return "black"
    if (s < 0.08f && v > 0.65f) return "white"
    if (s < 0.08f) return "gray"

    // Chromatic colors by hue
    return when {
        h < 15f || h >= 345f -> "red"
        h < 40f -> "orange"
        h < 70f -> "yellow"
        h < 165f -> "green"
        h < 195f -> "cyan"
        h < 260f -> "blue"
        h < 290f -> "purple"
        else -> "pink" // 290-345
    }
}
