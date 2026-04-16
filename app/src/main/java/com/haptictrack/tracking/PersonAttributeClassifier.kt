package com.haptictrack.tracking

import android.content.Context
import android.graphics.Bitmap
import android.graphics.Color
import android.graphics.RectF
import android.util.Log
import org.tensorflow.lite.Interpreter
import java.io.FileInputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.MappedByteBuffer
import java.nio.channels.FileChannel

/**
 * Person attribute classifier using Intel Crossroad-0230 (2.8 MB TFLite).
 *
 * Input: 160×80 person crop (height × width), RGB, 0-255 float32.
 * Outputs:
 *   - 8 binary attributes: is_male, has_bag, has_backpack, has_hat,
 *     has_longsleeves, has_longpants, has_longhair, has_coat_jacket
 *   - 2 color sampling points: upper body (x,y) and lower body (x,y)
 *     in normalized [0,1] coordinates relative to the crop.
 *
 * Only runs on "person" detections — skipped for non-person objects.
 */
class PersonAttributeClassifier(context: Context) {

    companion object {
        private const val TAG = "PersonAttr"
        private const val MODEL_ASSET = "person_attributes_crossroad_0230.tflite"
        private const val INPUT_HEIGHT = 160
        private const val INPUT_WIDTH = 80
        private const val INPUT_CHANNELS = 3
        private const val ATTR_THRESHOLD = 0.5f
    }

    private val interpreter: Interpreter

    init {
        val model = loadModelFile(context)
        val options = Interpreter.Options().apply {
            setNumThreads(2)
        }
        interpreter = Interpreter(model, options)
        Log.i(TAG, "Loaded Crossroad-0230: input=${INPUT_HEIGHT}x${INPUT_WIDTH}, 8 attrs + 2 color points")
    }

    /**
     * Classify a person crop and return attributes.
     * [bitmap] is the full frame, [normalizedBox] is the person's bounding box in [0,1].
     * Returns null for non-person labels or if the crop is too small.
     */
    fun classify(bitmap: Bitmap, normalizedBox: RectF, label: String?): PersonAttributes? {
        if (label != "person") return null

        val crop = cropAndResize(bitmap, normalizedBox) ?: return null

        try {
            val inputBuffer = bitmapToByteBuffer(crop)

            // Prepare output buffers matching model output shapes
            val colorPointTop = Array(1) { Array(1) { Array(1) { FloatArray(2) } } }
            val colorPointBottom = Array(1) { Array(1) { Array(1) { FloatArray(2) } } }
            val attributes = Array(1) { Array(1) { Array(1) { FloatArray(8) } } }

            val outputs = HashMap<Int, Any>()
            outputs[0] = colorPointTop
            outputs[1] = colorPointBottom
            outputs[2] = attributes

            interpreter.runForMultipleInputsOutputs(arrayOf(inputBuffer), outputs)

            val attrProbs = attributes[0][0][0]
            val topPoint = colorPointTop[0][0][0]   // (x, y) normalized
            val bottomPoint = colorPointBottom[0][0][0]

            // Sample clothing colors from the crop at the model's suggested points
            val upperColor = sampleColor(crop, topPoint[0], topPoint[1])
            val lowerColor = sampleColor(crop, bottomPoint[0], bottomPoint[1])

            return PersonAttributes(
                isMale = attrProbs[0] > ATTR_THRESHOLD,
                hasBag = attrProbs[1] > ATTR_THRESHOLD,
                hasBackpack = attrProbs[2] > ATTR_THRESHOLD,
                hasHat = attrProbs[3] > ATTR_THRESHOLD,
                hasLongSleeves = attrProbs[4] > ATTR_THRESHOLD,
                hasLongPants = attrProbs[5] > ATTR_THRESHOLD,
                hasLongHair = attrProbs[6] > ATTR_THRESHOLD,
                hasCoatJacket = attrProbs[7] > ATTR_THRESHOLD,
                upperColor = upperColor,
                lowerColor = lowerColor,
                rawProbabilities = attrProbs.copyOf()
            ).also {
                Log.d(TAG, "Classified: ${it.summary()} probs=[${attrProbs.joinToString { p -> "%.2f".format(p) }}]")
            }
        } catch (e: Exception) {
            Log.w(TAG, "Classification failed: ${e.message}")
            return null
        } finally {
            crop.recycle()
        }
    }

    fun shutdown() {
        interpreter.close()
    }

    private fun cropAndResize(bitmap: Bitmap, normalizedBox: RectF): Bitmap? {
        val imgW = bitmap.width
        val imgH = bitmap.height
        val left = (normalizedBox.left * imgW).toInt().coerceIn(0, imgW - 1)
        val top = (normalizedBox.top * imgH).toInt().coerceIn(0, imgH - 1)
        val right = (normalizedBox.right * imgW).toInt().coerceIn(left + 1, imgW)
        val bottom = (normalizedBox.bottom * imgH).toInt().coerceIn(top + 1, imgH)
        val w = right - left
        val h = bottom - top
        if (w < 10 || h < 20) return null  // too small to classify

        return try {
            val cropped = Bitmap.createBitmap(bitmap, left, top, w, h)
            Bitmap.createScaledBitmap(cropped, INPUT_WIDTH, INPUT_HEIGHT, true).also {
                if (it !== cropped) cropped.recycle()
            }
        } catch (e: Exception) {
            Log.w(TAG, "Crop failed: ${e.message}")
            null
        }
    }

    private fun bitmapToByteBuffer(bitmap: Bitmap): ByteBuffer {
        val buffer = ByteBuffer.allocateDirect(4 * INPUT_HEIGHT * INPUT_WIDTH * INPUT_CHANNELS)
        buffer.order(ByteOrder.nativeOrder())

        val pixels = IntArray(INPUT_WIDTH * INPUT_HEIGHT)
        bitmap.getPixels(pixels, 0, INPUT_WIDTH, 0, 0, INPUT_WIDTH, INPUT_HEIGHT)

        for (pixel in pixels) {
            // RGB 0-255 as float32
            buffer.putFloat(Color.red(pixel).toFloat())
            buffer.putFloat(Color.green(pixel).toFloat())
            buffer.putFloat(Color.blue(pixel).toFloat())
        }
        buffer.rewind()
        return buffer
    }

    /**
     * Sample the pixel color at a normalized (x, y) point in the crop bitmap
     * and quantize to a named color via HSV.
     */
    private fun sampleColor(crop: Bitmap, normX: Float, normY: Float): String? {
        val px = (normX * crop.width).toInt().coerceIn(0, crop.width - 1)
        val py = (normY * crop.height).toInt().coerceIn(0, crop.height - 1)

        // Sample a 5×5 patch around the point for robustness
        val hsv = FloatArray(3)
        var hSum = 0f; var sSum = 0f; var vSum = 0f; var count = 0
        for (dy in -2..2) {
            for (dx in -2..2) {
                val sx = (px + dx).coerceIn(0, crop.width - 1)
                val sy = (py + dy).coerceIn(0, crop.height - 1)
                Color.colorToHSV(crop.getPixel(sx, sy), hsv)
                hSum += hsv[0]; sSum += hsv[1]; vSum += hsv[2]; count++
            }
        }
        val h = hSum / count  // 0-360
        val s = sSum / count  // 0-1
        val v = vSum / count  // 0-1

        return quantizeColor(h, s, v)
    }

    private fun loadModelFile(context: Context): MappedByteBuffer {
        val fd = context.assets.openFd(MODEL_ASSET)
        val input = FileInputStream(fd.fileDescriptor)
        val channel = input.channel
        return channel.map(FileChannel.MapMode.READ_ONLY, fd.startOffset, fd.declaredLength)
    }
}

/**
 * Quantize HSV values to a named color string.
 * Covers the 11 basic color terms from Berlin & Kay.
 */
fun quantizeColor(h: Float, s: Float, v: Float): String {
    // Achromatic colors
    if (v < 0.15f) return "black"
    if (s < 0.10f && v > 0.85f) return "white"
    if (s < 0.15f) return "gray"

    // Chromatic colors by hue
    return when {
        h < 15f || h >= 345f -> "red"
        h < 40f -> "orange"
        h < 70f -> "yellow"
        h < 165f -> "green"
        h < 195f -> "cyan"
        h < 260f -> "blue"
        h < 290f -> "purple"
        h < 345f -> "pink"
        else -> "red"
    }
}
