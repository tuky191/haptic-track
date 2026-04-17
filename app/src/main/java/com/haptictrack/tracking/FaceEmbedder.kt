package com.haptictrack.tracking

import android.content.Context
import android.graphics.Bitmap
import android.graphics.Color
import android.graphics.RectF
import android.util.Log
import com.google.mediapipe.framework.image.BitmapImageBuilder
import com.google.mediapipe.tasks.core.BaseOptions
import com.google.mediapipe.tasks.vision.facedetector.FaceDetector
import org.tensorflow.lite.Interpreter
import java.nio.ByteBuffer
import java.nio.ByteOrder

/**
 * Face identity embedder using MobileFaceNet (192-dim).
 *
 * Detects faces with BlazeFace (shared model with PersonAttributeClassifier),
 * crops + resizes to 112x112, and computes a 192-dim L2-normalizable embedding.
 *
 * Preprocessing: (pixel - 127.5) / 128.0 → range [-1, +1] (InsightFace convention).
 *
 * Used for person-vs-person discrimination when a face is visible.
 * The embedding is added to the lock attributes progressively — not required at lock time.
 */
class FaceEmbedder(context: Context, sharedFaceDetector: FaceDetector? = null) {

    companion object {
        private const val TAG = "FaceEmbed"
        private const val MODEL_ASSET = "mobilefacenet.tflite"
        private const val FACE_MODEL_ASSET = "blaze_face_short_range.tflite"
        private const val INPUT_SIZE = 112
        private const val EMBEDDING_DIM = 192
        private const val FACE_MIN_CONFIDENCE = 0.5f
    }

    private val interpreter: Interpreter
    private val faceDetector: FaceDetector
    private val ownsFaceDetector: Boolean

    // Pre-allocated buffers — this MobileFaceNet variant has fixed batch=2
    private val inputBuffer: ByteBuffer = ByteBuffer.allocateDirect(4 * 2 * INPUT_SIZE * INPUT_SIZE * 3).apply {
        order(ByteOrder.nativeOrder())
    }
    private val outputArray = Array(2) { FloatArray(EMBEDDING_DIM) }

    init {
        val model = loadTfliteModel(context, MODEL_ASSET)
        interpreter = Interpreter(model, Interpreter.Options().apply { setNumThreads(2) })

        if (sharedFaceDetector != null) {
            faceDetector = sharedFaceDetector
            ownsFaceDetector = false
        } else {
            val faceOptions = FaceDetector.FaceDetectorOptions.builder()
                .setBaseOptions(BaseOptions.builder().setModelAssetPath(FACE_MODEL_ASSET).build())
                .setMinDetectionConfidence(FACE_MIN_CONFIDENCE)
                .build()
            faceDetector = FaceDetector.createFromOptions(context, faceOptions)
            ownsFaceDetector = true
        }

        Log.i(TAG, "Loaded MobileFaceNet (${EMBEDDING_DIM}-dim)${if (ownsFaceDetector) " + BlazeFace" else " (shared BlazeFace)"}")
    }

    /**
     * Detect the largest face in a person crop and compute its embedding.
     * Returns null if no face is detected or the crop is too small.
     *
     * [bitmap] is the full frame, [personBox] is the normalized person bounding box.
     */
    fun embedFace(bitmap: Bitmap, personBox: RectF): FloatArray? {
        val personCrop = cropNormalized(bitmap, personBox) ?: return null
        try {
            return embedFaceFromCrop(personCrop)
        } finally {
            personCrop.recycle()
        }
    }

    /**
     * Compute face embedding from an already-cropped person bitmap.
     * Returns null if no face is found. Does NOT recycle [personCrop].
     */
    @Synchronized
    fun embedFaceFromCrop(personCrop: Bitmap): FloatArray? {
        if (personCrop.width < 30 || personCrop.height < 30) return null
        try {
            val mpImage = BitmapImageBuilder(personCrop).build()
            val faces = faceDetector.detect(mpImage)
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
            val faceResized = Bitmap.createScaledBitmap(faceCrop, INPUT_SIZE, INPUT_SIZE, true)
            if (faceResized !== faceCrop) faceCrop.recycle()

            fillInputBuffer(faceResized)
            faceResized.recycle()

            interpreter.run(inputBuffer, outputArray)

            // L2 normalize the embedding
            val embedding = outputArray[0].copyOf()
            com.haptictrack.tracking.l2Normalize(embedding)

            Log.d(TAG, "Face embedding computed (norm after L2: ${"%.2f".format(l2Norm(embedding))})")
            return embedding
        } catch (e: Exception) {
            Log.w(TAG, "Face embedding failed: ${e.message}")
            return null
        }
    }

    fun close() {
        interpreter.close()
        if (ownsFaceDetector) faceDetector.close()
    }

    /** InsightFace preprocessing: (pixel - 127.5) / 128.0 → [-1, +1]. Fills both batch slots. */
    private fun fillInputBuffer(bitmap: Bitmap) {
        inputBuffer.rewind()
        val pixels = IntArray(INPUT_SIZE * INPUT_SIZE)
        bitmap.getPixels(pixels, 0, INPUT_SIZE, 0, 0, INPUT_SIZE, INPUT_SIZE)
        // Batch slot 0: actual face
        for (pixel in pixels) {
            inputBuffer.putFloat((Color.red(pixel) - 127.5f) / 128f)
            inputBuffer.putFloat((Color.green(pixel) - 127.5f) / 128f)
            inputBuffer.putFloat((Color.blue(pixel) - 127.5f) / 128f)
        }
        // Batch slot 1: zeros (unused, required by fixed batch=2 model)
        repeat(INPUT_SIZE * INPUT_SIZE * 3) { inputBuffer.putFloat(0f) }
        inputBuffer.rewind()
    }

}
