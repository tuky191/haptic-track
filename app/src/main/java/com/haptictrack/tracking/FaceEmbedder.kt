package com.haptictrack.tracking

import android.content.Context
import android.graphics.Bitmap
import android.graphics.Color
import android.graphics.PointF
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
 * Detects faces with BlazeFace inside a [CanonicalCrop] of the person, crops
 * the face from canonical pixel space, letterboxes the face to 112×112,
 * and computes a 192-dim L2-normalized embedding.
 *
 * Canonical input flow (#100):
 *  - Person canonical: square [PERSON_CANONICAL_SIZE]² letterbox of the person bbox.
 *    BlazeFace runs on this; keypoints return in canonical pixel space rather
 *    than the warped raw-bbox space they used to live in.
 *  - Face sub-canonical: 112×112 letterbox of the face bbox within the person
 *    canonical. No 5-point similarity transform yet — that's #93. The
 *    keypoints are surfaced via [debugFaceCrop] so #93 has them ready.
 *
 * Preprocessing: (pixel - 127.5) / 128.0 → range [-1, +1] (InsightFace convention).
 */
class FaceEmbedder(
    context: Context,
    sharedFaceDetector: FaceDetector? = null,
    private val cropper: CanonicalCropper = CanonicalCropper(),
) {

    companion object {
        private const val TAG = "FaceEmbed"
        private const val MODEL_ASSET = "mobilefacenet.tflite"
        private const val FACE_MODEL_ASSET = "blaze_face_short_range.tflite"
        const val INPUT_SIZE = 112
        private const val EMBEDDING_DIM = 192
        private const val FACE_MIN_CONFIDENCE = 0.5f

        /**
         * Square canonical size for the person crop fed to BlazeFace. 256 is
         * enough headroom for BlazeFace's internal 128² model and small
         * enough that letterbox padding cost is negligible. Chosen to be
         * larger than MNV3's 224 so faces are rendered at higher resolution.
         */
        const val PERSON_CANONICAL_SIZE = 256
        /** Min raw bbox dim for the person input — tiny persons can't yield faces. */
        private const val MIN_PERSON_SOURCE_PIXELS = 30
    }

    private val gpu: GpuInterpreter
    private val interpreter: Interpreter get() = gpu.interpreter
    private val faceDetector: FaceDetector
    private val ownsFaceDetector: Boolean

    // Pre-allocated buffers — this MobileFaceNet variant has fixed batch=2
    private val inputBuffer: ByteBuffer = ByteBuffer.allocateDirect(4 * 2 * INPUT_SIZE * INPUT_SIZE * 3).apply {
        order(ByteOrder.nativeOrder())
    }
    private val outputArray = Array(2) { FloatArray(EMBEDDING_DIM) }

    init {
        val model = loadTfliteModel(context, MODEL_ASSET)
        gpu = createGpuInterpreter(model, modelName = "MobileFaceNet", cpuThreads = 2)

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
     * Detect the largest face in a person bbox and compute its embedding.
     * Wrapper that builds the person canonical and invokes [embedFace].
     * Returns null if the person crop is too small or no face is found.
     */
    fun embedFace(bitmap: Bitmap, personBox: RectF): FloatArray? {
        val personCanonical = cropper.prepare(
            bitmap, personBox,
            targetWidth = PERSON_CANONICAL_SIZE, targetHeight = PERSON_CANONICAL_SIZE,
            minSourcePixels = MIN_PERSON_SOURCE_PIXELS,
        ) ?: return null
        return try {
            embedFace(personCanonical)
        } finally {
            personCanonical.bitmap.recycle()
        }
    }

    /**
     * Compute face embedding from a prepared person [CanonicalCrop]. Runs
     * BlazeFace on the canonical, picks the largest face, builds a face
     * sub-canonical, runs MobileFaceNet. Caller owns [personCanonical] and
     * is responsible for recycling.
     */
    @Synchronized
    fun embedFace(personCanonical: CanonicalCrop): FloatArray? {
        val personCrop = personCanonical.bitmap
        if (personCrop.width < 30 || personCrop.height < 30) return null
        return try {
            val mpImage = BitmapImageBuilder(personCrop).build()
            val faces = synchronized(faceDetector) { faceDetector.detect(mpImage) }
            if (faces.detections().isEmpty()) return null

            // Largest face wins.
            val face = faces.detections().maxByOrNull {
                it.boundingBox().width() * it.boundingBox().height()
            }!!
            val faceNormBox = normalizeFaceBox(face.boundingBox(), personCrop.width, personCrop.height)
                ?: return null

            // Build the face sub-canonical from the person canonical bitmap.
            val faceCanonical = cropper.prepare(
                personCrop, faceNormBox,
                targetWidth = INPUT_SIZE, targetHeight = INPUT_SIZE,
                paddingFraction = 0f,    // BlazeFace bbox already includes some context
                minSourcePixels = 16,    // faces can be small; let MobileFaceNet decide
            ) ?: return null

            try {
                fillInputBuffer(faceCanonical.bitmap)
                interpreter.run(inputBuffer, outputArray)

                val embedding = outputArray[0].copyOf()
                com.haptictrack.tracking.l2Normalize(embedding)

                Log.d(TAG, "Face embedding computed (norm after L2: ${"%.2f".format(l2Norm(embedding))})")
                embedding
            } finally {
                faceCanonical.bitmap.recycle()
            }
        } catch (e: Exception) {
            Log.w(TAG, "Face embedding failed: ${e.message}")
            null
        }
    }

    fun close() {
        gpu.close()
        if (ownsFaceDetector) faceDetector.close()
    }

    /**
     * Audit/debug only — runs BlazeFace on a person canonical and returns
     * the 112×112 face canonical fed to MobileFaceNet plus the face bbox
     * and keypoints in person-canonical pixel coordinates. The current
     * pipeline ignores the keypoints, so visualizing them is the whole
     * point — we can see what alignment we're throwing away (#93 will use them).
     * Caller must recycle [DebugFaceCrop.faceCrop] and [DebugFaceCrop.personCrop].
     */
    data class DebugFaceCrop(
        /** The person canonical fed to BlazeFace. Caller recycles. */
        val personCrop: Bitmap,
        /** BlazeFace bbox in personCrop pixel space, or null if no face detected. */
        val faceBoxOnPerson: RectF?,
        /** BlazeFace keypoints in personCrop pixel space (typically 6: 2 eyes, nose, mouth, 2 ears). */
        val keypoints: List<PointF>,
        /** The 112×112 face canonical fed to MobileFaceNet, or null if no face. Caller recycles. */
        val faceCrop: Bitmap?
    )

    fun debugFaceCrop(bitmap: Bitmap, personBox: RectF): DebugFaceCrop? {
        val personCanonical = cropper.prepare(
            bitmap, personBox,
            targetWidth = PERSON_CANONICAL_SIZE, targetHeight = PERSON_CANONICAL_SIZE,
            minSourcePixels = MIN_PERSON_SOURCE_PIXELS,
        ) ?: return null
        val personCrop = personCanonical.bitmap
        return try {
            val mpImage = BitmapImageBuilder(personCrop).build()
            val faces = synchronized(faceDetector) { faceDetector.detect(mpImage) }
            if (faces.detections().isEmpty()) {
                return DebugFaceCrop(personCrop, null, emptyList(), null)
            }
            val face = faces.detections().maxByOrNull {
                it.boundingBox().width() * it.boundingBox().height()
            }!!
            val fb = face.boundingBox()
            val faceBox = RectF(fb.left, fb.top, fb.right, fb.bottom)
            val kps = face.keypoints().orElse(emptyList()).map { kp ->
                PointF(kp.x() * personCrop.width, kp.y() * personCrop.height)
            }
            val faceNormBox = normalizeFaceBox(fb, personCrop.width, personCrop.height)
            val faceSub = if (faceNormBox != null) {
                cropper.prepare(
                    personCrop, faceNormBox,
                    targetWidth = INPUT_SIZE, targetHeight = INPUT_SIZE,
                    paddingFraction = 0f,
                    minSourcePixels = 16,
                )?.bitmap
            } else null
            DebugFaceCrop(personCrop, faceBox, kps, faceSub)
        } catch (e: Exception) {
            personCanonical.bitmap.recycle()
            null
        }
    }

    private fun normalizeFaceBox(fb: android.graphics.RectF, w: Int, h: Int): RectF? {
        val l = (fb.left / w).coerceIn(0f, 1f)
        val t = (fb.top / h).coerceIn(0f, 1f)
        val r = (fb.right / w).coerceIn(0f, 1f)
        val b = (fb.bottom / h).coerceIn(0f, 1f)
        if (r - l <= 0f || b - t <= 0f) return null
        return RectF(l, t, r, b)
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
