package com.haptictrack.tracking

import android.content.Context
import android.graphics.Bitmap
import android.graphics.Color
import android.graphics.PointF
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
data class FaceFramingLocal(val faceBoxInPerson: RectF, val yawDeg: Float?)

class FaceEmbedder(
    context: Context,
    sharedFaceDetector: FaceDetector? = null,
    private val cropper: CanonicalCropper = CanonicalCropper(),
    /** Optional gender/age classifier — when set, [classifyAttributes] is available. */
    private val attributeClassifier: FaceAttributeClassifier? = null,
) {

    companion object {
        private const val TAG = "FaceEmbed"
        private const val MODEL_ASSET = "mobilefacenet.tflite"
        private const val FACE_MODEL_ASSET = "blaze_face_short_range.tflite"
        const val INPUT_SIZE = 112
        /** Attribute-classifier crop size (FairFace = 224) — aligned separately from the identity crop. */
        private const val ATTR_SIZE = 224
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
    private val appContext = context.applicationContext

    /** Debug: when true, classifyAttributes saves the exact 96² crops it feeds genderage. */
    var debugSaveAttributeCrops = false

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
            faceDetector = try {
                FaceDetector.createFromOptions(context, FaceDetector.FaceDetectorOptions.builder()
                    .setBaseOptions(BaseOptions.builder().setModelAssetPath(FACE_MODEL_ASSET)
                        .setDelegate(Delegate.GPU).build())
                    .setMinDetectionConfidence(FACE_MIN_CONFIDENCE).build())
            } catch (e: Exception) {
                Log.w(TAG, "BlazeFace GPU failed, falling back to CPU: ${e.message}")
                FaceDetector.createFromOptions(context, FaceDetector.FaceDetectorOptions.builder()
                    .setBaseOptions(BaseOptions.builder().setModelAssetPath(FACE_MODEL_ASSET).build())
                    .setMinDetectionConfidence(FACE_MIN_CONFIDENCE).build())
            }
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

    /**
     * Detect the largest face in a person bbox and classify gender/age. Unlike
     * [embedFace] (which letterboxes the face box — fine for MobileFaceNet
     * identity), the FairFace age head is alignment-sensitive, so we warp the
     * face to the ArcFace template using BlazeFace's eye keypoints — a 2-point
     * (eyes-only) similarity, scaled to ATTR_SIZE. NB: the off-device FairFace
     * eval (tools/age_gender_eval/eval_fairface.py) used a 5-point warp, so
     * on-device attribute accuracy should be re-confirmed against it. Falls back
     * to a letterbox crop if keypoints are unavailable. Returns null if no face
     * / no classifier / crop too small.
     */
    @Synchronized
    fun classifyAttributes(bitmap: Bitmap, personBox: RectF): FaceAttributes? {
        val classifier = attributeClassifier ?: return null
        val personCanonical = cropper.prepare(
            bitmap, personBox,
            targetWidth = PERSON_CANONICAL_SIZE, targetHeight = PERSON_CANONICAL_SIZE,
            minSourcePixels = MIN_PERSON_SOURCE_PIXELS,
        ) ?: return null
        val personCrop = personCanonical.bitmap
        return try {
            val mpImage = BitmapImageBuilder(personCrop).build()
            val faces = synchronized(faceDetector) { faceDetector.detect(mpImage) }
            val face = faces.detections().maxByOrNull {
                it.boundingBox().width() * it.boundingBox().height()
            } ?: return null

            val kps = face.keypoints().orElse(emptyList())
            val aligned = alignFaceToTemplate(personCrop, kps)
            // Fallback letterbox crop too (for A/B comparison of alignment effect).
            val faceNormBox = normalizeFaceBox(face.boundingBox(), personCrop.width, personCrop.height)
            val letterbox = faceNormBox?.let {
                cropper.prepare(personCrop, it, targetWidth = ATTR_SIZE, targetHeight = ATTR_SIZE,
                    paddingFraction = 0f, minSourcePixels = 16)?.bitmap
            }
            if (debugSaveAttributeCrops) saveDebugCrops(personCrop, aligned, letterbox, kps)

            return when {
                aligned != null -> try { classifier.classify(aligned) } finally {
                    aligned.recycle(); letterbox?.recycle()
                }
                letterbox != null -> try { classifier.classify(letterbox) } finally { letterbox.recycle() }
                else -> null
            }
        } catch (e: Exception) {
            Log.w(TAG, "Attribute classify failed: ${e.message}")
            null
        } finally {
            personCrop.recycle()
        }
    }

    /**
     * Warp the face in [personCrop] to a ATTR_SIZE² ArcFace-aligned crop
     * using BlazeFace eye keypoints (index 0 = right eye, 1 = left eye, in
     * normalized personCrop coords). A 2-point similarity (Android setPolyToPoly,
     * pointCount=2) places the eyes on the template eye line — rotation + scale +
     * position — which is what genderage's age head needs. Null if <2 keypoints.
     */
    private fun alignFaceToTemplate(personCrop: Bitmap, kps: List<com.google.mediapipe.tasks.components.containers.NormalizedKeypoint>): Bitmap? {
        if (kps.size < 2) return null
        val w = personCrop.width; val h = personCrop.height
        val rightEyeX = kps[0].x() * w; val rightEyeY = kps[0].y() * h
        val leftEyeX = kps[1].x() * w; val leftEyeY = kps[1].y() * h
        // ArcFace eye template (112) scaled to ATTR_SIZE. Template index 0 is
        // image-left = subject's right eye = BlazeFace kp[0].
        val s = ATTR_SIZE / 112f
        val src = floatArrayOf(rightEyeX, rightEyeY, leftEyeX, leftEyeY)
        val dst = floatArrayOf(38.2946f * s, 51.6963f * s, 73.5318f * s, 51.5014f * s)
        val m = android.graphics.Matrix()
        if (!m.setPolyToPoly(src, 0, dst, 0, 2)) return null
        val out = Bitmap.createBitmap(ATTR_SIZE, ATTR_SIZE, Bitmap.Config.ARGB_8888)
        android.graphics.Canvas(out).drawBitmap(personCrop, m, android.graphics.Paint(android.graphics.Paint.FILTER_BITMAP_FLAG))
        return out
    }

    private fun saveDebugCrops(
        personCrop: Bitmap, aligned: Bitmap?, letterbox: Bitmap?,
        kps: List<com.google.mediapipe.tasks.components.containers.NormalizedKeypoint>
    ) {
        try {
            val dir = java.io.File(appContext.getExternalFilesDir(null), "sentry_debug").apply { mkdirs() }
            fun save(bmp: Bitmap?, name: String) = bmp?.let {
                java.io.File(dir, name).outputStream().use { os -> it.compress(Bitmap.CompressFormat.PNG, 100, os) }
            }
            save(personCrop, "person.png")
            save(aligned, "aligned.png")
            save(letterbox, "letterbox.png")
            java.io.File(dir, "keypoints.txt").writeText(
                "personCrop=${personCrop.width}x${personCrop.height}\n" +
                kps.mapIndexed { i, k -> "kp$i=${k.x()},${k.y()}" }.joinToString("\n")
            )
            Log.i(TAG, "Saved attribute debug crops to ${dir.absolutePath} (kps=${kps.size})")
        } catch (e: Exception) {
            Log.w(TAG, "saveDebugCrops failed: ${e.message}")
        }
    }

    /** Face box (normalized within personBox) + coarse yaw for the largest face. Null if none. */
    @Synchronized
    fun detectFaceFraming(bitmap: Bitmap, personBox: RectF): FaceFramingLocal? {
        val personCanonical = cropper.prepare(
            bitmap, personBox,
            targetWidth = PERSON_CANONICAL_SIZE, targetHeight = PERSON_CANONICAL_SIZE,
            paddingFraction = 0f, minSourcePixels = MIN_PERSON_SOURCE_PIXELS,
        ) ?: return null
        val personCrop = personCanonical.bitmap
        return try {
            val mpImage = BitmapImageBuilder(personCrop).build()
            val faces = synchronized(faceDetector) { faceDetector.detect(mpImage) }
            val face = faces.detections().maxByOrNull {
                it.boundingBox().width() * it.boundingBox().height()
            } ?: return null
            val box = normalizeFaceBox(face.boundingBox(), personCrop.width, personCrop.height)
                ?: return null
            val kps = face.keypoints().orElse(emptyList())
            val yaw = if (kps.size >= 3)
                estimateYawDeg(
                    PointF(kps[0].x(), kps[0].y()),
                    PointF(kps[1].x(), kps[1].y()),
                    PointF(kps[2].x(), kps[2].y()),
                ) else null
            FaceFramingLocal(box, yaw)
        } catch (e: Exception) {
            Log.w(TAG, "detectFaceFraming failed: ${e.message}"); null
        } finally {
            personCrop.recycle()
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
