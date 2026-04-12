package com.haptictrack.tracking

import android.content.Context
import android.graphics.Bitmap
import android.graphics.Matrix
import android.graphics.RectF
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.ImageProxy
import com.google.mediapipe.framework.image.BitmapImageBuilder
import com.google.mediapipe.tasks.core.BaseOptions
import com.google.mediapipe.tasks.vision.objectdetector.ObjectDetector
import com.google.mediapipe.tasks.vision.objectdetector.ObjectDetectorResult

class ObjectTracker(
    context: Context,
    val reacquisition: ReacquisitionEngine = ReacquisitionEngine(),
    val filter: DetectionFilter = DetectionFilter(),
    private val frameTracker: FrameToFrameTracker = FrameToFrameTracker(),
    private val appearanceEmbedder: AppearanceEmbedder = AppearanceEmbedder(context)
) {

    private val detector: ObjectDetector

    // Keep last frame for computing embedding when user taps to lock
    private val lastFrameLock = Any()
    private var lastFrameBitmap: Bitmap? = null

    /** Callback: (displayObjects, lockedObject, imageWidth, imageHeight) */
    var onDetectionResult: ((List<TrackedObject>, TrackedObject?, Int, Int) -> Unit)? = null

    init {
        val baseOptions = BaseOptions.builder()
            .setModelAssetPath("efficientdet-lite0.tflite")
            .build()

        val options = ObjectDetector.ObjectDetectorOptions.builder()
            .setBaseOptions(baseOptions)
            .setScoreThreshold(0.5f)
            .setMaxResults(5)
            .build()

        detector = ObjectDetector.createFromOptions(context, options)
    }

    /**
     * Lock onto an object. If a frame bitmap is available, computes and stores
     * the visual embedding for identity-aware re-acquisition.
     */
    fun lockOnObject(trackingId: Int, boundingBox: RectF, label: String?) {
        val embedding = synchronized(lastFrameLock) {
            lastFrameBitmap?.let { bmp ->
                appearanceEmbedder.embed(bmp, boundingBox)
            }
        }
        reacquisition.lock(trackingId, boundingBox, label, embedding)
    }

    fun clearLock() {
        reacquisition.clear()
    }

    val analyzer = ImageAnalysis.Analyzer { imageProxy ->
        processImage(imageProxy)
    }

    private fun processImage(imageProxy: ImageProxy) {
        val bitmap = imageProxyToBitmap(imageProxy)
        if (bitmap == null) {
            imageProxy.close()
            return
        }

        val frameWidth = bitmap.width
        val frameHeight = bitmap.height

        try {
            val mpImage = BitmapImageBuilder(bitmap).build()
            val result: ObjectDetectorResult = detector.detect(mpImage)

            val rawDetections = result.detections().map { detection ->
                val box = detection.boundingBox()
                TrackedObject(
                    id = -1, // MediaPipe doesn't provide tracking IDs
                    boundingBox = RectF(
                        box.left / frameWidth.toFloat(),
                        box.top / frameHeight.toFloat(),
                        box.right / frameWidth.toFloat(),
                        box.bottom / frameHeight.toFloat()
                    ),
                    label = detection.categories().firstOrNull()?.categoryName(),
                    confidence = detection.categories().firstOrNull()?.score() ?: 0f
                )
            }

            // Assign stable tracking IDs via frame-to-frame IoU matching
            val tracked = frameTracker.assignIds(rawDetections)

            // Compute embeddings when re-acquiring (not every frame — only when searching)
            val withEmbeddings = if (reacquisition.isSearching && reacquisition.lockedEmbedding != null) {
                tracked.map { obj ->
                    val emb = appearanceEmbedder.embed(bitmap, obj.boundingBox)
                    if (emb != null) obj.copy(embedding = emb) else obj
                }
            } else tracked

            // Re-acquisition
            val lockedObject = reacquisition.processFrame(withEmbeddings)

            // Filter for display
            val displayObjects = filter.filter(withEmbeddings)

            onDetectionResult?.invoke(displayObjects, lockedObject, frameWidth, frameHeight)

            // Keep a copy of the frame for embedding on tap-to-lock
            synchronized(lastFrameLock) {
                lastFrameBitmap?.recycle()
                lastFrameBitmap = bitmap.copy(bitmap.config ?: Bitmap.Config.ARGB_8888, false)
            }
        } finally {
            imageProxy.close()
            bitmap.recycle()
        }
    }

    private fun imageProxyToBitmap(imageProxy: ImageProxy): Bitmap? {
        val bitmap = imageProxy.toBitmap()
        val rotation = imageProxy.imageInfo.rotationDegrees
        if (rotation == 0) return bitmap

        val matrix = Matrix().apply { postRotate(rotation.toFloat()) }
        val rotated = Bitmap.createBitmap(bitmap, 0, 0, bitmap.width, bitmap.height, matrix, true)
        if (rotated !== bitmap) bitmap.recycle()
        return rotated
    }

    fun shutdown() {
        detector.close()
        appearanceEmbedder.shutdown()
        synchronized(lastFrameLock) {
            lastFrameBitmap?.recycle()
            lastFrameBitmap = null
        }
    }
}
