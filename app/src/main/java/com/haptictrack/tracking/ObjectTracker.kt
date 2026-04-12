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
    private val frameTracker: FrameToFrameTracker = FrameToFrameTracker()
) {

    private val detector: ObjectDetector

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

    fun lockOnObject(trackingId: Int, boundingBox: RectF, label: String?) {
        reacquisition.lock(trackingId, boundingBox, label)
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

            // Re-acquisition
            val lockedObject = reacquisition.processFrame(tracked)

            // Filter for display
            val displayObjects = filter.filter(tracked)

            onDetectionResult?.invoke(displayObjects, lockedObject, frameWidth, frameHeight)
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
    }
}
