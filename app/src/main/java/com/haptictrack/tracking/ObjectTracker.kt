package com.haptictrack.tracking

import android.graphics.RectF
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.ImageProxy
import com.google.mlkit.vision.common.InputImage
import com.google.mlkit.vision.objects.ObjectDetection
import com.google.mlkit.vision.objects.ObjectDetector
import com.google.mlkit.vision.objects.defaults.ObjectDetectorOptions

class ObjectTracker(
    val reacquisition: ReacquisitionEngine = ReacquisitionEngine()
) {

    private val detector: ObjectDetector

    var onDetectionResult: ((List<TrackedObject>, TrackedObject?) -> Unit)? = null

    init {
        val options = ObjectDetectorOptions.Builder()
            .setDetectorMode(ObjectDetectorOptions.STREAM_MODE)
            .enableMultipleObjects()
            .enableClassification()
            .build()
        detector = ObjectDetection.getClient(options)
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

    @androidx.camera.core.ExperimentalGetImage
    private fun processImage(imageProxy: ImageProxy) {
        val mediaImage = imageProxy.image ?: run {
            imageProxy.close()
            return
        }

        val inputImage = InputImage.fromMediaImage(mediaImage, imageProxy.imageInfo.rotationDegrees)
        val imageWidth = inputImage.width.toFloat()
        val imageHeight = inputImage.height.toFloat()

        detector.process(inputImage)
            .addOnSuccessListener { detectedObjects ->
                val tracked = detectedObjects.map { obj ->
                    TrackedObject(
                        id = obj.trackingId ?: -1,
                        boundingBox = RectF(
                            obj.boundingBox.left / imageWidth,
                            obj.boundingBox.top / imageHeight,
                            obj.boundingBox.right / imageWidth,
                            obj.boundingBox.bottom / imageHeight
                        ),
                        label = obj.labels.firstOrNull()?.text,
                        confidence = obj.labels.firstOrNull()?.confidence ?: 0f
                    )
                }

                val lockedObject = reacquisition.processFrame(tracked)
                onDetectionResult?.invoke(tracked, lockedObject)
            }
            .addOnCompleteListener {
                imageProxy.close()
            }
    }

    fun shutdown() {
        detector.close()
    }
}
