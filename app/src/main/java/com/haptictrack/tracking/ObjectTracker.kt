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
    private val appearanceEmbedder: AppearanceEmbedder = AppearanceEmbedder(context),
    val debugCapture: DebugFrameCapture = DebugFrameCapture(context),
    /** Provides physical device orientation; set from ViewModel. */
    var deviceRotationProvider: (() -> Int)? = null
) {

    private val detector: ObjectDetector

    // Keep last frame for computing embedding when user taps to lock
    private val lastFrameLock = Any()
    private var lastFrameBitmap: Bitmap? = null

    // Track previous state to detect transitions
    private var previouslyLocked: Boolean = false
    private var previousFramesLost: Int = 0

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

        // Capture debug frame on lock
        synchronized(lastFrameLock) {
            lastFrameBitmap?.let { bmp ->
                val locked = TrackedObject(trackingId, boundingBox, label)
                debugCapture.capture("LOCK", bmp, listOf(locked), lockedObject = locked,
                    extraInfo = "id=$trackingId label=$label hasEmbed=${embedding != null}")
            }
        }
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

            val deviceRot = deviceRotationProvider?.invoke() ?: 0

            val rawDetections = result.detections().map { detection ->
                val box = detection.boundingBox()
                val normLeft = box.left / frameWidth.toFloat()
                val normTop = box.top / frameHeight.toFloat()
                val normRight = box.right / frameWidth.toFloat()
                val normBottom = box.bottom / frameHeight.toFloat()

                // Detection ran on a rotated bitmap. Map coordinates back
                // to the original screen orientation by applying the inverse
                // of the device rotation we added.
                val screenBox = unmapRotation(normLeft, normTop, normRight, normBottom, deviceRot)

                TrackedObject(
                    id = -1,
                    boundingBox = screenBox,
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

            // Snapshot state before processing
            val wasSearching = reacquisition.isSearching
            val prevLost = reacquisition.framesLost

            // Re-acquisition
            val lockedObject = reacquisition.processFrame(withEmbeddings)

            // Debug frame capture on tracking events
            captureDebugFrame(bitmap, withEmbeddings, lockedObject, wasSearching, prevLost)

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

        // CameraX rotationDegrees assumes the display matches the activity's
        // declared orientation (portrait). It does NOT account for the user
        // physically holding the phone in landscape or upside down.
        // We add the physical device rotation so the image fed to the detector
        // is always upright relative to the real world.
        val cameraRotation = imageProxy.imageInfo.rotationDegrees
        val deviceRot = deviceRotationProvider?.invoke() ?: 0
        val rotation = (cameraRotation + deviceRot) % 360

        if (rotation == 0) return bitmap

        val matrix = Matrix().apply { postRotate(rotation.toFloat()) }
        val rotated = Bitmap.createBitmap(bitmap, 0, 0, bitmap.width, bitmap.height, matrix, true)
        if (rotated !== bitmap) bitmap.recycle()
        return rotated
    }

    private fun captureDebugFrame(
        bitmap: Bitmap,
        detections: List<TrackedObject>,
        lockedObject: TrackedObject?,
        wasSearching: Boolean,
        prevFramesLost: Int
    ) {
        val nowLost = reacquisition.framesLost

        when {
            // Just re-acquired
            wasSearching && lockedObject != null && nowLost == 0 -> {
                debugCapture.capture(
                    "REACQUIRE", bitmap, detections,
                    lockedObject = lockedObject,
                    extraInfo = "after $prevFramesLost frames, id=${lockedObject.id} label=${lockedObject.label}"
                )
            }
            // Just lost (first frame)
            nowLost == 1 && prevFramesLost == 0 -> {
                debugCapture.capture(
                    "LOST", bitmap, detections,
                    lastKnownBox = reacquisition.lastKnownBox,
                    extraInfo = "label=${reacquisition.lockedLabel}"
                )
            }
            // Searching: capture every 10th frame, or whenever a same-label candidate exists
            // but didn't match — this is the key diagnostic frame
            wasSearching && lockedObject == null && nowLost > 0 -> {
                val hasSameLabelCandidate = detections.any { d ->
                    d.label != null && d.label == reacquisition.lockedLabel
                }
                if (hasSameLabelCandidate || nowLost % 10 == 0) {
                    val candidateInfo = detections
                        .filter { it.label == reacquisition.lockedLabel }
                        .joinToString(", ") { "#${it.id} ${it.confidence.times(100).toInt()}%" }
                    debugCapture.capture(
                        "SEARCH", bitmap, detections,
                        lastKnownBox = reacquisition.lastKnownBox,
                        extraInfo = "frame=$nowLost match=[${candidateInfo.ifEmpty { "none" }}]"
                    )
                }
            }
            // Timed out
            nowLost == reacquisition.maxFramesLost + 1 -> {
                debugCapture.capture(
                    "TIMEOUT", bitmap, detections,
                    lastKnownBox = reacquisition.lastKnownBox,
                    extraInfo = "gave up on ${reacquisition.lockedLabel}"
                )
            }
        }
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

/**
 * Reverse-maps normalized bounding box coordinates from the rotated
 * detection image back to the original screen coordinate space.
 *
 * After rotating the bitmap by [deviceRot] degrees for detection,
 * we apply the inverse rotation to the detected box coordinates.
 */
internal fun unmapRotation(
    left: Float, top: Float, right: Float, bottom: Float, deviceRot: Int
): RectF {
    return when (deviceRot) {
        0 -> RectF(left, top, right, bottom)
        180 -> RectF(1f - right, 1f - bottom, 1f - left, 1f - top)
        90 -> {
            // Detection image was rotated 90° CW extra.
            // Inverse: rotate 90° CCW → (x,y) → (y, 1-x)
            RectF(top, 1f - right, bottom, 1f - left)
        }
        270 -> {
            // Detection image was rotated 270° CW (= 90° CCW) extra.
            // Inverse: rotate 90° CW → (x,y) → (1-y, x)
            RectF(1f - bottom, left, 1f - top, right)
        }
        else -> RectF(left, top, right, bottom)
    }
}
