package com.haptictrack.tracking

import android.content.Context
import android.graphics.Bitmap
import android.graphics.Matrix
import android.graphics.PointF
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
    private val visualTracker: VisualTracker = VisualTracker(context),
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

    // Count frames where visual tracker has no detector confirmation
    private var vtUnconfirmedFrames = 0
    private var vtConfirmedFrames = 0
    private val VT_MAX_UNCONFIRMED = 10  // ~0.3s at 30fps

    /** Callback: (displayObjects, lockedObject, imageWidth, imageHeight, contour) */
    var onDetectionResult: ((List<TrackedObject>, TrackedObject?, Int, Int, List<PointF>) -> Unit)? = null

    // Contour extraction — disabled until the UI uses it (saves CPU/battery).
    // Enable by setting to true when contour-based overlay is implemented.
    private val contourEnabled = false
    private var cachedContour: List<PointF> = emptyList()
    private var contourFrameCount = 0
    private val CONTOUR_UPDATE_INTERVAL_VT = 2
    private val CONTOUR_UPDATE_INTERVAL_DET = 3

    init {
        val baseOptions = BaseOptions.builder()
            .setModelAssetPath("efficientdet-lite2.tflite")
            .build()

        val options = ObjectDetector.ObjectDetectorOptions.builder()
            .setBaseOptions(baseOptions)
            .setScoreThreshold(0.5f)
            .setMaxResults(5)
            .build()

        detector = ObjectDetector.createFromOptions(context, options)

        // Wire ReacquisitionEngine logs to session logger
        reacquisition.sessionLogger = { msg -> debugCapture.log("[Reacq] $msg") }
    }

    /**
     * Lock onto an object. If a frame bitmap is available, computes and stores
     * the visual embedding for identity-aware re-acquisition.
     */
    fun lockOnObject(trackingId: Int, boundingBox: RectF, label: String?) {
        synchronized(lastFrameLock) {
            val bmp = lastFrameBitmap ?: run {
                reacquisition.lock(trackingId, boundingBox, label, emptyList())
                return
            }

            val augResult = appearanceEmbedder.embedWithAugmentations(bmp, boundingBox)
            // Compute color histogram on the masked crop (single segmentation pass)
            val colorHist = if (augResult.maskedCrop != null) {
                val fullBox = RectF(0f, 0f, 1f, 1f) // crop is already the object
                computeColorHistogram(augResult.maskedCrop, fullBox).also { augResult.maskedCrop.recycle() }
            } else computeColorHistogram(bmp, boundingBox)
            reacquisition.lock(trackingId, boundingBox, label, augResult.embeddings, colorHist)
            visualTracker.init(bmp, boundingBox)

            debugCapture.startSession(label, trackingId)
            debugCapture.log("LOCK id=$trackingId label=$label box=${boundingBox} gallery=${augResult.embeddings.size} colorHist=${colorHist != null}")

            val locked = TrackedObject(trackingId, boundingBox, label)
            debugCapture.capture(DebugEvent.LOCK, bmp, listOf(locked), lockedObject = locked,
                extraInfo = "id=$trackingId label=$label gallery=${augResult.embeddings.size}")
        }
    }

    fun clearLock() {
        debugCapture.log("CLEAR by user")
        debugCapture.endSession()
        reacquisition.clear()
        visualTracker.stop()
        vtUnconfirmedFrames = 0
        vtConfirmedFrames = 0
        cachedContour = emptyList()
        contourFrameCount = 0
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
            // --- Visual tracker: primary frame-to-frame tracking ---
            // When active, it tracks the locked object by pixel correlation.
            // Cross-checked against the detector to prevent drift.
            if (visualTracker.isActive && reacquisition.isLocked && reacquisition.framesLost == 0) {
                val vtResult = visualTracker.update(bitmap)
                if (vtResult != null) {
                    // Run detector anyway to validate and for display
                    val detections = runDetector(bitmap, frameWidth, frameHeight)
                    val tracked = frameTracker.assignIds(detections)

                    // VT returns coords in rotated-image space; unmap to screen space
                    // to match detector boxes (which are already unmapped).
                    val deviceRot = deviceRotationProvider?.invoke() ?: 0
                    val rawBox = vtResult.boundingBox
                    val vtBox = unmapRotation(rawBox.left, rawBox.top, rawBox.right, rawBox.bottom, deviceRot)
                    val confirmed = tracked.any { det ->
                        det.label == reacquisition.lockedLabel &&
                        FrameToFrameTracker.computeIou(det.boundingBox, vtBox) > 0.15f
                    }

                    if (confirmed) {
                        vtUnconfirmedFrames = 0
                        // Sync lastKnownBox in screen coords when detector confirms
                        reacquisition.updateFromVisualTracker(vtBox)

                        // Accumulate embedding from current angle (every 30 confirmed frames ≈ 1s)
                        // Use raw rotated-image coords for embedding (embedder works on rotated bitmap)
                        vtConfirmedFrames++
                        if (vtConfirmedFrames % 30 == 0) {
                            // Use fallback: during confirmed tracking, even an unmasked
                            // embedding is better than no embedding for gallery diversity
                            val emb = appearanceEmbedder.embedWithFallback(bitmap, rawBox)
                            if (emb != null) {
                                reacquisition.addEmbedding(emb)
                                android.util.Log.d("AppearEmbed", "Gallery +1 → ${reacquisition.embeddingGallery.size} (accumulated)")
                            }
                        }
                    } else {
                        vtUnconfirmedFrames++
                    }

                    // If too many frames without detector confirmation, tracker is drifting
                    if (vtUnconfirmedFrames > VT_MAX_UNCONFIRMED) {
                        android.util.Log.w("VisualTracker", "DRIFT detected — $vtUnconfirmedFrames unconfirmed frames, stopping")
                        visualTracker.stop()
                        vtUnconfirmedFrames = 0
                        // Fall through to detector path below
                    } else {
                        val lockedObj = TrackedObject(
                            id = reacquisition.lockedId ?: -1,
                            boundingBox = vtBox,
                            label = reacquisition.lastKnownLabel,
                            confidence = vtResult.confidence
                        )
                        // Include the visual tracker's box, but remove detector
                        // boxes that overlap it to avoid duplicate rectangles.
                        val displayObjects = filter.filter(tracked)
                            .filter { FrameToFrameTracker.computeIou(it.boundingBox, vtBox) < 0.3f } + lockedObj

                        // Update contour periodically, unmap from rotated to screen coords
                        if (contourEnabled) {
                            contourFrameCount++
                            if (contourFrameCount % CONTOUR_UPDATE_INTERVAL_VT == 0) {
                                cachedContour = unmapContour(
                                    appearanceEmbedder.extractContour(bitmap, rawBox),
                                    deviceRot
                                )
                            }
                        }

                        onDetectionResult?.invoke(displayObjects, lockedObj, frameWidth, frameHeight, cachedContour)

                        synchronized(lastFrameLock) {
                            lastFrameBitmap?.recycle()
                            lastFrameBitmap = bitmap.copy(bitmap.config ?: Bitmap.Config.ARGB_8888, false)
                        }
                        return
                    }
                } else {
                    // Visual tracker lost the object — fall through to detector
                    visualTracker.stop()
                    vtUnconfirmedFrames = 0
                }
            }

            // --- Detector path: detection + re-acquisition ---
            val detections = runDetector(bitmap, frameWidth, frameHeight)
            val tracked = frameTracker.assignIds(detections)

            // Compute embeddings when searching OR when the direct ID match might
            // fail (framesLost==0 but tracking ID likely changed after visual tracker
            // handoff). Without embeddings, two same-label objects are indistinguishable.
            val needEmbeddings = reacquisition.hasEmbeddings &&
                (reacquisition.isSearching || (reacquisition.isLocked && reacquisition.framesLost == 0))
            val withEmbeddings = if (needEmbeddings) {
                tracked.map { obj ->
                    // Single segmentation pass: get both embedding and masked crop
                    val result = appearanceEmbedder.embedAndCrop(bitmap, obj.boundingBox)
                    val hist = if (result.maskedCrop != null) {
                        val fullBox = RectF(0f, 0f, 1f, 1f)
                        computeColorHistogram(result.maskedCrop, fullBox).also { result.maskedCrop.recycle() }
                    } else computeColorHistogram(bitmap, obj.boundingBox)
                    obj.copy(embedding = result.embedding, colorHistogram = hist)
                }
            } else tracked

            // Snapshot state before processing
            val wasSearching = reacquisition.isSearching
            val prevLost = reacquisition.framesLost

            // Re-acquisition
            val lockedObject = reacquisition.processFrame(withEmbeddings)

            // If re-acquired, re-initialize visual tracker
            if (wasSearching && lockedObject != null && reacquisition.framesLost == 0) {
                visualTracker.init(bitmap, lockedObject.boundingBox)
            }

            // Save lastFrameBitmap before debug capture to avoid a third bitmap copy.
            // Debug capture draws annotations directly onto a mutable copy of this frame.
            synchronized(lastFrameLock) {
                lastFrameBitmap?.recycle()
                lastFrameBitmap = bitmap.copy(bitmap.config ?: Bitmap.Config.ARGB_8888, false)
            }

            // Debug frame capture on tracking events
            captureDebugFrame(bitmap, withEmbeddings, lockedObject, wasSearching, prevLost)

            // Filter for display
            val displayObjects = filter.filter(withEmbeddings)

            // Update contour on re-acquire or periodically (gated — disabled until UI uses it)
            if (contourEnabled) {
                val deviceRot = deviceRotationProvider?.invoke() ?: 0
                if (lockedObject != null && reacquisition.framesLost == 0) {
                    contourFrameCount++
                    if (contourFrameCount % CONTOUR_UPDATE_INTERVAL_DET == 0 || (wasSearching && reacquisition.framesLost == 0)) {
                        val screenBox = lockedObject.boundingBox
                        val rotatedBox = mapToRotated(screenBox.left, screenBox.top, screenBox.right, screenBox.bottom, deviceRot)
                        cachedContour = unmapContour(
                            appearanceEmbedder.extractContour(bitmap, rotatedBox),
                            deviceRot
                        )
                    }
                } else if (!reacquisition.isLocked) {
                    cachedContour = emptyList()
                    contourFrameCount = 0
                }
            }

            onDetectionResult?.invoke(displayObjects, lockedObject, frameWidth, frameHeight, cachedContour)
        } finally {
            imageProxy.close()
            bitmap.recycle()
        }
    }

    private fun runDetector(bitmap: Bitmap, frameWidth: Int, frameHeight: Int): List<TrackedObject> {
        val mpImage = BitmapImageBuilder(bitmap).build()
        val result: ObjectDetectorResult = detector.detect(mpImage)
        val deviceRot = deviceRotationProvider?.invoke() ?: 0

        return result.detections().map { detection ->
            val box = detection.boundingBox()
            val normLeft = box.left / frameWidth.toFloat()
            val normTop = box.top / frameHeight.toFloat()
            val normRight = box.right / frameWidth.toFloat()
            val normBottom = box.bottom / frameHeight.toFloat()

            val screenBox = unmapRotation(normLeft, normTop, normRight, normBottom, deviceRot)

            TrackedObject(
                id = -1,
                boundingBox = screenBox,
                label = detection.categories().firstOrNull()?.categoryName(),
                confidence = detection.categories().firstOrNull()?.score() ?: 0f
            )
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
                debugCapture.log("REACQUIRE id=${lockedObject.id} label=${lockedObject.label} after $prevFramesLost frames box=${lockedObject.boundingBox}")
                debugCapture.capture(
                    DebugEvent.REACQUIRE, bitmap, detections,
                    lockedObject = lockedObject,
                    extraInfo = "after $prevFramesLost frames, id=${lockedObject.id} label=${lockedObject.label}"
                )
            }
            // Just lost (first frame)
            nowLost == 1 && prevFramesLost == 0 -> {
                debugCapture.log("LOST lockedLabel=${reacquisition.lockedLabel} ${detections.size} candidates")
                debugCapture.capture(
                    DebugEvent.LOST, bitmap, detections,
                    lastKnownBox = reacquisition.lastKnownBox,
                    extraInfo = "label=${reacquisition.lockedLabel}"
                )
            }
            // Searching
            wasSearching && lockedObject == null && nowLost > 0 -> {
                val hasSameLabelCandidate = detections.any { d ->
                    d.label != null && d.label == reacquisition.lockedLabel
                }
                if (nowLost % 10 == 1) {
                    debugCapture.log("SEARCH frame=$nowLost candidates=${detections.size} sameLabelMatch=$hasSameLabelCandidate")
                }
                if (hasSameLabelCandidate || nowLost % 10 == 0) {
                    val candidateInfo = detections
                        .filter { it.label == reacquisition.lockedLabel }
                        .joinToString(", ") { "#${it.id} ${it.confidence.times(100).toInt()}%" }
                    debugCapture.capture(
                        DebugEvent.SEARCH, bitmap, detections,
                        lastKnownBox = reacquisition.lastKnownBox,
                        extraInfo = "frame=$nowLost match=[${candidateInfo.ifEmpty { "none" }}]"
                    )
                }
            }
            // Timed out
            nowLost == reacquisition.maxFramesLost + 1 -> {
                debugCapture.log("TIMEOUT after ${reacquisition.maxFramesLost} frames, gave up on ${reacquisition.lockedLabel}")
                debugCapture.endSession()
                debugCapture.capture(
                    DebugEvent.TIMEOUT, bitmap, detections,
                    lastKnownBox = reacquisition.lastKnownBox,
                    extraInfo = "gave up on ${reacquisition.lockedLabel}"
                )
            }
        }
    }

    fun shutdown() {
        debugCapture.endSession()
        detector.close()
        appearanceEmbedder.shutdown()
        visualTracker.stop()
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

/**
 * Forward-maps screen coordinates to rotated-image space (inverse of unmapRotation).
 * Used to convert screen-space bounding boxes back to the rotated bitmap's coordinate system.
 */
internal fun mapToRotated(
    left: Float, top: Float, right: Float, bottom: Float, deviceRot: Int
): RectF {
    // The forward map is the inverse of the unmap.
    // unmapRotation undoes the rotation, so mapToRotated re-applies it.
    return when (deviceRot) {
        0 -> RectF(left, top, right, bottom)
        180 -> RectF(1f - right, 1f - bottom, 1f - left, 1f - top)
        90 -> {
            // Inverse of unmap 90°: (x,y) → (1-y, x)
            RectF(1f - bottom, left, 1f - top, right)
        }
        270 -> {
            // Inverse of unmap 270°: (x,y) → (y, 1-x)
            RectF(top, 1f - right, bottom, 1f - left)
        }
        else -> RectF(left, top, right, bottom)
    }
}

/**
 * Unmap a single normalized point from rotated-image space to screen space.
 */
internal fun unmapPoint(x: Float, y: Float, deviceRot: Int): PointF {
    return when (deviceRot) {
        0 -> PointF(x, y)
        180 -> PointF(1f - x, 1f - y)
        90 -> PointF(y, 1f - x)      // inverse of 90° CW
        270 -> PointF(1f - y, x)      // inverse of 270° CW
        else -> PointF(x, y)
    }
}

/**
 * Apply unmapRotation to each contour point (normalized [0,1] coords).
 */
internal fun unmapContour(
    contour: List<PointF>, deviceRot: Int
): List<PointF> {
    if (deviceRot == 0 || contour.isEmpty()) return contour
    return contour.map { pt -> unmapPoint(pt.x, pt.y, deviceRot) }
}
