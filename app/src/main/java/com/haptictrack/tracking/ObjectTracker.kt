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
import com.google.mediapipe.tasks.core.Delegate
import com.google.mediapipe.tasks.vision.objectdetector.ObjectDetector
import com.google.mediapipe.tasks.vision.objectdetector.ObjectDetectorResult

class ObjectTracker(
    context: Context,
    private val onLoadingStatus: ((String) -> Unit)? = null,
    val reacquisition: ReacquisitionEngine = ReacquisitionEngine(),
    val filter: DetectionFilter = DetectionFilter(),
    private val frameTracker: FrameToFrameTracker = FrameToFrameTracker(),
    val debugCapture: DebugFrameCapture = DebugFrameCapture(context),
    private val visualTracker: VisualTracker = VisualTracker(context),
    /** Provides physical device orientation; set from ViewModel. */
    var deviceRotationProvider: (() -> Int)? = null
) {

    private val detector: ObjectDetector
    private val labelEnricher: Yolov8Detector
    private val appearanceEmbedder: AppearanceEmbedder
    private val personClassifier: PersonAttributeClassifier
    private val faceEmbedder: FaceEmbedder
    private val personReId: PersonReIdEmbedder
    private val scenarioRecorder = ScenarioRecorder()

    // Keep last frame for computing embedding when user taps to lock
    private val lastFrameLock = Any()
    private var lastFrameBitmap: Bitmap? = null

    // Track previous state to detect transitions
    private var previouslyLocked: Boolean = false
    private var previousFramesLost: Int = 0

    // Count frames where visual tracker has no detector confirmation
    private var vtUnconfirmedFrames = 0
    private var vtConfirmedFrames = 0
    private val VT_MAX_UNCONFIRMED = 10  // ~0.3s at 30fps (baseline for slow subjects)

    /** Skip detector every other frame when VT is confident and recently confirmed. */
    private var vtFrameCounter = 0
    private val VT_SKIP_INTERVAL = 2  // run detector every Nth frame when skipping
    private val VT_SKIP_MIN_CONFIRMED = 5  // need this many confirmations before skipping

    /** Tracks object velocity for adaptive drift detection and position prediction. */
    private val velocityEstimator = VelocityEstimator()
    private val VT_ADAPTIVE_UNCONFIRMED_HIGH = 5      // ~165ms for fast subjects
    private val VT_ADAPTIVE_UNCONFIRMED_VERY_HIGH = 3  // ~100ms for very fast subjects

    /** Async embedding pipeline: compute embeddings on background thread, use results next frame. */
    private val embeddingExecutor = java.util.concurrent.Executors.newSingleThreadExecutor()
    private var pendingEmbeddings: java.util.concurrent.Future<Map<Int, EmbeddingResult>>? = null

    /** Cached embedding results from previous frame, keyed by detection ID. */
    data class EmbeddingResult(
        val embedding: FloatArray? = null,
        val colorHistogram: FloatArray? = null,
        val personAttributes: PersonAttributes? = null,
        val reIdEmbedding: FloatArray? = null,
        val faceEmbedding: FloatArray? = null,
        /** Bounding box at computation time, for IoU-based matching to next frame. */
        val cachedBox: RectF? = null
    )

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
        onLoadingStatus?.invoke("Loading embedder (GPU)...")
        appearanceEmbedder = AppearanceEmbedder(context)

        onLoadingStatus?.invoke("Loading person classifier (GPU)...")
        personClassifier = PersonAttributeClassifier(context)

        onLoadingStatus?.invoke("Loading detector (GPU)...")
        val baseOptions = BaseOptions.builder()
            .setModelAssetPath("efficientdet-lite2-fp16.tflite")
            .setDelegate(Delegate.GPU)
            .build()

        val options = ObjectDetector.ObjectDetectorOptions.builder()
            .setBaseOptions(baseOptions)
            .setScoreThreshold(0.5f)
            .setMaxResults(5)
            .build()

        detector = ObjectDetector.createFromOptions(context, options)

        onLoadingStatus?.invoke("Loading YOLOv8 (GPU)...")
        labelEnricher = Yolov8Detector(context)

        onLoadingStatus?.invoke("Loading face models (GPU)...")
        faceEmbedder = FaceEmbedder(context, personClassifier.faceDetector)
        personReId = PersonReIdEmbedder(context)

        // Wire ReacquisitionEngine logs to session logger
        reacquisition.sessionLogger = { msg -> debugCapture.log("[Reacq] $msg") }
        onLoadingStatus?.invoke("Ready")
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

            // Enrich coarse COCO label with finer OIV7 label (one-shot, ~270ms)
            val enrichedLabel = labelEnricher.enrichLabel(bmp, boundingBox, label) ?: label

            val augResult = appearanceEmbedder.embedWithAugmentations(bmp, boundingBox)
            // Compute color histogram on the masked crop (single segmentation pass)
            val colorHist = if (augResult.maskedCrop != null) {
                val fullBox = RectF(0f, 0f, 1f, 1f) // crop is already the object
                computeColorHistogram(augResult.maskedCrop, fullBox).also { augResult.maskedCrop.recycle() }
            } else computeColorHistogram(bmp, boundingBox)
            // Classify person attributes at lock time (use original COCO label for "person" check)
            val personAttrs = personClassifier.classify(bmp, boundingBox, label)
            // Person re-ID + face embeddings (only for person labels)
            val isPerson = label == "person"
            val reIdEmb = if (isPerson) personReId.embed(bmp, boundingBox) else null
            val faceEmb = if (isPerson) faceEmbedder.embedFace(bmp, boundingBox) else null
            reacquisition.lock(trackingId, boundingBox, enrichedLabel, augResult.embeddings, colorHist, personAttrs,
                cocoLabel = label, reIdEmbedding = reIdEmb, faceEmbedding = faceEmb)
            visualTracker.init(bmp, boundingBox)

            debugCapture.startSession(enrichedLabel, trackingId)
            val attrStr = personAttrs?.summary() ?: "n/a"
            debugCapture.log("LOCK id=$trackingId label=$enrichedLabel (coco=$label) box=${boundingBox} gallery=${augResult.embeddings.size} colorHist=${colorHist != null} attrs=\"$attrStr\"")

            // Start scenario recording for replay testing
            debugCapture.sessionDir?.let { dir ->
                scenarioRecorder.start(dir, trackingId, enrichedLabel, label,
                    boundingBox, augResult.embeddings, colorHist, personAttrs)
            }

            val locked = TrackedObject(trackingId, boundingBox, enrichedLabel)
            debugCapture.capture(DebugEvent.LOCK, bmp, listOf(locked), lockedObject = locked,
                extraInfo = "id=$trackingId label=$label gallery=${augResult.embeddings.size}")
        }
    }

    /**
     * Prepare for a camera rebind. Stops visual tracker and forces
     * re-acquisition engine into search mode so it recovers on the
     * first frame after the camera restarts.
     */
    fun prepareForRebind() {
        if (!reacquisition.isLocked) return
        visualTracker.stop()
        vtUnconfirmedFrames = 0
        vtFrameCounter = 0
        velocityEstimator.reset()
        pendingEmbeddings?.cancel(true)
        pendingEmbeddings = null
        reacquisition.prepareForRebind()
    }

    fun clearLock() {
        scenarioRecorder.recordEvent("CLEAR")
        scenarioRecorder.stop()
        debugCapture.log("CLEAR by user")
        debugCapture.endSession()
        reacquisition.clear()
        visualTracker.stop()
        vtUnconfirmedFrames = 0
        vtConfirmedFrames = 0
        vtFrameCounter = 0
        velocityEstimator.reset()
        pendingEmbeddings?.cancel(true)
        pendingEmbeddings = null
        cachedContour = emptyList()
        contourFrameCount = 0
    }

    val analyzer = ImageAnalysis.Analyzer { imageProxy ->
        processImage(imageProxy)
    }

    /**
     * Process a display-oriented bitmap from PreviewView.getBitmap().
     * Unlike the ImageProxy path, the bitmap is already in screen orientation —
     * no rotation is applied, and detector coordinates are used as-is (deviceRot=0).
     */
    fun processBitmap(bitmap: Bitmap) {
        processBitmapInternal(bitmap, deviceRotation = 0, imageProxy = null)
    }

    private fun processImage(imageProxy: ImageProxy) {
        val bitmap = imageProxyToBitmap(imageProxy)
        if (bitmap == null) {
            imageProxy.close()
            return
        }
        processBitmapInternal(bitmap, deviceRotation = deviceRotationProvider?.invoke() ?: 0, imageProxy = imageProxy)
    }

    private fun processBitmapInternal(bitmap: Bitmap, deviceRotation: Int, imageProxy: ImageProxy?) {
        val frameWidth = bitmap.width
        val frameHeight = bitmap.height

        try {
            // --- Visual tracker: primary frame-to-frame tracking ---
            // When active, it tracks the locked object by pixel correlation.
            // Cross-checked against the detector to prevent drift.
            if (visualTracker.isActive && reacquisition.isLocked && reacquisition.framesLost == 0) {
                val vtResult = visualTracker.update(bitmap)
                if (vtResult != null) {
                    // VT returns coords in rotated-image space; unmap to screen space
                    val deviceRot = deviceRotation
                    val rawBox = vtResult.boundingBox
                    val vtBox = unmapRotation(rawBox.left, rawBox.top, rawBox.right, rawBox.bottom, deviceRot)

                    // Feed position to velocity estimator for adaptive drift detection
                    velocityEstimator.update(vtBox.centerX(), vtBox.centerY())

                    // Skip detector when VT is confident and recently confirmed.
                    // Saves ~35ms per skipped frame. Still run every Nth frame for drift detection.
                    vtFrameCounter++
                    val canSkip = vtConfirmedFrames >= VT_SKIP_MIN_CONFIRMED &&
                        vtUnconfirmedFrames == 0 &&
                        vtResult.confidence > 0.6f
                    val skipDetector = canSkip && (vtFrameCounter % VT_SKIP_INTERVAL != 0)

                    val tracked = if (skipDetector) {
                        emptyList()
                    } else {
                        val detections = runDetector(bitmap, frameWidth, frameHeight, deviceRotation)
                        frameTracker.assignIds(detections)
                    }

                    val confirmed = if (skipDetector) {
                        true  // trust VT on skipped frames
                    } else {
                        tracked.any { det ->
                            det.label == reacquisition.lockedLabel &&
                            FrameToFrameTracker.computeIou(det.boundingBox, vtBox) > 0.15f
                        }
                    }

                    if (confirmed) {
                        vtUnconfirmedFrames = 0
                        // Sync lastKnownBox and velocity in screen coords when detector confirms
                        reacquisition.updateFromVisualTracker(vtBox)
                        reacquisition.updateVelocity(velocityEstimator.velocityX, velocityEstimator.velocityY)

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
                            // Progressive face embedding: try to capture face during tracking
                            // Use rawBox (rotated-image coords) since bitmap is the rotated image
                            if (reacquisition.lockedFaceEmbedding == null && reacquisition.lockedReIdEmbedding != null) {
                                val faceEmb = faceEmbedder.embedFace(bitmap, rawBox)
                                if (faceEmb != null) reacquisition.addFaceEmbedding(faceEmb)
                            }
                        }
                    } else {
                        vtUnconfirmedFrames++
                    }

                    // If too many frames without detector confirmation, tracker is drifting.
                    // Adaptive: fast-moving subjects get shorter tolerance (drift is more costly).
                    val adaptiveMaxUnconfirmed = when {
                        velocityEstimator.isVeryHighVelocity() -> VT_ADAPTIVE_UNCONFIRMED_VERY_HIGH
                        velocityEstimator.isHighVelocity() -> VT_ADAPTIVE_UNCONFIRMED_HIGH
                        else -> VT_MAX_UNCONFIRMED
                    }
                    if (vtUnconfirmedFrames > adaptiveMaxUnconfirmed) {
                        android.util.Log.w("VisualTracker", "DRIFT detected — $vtUnconfirmedFrames unconfirmed frames (max=$adaptiveMaxUnconfirmed, speed=${velocityEstimator.speed}), stopping")
                        visualTracker.stop()
                        vtUnconfirmedFrames = 0
                        vtFrameCounter = 0
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
            val detections = runDetector(bitmap, frameWidth, frameHeight, deviceRotation)
            // Use velocity-shifted matching when tracking a fast-moving subject.
            // Shifts previous frame's boxes by velocity before IoU matching,
            // preventing ID churn from large displacements between frames.
            val tracked = if (velocityEstimator.speed > 0.01f && reacquisition.isLocked) {
                frameTracker.assignIdsWithVelocity(
                    detections, velocityEstimator.velocityX, velocityEstimator.velocityY
                )
            } else {
                frameTracker.assignIds(detections)
            }

            // --- Embedding pipeline (async): use previous frame's results, kick off current ---
            val hasDirectMatch = reacquisition.lockedId != null &&
                tracked.any { it.id == reacquisition.lockedId }
            val needEmbeddings = reacquisition.hasEmbeddings &&
                reacquisition.isSearching && !hasDirectMatch

            // Enrich labels with YOLOv8 during search (every 10 frames, not every frame)
            val searchFrame = reacquisition.framesLost
            val shouldEnrichLabels = reacquisition.isSearching && tracked.isNotEmpty() &&
                (searchFrame <= 1 || searchFrame % 10 == 0)
            val enrichedIds = if (shouldEnrichLabels) labelEnricher.enrichLabels(bitmap, tracked) else emptyMap()
            val withLabels = if (enrichedIds.isNotEmpty()) {
                tracked.map { obj ->
                    val enriched = enrichedIds[obj.id]
                    if (enriched != null) obj.copy(label = enriched) else obj
                }
            } else tracked

            val withEmbeddings = if (needEmbeddings) {
                // Collect any ready results from previous frame's async embedding computation.
                // Match by best IoU overlap (not ID) since FTF IDs may change between frames.
                val cachedResults = try {
                    pendingEmbeddings?.takeIf { it.isDone }?.get() ?: emptyMap()
                } catch (e: Exception) { emptyMap() }

                // Merge cached embeddings into current detections by best IoU match
                val cachedEntries = cachedResults.entries.toList()
                val merged = withLabels.map { obj ->
                    val bestMatch = cachedEntries
                        .filter { (_, result) -> result.cachedBox != null }
                        .maxByOrNull { (_, result) ->
                            computeIou(obj.boundingBox, result.cachedBox!!)
                        }
                    val iou = bestMatch?.let { (_, result) ->
                        computeIou(obj.boundingBox, result.cachedBox!!)
                    } ?: 0f
                    if (bestMatch != null && iou > 0.3f) {
                        val cached = bestMatch.value
                        obj.copy(
                            embedding = cached.embedding,
                            colorHistogram = cached.colorHistogram,
                            personAttributes = cached.personAttributes,
                            reIdEmbedding = cached.reIdEmbedding,
                            faceEmbedding = cached.faceEmbedding
                        )
                    } else obj
                }

                // Kick off async embedding computation for current frame's detections.
                // Results will be available next frame via pendingEmbeddings.
                val bitmapCopy = bitmap.copy(bitmap.config ?: Bitmap.Config.ARGB_8888, false)
                val detectionsSnapshot = withLabels.toList()
                pendingEmbeddings = embeddingExecutor.submit(java.util.concurrent.Callable {
                    try {
                        computeEmbeddingsSync(bitmapCopy, detectionsSnapshot)
                    } finally {
                        bitmapCopy.recycle()
                    }
                })

                merged
            } else {
                pendingEmbeddings = null // clear when not searching
                withLabels
            }

            // Snapshot state before processing
            val wasSearching = reacquisition.isSearching
            val prevLost = reacquisition.framesLost

            // Record frame for scenario replay (before processFrame consumes it)
            scenarioRecorder.recordFrame(withEmbeddings)

            // Re-acquisition
            val lockedObject = reacquisition.processFrame(withEmbeddings)

            // Record events for scenario replay
            val nowLost = reacquisition.framesLost
            when {
                wasSearching && lockedObject != null && nowLost == 0 ->
                    scenarioRecorder.recordEvent("REACQUIRE", org.json.JSONObject().apply {
                        put("id", lockedObject.id); put("label", lockedObject.label ?: "null")
                    })
                nowLost == 1 && prevLost == 0 ->
                    scenarioRecorder.recordEvent("LOST")
                reacquisition.hasTimedOut && prevLost <= reacquisition.maxFramesLost ->
                    scenarioRecorder.recordEvent("TIMEOUT").also { scenarioRecorder.stop() }
            }

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
                val deviceRot = deviceRotation
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
            imageProxy?.close()
            bitmap.recycle()
        }
    }

    private fun runDetector(bitmap: Bitmap, frameWidth: Int, frameHeight: Int, deviceRot: Int): List<TrackedObject> {
        val mpImage = BitmapImageBuilder(bitmap).build()
        val result: ObjectDetectorResult = detector.detect(mpImage)

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

    /**
     * Compute embeddings synchronously for a list of detections. Runs on [embeddingExecutor].
     * Returns a map of detection ID → embedding results.
     */
    private fun computeEmbeddingsSync(bitmap: Bitmap, detections: List<TrackedObject>): Map<Int, EmbeddingResult> {
        val results = mutableMapOf<Int, EmbeddingResult>()
        // First pass: embeddings + color histograms for all
        for (obj in detections) {
            val embResult = appearanceEmbedder.embedAndCrop(bitmap, obj.boundingBox, fallback = true)
            val hist = if (embResult.maskedCrop != null) {
                val fullBox = RectF(0f, 0f, 1f, 1f)
                computeColorHistogram(embResult.maskedCrop, fullBox).also { embResult.maskedCrop.recycle() }
            } else computeColorHistogram(bitmap, obj.boundingBox)
            results[obj.id] = EmbeddingResult(embedding = embResult.embedding, colorHistogram = hist, cachedBox = RectF(obj.boundingBox))
        }
        // Second pass: classify top-2 person candidates
        val personCandidates = detections
            .filter { it.label == "person" && results[it.id]?.embedding != null }
            .sortedByDescending {
                val emb = results[it.id]?.embedding
                if (emb != null) reacquisition.bestGallerySimilarity(emb) else 0f
            }
            .take(2)
            .map { it.id }
            .toSet()
        for (id in personCandidates) {
            val obj = detections.find { it.id == id } ?: continue
            val attrs = personClassifier.classify(bitmap, obj.boundingBox, obj.label)
            val reIdEmb = personReId.embed(bitmap, obj.boundingBox)
            val faceEmb = if (reacquisition.lockedFaceEmbedding != null) {
                faceEmbedder.embedFace(bitmap, obj.boundingBox)
            } else null
            val existing = results[id] ?: EmbeddingResult(cachedBox = RectF(obj.boundingBox))
            results[id] = existing.copy(personAttributes = attrs, reIdEmbedding = reIdEmb, faceEmbedding = faceEmb)
        }
        return results
    }

    fun shutdown() {
        scenarioRecorder.stop()
        debugCapture.endSession()
        embeddingExecutor.shutdownNow()
        detector.close()
        labelEnricher.close()
        faceEmbedder.close()
        personReId.close()
        personClassifier.shutdown()
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
