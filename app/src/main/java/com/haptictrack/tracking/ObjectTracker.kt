package com.haptictrack.tracking

import android.content.Context
import android.graphics.Bitmap
import android.graphics.PointF
import android.graphics.RectF
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
    private val appearanceEmbedder: AppearanceEmbedder
    private val personClassifier: PersonAttributeClassifier
    private val faceEmbedder: FaceEmbedder
    private val personReId: PersonReIdEmbedder
    private val scenarioRecorder = ScenarioRecorder()

    // Keep last frame for computing embedding when user taps to lock
    private val lastFrameLock = Any()
    private var lastFrameBitmap: Bitmap? = null

    /**
     * Called to hand a no-longer-needed bitmap back to the frame reader's pool.
     * When null (tests), we skip pool return — the test owns bitmap lifetime
     * and recycles its own copies. In production, CameraViewModel wires this
     * to `CameraManager.releaseAnalysisBitmap`.
     */
    var bitmapRecycler: ((Bitmap) -> Unit)? = null
    /** Device rotation when lastFrameBitmap was captured — needed to map screen-space
     *  detection boxes back to rotated-image space for embedding. */
    private var lastFrameDeviceRotation: Int = 0

    // Track previous state to detect transitions
    private var previouslyLocked: Boolean = false
    private var previousFramesLost: Int = 0

    // Count frames where visual tracker has no detector confirmation
    private var vtUnconfirmedFrames = 0
    private var vtConfirmedFrames = 0
    private val VT_MAX_UNCONFIRMED = 10  // ~0.3s at 30fps (baseline for slow subjects)

    /** Skip detector every other frame when VT is confident and recently confirmed. */
    private var vtFrameCounter = 0
    private val VT_SKIP_INTERVAL_BASE = 2  // default: run detector every 2nd frame (50% skip)
    private val VT_SKIP_INTERVAL_STABLE = 3  // highly confident + long-stable: every 3rd frame (67% skip)
    private val VT_SKIP_MIN_CONFIRMED = 5  // need this many confirmations before skipping at all
    private val VT_SKIP_STABLE_CONFIRMED = 10  // confirmations required for wider skip interval
    private val VT_SKIP_STABLE_CONFIDENCE = 0.7f  // VT confidence required for wider skip interval

    /** Tracks object velocity for adaptive drift detection and position prediction. */
    private val velocityEstimator = VelocityEstimator()
    private val VT_ADAPTIVE_UNCONFIRMED_HIGH = 5      // ~165ms for fast subjects
    private val VT_ADAPTIVE_UNCONFIRMED_VERY_HIGH = 3  // ~100ms for very fast subjects

    /**
     * Template self-verification (issue #45, R3 from research):
     * Every N frames during VT tracking, embed the current VT crop and compare
     * it against the lock gallery. Low similarity means the crop no longer
     * resembles the locked object → VT has drifted. Detector-independent,
     * so it catches drift even when the detector can't see the target
     * (small objects, uniform surfaces, label flicker).
     *
     * Asymmetric thresholds (hysteresis): increment the counter when sim falls
     * below TEMPLATE_SIM_LOW, reset only when sim rises above TEMPLATE_SIM_OK.
     * This prevents oscillation around a single threshold from masking drift.
     */
    private val TEMPLATE_CHECK_INTERVAL = 5    // embed VT crop every N frames
    private val TEMPLATE_SIM_LOW = 0.4f        // below this = drift suspicion (increment)
    private val TEMPLATE_SIM_OK = 0.5f         // above this = safe (reset counter)
    private val TEMPLATE_MISMATCH_MAX = 3      // consecutive low-sim checks → kill (≈0.5s)
    private var templateMismatchCount = 0

    /**
     * VT box size sanity check: the tracker's output box shouldn't grow
     * dramatically relative to the box it was initialized with. Objects don't
     * physically grow 5x in a fraction of a second — if VT is producing a
     * much larger box, it has latched onto a larger texture pattern (e.g.
     * the whole screen) instead of the locked object.
     */
    private val VT_BOX_AREA_MAX_RATIO = 5f
    private var vtLockedBoxArea = 0f  // area of VT's init box (normalized 0..1)

    /** Async embedding pipeline: compute embeddings on background thread, use results next frame. */
    private val embeddingExecutor = java.util.concurrent.Executors.newSingleThreadExecutor { r -> Thread(r, "EmbeddingAsync") }
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

    /**
     * Sync-fallback dedup cache. During search the sync fallback was firing on the
     * same candidate ID every frame (~12 rejects/sec = ~280ms/sec of sync embed),
     * because async results had moved out of IoU>0.3 range or hadn't arrived in time.
     * We remember the last sync result per ID and reuse it while the box is still
     * close to where we embedded it. Cleared on every lock/reacq transition.
     */
    private data class SyncEmbedEntry(
        val embedding: FloatArray,
        val colorHistogram: FloatArray?,
        val cachedBox: RectF,
        val framesLost: Int
    )
    private val recentSyncEmbeds = mutableMapOf<Int, SyncEmbedEntry>()
    private val SYNC_EMBED_CACHE_FRAMES = 30  // ~1-2s at typical search fps; IoU is the real staleness guard
    private val SYNC_EMBED_CACHE_IOU = 0.5f  // box must overlap this much with cached box

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

        onLoadingStatus?.invoke("Loading face models (GPU)...")
        faceEmbedder = FaceEmbedder(context, personClassifier.faceDetector)
        personReId = PersonReIdEmbedder(context)

        // Wire ReacquisitionEngine logs to session logger
        reacquisition.sessionLogger = { msg -> debugCapture.log("[Reacq] $msg") }
        onLoadingStatus?.invoke("Ready")
    }

    /** Detections from the last processed frame — used to collect scene negatives at lock time. */
    @Volatile private var lastDetections: List<TrackedObject> = emptyList()

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
            // Classify person attributes at lock time (use original COCO label for "person" check)
            val personAttrs = personClassifier.classify(bmp, boundingBox, label)
            // Person re-ID + face embeddings (only for person labels)
            val isPerson = label == "person"
            val reIdEmb = if (isPerson) personReId.embed(bmp, boundingBox) else null
            val faceEmb = if (isPerson) faceEmbedder.embedFace(bmp, boundingBox) else null
            reacquisition.lock(trackingId, boundingBox, label, augResult.embeddings, colorHist, personAttrs,
                cocoLabel = label, reIdEmbedding = reIdEmb, faceEmbedding = faceEmb)
            visualTracker.init(bmp, boundingBox)
            vtLockedBoxArea = boundingBox.width() * boundingBox.height()

            debugCapture.startSession(label, trackingId)
            val attrStr = personAttrs?.summary() ?: "n/a"
            debugCapture.log("LOCK id=$trackingId label=$label box=${boundingBox} gallery=${augResult.embeddings.size} colorHist=${colorHist != null} attrs=\"$attrStr\"")

            // Start scenario recording for replay testing
            debugCapture.sessionDir?.let { dir ->
                scenarioRecorder.start(dir, trackingId, label, label,
                    boundingBox, augResult.embeddings, colorHist, personAttrs)
            }

            // Collect scene negatives from other objects visible at lock time.
            // lastDetections has screen-space boxes but bmp is the rotated bitmap,
            // so convert back to rotated-image space before cropping.
            val lockDevRot = lastFrameDeviceRotation
            for (det in lastDetections) {
                if (det.id == trackingId) continue
                val rotBox = mapToRotated(det.boundingBox.left, det.boundingBox.top,
                    det.boundingBox.right, det.boundingBox.bottom, lockDevRot)
                val negEmb = appearanceEmbedder.embed(bmp, rotBox)
                if (negEmb != null) reacquisition.addSceneNegative(negEmb)
            }

            val locked = TrackedObject(trackingId, boundingBox, label)
            debugCapture.capture(DebugEvent.LOCK, bmp, listOf(locked), lockedObject = locked,
                extraInfo = "id=$trackingId label=$label gallery=${augResult.embeddings.size}")
        }
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
        templateMismatchCount = 0
        vtLockedBoxArea = 0f
        velocityEstimator.reset()
        pendingEmbeddings?.cancel(true)
        pendingEmbeddings = null
        recentSyncEmbeds.clear()
        cachedContour = emptyList()
        contourFrameCount = 0
    }

    /**
     * Process a display-oriented bitmap from the SurfaceTexture GL pipeline.
     * The bitmap pixels are in display (phone-top) orientation. The detector
     * reads [deviceRotationProvider] internally and rotates the bitmap to
     * physical-upright before inference; VT and the rest of the pipeline
     * continue operating in display-image coordinates.
     */
    fun processBitmap(bitmap: Bitmap) {
        processBitmapInternal(bitmap, deviceRotation = 0)
    }

    private fun processBitmapInternal(bitmap: Bitmap, deviceRotation: Int) {
        val frameWidth = bitmap.width
        val frameHeight = bitmap.height

        // Bitmap ownership: we retain `bitmap` as lastFrameBitmap at the end of
        // processing, swapping out the previous lastFrameBitmap which then goes
        // back to the pool. Previously we did bitmap.copy() (~4ms) every frame
        // just to keep a reference for lockOnObject and debug capture — that was
        // wasteful since the input is already a stable snapshot for this frame.
        //
        // `releaseOnExit` tracks which bitmap to return to the pool in `finally`:
        //   - initially the input (so exceptions still release it)
        //   - after retention, the old lastFrameBitmap (release instead of input)
        var releaseOnExit: Bitmap? = bitmap
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
                    // Widen the interval once VT has been confirmed for a long stretch at high
                    // confidence — the detector's role at that point is a slow safety net,
                    // template self-verification (every 5 frames) is the primary drift signal.
                    vtFrameCounter++
                    val canSkip = vtConfirmedFrames >= VT_SKIP_MIN_CONFIRMED &&
                        vtUnconfirmedFrames == 0 &&
                        vtResult.confidence > 0.6f
                    val skipInterval = if (
                        vtConfirmedFrames >= VT_SKIP_STABLE_CONFIRMED &&
                        vtResult.confidence > VT_SKIP_STABLE_CONFIDENCE
                    ) VT_SKIP_INTERVAL_STABLE else VT_SKIP_INTERVAL_BASE
                    val skipDetector = canSkip && (vtFrameCounter % skipInterval != 0)

                    val tracked = if (skipDetector) {
                        emptyList()
                    } else {
                        val detections = runDetector(bitmap)
                        frameTracker.assignIds(detections)
                    }

                    val confirmed = if (skipDetector) {
                        true  // trust VT on skipped frames
                    } else {
                        // Confirmation = detection overlaps VT's position.
                        // Accept same-category at low IoU (0.15) or any category at high IoU (0.4).
                        // High IoU means it's the same object regardless of label.
                        // Low IoU with different category (e.g. person walking past a cup) doesn't confirm.
                        val lockedIsPerson = reacquisition.lockedIsPerson
                        tracked.any { det ->
                            val iou = FrameToFrameTracker.computeIou(det.boundingBox, vtBox)
                            val sameCategory = (det.label == "person") == lockedIsPerson
                            (sameCategory && iou > 0.15f) || iou > 0.4f
                        }
                    }
                    // Distinguish non-confirmation from contradiction:
                    // If detector found nothing at all, that's not evidence of drift —
                    // it's poor detection conditions. Only count unconfirmed when
                    // detector found detections but none match our tracked position.
                    val detectorFoundSomething = skipDetector || tracked.isNotEmpty()

                    if (confirmed) {
                        vtUnconfirmedFrames = 0
                        // Sync lastKnownBox and velocity in screen coords when detector confirms
                        reacquisition.updateFromVisualTracker(vtBox)
                        reacquisition.updateVelocity(velocityEstimator.velocityX, velocityEstimator.velocityY)

                        // Accumulate embedding from current angle (every 15 confirmed frames ≈ 0.5s)
                        // Use raw rotated-image coords for embedding (embedder works on rotated bitmap)
                        vtConfirmedFrames++
                        if (vtConfirmedFrames % 15 == 0) {
                            // Use fallback: during confirmed tracking, even an unmasked
                            // embedding is better than no embedding for gallery diversity
                            val emb = appearanceEmbedder.embedWithFallback(bitmap, rawBox)
                            if (emb != null) {
                                // Diversity check: only add if meaningfully different from centroid
                                val centroidSim = reacquisition.centroidSimilarity(emb)
                                if (centroidSim < 0.92f || reacquisition.embeddingGallery.size < 8) {
                                    reacquisition.addEmbedding(emb)
                                    android.util.Log.d("AppearEmbed", "Gallery +1 → ${reacquisition.embeddingGallery.size} (centroidSim=${centroidSim})")
                                }
                            }
                            // Progressive face embedding: try to capture face during tracking
                            // Use rawBox (rotated-image coords) since bitmap is the rotated image
                            if (reacquisition.lockedFaceEmbedding == null && reacquisition.lockedReIdEmbedding != null) {
                                val faceEmb = faceEmbedder.embedFace(bitmap, rawBox)
                                if (faceEmb != null) reacquisition.addFaceEmbedding(faceEmb)
                            }
                        }

                        // Collect scene negatives every 5 confirmed frames.
                        // tracked has screen-space boxes but bitmap is rotated,
                        // so convert back to rotated-image space before cropping.
                        if (vtConfirmedFrames % 5 == 0) {
                            for (det in tracked) {
                                if (det.id == reacquisition.lockedId) continue
                                val rotBox = mapToRotated(det.boundingBox.left, det.boundingBox.top,
                                    det.boundingBox.right, det.boundingBox.bottom, deviceRot)
                                val negEmb = appearanceEmbedder.embed(bitmap, rotBox)
                                if (negEmb != null) reacquisition.addSceneNegative(negEmb)
                            }
                        }
                    } else {
                        // Only count as unconfirmed if detector actually found detections
                        // (found objects but none match VT position = likely drift).
                        // Empty detector output = poor conditions, not drift evidence.
                        if (detectorFoundSomething) {
                            vtUnconfirmedFrames++
                        }
                    }

                    // --- Template self-verification (R3) ---
                    // Primary drift signal: compare current VT crop's embedding to the
                    // lock gallery. Detector-independent, so it catches drift on small
                    // or uniform objects the detector struggles with. Skipped on
                    // detector-skip frames (no bitmap cost budget).
                    //
                    // Effective rate is every ~5 frames normally, ~10 frames when VT
                    // frame skipping is active (skipDetector=true on alternate frames).
                    //
                    // Hysteresis: increment when sim < TEMPLATE_SIM_LOW, reset only
                    // when sim > TEMPLATE_SIM_OK. Values in between (the dead zone)
                    // leave the counter unchanged so marginal oscillation can't mask
                    // real drift.
                    if (!skipDetector &&
                        reacquisition.hasEmbeddings &&
                        vtFrameCounter % TEMPLATE_CHECK_INTERVAL == 0) {
                        val curEmb = appearanceEmbedder.embedWithFallback(bitmap, rawBox)
                        if (curEmb != null) {
                            val sim = reacquisition.bestGallerySimilarity(curEmb)
                            when {
                                sim < TEMPLATE_SIM_LOW -> {
                                    templateMismatchCount++
                                    android.util.Log.d("VisualTracker", "Template mismatch $templateMismatchCount/$TEMPLATE_MISMATCH_MAX (sim=${"%.3f".format(sim)})")
                                }
                                sim > TEMPLATE_SIM_OK -> templateMismatchCount = 0
                                else -> { /* dead zone — hold the counter */ }
                            }
                        }
                    }

                    // VT box size sanity: if the tracker's output is much larger than
                    // what we initialized it with, VT has latched onto background
                    // texture (classic "expands to whole screen" drift).
                    val vtBoxArea = vtBox.width() * vtBox.height()
                    val sizeDrift = vtLockedBoxArea > 0f &&
                        vtBoxArea > vtLockedBoxArea * VT_BOX_AREA_MAX_RATIO

                    // If too many frames without detector confirmation, tracker is drifting.
                    // Adaptive: fast-moving subjects get shorter tolerance (drift is more costly).
                    val adaptiveMaxUnconfirmed = when {
                        velocityEstimator.isVeryHighVelocity() -> VT_ADAPTIVE_UNCONFIRMED_VERY_HIGH
                        velocityEstimator.isHighVelocity() -> VT_ADAPTIVE_UNCONFIRMED_HIGH
                        else -> VT_MAX_UNCONFIRMED
                    }
                    val templateDrift = templateMismatchCount >= TEMPLATE_MISMATCH_MAX
                    val detectorDrift = vtUnconfirmedFrames > adaptiveMaxUnconfirmed
                    if (sizeDrift || templateDrift || detectorDrift) {
                        val reason = when {
                            sizeDrift -> "box expanded ${"%.1fx".format(vtBoxArea / vtLockedBoxArea)} (area=${"%.3f".format(vtBoxArea)})"
                            templateDrift -> "template mismatch ${templateMismatchCount}x"
                            else -> "$vtUnconfirmedFrames unconfirmed frames (max=$adaptiveMaxUnconfirmed)"
                        }
                        android.util.Log.w("VisualTracker", "DRIFT detected — $reason (conf=${vtResult.confidence}, speed=${velocityEstimator.speed}), stopping")
                        visualTracker.stop()
                        vtUnconfirmedFrames = 0
                        vtFrameCounter = 0
                        templateMismatchCount = 0
                        vtLockedBoxArea = 0f
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

                        lastDetections = displayObjects
                        onDetectionResult?.invoke(displayObjects, lockedObj, frameWidth, frameHeight, cachedContour)

                        synchronized(lastFrameLock) {
                            val previous = lastFrameBitmap
                            lastFrameBitmap = bitmap
                            lastFrameDeviceRotation = deviceRotation
                            releaseOnExit = previous  // release old, retain new
                        }
                        return
                    }
                } else {
                    // Visual tracker lost the object — fall through to detector
                    visualTracker.stop()
                    vtUnconfirmedFrames = 0
                    templateMismatchCount = 0
                    vtLockedBoxArea = 0f
                }
            }

            // --- Detector path: detection + re-acquisition ---
            val detections = runDetector(bitmap)
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

            // YOLOv8 label enrichment skipped during search — person/not-person gate
            // doesn't need specific labels, and COCO labels from EfficientDet suffice
            // for the binary category check. Saves ~270ms per enrichment call.
            val withLabels = tracked

            val withEmbeddings = if (needEmbeddings) {
                // Collect results from previous frame's async embedding computation.
                // Wait up to 50ms — the processing thread is decoupled from the GL thread,
                // so blocking briefly is fine. Better to wait for embeddings than reacquire
                // without identity verification.
                val cachedResults = try {
                    pendingEmbeddings?.get(50, java.util.concurrent.TimeUnit.MILLISECONDS) ?: emptyMap()
                } catch (e: java.util.concurrent.TimeoutException) { emptyMap() }
                catch (e: Exception) { emptyMap() }

                // Merge cached embeddings into current detections. Prefer ID match
                // (FTFTracker preserves IDs across frames for the same physical object —
                // trust it), fall back to greedy IoU match for re-IDed detections.
                // Prefer-by-ID avoids throwing away async work when fast motion pushes
                // box-to-box IoU below 0.3 between consecutive frames.
                val cachedEntries = cachedResults.entries.toList()
                val usedCacheKeys = mutableSetOf<Int>()
                val merged = withLabels.map { obj ->
                    val byId = if (obj.id >= 0) cachedResults[obj.id] else null
                    val cached: EmbeddingResult? = if (byId != null) {
                        usedCacheKeys.add(obj.id)
                        byId
                    } else {
                        val bestMatch = cachedEntries
                            .filter { (id, result) -> result.cachedBox != null && id !in usedCacheKeys }
                            .maxByOrNull { (_, result) ->
                                computeIou(obj.boundingBox, result.cachedBox!!)
                            }
                        val iou = bestMatch?.let { (_, result) ->
                            computeIou(obj.boundingBox, result.cachedBox!!)
                        } ?: 0f
                        if (bestMatch != null && iou > 0.3f) {
                            usedCacheKeys.add(bestMatch.key)
                            bestMatch.value
                        } else null
                    }
                    if (cached != null) {
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
                // Only submit if previous task is done — don't overwrite in-flight work
                // or its results are lost before we can collect them.
                val prevDone = pendingEmbeddings?.isDone ?: true
                if (prevDone) {
                    val bitmapCopy = bitmap.copy(bitmap.config ?: Bitmap.Config.ARGB_8888, false)
                    val detectionsSnapshot = withLabels.toList()
                    pendingEmbeddings = embeddingExecutor.submit(java.util.concurrent.Callable {
                        try {
                            computeEmbeddingsSync(bitmapCopy, detectionsSnapshot)
                        } finally {
                            bitmapCopy.recycle()
                        }
                    })
                }

                // --- Synchronous embedding fallback for candidates that missed the cache ---
                // During rotation, IoU between consecutive frames drops below 0.3 so the
                // async cache can't merge. Without embeddings, the ReacquisitionEngine's
                // identity gate hard-rejects these candidates, causing multi-second gaps.
                // Fix: compute embedding synchronously for the single best same-category
                // candidate that has no embedding. One-time ~23ms cost per new candidate.
                //
                // Dedup: a short-retention cache (recentSyncEmbeds) avoids re-embedding the
                // same ID on consecutive frames — the same bad candidate used to cost ~23ms
                // every frame because its sim never crossed the override threshold and
                // the engine rejected it, leaving it "needsSync" again next frame.
                val withSyncFallback = if (reacquisition.hasEmbeddings) {
                    val refBox = reacquisition.lastKnownBox
                    val lockedIsPerson = reacquisition.lockedIsPerson
                    // Find same-category candidates missing embeddings
                    val needsSync = merged.filter { obj ->
                        obj.embedding == null && obj.id >= 0 &&
                            ((obj.label == "person") == lockedIsPerson)
                    }
                    if (needsSync.isNotEmpty() && refBox != null) {
                        // Pick the closest one by center distance to last known position
                        val best = needsSync.minByOrNull { obj ->
                            val dx = obj.boundingBox.centerX() - refBox.centerX()
                            val dy = obj.boundingBox.centerY() - refBox.centerY()
                            dx * dx + dy * dy
                        }!!

                        // Dedup: reuse a recent sync result if this ID was embedded in
                        // the last SYNC_EMBED_CACHE_FRAMES frames and its box hasn't moved.
                        // Prune stale entries first so the map doesn't grow unbounded across
                        // long searches where FTFTracker keeps minting new IDs.
                        val currFramesLost = reacquisition.framesLost
                        recentSyncEmbeds.entries.removeAll { (_, entry) ->
                            currFramesLost - entry.framesLost >= SYNC_EMBED_CACHE_FRAMES
                        }
                        val cached = recentSyncEmbeds[best.id]
                        val reusable = cached != null &&
                            computeIou(best.boundingBox, cached.cachedBox) > SYNC_EMBED_CACHE_IOU

                        val (embedding, hist) = if (reusable) {
                            cached!!.embedding to cached.colorHistogram
                        } else {
                            // Compute embedding synchronously (raw crop fallback, ~8-23ms)
                            val embResult = appearanceEmbedder.embedAndCrop(bitmap, best.boundingBox, fallback = true)
                            val computedHist = if (embResult.maskedCrop != null) {
                                val fullBox = RectF(0f, 0f, 1f, 1f)
                                computeColorHistogram(embResult.maskedCrop, fullBox).also { embResult.maskedCrop.recycle() }
                            } else computeColorHistogram(bitmap, best.boundingBox)
                            if (embResult.embedding != null) {
                                android.util.Log.d("EmbedSync", "Sync embedding for id=${best.id} label=${best.label}")
                                recentSyncEmbeds[best.id] = SyncEmbedEntry(
                                    embedding = embResult.embedding,
                                    colorHistogram = computedHist,
                                    cachedBox = RectF(best.boundingBox),
                                    framesLost = currFramesLost
                                )
                            }
                            embResult.embedding to computedHist
                        }

                        if (embedding != null) {
                            merged.map { obj ->
                                if (obj.id == best.id) obj.copy(
                                    embedding = embedding,
                                    colorHistogram = hist
                                ) else obj
                            }
                        } else merged
                    } else merged
                } else merged

                withSyncFallback
            } else {
                pendingEmbeddings = null // clear when not searching
                recentSyncEmbeds.clear()
                withLabels
            }

            // Snapshot state before processing
            val wasSearching = reacquisition.isSearching
            val prevLost = reacquisition.framesLost

            // Filter before scoring — removes phantom full-screen detections
            // (e.g. "airplane" at [0,0,1,0.7]) that appear during rotation/motion blur.
            val filtered = filter.filter(withEmbeddings)

            // Record frame for scenario replay (before processFrame consumes it)
            scenarioRecorder.recordFrame(filtered)

            // Re-acquisition
            val lockedObject = reacquisition.processFrame(filtered)

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
                vtLockedBoxArea = lockedObject.boundingBox.width() * lockedObject.boundingBox.height()
                templateMismatchCount = 0
            }

            // Retain the frame for debug capture and future lockOnObject calls.
            // Debug capture draws annotations onto its own mutable copy of this bitmap,
            // so concurrent access is safe.
            synchronized(lastFrameLock) {
                val previous = lastFrameBitmap
                lastFrameBitmap = bitmap
                lastFrameDeviceRotation = deviceRotation
                releaseOnExit = previous  // release old, retain new
            }

            // Debug frame capture on tracking events.
            // Uses unfiltered `withEmbeddings` on purpose — debug overlays should
            // show what the detector produced, including phantoms that the filter
            // removed, so we can diagnose when the filter is too aggressive.
            captureDebugFrame(bitmap, withEmbeddings, lockedObject, wasSearching, prevLost)

            // Filter for display
            // Already filtered before processFrame — reuse for display
            val displayObjects = filtered

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

            lastDetections = displayObjects
            onDetectionResult?.invoke(displayObjects, lockedObject, frameWidth, frameHeight, cachedContour)
        } finally {
            releaseOnExit?.let { bitmapRecycler?.invoke(it) ?: it.recycle() }
        }
    }

    /**
     * Reusable buffer for the physical-upright rotation in [runDetector].
     * Allocated once per (width,height) combination and drawn into each frame —
     * previously we allocated a fresh bitmap every non-portrait frame (~5-10ms).
     */
    private var uprightBuffer: Bitmap? = null

    private fun runDetector(bitmap: Bitmap): List<TrackedObject> {
        // EfficientDet fails on rotated scenes — pre-rotate the bitmap to
        // physical-upright before inference when the phone is not in portrait.
        // The rest of the pipeline operates in display-image coords; we remap
        // detector output back to display-image (screen) space via unmapRotation.
        val actualRot = deviceRotationProvider?.invoke() ?: 0
        val upright = if (actualRot == 0) bitmap else rotateIntoBuffer(bitmap, -actualRot)
        val uprightW = upright.width
        val uprightH = upright.height
        val mpImage = BitmapImageBuilder(upright).build()
        val result: ObjectDetectorResult = detector.detect(mpImage)

        val detections = result.detections().map { detection ->
            val box = detection.boundingBox()
            val normLeft = box.left / uprightW.toFloat()
            val normTop = box.top / uprightH.toFloat()
            val normRight = box.right / uprightW.toFloat()
            val normBottom = box.bottom / uprightH.toFloat()

            val screenBox = unmapRotation(normLeft, normTop, normRight, normBottom, actualRot)

            TrackedObject(
                id = -1,
                boundingBox = screenBox,
                label = detection.categories().firstOrNull()?.categoryName(),
                confidence = detection.categories().firstOrNull()?.score() ?: 0f
            )
        }

        // No recycle here — uprightBuffer is reused across frames.
        return detections
    }

    /**
     * Rotate [src] by [rotDegrees] into a pooled buffer bitmap, reallocating only
     * when the target dimensions change (i.e. rotation flipped between 90/270 and
     * 0/180, or the input resolution changed).
     *
     * [rotDegrees] is in {90, 180, 270, -90, -180, -270}; we normalize to 0..359.
     */
    private fun rotateIntoBuffer(src: Bitmap, rotDegrees: Int): Bitmap {
        val rot = ((rotDegrees % 360) + 360) % 360
        val targetW = if (rot == 90 || rot == 270) src.height else src.width
        val targetH = if (rot == 90 || rot == 270) src.width else src.height

        var buffer = uprightBuffer
        if (buffer == null || buffer.isRecycled ||
            buffer.width != targetW || buffer.height != targetH) {
            buffer?.recycle()
            buffer = Bitmap.createBitmap(targetW, targetH, Bitmap.Config.ARGB_8888)
            uprightBuffer = buffer
        }

        val canvas = android.graphics.Canvas(buffer)
        canvas.drawColor(android.graphics.Color.BLACK, android.graphics.PorterDuff.Mode.SRC)
        val matrix = android.graphics.Matrix().apply {
            postRotate(rot.toFloat(), src.width / 2f, src.height / 2f)
            // After rotation around the source center, translate so the rotated
            // image's top-left aligns with the buffer's top-left.
            val dx = (targetW - src.width) / 2f
            val dy = (targetH - src.height) / 2f
            postTranslate(dx, dy)
        }
        canvas.drawBitmap(src, matrix, null)
        return buffer
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
        debugCapture.shutdown()
        embeddingExecutor.shutdownNow()
        detector.close()
        faceEmbedder.close()
        personReId.close()
        personClassifier.shutdown()
        appearanceEmbedder.shutdown()
        visualTracker.stop()
        synchronized(lastFrameLock) {
            lastFrameBitmap?.recycle()
            lastFrameBitmap = null
        }
        uprightBuffer?.recycle()
        uprightBuffer = null
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
