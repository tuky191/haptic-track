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
    val reacquisition: ReacquisitionEngine = ReacquisitionEngine.create(context),
    val filter: DetectionFilter = DetectionFilter(),
    private val frameTracker: FrameToFrameTracker = FrameToFrameTracker(),
    val debugCapture: DebugFrameCapture = DebugFrameCapture(context),
    private val visualTracker: VisualTracker = VisualTracker(context),
    /** Provides physical device orientation; set from ViewModel. */
    var deviceRotationProvider: (() -> Int)? = null
) {

    companion object {
        /** Sample the locked object every Nth confirmed frame and submit to [auditExecutor]. */
        private const val AUDIT_STABILITY_INTERVAL = 5
        /** Save a crop composite every Nth confirmed frame (subset of stability cadence). */
        private const val AUDIT_COMPOSITE_INTERVAL = 30
    }

    private val detector: ObjectDetector
    private val appearanceEmbedder: AppearanceEmbedder
    private val personClassifier: PersonAttributeClassifier
    private val faceEmbedder: FaceEmbedder
    private val personReId: PersonReIdEmbedder
    private val scenarioRecorder = ScenarioRecorder()

    /** Embedding-input audit (#92): periodic crops + per-embedder stability log. */
    private val cropDebugCapture: CropDebugCapture
    private val stabilityLogger = EmbeddingStabilityLogger()
    /**
     * Most recent stability summary written by [clearLock]. Read by tests
     * (see [VideoReplayTest]) to aggregate per-video noise floors.
     */
    var lastEmbeddingStabilitySummary: org.json.JSONObject? = null
        private set
    /**
     * Audit work (3 embedder calls + optional composite encode) runs off the
     * processing thread on a low-priority single-thread pool. Bounded queue +
     * DiscardOldest means we drop audit samples under backpressure rather than
     * stall the tracking pipeline. The single-thread pool serializes all audit
     * work — including the composite-write path that [CropDebugCapture] now
     * shares with us via constructor injection — so audit never adds more than
     * one concurrent caller into the production-shared embedder monitors.
     */
    private val auditExecutor = java.util.concurrent.ThreadPoolExecutor(
        1, 1, 0L, java.util.concurrent.TimeUnit.MILLISECONDS,
        java.util.concurrent.LinkedBlockingQueue<Runnable>(4),
        { r -> Thread(r, "AuditEmbed").apply { priority = Thread.MIN_PRIORITY } },
        java.util.concurrent.ThreadPoolExecutor.DiscardOldestPolicy()
    )

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
     * Hysteresis: increment when sim < TEMPLATE_SIM_LOW, decrement (not reset)
     * when sim > TEMPLATE_SIM_OK. The decrement-instead-of-reset semantics
     * (#80 fix) means a single high-sim flicker doesn't undo two prior bad
     * frames. Counter still trips at TEMPLATE_MISMATCH_MAX consecutive-ish
     * low frames, but is robust to occasional flukes — exactly the case
     * where the camera pans to uniform backdrop and sim flickers between
     * 0.3 and 0.5.
     */
    private val TEMPLATE_CHECK_INTERVAL = 5    // embed VT crop every N frames
    // TEMPLATE_SIM_LOW / TEMPLATE_SIM_OK are dynamic per-lock (#110): derived
    // from [ReacquisitionEngine.lockSelfFloor] so the drift-detection band
    // adapts to the embedder's same-object cosine band for THIS lock. Static
    // 0.4 / 0.5 (the prior values) sat above the chair's live-vs-gallery
    // band (0.3-0.5), killing VT before the accumulator could grow the
    // gallery and dropping chair tracking from 76% to 25-42%. Person/OSNet
    // sits at 0.6+ and the upper clamp keeps the gate well-protected there.
    private val TEMPLATE_MISMATCH_MAX = 3      // mismatches → kill (≈0.5s)
    private var templateMismatchCount = 0

    /**
     * VT box size sanity check: the tracker's output box shouldn't grow
     * dramatically relative to the box it was initialized with. Objects don't
     * physically grow 5x in a fraction of a second — if VT is producing a
     * much larger box, it has latched onto a larger texture pattern (e.g.
     * the whole screen) instead of the locked object.
     */
    private val VT_BOX_AREA_MAX_RATIO = 5f
    /**
     * Detector-anchor reseat thresholds (#110). When the detector confirms VT
     * but two conditions hold, VT has latched onto background texture and we
     * reseat the tracker with the detector's box:
     *
     *   1. VT has GROWN since its init (`vtArea / vtLockedBoxArea > GROWTH`)
     *   2. VT is also larger than the detector reports for the same object
     *      (`vtArea / detArea > DISAGREEMENT`)
     *
     * Both conditions are needed because the two fail differently:
     *   - Condition 1 alone fires on legitimate zoom (subject approaches
     *     camera, both VT and detector grow in lockstep — detector match
     *     stays close to VT, no runaway).
     *   - Condition 2 alone fires on sizing-convention mismatch (e.g. person
     *     detector outputs head+torso but VT tracks the full body lock box
     *     — natural disagreement at every frame, not a runaway).
     *
     * Their intersection captures only the runaway case: VT grew AND the
     * detector says we should be smaller. Empirically: chair box expands
     * 1x → 5x while detector stays at small chair-only box → both fire.
     * Person tracking: VT stays close to its init size → condition 1 stays
     * below 1.5x → no reseat.
     *
     * Doesn't fire when detector misses for the frame (no reference box).
     * The existing 5x [VT_BOX_AREA_MAX_RATIO] remains the catch-all for
     * sustained detector-missing windows.
     */
    private val VT_RESEAT_GROWTH_RATIO = 1.3f       // condition 1
    private val VT_RESEAT_DISAGREEMENT_RATIO = 1.3f // condition 2
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
        val reIdEmbedding: FloatArray? = null,
        val faceEmbedding: FloatArray? = null,
        val cachedBox: RectF,
        val framesLost: Int
    )
    private val recentSyncEmbeds = mutableMapOf<Int, SyncEmbedEntry>()
    private val SYNC_EMBED_CACHE_FRAMES = 30  // ~1-2s at typical search fps; IoU is the real staleness guard
    private val SYNC_EMBED_CACHE_IOU = 0.5f  // box must overlap this much with cached box

    /**
     * Off-thread lock-on. The full lock burst (5 augmented embeddings + segmenter
     * + person classifier + re-id + face + scene negatives) is ~150-250ms of GPU
     * work that used to run on the processing thread under lastFrameLock, blocking
     * frame processing for the full duration. Now we snapshot a bitmap, release
     * the lock immediately, and run the ML on a dedicated thread; the processing
     * thread picks the result up at the next frame boundary.
     */
    private val lockExecutor = java.util.concurrent.Executors.newSingleThreadExecutor { r -> Thread(r, "LockOnObject") }
    private var pendingLockFuture: java.util.concurrent.Future<*>? = null
    private val pendingLockResult = java.util.concurrent.atomic.AtomicReference<LockResult?>()

    /** Result of off-thread lock ML; applied on the processing thread. */
    private data class LockResult(
        val trackingId: Int,
        val boundingBox: RectF,
        val label: String?,
        val gallery: List<FloatArray>,
        val colorHist: FloatArray?,
        val personAttrs: PersonAttributes?,
        val reIdEmb: FloatArray?,
        val faceEmb: FloatArray?,
        val sceneNegatives: List<FloatArray>,
        /** Face+body embeddings for OTHER persons visible at lock time — seeds
         *  the [SessionRoster] (#108) so the open-set rejection has data from
         *  frame 0 of search, not after VT-confirmed-tracking accumulation. */
        val sceneRosterObservations: List<Pair<FloatArray?, FloatArray?>>,
        val deviceRotation: Int,
        /** Snapshot bitmap; ownership transfers to whoever applies/cancels this result. */
        val snapshotBmp: Bitmap
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

        onLoadingStatus?.invoke("Loading face models (GPU)...")
        faceEmbedder = FaceEmbedder(context, personClassifier.faceDetector)
        personReId = PersonReIdEmbedder(context)

        cropDebugCapture = CropDebugCapture(appearanceEmbedder, personReId, faceEmbedder, auditExecutor)
        stabilityLogger.samplingIntervalFrames = AUDIT_STABILITY_INTERVAL

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
        // Snapshot the latest frame + detections atomically, then release the
        // lastFrameLock immediately so the processing thread can keep producing
        // frames during the lock burst.
        val snapshotBmp: Bitmap
        val snapshotDevRot: Int
        val snapshotDetections: List<TrackedObject>
        synchronized(lastFrameLock) {
            val src = lastFrameBitmap ?: run {
                reacquisition.lock(trackingId, boundingBox, label, emptyList())
                return
            }
            snapshotBmp = src.copy(src.config ?: Bitmap.Config.ARGB_8888, false)
            snapshotDevRot = lastFrameDeviceRotation
            snapshotDetections = lastDetections.toList()
        }

        // Cancel any in-flight prior lock and discard its result if not yet applied.
        pendingLockFuture?.cancel(true)
        pendingLockResult.getAndSet(null)?.snapshotBmp?.recycle()

        pendingLockFuture = lockExecutor.submit {
            try {
                val augResult = appearanceEmbedder.embedWithAugmentations(snapshotBmp, boundingBox)
                val colorHist = if (augResult.maskedCrop != null) {
                    val fullBox = RectF(0f, 0f, 1f, 1f)
                    computeColorHistogram(augResult.maskedCrop, fullBox).also { augResult.maskedCrop.recycle() }
                } else computeColorHistogram(snapshotBmp, boundingBox)
                val personAttrs = personClassifier.classify(snapshotBmp, boundingBox, label)
                val isPerson = label == "person"
                val reIdEmb = if (isPerson) personReId.embed(snapshotBmp, boundingBox) else null
                val faceEmb = if (isPerson) faceEmbedder.embedFace(snapshotBmp, boundingBox) else null

                // Scene negatives: embed every other detection visible at lock time.
                // boundingBox is screen-space; remap to rotated-image space to crop.
                // ALSO seed the SessionRoster (#108) with face+body of any other
                // persons in frame so open-set rejection has data from frame 0 of
                // search rather than waiting on VT-confirmed accumulation.
                val sceneNegs = mutableListOf<FloatArray>()
                val rosterObservations = mutableListOf<Pair<FloatArray?, FloatArray?>>()
                for (det in snapshotDetections) {
                    if (det.id == trackingId) continue
                    val rotBox = mapToRotated(det.boundingBox.left, det.boundingBox.top,
                        det.boundingBox.right, det.boundingBox.bottom, snapshotDevRot)
                    val negEmb = appearanceEmbedder.embed(snapshotBmp, rotBox)
                    if (negEmb != null) sceneNegs.add(negEmb)
                    if (det.label == "person" && isPerson) {
                        val otherFace = faceEmbedder.embedFace(snapshotBmp, rotBox)
                        val otherBody = personReId.embed(snapshotBmp, rotBox)
                        if (otherFace != null || otherBody != null) {
                            rosterObservations.add(otherFace to otherBody)
                        }
                    }
                }

                val result = LockResult(
                    trackingId = trackingId,
                    boundingBox = boundingBox,
                    label = label,
                    gallery = augResult.embeddings,
                    colorHist = colorHist,
                    personAttrs = personAttrs,
                    reIdEmb = reIdEmb,
                    faceEmb = faceEmb,
                    sceneNegatives = sceneNegs,
                    sceneRosterObservations = rosterObservations,
                    deviceRotation = snapshotDevRot,
                    snapshotBmp = snapshotBmp
                )
                // Hand off; if a previous result is still pending, recycle its bitmap.
                pendingLockResult.getAndSet(result)?.snapshotBmp?.recycle()
            } catch (t: Throwable) {
                android.util.Log.e("ObjectTracker", "Lock ML failed: ${t.message}", t)
                snapshotBmp.recycle()
            }
        }
    }

    /**
     * Apply a completed off-thread lock result on the processing thread. Called at
     * the start of [processBitmapInternal] so all state mutation happens on the
     * single thread that reads it. Recycles the snapshot bitmap once consumed.
     */
    private fun applyPendingLockIfAny() {
        val result = pendingLockResult.getAndSet(null) ?: return
        try {
            reacquisition.lock(result.trackingId, result.boundingBox, result.label,
                result.gallery, result.colorHist, result.personAttrs,
                cocoLabel = result.label, reIdEmbedding = result.reIdEmb, faceEmbedding = result.faceEmb)
            visualTracker.init(result.snapshotBmp, result.boundingBox)
            vtLockedBoxArea = result.boundingBox.width() * result.boundingBox.height()

            for (neg in result.sceneNegatives) reacquisition.addSceneNegative(neg)
            for ((face, body) in result.sceneRosterObservations) {
                reacquisition.observePerson(face, body)
            }

            debugCapture.startSession(result.label, result.trackingId)
            val attrStr = result.personAttrs?.summary() ?: "n/a"
            debugCapture.log("LOCK id=${result.trackingId} label=${result.label} box=${result.boundingBox} gallery=${result.gallery.size} colorHist=${result.colorHist != null} attrs=\"$attrStr\"")

            debugCapture.sessionDir?.let { dir ->
                scenarioRecorder.start(dir, result.trackingId, result.label, result.label,
                    result.boundingBox, result.gallery, result.colorHist, result.personAttrs)
            }

            val locked = TrackedObject(result.trackingId, result.boundingBox, result.label)
            debugCapture.capture(DebugEvent.LOCK, result.snapshotBmp, listOf(locked), lockedObject = locked,
                extraInfo = "id=${result.trackingId} label=${result.label} gallery=${result.gallery.size}")

            // Audit instrumentation (#92): seed the stability ring with lock-time
            // embeddings so frame 0 has something to compare frame 5 against.
            cropDebugCapture.startSession(debugCapture.sessionDir)
            stabilityLogger.clear()
            val isPerson = result.label == "person"
            cropDebugCapture.capture("LOCK", 0, result.snapshotBmp, result.boundingBox, isPerson, result.label)
            stabilityLogger.record("mnv3", 0, result.gallery.firstOrNull())
            if (isPerson) {
                stabilityLogger.record("osnet", 0, result.reIdEmb)
                stabilityLogger.record("face", 0, result.faceEmb)
            }
        } finally {
            result.snapshotBmp.recycle()
        }
    }

    fun clearLock() {
        scenarioRecorder.recordEvent("CLEAR")
        scenarioRecorder.stop()
        debugCapture.log("CLEAR by user")
        // Audit (#92): drain pending audit work so the JSON includes every
        // sample queued during the session, then flush BEFORE the debug
        // session dir is nulled.
        try {
            auditExecutor.submit {}.get(2, java.util.concurrent.TimeUnit.SECONDS)
        } catch (e: Exception) {
            android.util.Log.w("AuditEmbed", "Audit drain timeout: ${e.message}")
        }
        val summary = stabilityLogger.flush(debugCapture.sessionDir)
        if (summary != null) lastEmbeddingStabilitySummary = summary
        cropDebugCapture.endSession()
        stabilityLogger.clear()
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
        pendingLockFuture?.cancel(true)
        pendingLockFuture = null
        pendingLockResult.getAndSet(null)?.snapshotBmp?.recycle()
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
        // Apply any off-thread lock that completed since the last frame.
        applyPendingLockIfAny()

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
                    // VT returns coords in rotated-image space; unmap to screen space.
                    // Mutable so the detector-anchor reseat (#110) can swap them to
                    // the detector's box when VT runs away onto background texture.
                    val deviceRot = deviceRotation
                    var rawBox = vtResult.boundingBox
                    var vtBox = unmapRotation(rawBox.left, rawBox.top, rawBox.right, rawBox.bottom, deviceRot)

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

                    val lockedIsPerson = reacquisition.lockedIsPerson
                    val matchedDet: TrackedObject? = if (skipDetector) null else {
                        // Confirmation = detection overlaps VT's position.
                        // Accept same-category at low IoU (0.15) or any category at high IoU (0.4).
                        // High IoU means it's the same object regardless of label.
                        // Low IoU with different category (e.g. person walking past a cup) doesn't confirm.
                        //
                        // Among multiple matches we pick the SMALLEST-area detection (#110): when
                        // VT is running away onto background texture, both a tight chair box and
                        // an inflated envelope box may match VT's IoU range. The tight box is the
                        // truer ground-truth size for the runaway-vs-detector size comparison
                        // below, and is also the better reseat target.
                        tracked
                            .filter { det ->
                                val iou = FrameToFrameTracker.computeIou(det.boundingBox, vtBox)
                                val sameCategory = (det.label == "person") == lockedIsPerson
                                (sameCategory && iou > 0.15f) || iou > 0.4f
                            }
                            .minByOrNull { it.boundingBox.width() * it.boundingBox.height() }
                    }
                    val confirmed = skipDetector || matchedDet != null
                    // Distinguish non-confirmation from contradiction:
                    // If detector found nothing at all, that's not evidence of drift —
                    // it's poor detection conditions. Only count unconfirmed when
                    // detector found detections but none match our tracked position.
                    val detectorFoundSomething = skipDetector || tracked.isNotEmpty()

                    if (confirmed) {
                        vtUnconfirmedFrames = 0

                        // Detector-anchor reseat (#110): when the detector confirms VT
                        // but reports a much smaller box for the same object, VT is
                        // running away onto background texture. Reseat the tracker so
                        // subsequent updates start from the detector's ground-truth size.
                        // No-op when:
                        //   - detector skipped this frame (matchedDet null)
                        //   - VT and detector agree on size (legitimate zoom)
                        //   - detector reports LARGER box (some other path; not our concern)
                        if (matchedDet != null) {
                            val detBox = matchedDet.boundingBox
                            val detArea = detBox.width() * detBox.height()
                            val vtArea = vtBox.width() * vtBox.height()
                            val growthRatio = if (vtLockedBoxArea > 0f) vtArea / vtLockedBoxArea else 1f
                            val disagreementRatio = if (detArea > 0f) vtArea / detArea else 1f
                            // Both conditions: VT has grown since init AND VT is
                            // larger than detector. Either alone is a false positive
                            // (legitimate zoom or sizing-convention mismatch).
                            if (growthRatio > VT_RESEAT_GROWTH_RATIO &&
                                disagreementRatio > VT_RESEAT_DISAGREEMENT_RATIO) {
                                visualTracker.init(bitmap, detBox)
                                vtLockedBoxArea = detArea
                                templateMismatchCount = 0
                                // Reset the confirmed-frame counter so accumulation
                                // doesn't immediately fire on this frame's drifted
                                // crop (rotated-coord rawBox is the inflated VT box).
                                vtConfirmedFrames = 0
                                // Swap working boxes for downstream embedding/sync so
                                // this frame's accumulator and template-check operate
                                // on detector-confirmed pixels, not the runaway crop.
                                vtBox = detBox
                                rawBox = mapToRotated(detBox.left, detBox.top, detBox.right, detBox.bottom, deviceRot)
                                debugCapture.log("VT_RESEAT vtArea=${"%.3f".format(vtArea)} detArea=${"%.3f".format(detArea)} growth=${"%.2fx".format(growthRatio)} disagreement=${"%.2fx".format(disagreementRatio)} → reinit on detBox=[${"%.2f,%.2f,%.2f,%.2f".format(detBox.left, detBox.top, detBox.right, detBox.bottom)}]")
                                android.util.Log.d("VisualTracker", "VT_RESEAT growth=${"%.2fx".format(growthRatio)} disagreement=${"%.2fx".format(disagreementRatio)}")
                            }
                        }

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
                                // Two-gate accumulation. Both must hold:
                                //   1. Diverse from the existing centroid — don't add near-
                                //      duplicates that cost RAM without adding identity coverage.
                                //   2. Still close to the original lock identity — don't poison
                                //      the gallery with a drifted VT crop. Without this clause
                                //      a brief VT drift adds non-lock embeddings during the
                                //      lock window and a later candidate matches against THOSE
                                //      at high cosine, defeating identity discrimination.
                                //
                                // Old gate: `centroidSim < 0.92 OR gallery.size < 8`. The
                                // size escape hatch forced gallery growth to 8 regardless of
                                // identity — the structural source of the pollution. On a
                                // diagnostic where a small distant kid was locked and the
                                // camera panned, lockSim≈0.06 entries were waved through and
                                // a different person scored sim=0.864 against the polluted
                                // gallery vs 0.541 against the raw lock-only embeddings.
                                //
                                // The lockSim threshold is per-lock adaptive (#110): tight-
                                // distribution embedders (OSNet/persons) clamp at 0.50,
                                // loose ones (MNV3/chair) drop to ~0.32-0.42. Fixed 0.5
                                // was cutting through the same-object band of generic
                                // classes — gallery never grew past 5, tracking 76% → 25%.
                                val centroidSim = reacquisition.centroidSimilarity(emb)
                                val lockSim = reacquisition.bestLockGallerySimilarity(emb)
                                val lockSimFloor = reacquisition.lockSelfFloor
                                val boxStr = "%.2f,%.2f,%.2f,%.2f".format(rawBox.left, rawBox.top, rawBox.right, rawBox.bottom)
                                if (centroidSim < 0.92f && lockSim > lockSimFloor) {
                                    reacquisition.addEmbedding(emb)
                                    android.util.Log.d("AppearEmbed", "Gallery +1 → ${reacquisition.embeddingGallery.size} (centroidSim=${"%.3f".format(centroidSim)}, lockSim=${"%.3f".format(lockSim)}, floor=${"%.3f".format(lockSimFloor)})")
                                    debugCapture.log("ACCUM +1 vtFrame=$vtConfirmedFrames gallery=${reacquisition.embeddingGallery.size} lockSim=${"%.3f".format(lockSim)} floor=${"%.3f".format(lockSimFloor)} centroidSim=${"%.3f".format(centroidSim)} box=[$boxStr]")
                                } else {
                                    val why = if (lockSim <= lockSimFloor) "lockSim<=floor(${"%.3f".format(lockSimFloor)})" else "centroidSim>=0.92"
                                    debugCapture.log("ACCUM skip vtFrame=$vtConfirmedFrames gallery=${reacquisition.embeddingGallery.size} lockSim=${"%.3f".format(lockSim)} floor=${"%.3f".format(lockSimFloor)} centroidSim=${"%.3f".format(centroidSim)} box=[$boxStr] ($why)")
                                }
                            }
                            // Progressive face embedding: try to capture face during tracking
                            // Use rawBox (rotated-image coords) since bitmap is the rotated image
                            if (reacquisition.lockedFaceEmbedding == null && reacquisition.lockedReIdEmbedding != null) {
                                val faceEmb = faceEmbedder.embedFace(bitmap, rawBox)
                                if (faceEmb != null) reacquisition.addFaceEmbedding(faceEmb)
                            }

                            // Augment the SessionRoster's lock-slot body gallery (#108)
                            // every 5 confirmed VT frames. Without this the lock slot
                            // stays at 1 body sample from seedLock while non-lock slots
                            // accumulate continuously, creating an asymmetric ROSTER_REJECT
                            // disadvantage for the locked subject.
                            if (vtConfirmedFrames % 5 == 0 &&
                                reacquisition.lockedIsPerson &&
                                reacquisition.lockedReIdEmbedding != null) {
                                val lockBodyEmb = personReId.embed(bitmap, rawBox)
                                if (lockBodyEmb != null) reacquisition.augmentLockReId(lockBodyEmb)
                            }
                        }

                        // Collect scene negatives + roster observations every 5 confirmed
                        // frames. tracked has screen-space boxes but bitmap is rotated,
                        // so convert back to rotated-image space before cropping.
                        //
                        // The roster (#108) accepts partial observations — face-only or
                        // body-only — so we no longer drop a candidate when one modality
                        // fails. Either signal alone is sufficient evidence that someone
                        // other than the lock is in the scene.
                        //
                        // Cost: face+body inference is ~16-20ms per non-lock person.
                        // Tried 15-frame cadence earlier; real footage with short
                        // multi-person lock windows under-observed the distractor.
                        // Async batching is a follow-up optimization once we can
                        // measure real device fps cost.
                        if (vtConfirmedFrames % 5 == 0) {
                            val collectRoster = reacquisition.lockedIsPerson
                            for (det in tracked) {
                                if (det.id == reacquisition.lockedId) continue
                                val rotBox = mapToRotated(det.boundingBox.left, det.boundingBox.top,
                                    det.boundingBox.right, det.boundingBox.bottom, deviceRot)
                                val negEmb = appearanceEmbedder.embed(bitmap, rotBox)
                                if (negEmb != null) reacquisition.addSceneNegative(negEmb)

                                if (collectRoster && det.label == "person") {
                                    val faceEmb = faceEmbedder.embedFace(bitmap, rotBox)
                                    val bodyEmb = personReId.embed(bitmap, rotBox)
                                    if (faceEmb != null || bodyEmb != null) {
                                        reacquisition.observePerson(faceEmb, bodyEmb)
                                    }
                                }
                            }
                        }

                        // Audit instrumentation (#92): periodically run all three
                        // embedders on the locked object's current VT crop and
                        // record into the stability logger. The production
                        // pipeline never embeds OSNet on the lock during VT
                        // tracking, so we need our own sampling cadence to get
                        // a per-embedder noise floor. Cost ≈ 17ms per sample on
                        // person locks; gated by AUDIT_ENABLED for release builds.
                        if (CropDebugCapture.AUDIT_ENABLED &&
                            vtConfirmedFrames % AUDIT_STABILITY_INTERVAL == 0) {
                            runAuditSample(bitmap, rawBox)
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
                    // Hysteresis: increment when sim < TEMPLATE_SIM_LOW, decrement
                    // (floored at 0) when sim > TEMPLATE_SIM_OK. Values in between
                    // (the dead zone) leave the counter unchanged. Decrement instead
                    // of full reset means a single high-sim flicker doesn't erase
                    // two prior bad frames — fixes the slow-drift case in #80 where
                    // panning to a uniform backdrop produced 0.3 / 0.5 / 0.3 / 0.5
                    // patterns that kept the counter pinned at 1/3.
                    if (!skipDetector &&
                        reacquisition.hasEmbeddings &&
                        vtFrameCounter % TEMPLATE_CHECK_INTERVAL == 0) {
                        val curEmb = appearanceEmbedder.embedWithFallback(bitmap, rawBox)
                        if (curEmb != null) {
                            // Use lock-time gallery only — accumulated entries
                            // can include drifted VT crops that mask the drift
                            // signal at sim=1.0. (#80)
                            val sim = reacquisition.bestLockGallerySimilarity(curEmb)
                            // Dynamic thresholds (#110): track LOW/OK to the
                            // per-lock floor so the dead zone sits inside the
                            // class's live-vs-gallery band. Hysteresis still
                            // requires LOW < OK; the ratio constant guarantees this.
                            val templateOk = reacquisition.lockSelfFloor
                            val templateLow = templateOk * ReacquisitionEngine.TEMPLATE_SIM_LOW_RATIO
                            when {
                                sim < templateLow -> {
                                    templateMismatchCount++
                                    android.util.Log.d("VisualTracker", "Template mismatch $templateMismatchCount/$TEMPLATE_MISMATCH_MAX (sim=${"%.3f".format(sim)} low=${"%.3f".format(templateLow)})")
                                    debugCapture.log("TEMPLATE mismatch vtFrame=$vtConfirmedFrames count=$templateMismatchCount/$TEMPLATE_MISMATCH_MAX lockSim=${"%.3f".format(sim)} low=${"%.3f".format(templateLow)}")
                                }
                                sim > templateOk -> {
                                    if (templateMismatchCount > 0) {
                                        templateMismatchCount--
                                        android.util.Log.d("VisualTracker", "Template recovery $templateMismatchCount/$TEMPLATE_MISMATCH_MAX (sim=${"%.3f".format(sim)} ok=${"%.3f".format(templateOk)})")
                                        debugCapture.log("TEMPLATE recovery vtFrame=$vtConfirmedFrames count=$templateMismatchCount/$TEMPLATE_MISMATCH_MAX lockSim=${"%.3f".format(sim)} ok=${"%.3f".format(templateOk)}")
                                    }
                                }
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
                        debugCapture.log("DRIFT vtFrame=$vtConfirmedFrames reason=\"$reason\" conf=${"%.2f".format(vtResult.confidence)} speed=${"%.3f".format(velocityEstimator.speed)}")
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

                // Merge cached embeddings into current detections by ID match only.
                // FTFTracker preserves IDs across frames for the same physical object,
                // so an ID hit is identity-safe. The previous IoU>0.3 fallback was
                // removed (#116): when the lock person left frame, their cache entry
                // could be inherited by any new detection at IoU>0.3 with the stale
                // cachedBox — including a totally different person who walked into
                // the area. The candidate's reIdEmbedding became a literal copy of
                // the lock's, producing reId=1.000 in scoring and pulling the
                // cascade above its 0.45 admit threshold even when MNV3 sim was
                // 0.15 (clear impostor). The sync-fallback path below (line ~836)
                // still recovers async work for the single best same-category
                // candidate that has no embedding, by computing fresh OSNet on its
                // actual bbox rather than inheriting from spatial neighbors.
                val merged = withLabels.map { obj ->
                    val cached = if (obj.id >= 0) cachedResults[obj.id] else null
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

                        val syncEntry: SyncEmbedEntry? = if (reusable) {
                            cached
                        } else {
                            // Compute embedding synchronously (raw crop fallback, ~8-23ms)
                            val embResult = appearanceEmbedder.embedAndCrop(bitmap, best.boundingBox, fallback = true)
                            val computedHist = if (embResult.maskedCrop != null) {
                                val fullBox = RectF(0f, 0f, 1f, 1f)
                                computeColorHistogram(embResult.maskedCrop, fullBox).also { embResult.maskedCrop.recycle() }
                            } else computeColorHistogram(bitmap, best.boundingBox)
                            if (embResult.embedding != null) {
                                // Person candidates also need OSNet (and face when locked
                                // has a face) so the new OSNet-gated path (#67) can fire
                                // on freshly-embedded candidates without waiting for async.
                                val isPersonCandidate = lockedIsPerson && best.label == "person"
                                val reIdEmb = if (isPersonCandidate) personReId.embed(bitmap, best.boundingBox) else null
                                val faceEmb = if (isPersonCandidate && reacquisition.lockedFaceEmbedding != null) {
                                    faceEmbedder.embedFace(bitmap, best.boundingBox)
                                } else null
                                android.util.Log.d("EmbedSync", "Sync embedding for id=${best.id} label=${best.label} reId=${reIdEmb != null} face=${faceEmb != null}")
                                val entry = SyncEmbedEntry(
                                    embedding = embResult.embedding,
                                    colorHistogram = computedHist,
                                    reIdEmbedding = reIdEmb,
                                    faceEmbedding = faceEmb,
                                    cachedBox = RectF(best.boundingBox),
                                    framesLost = currFramesLost
                                )
                                recentSyncEmbeds[best.id] = entry
                                entry
                            } else null
                        }

                        if (syncEntry != null) {
                            merged.map { obj ->
                                if (obj.id == best.id) obj.copy(
                                    embedding = syncEntry.embedding,
                                    colorHistogram = syncEntry.colorHistogram,
                                    reIdEmbedding = syncEntry.reIdEmbedding ?: obj.reIdEmbedding,
                                    faceEmbedding = syncEntry.faceEmbedding ?: obj.faceEmbedding
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
        // Second pass: classify all person candidates. Previously this was filtered
        // to top-2 by MobileNetV3 sim, which silently dropped the right person from
        // OSNet computation when MobileNetV3 ranked the wrong candidate higher
        // (#67). With the OSNet gate, every person candidate needs its OSNet
        // embedding so the engine can decide on identity.
        val personIds = detections
            .filter { it.label == "person" && results[it.id]?.embedding != null }
            .map { it.id }
        for (id in personIds) {
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

    /**
     * Audit instrumentation (#92): snapshot the current frame + locked-object
     * box on the processing thread, then offload all embedder calls + the
     * composite-PNG encode to [auditExecutor]. Cost on the processing thread
     * is just one bitmap.copy (~4ms per audit fire ≈ 0.8ms/frame averaged at
     * the 5-frame cadence) — meaningfully cheaper than running the embedders
     * synchronously, and isolates audit overhead from production timing so
     * tests don't see audit-induced hangs from GPU/embedder contention.
     */
    private fun runAuditSample(bitmap: Bitmap, rawBox: RectF) {
        val isPerson = reacquisition.lockedIsPerson
        val frame = vtConfirmedFrames
        val saveComposite = frame % AUDIT_COMPOSITE_INTERVAL == 0
        val label = reacquisition.lastKnownLabel
        val snapshot = try {
            bitmap.copy(Bitmap.Config.ARGB_8888, false) ?: return
        } catch (e: Throwable) { return }
        val box = RectF(rawBox)
        try {
            auditExecutor.execute {
                try {
                    val mnv3 = appearanceEmbedder.embedWithFallback(snapshot, box)
                    val osnet = if (isPerson) personReId.embed(snapshot, box) else null
                    val face = if (isPerson) faceEmbedder.embedFace(snapshot, box) else null
                    stabilityLogger.record("mnv3", frame, mnv3)
                    if (isPerson) {
                        stabilityLogger.record("osnet", frame, osnet)
                        stabilityLogger.record("face", frame, face)
                    }
                    if (saveComposite) {
                        cropDebugCapture.capture("VT", frame, snapshot, box, isPerson, label)
                    }
                } catch (e: Throwable) {
                    android.util.Log.w("AuditEmbed", "Audit sample failed: ${e.message}")
                } finally {
                    snapshot.recycle()
                }
            }
        } catch (e: java.util.concurrent.RejectedExecutionException) {
            // Executor shut down concurrently — drop the snapshot.
            snapshot.recycle()
        }
    }

    fun shutdown() {
        scenarioRecorder.stop()
        debugCapture.shutdown()
        cropDebugCapture.shutdown()
        auditExecutor.shutdownNow()
        embeddingExecutor.shutdownNow()
        lockExecutor.shutdownNow()
        pendingLockResult.getAndSet(null)?.snapshotBmp?.recycle()
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
