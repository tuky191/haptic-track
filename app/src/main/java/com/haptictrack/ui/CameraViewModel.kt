package com.haptictrack.ui

import android.app.Application
import android.graphics.RectF
import android.util.Log
import androidx.camera.video.VideoRecordEvent
import androidx.camera.view.PreviewView
import androidx.lifecycle.AndroidViewModel
import androidx.lifecycle.LifecycleOwner
import androidx.lifecycle.viewModelScope
import com.haptictrack.camera.CameraManager
import com.haptictrack.camera.DeviceOrientationListener
import com.haptictrack.camera.RecordingManager
import com.haptictrack.haptics.HapticFeedbackManager
import com.haptictrack.tracking.ObjectTracker
import com.haptictrack.tracking.TrackedObject
import com.haptictrack.tracking.TrackingStatus
import com.haptictrack.tracking.TrackingUiState
import com.haptictrack.tracking.CaptureMode
import com.haptictrack.tracking.TrackingFilter
import com.haptictrack.tracking.labelMatchesFilter
import com.haptictrack.tracking.SentryController
import com.haptictrack.tracking.SentryCriteria
import com.haptictrack.tracking.GenderFilter
import com.haptictrack.zoom.ZoomController
import java.io.File
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.flow.update
import kotlinx.coroutines.launch

class CameraViewModel(application: Application) : AndroidViewModel(application) {

    internal val cameraManager = CameraManager(application)
    private val recordingManager = RecordingManager(application)
    private lateinit var objectTracker: ObjectTracker
    private val hapticManager = HapticFeedbackManager(application)
    private val zoomController = ZoomController()
    private val orientationListener = DeviceOrientationListener(application)
    private var sentry: SentryController? = null
    private val sentryLogger = com.haptictrack.tracking.SentryLogger(application)
    private var sentryInspected = 0
    private var sentryMatched = 0

    private val _uiState = MutableStateFlow(TrackingUiState())
    val uiState: StateFlow<TrackingUiState> = _uiState.asStateFlow()

    /** Viewfinder frame from SurfaceTexture GL thread — always active. */
    private val _viewfinderBitmap = MutableStateFlow<android.graphics.Bitmap?>(null)
    val viewfinderBitmap: StateFlow<android.graphics.Bitmap?> = _viewfinderBitmap.asStateFlow()

    companion object {
        private const val TAG = "CameraVM"
        /** Tap target padding in normalized coordinates (~3% of screen on each side). */
        private const val TAP_PADDING = 0.03f
        private const val GYRO_TC_MAX = 1.00       // time constant at strength=0 (most laggy)
        private const val GYRO_TC_RANGE = 0.60      // TC swing: 1.00 - 0.60 = 0.40 at strength=1
        private const val GYRO_CROP_MIN = 1.15f     // crop zoom at strength=0 (light stabilization)
        private const val GYRO_CROP_RANGE = 0.30f   // crop swing: 1.15 + 0.30 = 1.45 at strength=1
        /** Sentry age-group presets: label → inclusive [min,max] years. */
        private val AGE_GROUPS = listOf(
            "Any" to (0 to 120), "Child" to (0 to 14),
            "Teen+Adult" to (15 to 45), "Senior" to (46 to 120)
        )
    }

    /** Smooths idle detections by keeping objects alive for a few frames after they disappear. */
    private val recentDetections = mutableMapOf<Int, Pair<TrackedObject, Int>>() // id → (object, framesRemaining)
    private val IDLE_PERSIST_FRAMES = 5

    init {
        orientationListener.start()
        setGyroStrength(_uiState.value.gyroStrength)
        // Gyro EIS default is device-gated (off on S26 — hardware OIS wins);
        // sync the UI toggle to the stabilizer's actual state.
        _uiState.update { it.copy(gyroEis = cameraManager.gyroStabilizer.enabled) }

        // Load ML models on background thread — takes ~20s with GPU delegate init
        viewModelScope.launch(Dispatchers.Default) {
            _uiState.update { it.copy(loadingStatus = "Loading ML models...") }
            val tracker = ObjectTracker(getApplication(), onLoadingStatus = { status ->
                _uiState.update { it.copy(loadingStatus = status) }
            })
            tracker.deviceRotationProvider = { orientationListener.deviceRotation }
            tracker.onSessionDir = { dir ->
                if (dir != null) cameraManager.gyroStabilizer.startSessionLog(dir)
                else cameraManager.gyroStabilizer.endSessionLog()
            }

            tracker.onDetectionResult = { allObjects, lockedObject, imgWidth, imgHeight, contour ->
                val previousStatus = _uiState.value.status

                val status = when {
                    previousStatus == TrackingStatus.IDLE -> TrackingStatus.IDLE
                    lockedObject != null -> TrackingStatus.LOCKED
                    previousStatus == TrackingStatus.LOCKED || previousStatus == TrackingStatus.LOST -> TrackingStatus.LOST
                    else -> TrackingStatus.SEARCHING
                }

                // Sentry: drive the active scan from IDLE on this frame's person detections.
                // onFrame may classify (sync, ~30ms during inspect) and lock on a match.
                // A match initiates an async lock (applies next frame), so on the match
                // frame we nudge status to SEARCHING — otherwise the machine pins IDLE
                // (its first rule) and the IDLE->LOCKED transition never fires.
                val sentryPhaseBefore = sentry?.phase
                if (_uiState.value.sentryEnabled && status == TrackingStatus.IDLE) {
                    sentry?.onFrame(allObjects.filter { it.label == "person" })
                }
                val sentryMatchedNow = sentryPhaseBefore != com.haptictrack.tracking.SentryPhase.MATCHED &&
                    sentry?.phase == com.haptictrack.tracking.SentryPhase.MATCHED
                val effectiveStatus = if (sentryMatchedNow) TrackingStatus.SEARCHING else status

                val driftX = lockedObject?.let {
                    (it.boundingBox.centerX() - 0.5f) * 2f
                } ?: 0f
                val driftY = lockedObject?.let {
                    (it.boundingBox.centerY() - 0.5f) * 2f
                } ?: 0f

                val targetZoom = if (lockedObject != null) {
                    zoomController.resetLossCounter()
                    cameraManager.gyroStabilizer.onTrackingUpdate(
                        lockedObject.boundingBox.centerX(),
                        lockedObject.boundingBox.centerY(),
                        lockedObject.boundingBox.width() * lockedObject.boundingBox.height()
                    )
                    zoomController.calculateZoom(
                        lockedObject.boundingBox,
                        cameraManager.getMinZoom(),
                        cameraManager.getMaxZoom()
                    ).also { cameraManager.setZoomTarget(it) }
                } else if (effectiveStatus == TrackingStatus.LOST) {
                    cameraManager.gyroStabilizer.clearTracking()
                    // Gradual zoom-out: delays 5 frames then pulls back 15% per frame.
                    // Gives reacquisition a chance at the original zoom before widening FOV.
                    zoomController.zoomOutForSearchGradual(
                        cameraManager.getMinZoom(),
                        cameraManager.getMaxZoom()
                    ).also { cameraManager.setZoomTarget(it) }
                } else null

                hapticManager.updateTrackingStatus(effectiveStatus, driftX, driftY)

                val displayObjects = if (effectiveStatus == TrackingStatus.IDLE) {
                    smoothIdleDetections(allObjects)
                } else {
                    recentDetections.clear()
                    allObjects
                }

                _uiState.update { current ->
                    current.copy(
                        status = effectiveStatus,
                        trackedObject = lockedObject ?: if (effectiveStatus == TrackingStatus.LOST) current.trackedObject else null,
                        detectedObjects = displayObjects,
                        sourceImageWidth = imgWidth,
                        sourceImageHeight = imgHeight,
                        currentZoomRatio = targetZoom ?: current.currentZoomRatio,
                        lockedContour = if (effectiveStatus == TrackingStatus.LOCKED) contour else
                            if (effectiveStatus == TrackingStatus.LOST) current.lockedContour else emptyList(),
                        sentryPhase = sentry?.phase ?: com.haptictrack.tracking.SentryPhase.OFF
                    )
                }
            }

            objectTracker = tracker
            sentry = SentryController(
                criteria = { _uiState.value.sentryCriteria },
                setZoomTarget = { cameraManager.setZoomTarget(it) },
                currentZoom = { cameraManager.gyroStabilizer.zoomRatio },
                minZoom = { cameraManager.getMinZoom() },
                maxZoom = { cameraManager.getMaxZoom() },
                classify = { obj -> tracker.classifyPersonAttributes(obj.boundingBox) },
                lock = { obj -> sentryLock(obj) },
                haptic = { cue -> hapticManager.sentryCue(cue) },
                onEvent = { type, box, attr, note ->
                    sentryLogger.event(type, box, attr, note)
                    if (type == "INSPECT_START") sentryInspected++
                    if (type == "MATCH") sentryMatched++
                    // Capture the frame at each decision point (re-croppable offline via the box).
                    if (type == "INSPECT_START" || type == "MATCH" || type == "REJECT") {
                        tracker.currentFrameForLog()?.let { sentryLogger.saveFrame(type, it) }
                    }
                },
            )
            // Wire the pool-release callback so the tracker can return the previous
            // lastFrameBitmap (and any un-retained frame) back to the pool.
            tracker.bitmapRecycler = { bmp -> cameraManager.releaseAnalysisBitmap(bmp) }
            // Analysis frames always come from SurfaceTexture — no ImageAnalysis needed.
            // The tracker retains each input as lastFrameBitmap and calls bitmapRecycler
            // on the previous frame, so the caller must not release here.
            cameraManager.onAnalysisFrame = { bitmap ->
                if (isTrackerReady) {
                    tracker.processBitmap(bitmap)
                } else {
                    cameraManager.releaseAnalysisBitmap(bitmap)
                }
            }
            cameraManager.onViewfinderFrame = { bitmap ->
                // Don't recycle previous bitmaps — Compose's RenderThread may still be
                // drawing them asynchronously even after StateFlow emits a new value.
                // At 480×640 ARGB (~1.2MB each), GC handles this fine on 8GB+ devices.
                _viewfinderBitmap.value = bitmap
            }
            _uiState.update { it.copy(isReady = true) }
            Log.i(TAG, "ML models loaded, tracking ready")
        }
    }

    fun startCamera(lifecycleOwner: LifecycleOwner, previewView: PreviewView) {
        cameraManager.startCamera(lifecycleOwner, previewView)
    }

    private val isTrackerReady get() = ::objectTracker.isInitialized

    fun onTapToLock(normalizedX: Float, normalizedY: Float) {
        if (!isTrackerReady) return
        // Ignore taps while already tracking — only Clear can reset
        if (_uiState.value.status != TrackingStatus.IDLE) return

        val objects = _uiState.value.detectedObjects

        // Expand each box by TAP_PADDING in normalized coords to make small objects easier to hit.
        // When multiple boxes overlap at the tap point, pick the smallest (most specific).
        val filter = _uiState.value.trackingFilter
        val tapped = objects
            .filter { it.id >= 0 && labelMatchesFilter(it.label, filter) && it.boundingBox.containsWithPadding(normalizedX, normalizedY, TAP_PADDING) }
            .minByOrNull { it.boundingBox.width() * it.boundingBox.height() }

        if (tapped != null) {
            objectTracker.lockOnObject(tapped.id, tapped.boundingBox, tapped.label)
            _uiState.update { it.copy(status = TrackingStatus.LOCKED, trackedObject = tapped) }
            if (!_uiState.value.isRecording) toggleRecording()
            // ISP tracker probe: register the same box with the Qualcomm hardware
            // tracker for side-by-side comparison (logcat tag IspTracker)
            cameraManager.ispTrackerRegister(tapped.boundingBox, cameraManager.gyroStabilizer.zoomRatio)
        }
    }

    /** Sentry auto-lock on a matched candidate. Mirrors onTapToLock minus the UI mutation
     *  (the next frame's normal flow reflects LOCKED once the async lock applies). */
    private fun sentryLock(obj: TrackedObject) {
        objectTracker.lockOnObject(obj.id, obj.boundingBox, obj.label)
        if (!_uiState.value.isRecording) toggleRecording()
        Log.i(TAG, "Sentry auto-locked person id=${obj.id}")
    }

    /** Toggle the sentry active auto-lock. */
    fun toggleSentry() {
        val newVal = !_uiState.value.sentryEnabled
        if (newVal) {
            sentryInspected = 0; sentryMatched = 0
            sentryLogger.arm(_uiState.value.sentryCriteria)
        } else {
            sentryLogger.disarm(matched = sentryMatched, inspected = sentryInspected)
        }
        sentry?.setEnabled(newVal)
        _uiState.update { it.copy(sentryEnabled = newVal, sentryPhase = sentry?.phase ?: com.haptictrack.tracking.SentryPhase.OFF) }
    }

    fun setSentryGender(gender: GenderFilter) {
        _uiState.update { it.copy(sentryCriteria = it.sentryCriteria.copy(gender = gender)) }
    }

    fun setSentryAgeRange(min: Int, max: Int) {
        _uiState.update { it.copy(sentryCriteria = it.sentryCriteria.copy(ageMin = min, ageMax = max)) }
    }

    fun cycleSentryGender() {
        val vals = GenderFilter.values()
        setSentryGender(vals[(_uiState.value.sentryCriteria.gender.ordinal + 1) % vals.size])
    }

    /** Cycle the age-group preset. */
    fun cycleSentryAgeGroup() {
        val c = _uiState.value.sentryCriteria
        val idx = AGE_GROUPS.indexOfFirst { it.second.first == c.ageMin && it.second.second == c.ageMax }
        val next = AGE_GROUPS[(if (idx < 0) 0 else (idx + 1)) % AGE_GROUPS.size]
        setSentryAgeRange(next.second.first, next.second.second)
    }

    /** Human label for the current age-group preset (or "Custom"). */
    fun sentryAgeLabel(): String {
        val c = _uiState.value.sentryCriteria
        return AGE_GROUPS.firstOrNull { it.second.first == c.ageMin && it.second.second == c.ageMax }?.first
            ?: "${c.ageMin}-${c.ageMax}"
    }

    /** Merge current detections with recently-seen ones to prevent flickering. */
    private fun smoothIdleDetections(current: List<TrackedObject>): List<TrackedObject> {
        val currentIds = current.map { it.id }.toSet()

        // Refresh current detections
        for (obj in current) {
            recentDetections[obj.id] = Pair(obj, IDLE_PERSIST_FRAMES)
        }

        // Decrement and prune stale entries
        val stale = mutableListOf<Int>()
        for ((id, pair) in recentDetections) {
            if (id !in currentIds) {
                val remaining = pair.second - 1
                if (remaining <= 0) stale.add(id)
                else recentDetections[id] = Pair(pair.first, remaining)
            }
        }
        stale.forEach { recentDetections.remove(it) }

        return recentDetections.values.map { it.first }
    }

    /**
     * Handle pinch-to-zoom gesture. [scaleFactor] is the incremental scale from the gesture
     * (e.g. 1.05 = 5% zoom in, 0.95 = 5% zoom out).
     */
    fun onPinchZoom(scaleFactor: Float) {
        val currentZoom = zoomController.getCurrentZoom()
        val newZoom = currentZoom * scaleFactor
        val appliedZoom = zoomController.setManualZoom(
            newZoom, cameraManager.getMinZoom(), cameraManager.getMaxZoom()
        )
        cameraManager.setZoomImmediate(appliedZoom)
        _uiState.update { it.copy(currentZoomRatio = appliedZoom, showZoomIndicator = true) }
    }

    /** Hide the zoom indicator (called after fade-out delay). */
    fun hideZoomIndicator() {
        _uiState.update { it.copy(showZoomIndicator = false) }
    }

    /**
     * Volume-down handler — three-stage cycle:
     * 1. Idle → lock on center object
     * 2. Tracking (not recording) → start recording
     * 3. Recording → stop recording + clear tracking
     *
     * Since lock now auto-starts recording, the normal cycle is just 2 presses:
     * idle → lock+record → stop+clear. Stage 2 is a safety net.
     */
    fun onVolumeDown() {
        if (!isTrackerReady) return
        val state = _uiState.value

        if (state.isRecording) {
            // Recording → stop recording (toggleRecording also clears tracking)
            toggleRecording()
            return
        }

        if (state.status != TrackingStatus.IDLE) {
            // Tracking but not recording (shouldn't normally happen) — start recording
            toggleRecording()
            return
        }

        // Idle → lock on center + start recording
        val objects = state.detectedObjects.filter { it.id >= 0 && labelMatchesFilter(it.label, state.trackingFilter) }
        if (objects.isEmpty()) return

        val closest = objects.minByOrNull { obj ->
            val cx = obj.boundingBox.centerX() - 0.5f
            val cy = obj.boundingBox.centerY() - 0.5f
            cx * cx + cy * cy
        } ?: return

        objectTracker.lockOnObject(closest.id, closest.boundingBox, closest.label)
        _uiState.update { it.copy(status = TrackingStatus.LOCKED, trackedObject = closest) }
        toggleRecording()
    }

    fun setHapticStrength(strength: Float) {
        val clamped = strength.coerceIn(0f, 1f)
        hapticManager.strength = clamped
        _uiState.update { it.copy(hapticStrength = clamped) }
    }

    fun cycleTrackingFilter() {
        val next = when (_uiState.value.trackingFilter) {
            TrackingFilter.ALL -> TrackingFilter.PERSON_ONLY
            TrackingFilter.PERSON_ONLY -> TrackingFilter.PETS
            TrackingFilter.PETS -> TrackingFilter.NON_PERSON_ONLY
            TrackingFilter.NON_PERSON_ONLY -> TrackingFilter.ALL
        }
        _uiState.update { it.copy(trackingFilter = next) }
    }

    fun toggleCaptureMode() {
        _uiState.update { current ->
            current.copy(
                captureMode = if (current.captureMode == CaptureMode.VIDEO) CaptureMode.PHOTO else CaptureMode.VIDEO
            )
        }
    }

    /** UI-only overlay; recording and pipeline are unaffected. */
    fun toggleStealthMode() {
        _uiState.update { it.copy(stealthMode = !it.stealthMode) }
    }

    /** Volume-up: toggle stealth mode. Entry/exit point for hands-free stealth. */
    fun onVolumeUp() {
        toggleStealthMode()
    }

    fun toggleIspStabilization() {
        val newValue = !_uiState.value.ispStabilization
        cameraManager.ispStabilizationEnabled = newValue
        _uiState.update { it.copy(ispStabilization = newValue) }
        cameraManager.rebind()
    }

    fun toggleGyroEis() {
        val newValue = !_uiState.value.gyroEis
        cameraManager.gyroStabilizer.enabled = newValue
        applyOisCompensation(gyroEis = newValue, oisToggle = _uiState.value.oisCompensation)
        _uiState.update { it.copy(gyroEis = newValue) }
    }

    fun toggleAdaptiveEis() {
        val newValue = !_uiState.value.adaptiveEis
        cameraManager.gyroStabilizer.adaptiveSmoothing = newValue
        _uiState.update { it.copy(adaptiveEis = newValue) }
    }

    fun toggleLeash() {
        val newValue = !_uiState.value.leashEnabled
        cameraManager.gyroStabilizer.leashEnabled = newValue
        _uiState.update { it.copy(leashEnabled = newValue) }
    }

    fun toggleOisCompensation() {
        val newValue = !_uiState.value.oisCompensation
        applyOisCompensation(gyroEis = _uiState.value.gyroEis, oisToggle = newValue)
        _uiState.update { it.copy(oisCompensation = newValue) }
    }

    private fun applyOisCompensation(gyroEis: Boolean, oisToggle: Boolean) {
        cameraManager.gyroStabilizer.oisCompensation = if (gyroEis && oisToggle) 0.40 else 1.0
    }

    fun toggleTranslationEis() {
        val newValue = !_uiState.value.translationEis
        cameraManager.setTranslationCorrectionEnabled(newValue)
        _uiState.update { it.copy(translationEis = newValue) }
    }

    fun toggleHorizonLock() {
        val newValue = !_uiState.value.horizonLock
        cameraManager.gyroStabilizer.horizonLockEnabled = newValue
        _uiState.update { it.copy(horizonLock = newValue) }
    }

    /** Recording preset: 4K30 ↔ FHD 1080p60 + vendor VDIS. Needs a rebind; blocked while recording. */
    fun toggleFhd60Vdis() {
        if (_uiState.value.isRecording) return
        val newValue = !_uiState.value.fhd60Vdis
        cameraManager.fhd60VdisPreset = newValue
        _uiState.update { it.copy(fhd60Vdis = newValue) }
        cameraManager.rebind()
    }

    fun setGyroStrength(strength: Float) {
        val clamped = strength.coerceIn(0f, 1f)
        val tc = GYRO_TC_MAX - GYRO_TC_RANGE * clamped
        val crop = GYRO_CROP_MIN + GYRO_CROP_RANGE * clamped
        cameraManager.gyroStabilizer.timeConstant = tc
        cameraManager.gyroStabilizer.cropZoom = crop
        Log.d(TAG, "Gyro strength=${"%.2f".format(clamped)} tc=${"%.3f".format(tc)} crop=${"%.2f".format(crop)}")
        _uiState.update { it.copy(gyroStrength = clamped) }
    }

    fun switchCamera() {
        clearTracking()
        cameraManager.switchCamera()
    }

    fun clearTracking() {
        if (!isTrackerReady) return
        if (recordingManager.isRecording) {
            recordingManager.stopRecording()
            cameraManager.gyroStabilizer.endBenchCapture()
        }
        objectTracker.clearLock()
        zoomController.reset()
        cameraManager.ispTrackerCancel()
        sentry?.onLockCleared()  // re-arm scanning if sentry still on
        hapticManager.updateTrackingStatus(TrackingStatus.IDLE)
        _uiState.update {
            TrackingUiState(status = TrackingStatus.IDLE, isRecording = false, captureMode = it.captureMode, stealthMode = it.stealthMode, isReady = it.isReady, ispStabilization = it.ispStabilization, gyroEis = it.gyroEis, gyroStrength = it.gyroStrength, adaptiveEis = it.adaptiveEis, leashEnabled = it.leashEnabled, oisCompensation = it.oisCompensation, translationEis = it.translationEis, horizonLock = it.horizonLock, fhd60Vdis = it.fhd60Vdis, trackingFilter = it.trackingFilter, hapticStrength = it.hapticStrength, sentryEnabled = it.sentryEnabled, sentryCriteria = it.sentryCriteria, sentryPhase = sentry?.phase ?: com.haptictrack.tracking.SentryPhase.OFF)
        }
    }

    @android.annotation.SuppressLint("MissingPermission")
    fun toggleRecording() {
        if (!isTrackerReady) return
        if (recordingManager.isRecording) {
            recordingManager.stopRecording()
            cameraManager.gyroStabilizer.endBenchCapture()
            if (_uiState.value.status != TrackingStatus.IDLE) {
                clearTracking()
            }
        } else {
            val ts = SimpleDateFormat("yyyyMMdd_HHmmss", Locale.US).format(Date())
            val benchDir = File(
                getApplication<Application>().getExternalFilesDir(null),
                "bench/session_$ts"
            ).also { it.mkdirs() }
            cameraManager.gyroStabilizer.startBenchCapture(benchDir)

            recordingManager.startRecording(cameraManager.videoCapture) { event ->
                when (event) {
                    is VideoRecordEvent.Start ->
                        _uiState.update { it.copy(isRecording = true, recordingError = false) }
                    is VideoRecordEvent.Finalize -> {
                        cameraManager.gyroStabilizer.endBenchCapture()
                        // A clean user stop finalizes with ERROR_NONE; any error here means the
                        // recording died unexpectedly (screen-off teardown, mic revoked, storage
                        // full). Surface it loudly so a hands-free failure isn't silent.
                        val errored = event.hasError()
                        if (errored) {
                            android.util.Log.w("Recording", "Finalize error=${event.error}: ${event.cause?.message}")
                            hapticManager.recordingFailureAlert()
                        }
                        _uiState.update { it.copy(isRecording = false, recordingError = errored) }
                    }
                }
            }
        }
    }

    /** Dismiss the recording-failure banner. */
    fun dismissRecordingError() {
        _uiState.update { it.copy(recordingError = false) }
    }

    override fun onCleared() {
        super.onCleared()
        orientationListener.stop()
        if (isTrackerReady) objectTracker.shutdown()
        hapticManager.shutdown()
        cameraManager.shutdown()
        if (recordingManager.isRecording) recordingManager.stopRecording()
    }
}

/** Check if a point falls inside the rect with padding on all sides. */
private fun RectF.containsWithPadding(x: Float, y: Float, padding: Float): Boolean {
    return x >= left - padding && x <= right + padding &&
           y >= top - padding && y <= bottom + padding
}
