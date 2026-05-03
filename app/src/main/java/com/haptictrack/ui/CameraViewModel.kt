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
import com.haptictrack.zoom.ZoomController
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

    private val _uiState = MutableStateFlow(TrackingUiState())
    val uiState: StateFlow<TrackingUiState> = _uiState.asStateFlow()

    /** Viewfinder frame from SurfaceTexture GL thread — always active. */
    private val _viewfinderBitmap = MutableStateFlow<android.graphics.Bitmap?>(null)
    val viewfinderBitmap: StateFlow<android.graphics.Bitmap?> = _viewfinderBitmap.asStateFlow()

    companion object {
        private const val TAG = "CameraVM"
        /** Tap target padding in normalized coordinates (~3% of screen on each side). */
        private const val TAP_PADDING = 0.03f
    }

    /** Smooths idle detections by keeping objects alive for a few frames after they disappear. */
    private val recentDetections = mutableMapOf<Int, Pair<TrackedObject, Int>>() // id → (object, framesRemaining)
    private val IDLE_PERSIST_FRAMES = 5

    init {
        orientationListener.start()

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

                val edgeProximity = lockedObject?.let {
                    zoomController.calculateEdgeProximity(it.boundingBox)
                } ?: 0f

                val targetZoom = if (lockedObject != null) {
                    zoomController.resetLossCounter()
                    zoomController.calculateZoom(
                        lockedObject.boundingBox,
                        cameraManager.getMinZoom(),
                        cameraManager.getMaxZoom()
                    ).also { cameraManager.setZoomRatio(it) }
                } else if (status == TrackingStatus.LOST) {
                    // Gradual zoom-out: delays 5 frames then pulls back 15% per frame.
                    // Gives reacquisition a chance at the original zoom before widening FOV.
                    zoomController.zoomOutForSearchGradual(
                        cameraManager.getMinZoom(),
                        cameraManager.getMaxZoom()
                    ).also { cameraManager.setZoomRatio(it) }
                } else null

                hapticManager.updateTrackingStatus(status, edgeProximity)

                val displayObjects = if (status == TrackingStatus.IDLE) {
                    smoothIdleDetections(allObjects)
                } else {
                    recentDetections.clear()
                    allObjects
                }

                _uiState.update { current ->
                    current.copy(
                        status = status,
                        trackedObject = lockedObject ?: if (status == TrackingStatus.LOST) current.trackedObject else null,
                        detectedObjects = displayObjects,
                        sourceImageWidth = imgWidth,
                        sourceImageHeight = imgHeight,
                        currentZoomRatio = targetZoom ?: current.currentZoomRatio,
                        lockedContour = if (status == TrackingStatus.LOCKED) contour else
                            if (status == TrackingStatus.LOST) current.lockedContour else emptyList()
                    )
                }
            }

            objectTracker = tracker
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
                    // Before models load we'd leak the bitmap — hand it straight back.
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
        val tapped = objects
            .filter { it.id >= 0 && it.boundingBox.containsWithPadding(normalizedX, normalizedY, TAP_PADDING) }
            .minByOrNull { it.boundingBox.width() * it.boundingBox.height() }

        if (tapped != null) {
            objectTracker.lockOnObject(tapped.id, tapped.boundingBox, tapped.label)
            _uiState.update { it.copy(status = TrackingStatus.LOCKED, trackedObject = tapped) }
            if (!_uiState.value.isRecording) toggleRecording()
        }
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
        cameraManager.setZoomRatio(appliedZoom)
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
        val objects = state.detectedObjects.filter { it.id >= 0 }
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

    fun toggleCaptureMode() {
        _uiState.update { current ->
            current.copy(
                captureMode = if (current.captureMode == CaptureMode.VIDEO) CaptureMode.PHOTO else CaptureMode.VIDEO
            )
        }
    }

    fun toggleStealthMode() {
        val entering = !_uiState.value.stealthMode
        if (!entering && _uiState.value.isRecording) {
            recordingManager.stopRecording()
        }
        _uiState.update { it.copy(stealthMode = entering) }
        // No camera rebind needed — SurfaceTexture pipeline is always active.
        // Stealth is purely a UI overlay change.
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
        _uiState.update { it.copy(gyroEis = newValue) }
    }

    fun setGyroStrength(strength: Float) {
        val clamped = strength.coerceIn(0f, 1f)
        val tc = 0.30 - 0.26 * clamped
        val crop = 1.05f + 0.15f * clamped
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
        objectTracker.clearLock()
        zoomController.reset()
        hapticManager.updateTrackingStatus(TrackingStatus.IDLE)
        _uiState.update {
            TrackingUiState(status = TrackingStatus.IDLE, isRecording = it.isRecording, captureMode = it.captureMode, stealthMode = it.stealthMode, isReady = it.isReady, ispStabilization = it.ispStabilization, gyroEis = it.gyroEis, gyroStrength = it.gyroStrength)
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
            if (cameraManager.gyroStabilizer.enabled) {
                val benchDir = java.io.File(
                    getApplication<Application>().getExternalFilesDir(null),
                    "bench/session_${java.text.SimpleDateFormat("yyyyMMdd_HHmmss", java.util.Locale.US).format(java.util.Date())}"
                ).also { it.mkdirs() }
                cameraManager.gyroStabilizer.startBenchCapture(benchDir)
            }

            recordingManager.startRecording(cameraManager.videoCapture) { event ->
                when (event) {
                    is VideoRecordEvent.Start ->
                        _uiState.update { it.copy(isRecording = true) }
                    is VideoRecordEvent.Finalize -> {
                        cameraManager.gyroStabilizer.endBenchCapture()
                        _uiState.update { it.copy(isRecording = false) }
                    }
                }
            }
        }
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
