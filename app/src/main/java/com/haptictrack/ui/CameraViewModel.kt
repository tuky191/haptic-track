package com.haptictrack.ui

import android.app.Application
import android.graphics.RectF
import androidx.camera.video.VideoRecordEvent
import androidx.camera.view.PreviewView
import androidx.lifecycle.AndroidViewModel
import androidx.lifecycle.LifecycleOwner
import com.haptictrack.camera.CameraManager
import com.haptictrack.camera.DeviceOrientationListener
import com.haptictrack.camera.RecordingManager
import com.haptictrack.haptics.HapticFeedbackManager
import com.haptictrack.tracking.CaptureMode
import com.haptictrack.tracking.ObjectTracker
import com.haptictrack.tracking.TrackedObject
import com.haptictrack.tracking.TrackingStatus
import com.haptictrack.tracking.TrackingUiState
import com.haptictrack.zoom.ZoomController
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.flow.update

class CameraViewModel(application: Application) : AndroidViewModel(application) {

    internal val cameraManager = CameraManager(application)
    private val recordingManager = RecordingManager(application)
    private val objectTracker = ObjectTracker(application)
    private val hapticManager = HapticFeedbackManager(application)
    private val zoomController = ZoomController()
    private val orientationListener = DeviceOrientationListener(application)

    private val _uiState = MutableStateFlow(TrackingUiState())
    val uiState: StateFlow<TrackingUiState> = _uiState.asStateFlow()

    /** Smooths idle detections by keeping objects alive for a few frames after they disappear. */
    private val recentDetections = mutableMapOf<Int, Pair<TrackedObject, Int>>() // id → (object, framesRemaining)
    private val IDLE_PERSIST_FRAMES = 5

    init {
        orientationListener.start()
        objectTracker.deviceRotationProvider = { orientationListener.deviceRotation }

        cameraManager.imageAnalysis.setAnalyzer(
            java.util.concurrent.Executors.newSingleThreadExecutor(),
            objectTracker.analyzer
        )

        objectTracker.onDetectionResult = { allObjects, lockedObject, imgWidth, imgHeight, contour ->
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
                zoomController.calculateZoom(
                    lockedObject.boundingBox,
                    cameraManager.getMinZoom(),
                    cameraManager.getMaxZoom()
                ).also { cameraManager.setZoomRatio(it) }
            } else if (status == TrackingStatus.LOST && previousStatus == TrackingStatus.LOCKED) {
                // Just lost — zoom out partially to widen field of view for re-acquisition
                zoomController.zoomOutForSearch(
                    cameraManager.getMinZoom(),
                    cameraManager.getMaxZoom()
                ).also { cameraManager.setZoomRatio(it) }
            } else null

            hapticManager.updateTrackingStatus(status, edgeProximity)

            // In idle state, smooth detections to prevent bracket flickering
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
    }

    fun startCamera(lifecycleOwner: LifecycleOwner, previewView: PreviewView) {
        cameraManager.startCamera(lifecycleOwner, previewView)
    }

    fun onTapToLock(normalizedX: Float, normalizedY: Float) {
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

    companion object {
        /** Tap target padding in normalized coordinates (~3% of screen on each side). */
        private const val TAP_PADDING = 0.03f
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

    /** Called when pinch gesture ends — starts the fade-out timer for the zoom indicator. */
    fun onPinchEnd() {
        // The indicator fade-out is handled by LaunchedEffect in CameraScreen
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
     */
    fun onVolumeDown() {
        val state = _uiState.value

        if (state.isRecording) {
            // Stage 3: stop recording and clear
            toggleRecording()
            clearTracking()
            return
        }

        if (state.status != TrackingStatus.IDLE) {
            // Stage 2: tracking but not recording — start recording
            toggleRecording()
            return
        }

        // Stage 1: idle — lock on center
        val objects = state.detectedObjects.filter { it.id >= 0 }
        if (objects.isEmpty()) return

        val closest = objects.minByOrNull { obj ->
            val cx = obj.boundingBox.centerX() - 0.5f
            val cy = obj.boundingBox.centerY() - 0.5f
            cx * cx + cy * cy
        } ?: return

        objectTracker.lockOnObject(closest.id, closest.boundingBox, closest.label)
        _uiState.update { it.copy(status = TrackingStatus.LOCKED, trackedObject = closest) }
    }

    fun toggleCaptureMode() {
        _uiState.update { current ->
            current.copy(
                captureMode = if (current.captureMode == CaptureMode.VIDEO) CaptureMode.PHOTO else CaptureMode.VIDEO
            )
        }
    }

    fun switchCamera() {
        clearTracking()
        cameraManager.switchCamera()
    }

    fun clearTracking() {
        objectTracker.clearLock()
        zoomController.reset()
        hapticManager.updateTrackingStatus(TrackingStatus.IDLE)
        _uiState.update {
            TrackingUiState(status = TrackingStatus.IDLE, isRecording = it.isRecording, captureMode = it.captureMode)
        }
    }

    @android.annotation.SuppressLint("MissingPermission")
    fun toggleRecording() {
        if (recordingManager.isRecording) {
            recordingManager.stopRecording()
            _uiState.update { it.copy(isRecording = false) }
        } else {
            recordingManager.startRecording(cameraManager.videoCapture) { event ->
                when (event) {
                    is VideoRecordEvent.Start ->
                        _uiState.update { it.copy(isRecording = true) }
                    is VideoRecordEvent.Finalize ->
                        _uiState.update { it.copy(isRecording = false) }
                }
            }
        }
    }

    override fun onCleared() {
        super.onCleared()
        orientationListener.stop()
        objectTracker.shutdown()
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