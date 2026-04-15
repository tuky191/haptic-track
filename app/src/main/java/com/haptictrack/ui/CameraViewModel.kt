package com.haptictrack.ui

import android.app.Application
import androidx.camera.video.VideoRecordEvent
import androidx.camera.view.PreviewView
import androidx.lifecycle.AndroidViewModel
import androidx.lifecycle.LifecycleOwner
import com.haptictrack.camera.CameraManager
import com.haptictrack.camera.DeviceOrientationListener
import com.haptictrack.camera.RecordingManager
import com.haptictrack.haptics.HapticFeedbackManager
import com.haptictrack.tracking.ObjectTracker
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

    init {
        orientationListener.start()
        objectTracker.deviceRotationProvider = { orientationListener.deviceRotation }

        cameraManager.imageAnalysis.setAnalyzer(
            java.util.concurrent.Executors.newSingleThreadExecutor(),
            objectTracker.analyzer
        )

        objectTracker.onDetectionResult = { allObjects, lockedObject, imgWidth, imgHeight ->
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

            _uiState.update { current ->
                current.copy(
                    status = status,
                    trackedObject = lockedObject ?: if (status == TrackingStatus.LOST) current.trackedObject else null,
                    detectedObjects = allObjects,
                    sourceImageWidth = imgWidth,
                    sourceImageHeight = imgHeight,
                    currentZoomRatio = targetZoom ?: current.currentZoomRatio
                )
            }
        }
    }

    fun startCamera(lifecycleOwner: LifecycleOwner, previewView: PreviewView) {
        cameraManager.startCamera(lifecycleOwner, previewView)
    }

    fun onTapToLock(normalizedX: Float, normalizedY: Float) {
        val objects = _uiState.value.detectedObjects

        // When multiple boxes overlap at the tap point, pick the smallest (most specific).
        // This prevents tapping a cup and accidentally locking onto the table underneath.
        val tapped = objects
            .filter { it.id >= 0 && it.boundingBox.contains(normalizedX, normalizedY) }
            .minByOrNull { it.boundingBox.width() * it.boundingBox.height() }

        if (tapped != null) {
            objectTracker.lockOnObject(tapped.id, tapped.boundingBox, tapped.label)
            _uiState.update { it.copy(status = TrackingStatus.LOCKED, trackedObject = tapped) }
        }
    }

    fun clearTracking() {
        objectTracker.clearLock()
        zoomController.reset()
        hapticManager.updateTrackingStatus(TrackingStatus.IDLE)
        _uiState.update {
            TrackingUiState(status = TrackingStatus.IDLE, isRecording = it.isRecording)
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