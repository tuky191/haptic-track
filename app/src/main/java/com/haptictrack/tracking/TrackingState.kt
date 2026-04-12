package com.haptictrack.tracking

import android.graphics.RectF

enum class TrackingStatus {
    IDLE,
    SEARCHING,
    LOCKED,
    LOST
}

data class TrackedObject(
    val id: Int,
    val boundingBox: RectF,
    val label: String? = null,
    val confidence: Float = 0f
)

data class TrackingUiState(
    val status: TrackingStatus = TrackingStatus.IDLE,
    val trackedObject: TrackedObject? = null,
    val isRecording: Boolean = false,
    val currentZoomRatio: Float = 1f,
    val detectedObjects: List<TrackedObject> = emptyList(),
    /** Source image width (post-rotation, i.e. portrait width). */
    val sourceImageWidth: Int = 0,
    /** Source image height (post-rotation, i.e. portrait height). */
    val sourceImageHeight: Int = 0
)
