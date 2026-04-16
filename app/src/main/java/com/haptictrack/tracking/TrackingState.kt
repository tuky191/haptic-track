package com.haptictrack.tracking

import android.graphics.PointF
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
    val confidence: Float = 0f,
    val embedding: FloatArray? = null,
    val colorHistogram: FloatArray? = null
) {
    // INVARIANT: embedding is excluded from equals/hashCode. Two TrackedObjects
    // differing only in embedding are considered equal. This is intentional —
    // embedding is transient ML output, not part of the object's identity for
    // UI diffing and collection operations.
    override fun equals(other: Any?): Boolean {
        if (this === other) return true
        if (other !is TrackedObject) return false
        return id == other.id && boundingBox == other.boundingBox &&
               label == other.label && confidence == other.confidence
    }

    override fun hashCode(): Int {
        var result = id
        result = 31 * result + boundingBox.hashCode()
        result = 31 * result + (label?.hashCode() ?: 0)
        result = 31 * result + confidence.hashCode()
        return result
    }
}

data class TrackingUiState(
    val status: TrackingStatus = TrackingStatus.IDLE,
    val trackedObject: TrackedObject? = null,
    val isRecording: Boolean = false,
    val currentZoomRatio: Float = 1f,
    val detectedObjects: List<TrackedObject> = emptyList(),
    /** Source image width (post-rotation, i.e. portrait width). */
    val sourceImageWidth: Int = 0,
    /** Source image height (post-rotation, i.e. portrait height). */
    val sourceImageHeight: Int = 0,
    /** Contour points of the locked object in normalized [0,1] coordinates. */
    val lockedContour: List<PointF> = emptyList()
)
