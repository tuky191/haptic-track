package com.haptictrack.tracking

import android.graphics.PointF
import android.graphics.RectF

enum class TrackingStatus {
    IDLE,
    SEARCHING,
    LOCKED,
    LOST
}

enum class CaptureMode {
    VIDEO,
    PHOTO
}

data class TrackedObject(
    val id: Int,
    val boundingBox: RectF,
    val label: String? = null,
    val confidence: Float = 0f,
    val embedding: FloatArray? = null,
    val colorHistogram: FloatArray? = null,
    /** OSNet person re-ID embedding (512-dim). Only computed for person candidates. */
    val reIdEmbedding: FloatArray? = null,
    /** EdgeFace-XS face embedding (512-dim). Only computed when face is visible. */
    val faceEmbedding: FloatArray? = null
) {
    // INVARIANT: embedding, colorHistogram, reIdEmbedding, and faceEmbedding are
    // excluded from equals/hashCode. These are transient ML output, not part of the
    // object's identity for UI diffing and collection operations.
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
    val lockedContour: List<PointF> = emptyList(),
    val captureMode: CaptureMode = CaptureMode.VIDEO,
    /** True when zoom indicator should be visible (during/after pinch). */
    val showZoomIndicator: Boolean = false,
    /** Stealth mode: preview hidden, screen stays black. */
    val stealthMode: Boolean = false,
    /** True once all ML models are loaded and ready. */
    val isReady: Boolean = false,
    /** Loading status messages shown during model init. */
    val loadingStatus: String = "Initializing..."
)
