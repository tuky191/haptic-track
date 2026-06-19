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

enum class TrackingFilter {
    ALL,
    PERSON_ONLY,
    PETS,
    NON_PERSON_ONLY
}

/** Sentry auto-lock gender criterion. */
enum class GenderFilter { ANY, MALE, FEMALE }

/**
 * Criteria the sentry auto-lock matches a person against. [ageMin]/[ageMax] are
 * inclusive years; default span accepts any age. A person matches when its
 * estimated gender and age both fall within the criteria.
 */
data class SentryCriteria(
    val gender: GenderFilter = GenderFilter.ANY,
    val ageMin: Int = 0,
    val ageMax: Int = 120,
) {
    fun matches(attr: FaceAttributes): Boolean {
        val genderOk = when (gender) {
            GenderFilter.ANY -> true
            GenderFilter.MALE -> attr.isMale
            GenderFilter.FEMALE -> !attr.isMale
        }
        return genderOk && attr.age in ageMin..ageMax
    }
}

/** Sentry state for UI display. */
enum class SentryPhase { OFF, SCANNING, INSPECTING, MATCHED }

private val ANIMAL_LABELS = setOf(
    "cat", "dog", "bird", "horse", "sheep", "cow",
    "elephant", "bear", "zebra", "giraffe"
)

fun labelMatchesFilter(label: String?, filter: TrackingFilter): Boolean = when (filter) {
    TrackingFilter.ALL -> true
    TrackingFilter.PERSON_ONLY -> label == "person"
    TrackingFilter.PETS -> label in ANIMAL_LABELS
    TrackingFilter.NON_PERSON_ONLY -> label != "person"
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
    /** MobileFaceNet face embedding (192-dim). Only computed when face is visible. */
    val faceEmbedding: FloatArray? = null,
    /** Gender/age estimate. Only computed when a face is visible and the sentry/debug attribute pass runs. */
    val faceAttributes: FaceAttributes? = null
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
    val loadingStatus: String = "Initializing...",
    /** ISP-level (vendor VDIS) stabilization toggle — on by default. */
    val ispStabilization: Boolean = true,
    /** Software gyro-based EIS toggle. */
    val gyroEis: Boolean = true,
    /** Gyro EIS strength 0.0–1.0 (0 = light, 1 = aggressive). */
    val gyroStrength: Float = 0.5f,
    /** Adaptive pan detection for gyro EIS. */
    val adaptiveEis: Boolean = true,
    /** Leash: limits smoothed-to-raw deviation. */
    val leashEnabled: Boolean = true,
    /** OIS compensation active (scale correction to avoid overcorrecting). */
    val oisCompensation: Boolean = true,
    /** Optical-flow translation correction on top of gyro rotation EIS. */
    val translationEis: Boolean = false,
    val horizonLock: Boolean = true,
    val fhd60Vdis: Boolean = false,
    /** Which object categories to show and allow tracking. */
    val trackingFilter: TrackingFilter = TrackingFilter.ALL,
    /** Haptic vibration strength 0.0–1.0. */
    val hapticStrength: Float = 0.5f,
    /** Sentry auto-lock: actively scan + zoom-inspect for a person matching criteria. */
    val sentryEnabled: Boolean = false,
    /** Default search: women, teen/adult (15-45). */
    val sentryCriteria: SentryCriteria = SentryCriteria(GenderFilter.FEMALE, 15, 45),
    /** Live sentry phase for UI display. */
    val sentryPhase: SentryPhase = SentryPhase.OFF
)
