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

/**
 * Person attributes from Crossroad-0230 classifier.
 * All boolean fields use threshold > 0.5 on model output probabilities.
 */
data class PersonAttributes(
    val isMale: Boolean,
    val hasBag: Boolean,
    val hasBackpack: Boolean,
    val hasHat: Boolean,
    val hasLongSleeves: Boolean,
    val hasLongPants: Boolean,
    val hasLongHair: Boolean,
    val hasCoatJacket: Boolean,
    /** Dominant upper-body clothing color name (e.g. "red", "blue", "black"). */
    val upperColor: String? = null,
    /** Dominant lower-body clothing color name. */
    val lowerColor: String? = null,
    /** Raw model probabilities for soft scoring (8 values, same order as boolean fields). */
    val rawProbabilities: FloatArray? = null
) {
    /** Human-readable summary, e.g. "man, long sleeves, backpack, hat, red/blue" */
    fun summary(): String {
        val parts = mutableListOf<String>()
        parts.add(if (isMale) "man" else "woman")
        if (hasCoatJacket) parts.add("jacket")
        if (hasLongSleeves) parts.add("long sleeves") else parts.add("short sleeves")
        if (hasLongPants) parts.add("pants") else parts.add("shorts/skirt")
        if (hasBackpack) parts.add("backpack")
        if (hasBag) parts.add("bag")
        if (hasHat) parts.add("hat")
        if (hasLongHair) parts.add("long hair")
        val colors = listOfNotNull(upperColor, lowerColor)
        if (colors.isNotEmpty()) parts.add(colors.joinToString("/"))
        return parts.joinToString(", ")
    }

    /** Similarity score between two attribute sets (0-1). */
    fun similarity(other: PersonAttributes): Float {
        if (rawProbabilities != null && other.rawProbabilities != null) {
            // Soft comparison: 1 - mean absolute difference of probabilities
            var diff = 0f
            for (i in rawProbabilities.indices) {
                diff += kotlin.math.abs(rawProbabilities[i] - other.rawProbabilities[i])
            }
            val attrSim = 1f - (diff / rawProbabilities.size)
            // Color match bonus
            val colorSim = when {
                upperColor != null && other.upperColor != null &&
                lowerColor != null && other.lowerColor != null ->
                    (if (upperColor == other.upperColor) 0.5f else 0f) +
                    (if (lowerColor == other.lowerColor) 0.5f else 0f)
                upperColor != null && other.upperColor != null ->
                    if (upperColor == other.upperColor) 1f else 0f
                else -> 0.5f // unknown, neutral
            }
            return attrSim * 0.6f + colorSim * 0.4f
        }
        // Hard comparison fallback: count matching booleans
        var matches = 0
        if (isMale == other.isMale) matches++
        if (hasBag == other.hasBag) matches++
        if (hasBackpack == other.hasBackpack) matches++
        if (hasHat == other.hasHat) matches++
        if (hasLongSleeves == other.hasLongSleeves) matches++
        if (hasLongPants == other.hasLongPants) matches++
        if (hasLongHair == other.hasLongHair) matches++
        if (hasCoatJacket == other.hasCoatJacket) matches++
        return matches / 8f
    }

    override fun equals(other: Any?): Boolean {
        if (this === other) return true
        if (other !is PersonAttributes) return false
        return isMale == other.isMale && hasBag == other.hasBag &&
               hasBackpack == other.hasBackpack && hasHat == other.hasHat &&
               hasLongSleeves == other.hasLongSleeves && hasLongPants == other.hasLongPants &&
               hasLongHair == other.hasLongHair && hasCoatJacket == other.hasCoatJacket &&
               upperColor == other.upperColor && lowerColor == other.lowerColor
    }

    override fun hashCode(): Int {
        var result = isMale.hashCode()
        result = 31 * result + hasBag.hashCode()
        result = 31 * result + hasBackpack.hashCode()
        result = 31 * result + hasHat.hashCode()
        result = 31 * result + hasLongSleeves.hashCode()
        result = 31 * result + hasLongPants.hashCode()
        result = 31 * result + hasLongHair.hashCode()
        result = 31 * result + hasCoatJacket.hashCode()
        result = 31 * result + (upperColor?.hashCode() ?: 0)
        result = 31 * result + (lowerColor?.hashCode() ?: 0)
        return result
    }
}

data class TrackedObject(
    val id: Int,
    val boundingBox: RectF,
    val label: String? = null,
    val confidence: Float = 0f,
    val embedding: FloatArray? = null,
    val colorHistogram: FloatArray? = null,
    val personAttributes: PersonAttributes? = null,
    /** OSNet person re-ID embedding (512-dim). Only computed for person candidates. */
    val reIdEmbedding: FloatArray? = null,
    /** MobileFaceNet face embedding (192-dim). Only computed when face is visible. */
    val faceEmbedding: FloatArray? = null
) {
    // INVARIANT: embedding, colorHistogram, personAttributes, reIdEmbedding, and
    // faceEmbedding are excluded from equals/hashCode. These are transient ML output,
    // not part of the object's identity for UI diffing and collection operations.
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
