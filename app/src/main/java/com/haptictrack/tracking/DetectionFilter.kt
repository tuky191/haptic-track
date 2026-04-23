package com.haptictrack.tracking

/**
 * Filters raw ML Kit detections to remove noise.
 * Pure logic, no Android dependencies — fully unit-testable.
 */
class DetectionFilter(
    /** Minimum confidence to show a detection. */
    val minConfidence: Float = 0.5f,
    /** Minimum bounding box area (normalized, 0..1). Rejects tiny slivers. */
    val minBoxArea: Float = 0.005f,
    /** Maximum bounding box area (normalized, 0..1). Rejects "whole frame" detections. */
    val maxBoxArea: Float = 0.5f,
    /** Minimum aspect ratio (width/height). Rejects extremely thin boxes. */
    val minAspectRatio: Float = 0.2f,
    /** Maximum aspect ratio (width/height). Rejects extremely wide boxes. */
    val maxAspectRatio: Float = 5.0f
) {

    /**
     * Filter a list of raw detections.
     * Returns only objects worth showing to the user.
     */
    fun filter(detections: List<TrackedObject>): List<TrackedObject> {
        return detections.filter { isValid(it) }
    }

    /**
     * Check if a single detection passes all quality filters.
     */
    fun isValid(obj: TrackedObject): Boolean {
        // Must have a valid tracking ID
        if (obj.id < 0) return false

        // Must be classified — unclassified objects are noise
        if (obj.label == null) return false

        // Must meet minimum confidence
        if (obj.confidence < minConfidence) return false

        val box = obj.boundingBox
        val width = box.width()
        val height = box.height()

        // Reject empty/invalid boxes
        if (width <= 0f || height <= 0f) return false

        // Reject too small or too large
        val area = width * height
        if (area < minBoxArea) return false
        if (area > maxBoxArea) return false

        // Reject extreme aspect ratios
        val aspectRatio = width / height
        if (aspectRatio < minAspectRatio) return false
        if (aspectRatio > maxAspectRatio) return false

        return true
    }
}