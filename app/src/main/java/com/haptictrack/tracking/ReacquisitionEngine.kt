package com.haptictrack.tracking

import android.graphics.RectF

/**
 * Pure logic for re-acquiring a lost tracking target.
 * No ML Kit or Android dependencies — fully unit-testable.
 *
 * When the locked object's tracking ID disappears (object left frame or ML Kit
 * reassigned IDs), this engine scores visible candidates against the last-known
 * appearance to find the most likely match.
 */
class ReacquisitionEngine(
    val maxFramesLost: Int = 60,
    val positionThreshold: Float = 0.25f,
    val sizeRatioThreshold: Float = 1.8f,
    val labelWeight: Float = 0.3f,
    val sizeWeight: Float = 0.25f,
    val minScoreThreshold: Float = 0.7f
) {

    var lockedId: Int? = null
        private set
    var lastKnownBox: RectF? = null
        private set
    var lastKnownLabel: String? = null
        private set
    var lastKnownSize: Float = 0f
        private set
    var framesLost: Int = 0
        private set

    val isLocked: Boolean get() = lockedId != null
    val isSearching: Boolean get() = lockedId != null && framesLost > 0 && framesLost <= maxFramesLost
    val hasTimedOut: Boolean get() = framesLost > maxFramesLost

    fun lock(trackingId: Int, boundingBox: RectF, label: String?) {
        lockedId = trackingId
        lastKnownBox = RectF(boundingBox)
        lastKnownLabel = label
        lastKnownSize = boundingBox.width() * boundingBox.height()
        framesLost = 0
    }

    fun clear() {
        lockedId = null
        lastKnownBox = null
        lastKnownLabel = null
        lastKnownSize = 0f
        framesLost = 0
    }

    /**
     * Process a new frame of detections. Returns the matched locked object, or null.
     *
     * If the original tracking ID is still present, returns it directly.
     * If not, attempts re-acquisition from candidates.
     */
    fun processFrame(detections: List<TrackedObject>): TrackedObject? {
        val lockId = lockedId ?: return null

        // Direct match by tracking ID
        val directMatch = detections.find { it.id == lockId }
        if (directMatch != null) {
            updateFromMatch(directMatch)
            return directMatch
        }

        // Object lost — attempt re-acquisition
        framesLost++
        if (framesLost > maxFramesLost) return null

        val reacquired = findBestCandidate(detections)
        if (reacquired != null) {
            lockedId = reacquired.id
            updateFromMatch(reacquired)
            return reacquired
        }

        return null
    }

    private fun updateFromMatch(obj: TrackedObject) {
        lastKnownBox = RectF(obj.boundingBox)
        if (obj.label != null) lastKnownLabel = obj.label
        lastKnownSize = obj.boundingBox.width() * obj.boundingBox.height()
        framesLost = 0
    }

    /**
     * Find the best re-acquisition candidate. Returns null if no candidate is good enough.
     */
    internal fun findBestCandidate(candidates: List<TrackedObject>): TrackedObject? {
        val refBox = lastKnownBox ?: return null
        if (candidates.isEmpty()) return null

        return candidates
            .filter { it.id >= 0 }
            .mapNotNull { candidate ->
                val score = scoreCandidate(candidate, refBox)
                if (score != null) Pair(candidate, score) else null
            }
            .maxByOrNull { it.second } // Higher score = better match
            ?.takeIf { it.second >= minScoreThreshold }
            ?.first
    }

    /**
     * Score a candidate for re-acquisition. Higher is better. Returns null if hard thresholds fail.
     *
     * Score components (each 0..1, weighted):
     * - Position similarity: inverse of distance between centers
     * - Size similarity: inverse of size ratio difference
     * - Label match: binary bonus
     */
    internal fun scoreCandidate(candidate: TrackedObject, refBox: RectF): Float? {
        val candBox = candidate.boundingBox

        // Hard threshold: distance between centers
        val dx = candBox.centerX() - refBox.centerX()
        val dy = candBox.centerY() - refBox.centerY()
        val distance = kotlin.math.sqrt(dx * dx + dy * dy)
        if (distance > positionThreshold) return null

        // Hard threshold: size ratio
        val candSize = candBox.width() * candBox.height()
        val sizeRatio = if (lastKnownSize > 0f && candSize > 0f) {
            if (candSize > lastKnownSize) candSize / lastKnownSize else lastKnownSize / candSize
        } else 1f
        if (sizeRatio > sizeRatioThreshold) return null

        // Soft scores
        val positionScore = 1f - (distance / positionThreshold)
        val sizeScore = 1f - ((sizeRatio - 1f) / (sizeRatioThreshold - 1f))
        val labelScore = if (lastKnownLabel != null && candidate.label == lastKnownLabel) 1f else 0f

        // Weighted combination — position is king, size and label are supporting
        val positionWeight = 1f - labelWeight - sizeWeight
        return (positionScore * positionWeight) + (sizeScore * sizeWeight) + (labelScore * labelWeight)
    }
}
