package com.haptictrack.tracking

import android.graphics.RectF
import android.util.Log

/**
 * Pure logic for re-acquiring a lost tracking target.
 *
 * When the locked object's tracking ID disappears, this engine scores visible
 * candidates against the last-known appearance to find the most likely match.
 *
 * Key design: position weight decays over time because a handheld camera moves.
 * After many lost frames, label + size dominate the score since the object will
 * reappear at a completely different screen position.
 *
 * All decisions are logged to Android logcat under tag "Reacq" for debugging.
 */
class ReacquisitionEngine(
    val maxFramesLost: Int = 450,
    val initialPositionThreshold: Float = 0.25f,
    val maxPositionThreshold: Float = 1.5f,
    val sizeRatioThreshold: Float = 2.0f,
    val minScoreThreshold: Float = 0.35f,
    val positionDecayFrames: Int = 30
) {

    companion object {
        private const val TAG = "Reacq"
        /** Embedding similarity above this bypasses position/size hard filters. */
        const val APPEARANCE_OVERRIDE_THRESHOLD = 0.5f
        /** Maximum embeddings to keep in gallery. */
        const val MAX_GALLERY_SIZE = 12
    }

    var lockedId: Int? = null
        private set
    var lockedLabel: String? = null
        private set
    /** Gallery of reference embeddings — augmented at lock time, accumulated during tracking. */
    var embeddingGallery: MutableList<FloatArray> = mutableListOf()
        private set

    /** Convenience: true if we have any reference embeddings. */
    val hasEmbeddings: Boolean get() = embeddingGallery.isNotEmpty()

    @Deprecated("Use embeddingGallery", ReplaceWith("embeddingGallery.firstOrNull()"))
    val lockedEmbedding: FloatArray? get() = embeddingGallery.firstOrNull()
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

    fun lock(trackingId: Int, boundingBox: RectF, label: String?, embedding: FloatArray? = null) {
        lock(trackingId, boundingBox, label, if (embedding != null) listOf(embedding) else emptyList())
    }

    fun lock(trackingId: Int, boundingBox: RectF, label: String?, embeddings: List<FloatArray>) {
        lockedId = trackingId
        lockedLabel = label
        embeddingGallery = embeddings.map { it.copyOf() }.toMutableList()
        lastKnownBox = RectF(boundingBox)
        lastKnownLabel = label
        lastKnownSize = boundingBox.width() * boundingBox.height()
        framesLost = 0
        Log.i(TAG, "LOCK id=$trackingId label=\"$label\" box=${fmtBox(boundingBox)} size=${fmtF(lastKnownSize)} gallery=${embeddingGallery.size}")
    }

    /** Add a new embedding to the gallery (e.g. from a confirmed visual tracker frame). */
    fun addEmbedding(embedding: FloatArray) {
        if (embeddingGallery.size >= MAX_GALLERY_SIZE) {
            // Keep first (lock-time augmented) and remove oldest accumulated
            val augmentedCount = 5  // original + 3 rotations + 1 flip
            if (embeddingGallery.size > augmentedCount) {
                embeddingGallery.removeAt(augmentedCount)
            } else {
                embeddingGallery.removeAt(embeddingGallery.size - 1)
            }
        }
        embeddingGallery.add(embedding.copyOf())
    }

    fun clear() {
        Log.i(TAG, "CLEAR (was id=$lockedId label=\"$lockedLabel\")")
        lockedId = null
        lockedLabel = null
        embeddingGallery.clear()
        lastKnownBox = null
        lastKnownLabel = null
        lastKnownSize = 0f
        framesLost = 0
    }

    fun processFrame(detections: List<TrackedObject>): TrackedObject? {
        val lockId = lockedId ?: return null

        // Direct match by tracking ID
        val directMatch = detections.find { it.id == lockId }
        if (directMatch != null) {
            if (framesLost > 0) {
                Log.d(TAG, "DIRECT_MATCH id=$lockId recovered after $framesLost lost frames, label=\"${directMatch.label}\"")
            }
            updateFromMatch(directMatch)
            return directMatch
        }

        // Object lost
        framesLost++
        if (framesLost == 1) {
            Log.w(TAG, "LOST id=$lockId (lockedLabel=\"$lockedLabel\") — starting search. ${detections.size} candidates in frame")
        }
        if (framesLost > maxFramesLost) {
            if (framesLost == maxFramesLost + 1) {
                Log.w(TAG, "TIMEOUT after $maxFramesLost frames. Giving up on lockedLabel=\"$lockedLabel\"")
            }
            return null
        }

        // Log candidates periodically (every 10 frames to avoid spam)
        if (framesLost % 10 == 1) {
            Log.d(TAG, "SEARCH frame=$framesLost posConf=${fmtF(positionConfidence())} posThresh=${fmtF(effectivePositionThreshold())} gallery=${embeddingGallery.size} candidates=${detections.size}")
            detections.forEach { d ->
                val simStr = if (hasEmbeddings && d.embedding != null) {
                    " sim=${fmtF(bestGallerySimilarity(d.embedding!!))}"
                } else ""
                Log.d(TAG, "  candidate id=${d.id} label=\"${d.label}\" conf=${fmtF(d.confidence)}$simStr box=${fmtBox(d.boundingBox)}")
            }
        }

        val reacquired = findBestCandidate(detections)
        if (reacquired != null) {
            Log.i(TAG, "REACQUIRE id=${reacquired.id} label=\"${reacquired.label}\" after $framesLost frames (lockedLabel=\"$lockedLabel\") box=${fmtBox(reacquired.boundingBox)}")
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

    /** Called by VisualTracker to keep last-known box in sync without full processFrame. */
    fun updateFromVisualTracker(boundingBox: RectF) {
        lastKnownBox = RectF(boundingBox)
        lastKnownSize = boundingBox.width() * boundingBox.height()
        framesLost = 0
    }

    internal fun positionConfidence(): Float {
        if (framesLost <= 0) return 1f
        return (1f - framesLost.toFloat() / positionDecayFrames).coerceIn(0f, 1f)
    }

    internal fun effectivePositionThreshold(): Float {
        val t = (framesLost.toFloat() / positionDecayFrames).coerceIn(0f, 1f)
        return initialPositionThreshold + t * (maxPositionThreshold - initialPositionThreshold)
    }

    internal fun findBestCandidate(candidates: List<TrackedObject>): TrackedObject? {
        val refBox = lastKnownBox ?: return null
        if (candidates.isEmpty()) return null

        val posConf = positionConfidence()
        val posThreshold = effectivePositionThreshold()

        // No hard label filter — label is a scoring factor, not a gate.
        // EfficientDet-Lite0 labels flicker (bowl↔potted plant↔toilet) so
        // blocking by label causes more harm than good. The embedding handles
        // identity; the label just adds a bonus to the score.
        val validCandidates = candidates.filter { it.id >= 0 }

        val logThis = framesLost % 10 == 1 || validCandidates.isNotEmpty()

        val scored = validCandidates.mapNotNull { candidate ->
            val score = scoreCandidate(candidate, refBox, posConf, posThreshold)
            val sim = if (hasEmbeddings && candidate.embedding != null) {
                bestGallerySimilarity(candidate.embedding!!)
            } else null
            if (score != null) {
                if (logThis) {
                    Log.d(TAG, "  scored id=${candidate.id} label=\"${candidate.label}\" score=${fmtF(score)} sim=${sim?.let { fmtF(it) } ?: "n/a"} (min=${fmtF(minScoreThreshold)})")
                }
                Pair(candidate, score)
            } else {
                if (logThis) {
                    Log.d(TAG, "  rejected id=${candidate.id} label=\"${candidate.label}\" sim=${sim?.let { fmtF(it) } ?: "n/a"} (hard threshold)")
                }
                null
            }
        }

        return scored
            .maxByOrNull { it.second }
            ?.takeIf { it.second >= minScoreThreshold }
            ?.first
    }

    /**
     * Effective size ratio threshold expands as frames are lost (camera moving closer/farther).
     */
    internal fun effectiveSizeRatioThreshold(): Float {
        val t = (framesLost.toFloat() / positionDecayFrames).coerceIn(0f, 1f)
        return sizeRatioThreshold + t * (sizeRatioThreshold * 2f)
    }

    internal fun scoreCandidate(
        candidate: TrackedObject,
        refBox: RectF,
        positionConfidence: Float = positionConfidence(),
        posThreshold: Float = effectivePositionThreshold()
    ): Float? {
        val candBox = candidate.boundingBox

        // Compute appearance similarity early — a strong visual match can
        // override geometric hard filters (position, size). This handles
        // scenarios like phone rotation where the last-known box is meaningless.
        val hasAppearance = hasEmbeddings && candidate.embedding != null
        val appearanceScore = if (hasAppearance) {
            bestGallerySimilarity(candidate.embedding!!)
                .coerceIn(0f, 1f)
        } else 0f
        val strongVisualMatch = hasAppearance && appearanceScore > APPEARANCE_OVERRIDE_THRESHOLD

        val dx = candBox.centerX() - refBox.centerX()
        val dy = candBox.centerY() - refBox.centerY()
        val distance = kotlin.math.sqrt(dx * dx + dy * dy)

        if (distance > posThreshold && !strongVisualMatch) return null
        if (distance > posThreshold && strongVisualMatch) {
            Log.d(TAG, "  OVERRIDE position: dist=${fmtF(distance)} > thresh=${fmtF(posThreshold)}, but sim=${fmtF(appearanceScore)}")
        }

        val candSize = candBox.width() * candBox.height()
        val sizeRatio = if (lastKnownSize > 0f && candSize > 0f) {
            if (candSize > lastKnownSize) candSize / lastKnownSize else lastKnownSize / candSize
        } else 1f
        val effectiveSizeThreshold = effectiveSizeRatioThreshold()
        if (sizeRatio > effectiveSizeThreshold && !strongVisualMatch) return null
        if (sizeRatio > effectiveSizeThreshold && strongVisualMatch) {
            Log.d(TAG, "  OVERRIDE size: ratio=${fmtF(sizeRatio)} > thresh=${fmtF(effectiveSizeThreshold)}, but sim=${fmtF(appearanceScore)}")
        }

        val positionScore = if (posThreshold > 0f) (1f - (distance / posThreshold)).coerceIn(0f, 1f) else 1f
        val sizeScore = (1f - ((sizeRatio - 1f) / (effectiveSizeThreshold - 1f).coerceAtLeast(0.01f))).coerceIn(0f, 1f)
        val labelScore = if (lastKnownLabel != null && candidate.label == lastKnownLabel) 1f else 0f

        // Weight distribution depends on whether we have appearance data.
        // With appearance: it gets the dominant share (replaces most of label weight
        // since visual identity is strictly more informative than category label).
        if (hasAppearance) {
            val basePositionWeight = 0.20f
            val baseSizeWeight = 0.15f
            val baseLabelWeight = 0.20f
            val baseAppearanceWeight = 0.45f

            val effectivePositionWeight = basePositionWeight * positionConfidence
            val redistributed = basePositionWeight * (1f - positionConfidence)
            val effectiveSizeWeight = baseSizeWeight + redistributed * 0.15f
            val effectiveLabelWeight = baseLabelWeight + redistributed * 0.15f
            val effectiveAppearanceWeight = baseAppearanceWeight + redistributed * 0.7f

            return (positionScore * effectivePositionWeight) +
                   (sizeScore * effectiveSizeWeight) +
                   (labelScore * effectiveLabelWeight) +
                   (appearanceScore * effectiveAppearanceWeight)
        } else {
            // Fallback: no embedding available, use original weights
            val basePositionWeight = 0.45f
            val baseSizeWeight = 0.25f
            val baseLabelWeight = 0.30f

            val effectivePositionWeight = basePositionWeight * positionConfidence
            val redistributed = basePositionWeight * (1f - positionConfidence)
            val effectiveSizeWeight = baseSizeWeight + redistributed * 0.4f
            val effectiveLabelWeight = baseLabelWeight + redistributed * 0.6f

            return (positionScore * effectivePositionWeight) +
                   (sizeScore * effectiveSizeWeight) +
                   (labelScore * effectiveLabelWeight)
        }
    }

    /** Best cosine similarity between a candidate and any embedding in the gallery. */
    internal fun bestGallerySimilarity(candidateEmbedding: FloatArray): Float {
        if (embeddingGallery.isEmpty()) return 0f
        return embeddingGallery.maxOf { cosineSimilarity(it, candidateEmbedding) }
    }

    private fun cosineSimilarity(a: FloatArray, b: FloatArray): Float {
        if (a.size != b.size) return 0f
        var dot = 0f
        for (i in a.indices) dot += a[i] * b[i]
        return dot
    }

    private fun fmtF(f: Float) = "%.3f".format(f)
    private fun fmtBox(b: RectF) = "[%.2f,%.2f,%.2f,%.2f]".format(b.left, b.top, b.right, b.bottom)
}
