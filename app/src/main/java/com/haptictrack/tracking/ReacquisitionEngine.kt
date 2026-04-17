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
    val minScoreThreshold: Float = 0.45f,
    val positionDecayFrames: Int = 30,
    /** Optional session logger — writes to both logcat and session log file. */
    var sessionLogger: ((String) -> Unit)? = null
) {

    companion object {
        private const val TAG = "Reacq"
        /** Embedding similarity above this bypasses position/size hard filters. */
        const val APPEARANCE_OVERRIDE_THRESHOLD = 0.7f
        /** Maximum embeddings to keep in gallery. */
        const val MAX_GALLERY_SIZE = 12
        /** Max times we re-acquire different objects before giving up. */
        const val MAX_REACQUISITION_HOPS = 3
    }

    var lockedId: Int? = null
        private set
    var lockedLabel: String? = null
        private set
    /** Original COCO label before enrichment — used for label matching during search. */
    var lockedCocoLabel: String? = null
        private set
    /** Gallery of reference embeddings — augmented at lock time, accumulated during tracking. */
    private var _embeddingGallery: MutableList<FloatArray> = mutableListOf()
    val embeddingGallery: List<FloatArray> get() = _embeddingGallery

    /** Convenience: true if we have any reference embeddings. */
    val hasEmbeddings: Boolean get() = _embeddingGallery.isNotEmpty()
    /** Reference color histogram from lock time. */
    var lockedColorHistogram: FloatArray? = null
        private set
    /** Reference person attributes from lock time. */
    var lockedPersonAttributes: PersonAttributes? = null
        private set
    var lastKnownBox: RectF? = null
        private set
    var lastKnownLabel: String? = null
        private set
    var lastKnownSize: Float = 0f
        private set
    var framesLost: Int = 0
        private set
    /** Number of times we've re-acquired a different object since the original lock. */
    var reacquisitionHops: Int = 0
        private set

    val isLocked: Boolean get() = lockedId != null
    val isSearching: Boolean get() = lockedId != null && framesLost > 0 && framesLost <= maxFramesLost
    val hasTimedOut: Boolean get() = framesLost > maxFramesLost

    fun lock(trackingId: Int, boundingBox: RectF, label: String?, embedding: FloatArray? = null) {
        lock(trackingId, boundingBox, label, if (embedding != null) listOf(embedding) else emptyList())
    }

    fun lock(trackingId: Int, boundingBox: RectF, label: String?, embeddings: List<FloatArray>,
             colorHist: FloatArray? = null, personAttrs: PersonAttributes? = null,
             cocoLabel: String? = null) {
        lockedId = trackingId
        lockedLabel = label
        lockedCocoLabel = cocoLabel
        _embeddingGallery = embeddings.map { it.copyOf() }.toMutableList()
        lockedColorHistogram = colorHist?.copyOf()
        lockedPersonAttributes = personAttrs
        lastKnownBox = RectF(boundingBox)
        lastKnownLabel = label
        lastKnownSize = boundingBox.width() * boundingBox.height()
        framesLost = 0
        reacquisitionHops = 0
        val attrStr = personAttrs?.summary() ?: "n/a"
        dualLog(Log.INFO, "LOCK id=$trackingId label=\"$label\" box=${fmtBox(boundingBox)} size=${fmtF(lastKnownSize)} gallery=${embeddingGallery.size} colorHist=${colorHist != null} attrs=\"$attrStr\"")
    }

    /** Add a new embedding to the gallery (e.g. from a confirmed visual tracker frame). */
    fun addEmbedding(embedding: FloatArray) {
        if (_embeddingGallery.size >= MAX_GALLERY_SIZE) {
            // Keep first (lock-time augmented) and remove oldest accumulated
            if (_embeddingGallery.size > LOCK_AUGMENTATION_COUNT) {
                _embeddingGallery.removeAt(LOCK_AUGMENTATION_COUNT)
            } else {
                _embeddingGallery.removeAt(_embeddingGallery.size - 1)
            }
        }
        _embeddingGallery.add(embedding.copyOf())
    }

    fun clear() {
        dualLog(Log.INFO, "CLEAR (was id=$lockedId label=\"$lockedLabel\" coco=\"$lockedCocoLabel\")")
        lockedId = null
        lockedLabel = null
        lockedCocoLabel = null
        _embeddingGallery.clear()
        lockedColorHistogram = null
        lockedPersonAttributes = null
        lastKnownBox = null
        lastKnownLabel = null
        lastKnownSize = 0f
        framesLost = 0
        reacquisitionHops = 0
    }

    fun processFrame(detections: List<TrackedObject>): TrackedObject? {
        val lockId = lockedId ?: return null

        // Direct match by tracking ID
        val directMatch = detections.find { it.id == lockId }
        if (directMatch != null) {
            if (framesLost > 0) {
                dualLog(Log.DEBUG, "DIRECT_MATCH id=$lockId recovered after $framesLost lost frames, label=\"${directMatch.label}\"")
            }
            updateFromMatch(directMatch)
            return directMatch
        }

        // Object lost
        framesLost++
        if (framesLost == 1) {
            dualLog(Log.WARN, "LOST id=$lockId (lockedLabel=\"$lockedLabel\") — starting search. ${detections.size} candidates in frame")
        }
        if (framesLost > maxFramesLost || reacquisitionHops >= MAX_REACQUISITION_HOPS) {
            if (framesLost == maxFramesLost + 1) {
                dualLog(Log.WARN, "TIMEOUT after $maxFramesLost frames. Giving up on lockedLabel=\"$lockedLabel\"")
            }
            if (reacquisitionHops >= MAX_REACQUISITION_HOPS && framesLost == 1) {
                dualLog(Log.WARN, "GIVE_UP after $reacquisitionHops re-acquisition hops — original object is gone")
                framesLost = maxFramesLost + 1  // force timeout state
            }
            return null
        }

        // Log candidates periodically (every 10 frames to avoid spam)
        if (framesLost % 10 == 1) {
            dualLog(Log.DEBUG, "SEARCH frame=$framesLost posConf=${fmtF(positionConfidence())} posThresh=${fmtF(effectivePositionThreshold())} gallery=${embeddingGallery.size} candidates=${detections.size}")
            detections.forEach { d ->
                val simStr = if (hasEmbeddings && d.embedding != null) {
                    " sim=${fmtF(bestGallerySimilarity(d.embedding!!))}"
                } else ""
                dualLog(Log.DEBUG, "  candidate id=${d.id} label=\"${d.label}\" conf=${fmtF(d.confidence)}$simStr box=${fmtBox(d.boundingBox)}")
            }
        }

        val reacquired = findBestCandidate(detections)
        if (reacquired != null) {
            // Only count as a hop if this looks like a different object.
            // High embedding similarity = likely same object after VT death, not a real hop.
            val sim = if (hasEmbeddings && reacquired.embedding != null) {
                bestGallerySimilarity(reacquired.embedding!!)
            } else 0f
            val isSameObject = sim >= APPEARANCE_OVERRIDE_THRESHOLD
            if (!isSameObject) reacquisitionHops++
            dualLog(Log.INFO, "REACQUIRE id=${reacquired.id} label=\"${reacquired.label}\" after $framesLost frames (lockedLabel=\"$lockedLabel\") hop=$reacquisitionHops/${MAX_REACQUISITION_HOPS} sim=${fmtF(sim)}${if (isSameObject) " (same-object)" else ""} box=${fmtBox(reacquired.boundingBox)}")
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
            val colorSim = if (lockedColorHistogram != null && candidate.colorHistogram != null) {
                histogramCorrelation(lockedColorHistogram!!, candidate.colorHistogram!!)
            } else null
            if (score != null) {
                if (logThis) {
                    val attrStr = if (lockedPersonAttributes != null && candidate.personAttributes != null) {
                        " attrs=${fmtF(lockedPersonAttributes!!.similarity(candidate.personAttributes!!))}"
                    } else ""
                    dualLog(Log.DEBUG, "  scored id=${candidate.id} label=\"${candidate.label}\" score=${fmtF(score)} sim=${sim?.let { fmtF(it) } ?: "n/a"} color=${colorSim?.let { fmtF(it) } ?: "n/a"}$attrStr (min=${fmtF(minScoreThreshold)})")
                }
                Pair(candidate, score)
            } else {
                if (logThis) {
                    dualLog(Log.DEBUG, "  rejected id=${candidate.id} label=\"${candidate.label}\" sim=${sim?.let { fmtF(it) } ?: "n/a"} (hard threshold)")
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
            dualLog(Log.DEBUG, "  OVERRIDE position: dist=${fmtF(distance)} > thresh=${fmtF(posThreshold)}, but sim=${fmtF(appearanceScore)}")
        }

        val candSize = candBox.width() * candBox.height()
        val sizeRatio = if (lastKnownSize > 0f && candSize > 0f) {
            if (candSize > lastKnownSize) candSize / lastKnownSize else lastKnownSize / candSize
        } else 1f
        val effectiveSizeThreshold = effectiveSizeRatioThreshold()
        if (sizeRatio > effectiveSizeThreshold && !strongVisualMatch) return null
        if (sizeRatio > effectiveSizeThreshold && strongVisualMatch) {
            dualLog(Log.DEBUG, "  OVERRIDE size: ratio=${fmtF(sizeRatio)} > thresh=${fmtF(effectiveSizeThreshold)}, but sim=${fmtF(appearanceScore)}")
        }

        val positionScore = if (posThreshold > 0f) (1f - (distance / posThreshold)).coerceIn(0f, 1f) else 1f
        val sizeScore = (1f - ((sizeRatio - 1f) / (effectiveSizeThreshold - 1f).coerceAtLeast(0.01f))).coerceIn(0f, 1f)
        // Label match: full bonus for match, penalty for mismatch (prevents
        // wrong-category objects from scoring high on color/position alone).
        // Matches against both the enriched label and the original COCO label,
        // since candidates may carry either depending on enrichment timing.
        val labelMatches = lockedLabel != null && (
            candidate.label == lockedLabel || candidate.label == lockedCocoLabel
        )
        val labelScore = when {
            lockedLabel == null -> 0.5f  // unknown label, neutral
            labelMatches -> 1f  // exact or COCO-parent match
            else -> -0.5f  // wrong label penalty
        }

        // Color histogram similarity — cheap but very effective for same-category discrimination
        val hasColor = lockedColorHistogram != null && candidate.colorHistogram != null
        val colorScore = if (hasColor) {
            histogramCorrelation(lockedColorHistogram!!, candidate.colorHistogram!!)
                .coerceIn(0f, 1f)
        } else 0f

        // Person attribute similarity — very discriminative for same-label "person" candidates
        val hasAttrs = lockedPersonAttributes != null && candidate.personAttributes != null
        val attrScore = if (hasAttrs) {
            lockedPersonAttributes!!.similarity(candidate.personAttributes!!)
        } else 0f

        // Weight distribution depends on available signals.
        if (hasAppearance) {
            // With person attributes, redistribute some weight from appearance/color to attrs.
            // Attributes are highly discriminative for persons (gender, clothing, accessories).
            val basePositionWeight = 0.15f
            val baseSizeWeight = 0.10f
            val baseLabelWeight = if (hasAttrs) 0.10f else 0.20f  // 600 OIV7 classes are reliable
            val baseAttrWeight = if (hasAttrs) 0.15f else 0f
            val baseAppearanceWeight = if (hasAttrs) 0.25f else 0.30f
            val baseColorWeight = if (hasColor) (if (hasAttrs) 0.15f else 0.25f) else 0f
            // Redistribute unused weights to appearance
            val unusedBase = 1f - basePositionWeight - baseSizeWeight - baseLabelWeight -
                             baseAttrWeight - baseAppearanceWeight - baseColorWeight
            val effectiveAppearanceBase = baseAppearanceWeight + unusedBase

            val effectivePositionWeight = basePositionWeight * positionConfidence
            val redistributed = basePositionWeight * (1f - positionConfidence)
            // Redistribute decayed position weight to active signals only.
            // Shares must sum to 1.0 regardless of which signals are available.
            val attrRedistShare = if (hasAttrs) 0.15f else 0f
            val colorRedistShare = if (hasColor) 0.20f else 0f
            val sizeRedistShare = 0.10f
            val labelRedistShare = 0.15f
            val appearRedistShare = 1f - sizeRedistShare - labelRedistShare - attrRedistShare - colorRedistShare
            val effectiveSizeWeight = baseSizeWeight + redistributed * sizeRedistShare
            val effectiveLabelWeight = baseLabelWeight + redistributed * labelRedistShare
            val effectiveAttrWeight = if (hasAttrs) baseAttrWeight + redistributed * attrRedistShare else 0f
            val effectiveAppearanceWeight = effectiveAppearanceBase + redistributed * appearRedistShare
            val effectiveColorWeight = if (hasColor) baseColorWeight + redistributed * colorRedistShare else 0f

            return (positionScore * effectivePositionWeight) +
                   (sizeScore * effectiveSizeWeight) +
                   (labelScore * effectiveLabelWeight) +
                   (attrScore * effectiveAttrWeight) +
                   (appearanceScore * effectiveAppearanceWeight) +
                   (colorScore * effectiveColorWeight)
        } else {
            // Fallback: no embedding available, use original weights
            val basePositionWeight = 0.40f
            val baseSizeWeight = 0.20f
            val baseLabelWeight = 0.40f

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
        return bestGallerySimilarity(candidateEmbedding, _embeddingGallery)
    }

    /** Log to both logcat and session file. */
    private fun dualLog(level: Int, msg: String) {
        Log.println(level, TAG, msg)
        sessionLogger?.invoke(msg)
    }

    private fun fmtF(f: Float) = "%.3f".format(f)
    private fun fmtBox(b: RectF) = "[%.2f,%.2f,%.2f,%.2f]".format(b.left, b.top, b.right, b.bottom)
}
