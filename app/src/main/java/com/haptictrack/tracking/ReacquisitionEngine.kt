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
    /** OSNet person re-ID embedding from lock time. */
    var lockedReIdEmbedding: FloatArray? = null
        private set
    /** MobileFaceNet face embedding — added progressively when face first appears. */
    var lockedFaceEmbedding: FloatArray? = null
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

    fun lock(trackingId: Int, boundingBox: RectF, label: String?, embedding: FloatArray? = null) {
        lock(trackingId, boundingBox, label, if (embedding != null) listOf(embedding) else emptyList())
    }

    fun lock(trackingId: Int, boundingBox: RectF, label: String?, embeddings: List<FloatArray>,
             colorHist: FloatArray? = null, personAttrs: PersonAttributes? = null,
             cocoLabel: String? = null, reIdEmbedding: FloatArray? = null,
             faceEmbedding: FloatArray? = null) {
        lockedId = trackingId
        lockedLabel = label
        lockedCocoLabel = cocoLabel
        _embeddingGallery = embeddings.map { it.copyOf() }.toMutableList()
        lockedColorHistogram = colorHist?.copyOf()
        lockedPersonAttributes = personAttrs
        lockedReIdEmbedding = reIdEmbedding?.copyOf()
        lockedFaceEmbedding = faceEmbedding?.copyOf()
        lastKnownBox = RectF(boundingBox)
        lastKnownLabel = label
        lastKnownSize = boundingBox.width() * boundingBox.height()
        framesLost = 0
        val attrStr = personAttrs?.summary() ?: "n/a"
        dualLog(Log.INFO, "LOCK id=$trackingId label=\"$label\" box=${fmtBox(boundingBox)} size=${fmtF(lastKnownSize)} gallery=${embeddingGallery.size} colorHist=${colorHist != null} attrs=\"$attrStr\"")
    }

    /** Add a face embedding progressively (e.g. when face first appears during tracking). */
    fun addFaceEmbedding(embedding: FloatArray) {
        if (lockedFaceEmbedding == null) {
            lockedFaceEmbedding = embedding.copyOf()
            dualLog(Log.INFO, "FACE_EMBED added (${embedding.size}-dim)")
        }
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
        lockedReIdEmbedding = null
        lockedFaceEmbedding = null
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
        if (framesLost > maxFramesLost) {
            if (framesLost == maxFramesLost + 1) {
                dualLog(Log.WARN, "TIMEOUT after $maxFramesLost frames. Giving up on lockedLabel=\"$lockedLabel\"")
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
            val sim = if (hasEmbeddings && reacquired.embedding != null) {
                bestGallerySimilarity(reacquired.embedding!!)
            } else 0f
            val reIdSim = if (lockedReIdEmbedding != null && reacquired.reIdEmbedding != null) {
                " reId=${fmtF(cosineSimilarity(lockedReIdEmbedding!!, reacquired.reIdEmbedding!!))}"
            } else ""
            val faceSim = if (lockedFaceEmbedding != null && reacquired.faceEmbedding != null) {
                " face=${fmtF(cosineSimilarity(lockedFaceEmbedding!!, reacquired.faceEmbedding!!))}"
            } else ""
            dualLog(Log.INFO, "REACQUIRE id=${reacquired.id} label=\"${reacquired.label}\" after $framesLost frames (lockedLabel=\"$lockedLabel\") sim=${fmtF(sim)}$reIdSim$faceSim box=${fmtBox(reacquired.boundingBox)}")
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

        // Label is a gate inside scoreCandidate(), not a pre-filter here.
        // Wrong-label candidates are rejected by the label gate unless a
        // strong embedding (>0.7) overrides — handles label flicker.
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
                    val reIdStr = if (lockedReIdEmbedding != null && candidate.reIdEmbedding != null) {
                        " reId=${fmtF(cosineSimilarity(lockedReIdEmbedding!!, candidate.reIdEmbedding!!))}"
                    } else ""
                    val faceStr = if (lockedFaceEmbedding != null && candidate.faceEmbedding != null) {
                        " face=${fmtF(cosineSimilarity(lockedFaceEmbedding!!, candidate.faceEmbedding!!))}"
                    } else ""
                    dualLog(Log.DEBUG, "  scored id=${candidate.id} label=\"${candidate.label}\" score=${fmtF(score)} sim=${sim?.let { fmtF(it) } ?: "n/a"} color=${colorSim?.let { fmtF(it) } ?: "n/a"}$attrStr$reIdStr$faceStr (min=${fmtF(minScoreThreshold)})")
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

        // --- Appearance signal (computed early for override checks) ---
        val hasAppearance = hasEmbeddings && candidate.embedding != null
        val appearanceScore = if (hasAppearance) {
            bestGallerySimilarity(candidate.embedding!!).coerceIn(0f, 1f)
        } else 0f
        val strongVisualMatch = hasAppearance && appearanceScore > APPEARANCE_OVERRIDE_THRESHOLD

        // --- GATE A: Position hard filter (with time decay) ---
        val dx = candBox.centerX() - refBox.centerX()
        val dy = candBox.centerY() - refBox.centerY()
        val distance = kotlin.math.sqrt(dx * dx + dy * dy)

        if (distance > posThreshold && !strongVisualMatch) return null
        if (distance > posThreshold && strongVisualMatch) {
            dualLog(Log.DEBUG, "  OVERRIDE position: dist=${fmtF(distance)} > thresh=${fmtF(posThreshold)}, but sim=${fmtF(appearanceScore)}")
        }

        // --- GATE A: Size hard filter (with time decay) ---
        val candSize = candBox.width() * candBox.height()
        val sizeRatio = if (lastKnownSize > 0f && candSize > 0f) {
            if (candSize > lastKnownSize) candSize / lastKnownSize else lastKnownSize / candSize
        } else 1f
        val effectiveSizeThreshold = effectiveSizeRatioThreshold()
        if (sizeRatio > effectiveSizeThreshold && !strongVisualMatch) return null
        if (sizeRatio > effectiveSizeThreshold && strongVisualMatch) {
            dualLog(Log.DEBUG, "  OVERRIDE size: ratio=${fmtF(sizeRatio)} > thresh=${fmtF(effectiveSizeThreshold)}, but sim=${fmtF(appearanceScore)}")
        }

        // --- GATE B: Label gate ---
        // Wrong-label candidates are rejected outright unless a strong embedding
        // indicates it's the same object with a flickered label. This prevents
        // cross-category leakage (e.g. person matching a locked chair via color).
        val labelMatches = lockedLabel != null && (
            candidate.label == lockedLabel || candidate.label == lockedCocoLabel
        )
        when {
            lockedLabel == null -> { /* pass: no label constraint */ }
            labelMatches -> { /* pass: correct label */ }
            strongVisualMatch -> {
                dualLog(Log.DEBUG, "  OVERRIDE label: \"${candidate.label}\" != \"$lockedLabel\", but sim=${fmtF(appearanceScore)}")
            }
            else -> return null  // REJECT: wrong label, no embedding override
        }

        // --- RANKING: score survivors for selection ---
        val positionScore = if (posThreshold > 0f) (1f - (distance / posThreshold)).coerceIn(0f, 1f) else 1f
        val sizeScore = (1f - ((sizeRatio - 1f) / (effectiveSizeThreshold - 1f).coerceAtLeast(0.01f))).coerceIn(0f, 1f)

        val hasColor = lockedColorHistogram != null && candidate.colorHistogram != null
        val colorScore = if (hasColor) {
            histogramCorrelation(lockedColorHistogram!!, candidate.colorHistogram!!).coerceIn(0f, 1f)
        } else 0f

        val hasAttrs = lockedPersonAttributes != null && candidate.personAttributes != null
        val attrScore = if (hasAttrs) {
            lockedPersonAttributes!!.similarity(candidate.personAttributes!!)
        } else 0f

        // Face embedding: strongest identity signal for persons (when available)
        val hasFace = lockedFaceEmbedding != null && candidate.faceEmbedding != null
        val faceScore = if (hasFace) {
            cosineSimilarity(lockedFaceEmbedding!!, candidate.faceEmbedding!!).coerceIn(0f, 1f)
        } else 0f

        // Re-ID embedding: strong person identity signal (when available)
        val hasReId = lockedReIdEmbedding != null && candidate.reIdEmbedding != null
        val reIdScore = if (hasReId) {
            cosineSimilarity(lockedReIdEmbedding!!, candidate.reIdEmbedding!!).coerceIn(0f, 1f)
        } else 0f

        // Small bonus for exact label match (vs passing via embedding override)
        val labelBonus = if (lockedLabel != null && labelMatches) 0.05f else 0f

        if (hasFace) {
            // Face available: strongest identity signal, dominates ranking.
            // Re-ID and generic embedding are secondary.
            val baseFaceW = 0.45f
            val baseReIdW = if (hasReId) 0.20f else 0f
            val baseEmbW = if (hasAppearance) 0.10f else 0f
            val basePosW = 0.05f * positionConfidence
            val baseColorW = if (hasColor) 0.10f else 0f
            val baseAttrW = if (hasAttrs) 0.05f else 0f
            val unused = (1f - baseFaceW - baseReIdW - baseEmbW - basePosW - baseColorW - baseAttrW).coerceAtLeast(0f)
            val effFaceW = baseFaceW + unused

            return (faceScore * effFaceW) +
                   (reIdScore * baseReIdW) +
                   (appearanceScore * baseEmbW) +
                   (positionScore * basePosW) +
                   (colorScore * baseColorW) +
                   (attrScore * baseAttrW) +
                   labelBonus
        } else if (hasReId) {
            // Re-ID available but no face: re-ID is primary, generic embedding secondary
            val baseReIdW = 0.40f
            val baseEmbW = if (hasAppearance) 0.20f else 0f
            val basePosW = 0.10f * positionConfidence
            val baseSizeW = 0.05f
            val baseColorW = if (hasColor) 0.15f else 0f
            val baseAttrW = if (hasAttrs) 0.10f else 0f
            val unused = (1f - baseReIdW - baseEmbW - basePosW - baseSizeW - baseColorW - baseAttrW).coerceAtLeast(0f)
            val effReIdW = baseReIdW + unused

            return (reIdScore * effReIdW) +
                   (appearanceScore * baseEmbW) +
                   (positionScore * basePosW) +
                   (sizeScore * baseSizeW) +
                   (colorScore * baseColorW) +
                   (attrScore * baseAttrW) +
                   labelBonus
        } else if (hasAppearance) {
            // Generic embedding only (non-person, or person without re-ID)
            val baseEmbW = 0.50f
            val basePosW = 0.15f * positionConfidence
            val baseSizeW = 0.10f
            val baseColorW = if (hasColor) 0.15f else 0f
            val baseAttrW = if (hasAttrs) 0.10f else 0f
            val unused = (1f - baseEmbW - basePosW - baseSizeW - baseColorW - baseAttrW).coerceAtLeast(0f)
            val effEmbW = baseEmbW + unused

            return (appearanceScore * effEmbW) +
                   (positionScore * basePosW) +
                   (sizeScore * baseSizeW) +
                   (colorScore * baseColorW) +
                   (attrScore * baseAttrW) +
                   labelBonus
        } else {
            // No embedding fallback: position + size only
            val basePosW = 0.50f * positionConfidence
            val baseSizeW = 0.25f
            val redistributed = 0.50f * (1f - positionConfidence)
            val effSizeW = baseSizeW + redistributed * 0.6f
            val baseBonus = 0.25f + redistributed * 0.4f

            return (positionScore * basePosW) +
                   (sizeScore * effSizeW) +
                   baseBonus +
                   labelBonus
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
