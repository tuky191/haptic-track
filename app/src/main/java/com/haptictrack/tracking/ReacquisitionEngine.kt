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
        /** Embedding similarity above this bypasses the label gate (cross-category protection). */
        const val APPEARANCE_OVERRIDE_THRESHOLD = 0.7f
        /** Embedding similarity above this bypasses position/size hard filters.
         *  Lower than label override because position rejection is about camera movement,
         *  not identity confusion — 0.55 is enough to say "same object, just moved." */
        const val GEOMETRIC_OVERRIDE_THRESHOLD = 0.55f
        /** Embedding similarity above this bypasses tentative confirmation.
         *  Higher than geometric override: overriding position/size is low-risk,
         *  but skipping multi-frame confirmation needs stronger evidence.
         *  Keyboard at sim=0.582 overrode geometric gates during phone rotation —
         *  tentative confirmation would have caught it (single-frame fluke). */
        const val TENTATIVE_BYPASS_THRESHOLD = 0.65f
        /** Minimum embedding similarity to consider a candidate at all.
         *  If the primary embedder says the candidate is a different object (sim < this),
         *  no amount of re-ID, attributes, or color can rescue it. */
        const val MIN_EMBEDDING_SIMILARITY = 0.15f
        /** Maximum embeddings to keep in gallery. */
        const val MAX_GALLERY_SIZE = 12
        /** Maximum negative examples to store. */
        const val MAX_NEGATIVE_EXAMPLES = 10
        /** Consecutive frames the same detection must win before committing (DeepSORT-style). */
        const val TENTATIVE_MIN_FRAMES = 3
        /** IoU threshold to consider a detection "the same" across consecutive frames. */
        const val TENTATIVE_IOU_THRESHOLD = 0.3f
        /** Lowe's ratio test: reject when secondBestSim / bestSim exceeds this (SIFT-style). */
        const val RATIO_TEST_THRESHOLD = 0.85f
    }

    var lockedId: Int? = null
        private set
    var lockedLabel: String? = null
        private set
    /** Original COCO label before enrichment — kept for logging/display. */
    var lockedCocoLabel: String? = null
        private set
    /** Binary category gate: person vs not-person. Replaces per-label matching. */
    var lockedIsPerson: Boolean = false
        private set
    /** Gallery of reference embeddings — augmented at lock time, accumulated during tracking. */
    private var _embeddingGallery: MutableList<FloatArray> = mutableListOf()
    val embeddingGallery: List<FloatArray> get() = _embeddingGallery

    /** Convenience: true if we have any reference embeddings. */
    val hasEmbeddings: Boolean get() = _embeddingGallery.isNotEmpty()
    /** Centroid (L2-normalized mean) of the gallery — stable identity representation. */
    private var _embeddingCentroid: FloatArray? = null
    private fun recomputeCentroid() {
        _embeddingCentroid = computeCentroid(_embeddingGallery)
        recomputeMinGallerySim()
    }

    /** Cosine similarity between a candidate and the gallery centroid. */
    fun centroidSimilarity(candidate: FloatArray): Float {
        val centroid = _embeddingCentroid ?: return 0f
        return cosineSimilarity(candidate, centroid)
    }

    /** Minimum pairwise sim within gallery — drives adaptive embedding floor. */
    private var _minGallerySim: Float = 1f
    private fun recomputeMinGallerySim() { _minGallerySim = minPairwiseSimilarity(_embeddingGallery) }

    /** Online classifier trained from gallery (positives) + scene negatives. */
    private val _classifier = OnlineClassifier()
    val classifierTrained: Boolean get() = _classifier.isTrained

    /** Retrain classifier when gallery or negatives change enough. */
    private var _lastTrainPositives = 0
    private var _lastTrainNegatives = 0
    private fun maybeRetrainClassifier() {
        val posCount = _embeddingGallery.size
        val negCount = _negativeExamples.size
        // Retrain when we have new data (at least 2 new examples since last train)
        if (posCount >= 3 && negCount >= 3 &&
            (posCount + negCount) - (_lastTrainPositives + _lastTrainNegatives) >= 2) {
            _classifier.train(_embeddingGallery, _negativeExamples)
            _lastTrainPositives = posCount
            _lastTrainNegatives = negCount
            if (_classifier.isTrained) {
                Log.d(TAG, "Classifier retrained: $posCount positives, $negCount negatives")
            }
        }
    }

    /** Classifier prediction for a candidate embedding. Returns [0, 1]. */
    fun classifierScore(candidate: FloatArray): Float = _classifier.predict(candidate)

    /** Embeddings of rejected candidates — sharpens identity boundary. */
    private val _negativeExamples = mutableListOf<FloatArray>()

    private fun addNegativeExample(embedding: FloatArray) {
        if (_negativeExamples.size >= MAX_NEGATIVE_EXAMPLES) _negativeExamples.removeAt(0)
        _negativeExamples.add(embedding.copyOf())
    }

    /** Add a scene negative — an embedding from a non-locked detection seen during tracking.
     *  Builds the negative prototype for discriminative scoring. */
    fun addSceneNegative(embedding: FloatArray) {
        val centroid = _embeddingCentroid ?: return
        val sim = cosineSimilarity(embedding, centroid)
        if (sim < 0.95f) {
            addNegativeExample(embedding)
            maybeRetrainClassifier()
            Log.d(TAG, "Scene negative added (sim=${"%.3f".format(sim)}) — total=${_negativeExamples.size}")
        }
    }

    /** Prototype margin: sim(candidate, pos_centroid) - sim(candidate, neg_centroid).
     *  Returns margin in [-1, 1]. Positive = closer to locked object than to scene.
     *  Returns 0 when no negatives exist (no discrimination possible). */
    private fun prototypeMargin(candidateEmbedding: FloatArray): Float {
        if (_negativeExamples.isEmpty() || _embeddingCentroid == null) return 0f
        val posSim = cosineSimilarity(candidateEmbedding, _embeddingCentroid!!)
        val negCentroid = computeCentroid(_negativeExamples) ?: return 0f
        val negSim = cosineSimilarity(candidateEmbedding, negCentroid)
        return posSim - negSim
    }
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
    /** Last known velocity (normalized units/frame) from VelocityEstimator. */
    var lastKnownVelocityX: Float = 0f
        private set
    var lastKnownVelocityY: Float = 0f
        private set
    var framesLost: Int = 0
        private set

    // --- Tentative confirmation state (DeepSORT-style) ---
    // A candidate must win scoring for TENTATIVE_MIN_FRAMES consecutive frames
    // before we commit to reacquisition. Eliminates single-frame flukes like
    // couch (sim=0.41) briefly appearing when tracking a chair.
    private var tentativeBox: RectF? = null
    private var tentativeCount: Int = 0

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
        lockedIsPerson = (cocoLabel ?: label) == "person"
        _embeddingGallery = embeddings.map { it.copyOf() }.toMutableList()
        recomputeCentroid()
        _negativeExamples.clear()
        _classifier.clear()
        _lastTrainPositives = 0
        _lastTrainNegatives = 0
        lockedColorHistogram = colorHist?.copyOf()
        lockedPersonAttributes = personAttrs
        lockedReIdEmbedding = reIdEmbedding?.copyOf()
        lockedFaceEmbedding = faceEmbedding?.copyOf()
        lastKnownBox = RectF(boundingBox)
        lastKnownLabel = label
        lastKnownSize = boundingBox.width() * boundingBox.height()
        lastKnownVelocityX = 0f
        lastKnownVelocityY = 0f
        framesLost = 0
        tentativeBox = null
        tentativeCount = 0
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
        recomputeCentroid()
        maybeRetrainClassifier()
    }

    fun clear() {
        dualLog(Log.INFO, "CLEAR (was id=$lockedId label=\"$lockedLabel\" coco=\"$lockedCocoLabel\")")
        lockedId = null
        lockedLabel = null
        lockedCocoLabel = null
        lockedIsPerson = false
        _embeddingGallery.clear()
        _embeddingCentroid = null
        _minGallerySim = 1f
        _negativeExamples.clear()
        _classifier.clear()
        _lastTrainPositives = 0
        _lastTrainNegatives = 0
        lockedColorHistogram = null
        lockedPersonAttributes = null
        lockedReIdEmbedding = null
        lockedFaceEmbedding = null
        lastKnownBox = null
        lastKnownLabel = null
        lastKnownVelocityX = 0f
        lastKnownVelocityY = 0f
        lastKnownSize = 0f
        framesLost = 0
        tentativeBox = null
        tentativeCount = 0
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

        val candidate = findBestCandidate(detections)
        if (candidate == null) {
            tentativeBox = null
            tentativeCount = 0
            return null
        }

        val sim = if (hasEmbeddings && candidate.embedding != null) {
            bestGallerySimilarity(candidate.embedding!!)
        } else 0f
        val strongMatch = sim >= APPEARANCE_OVERRIDE_THRESHOLD

        // --- Tentative confirmation (DeepSORT-style) ---
        // Don't commit on a single frame. Require the same detection to win
        // for TENTATIVE_MIN_FRAMES consecutive frames.
        //
        // Skip tentative when identity is confident:
        //   1. Strong embedding match (sim >= 0.7) — clearly the same object
        //   2. Trained classifier is confident (P >= 0.8) — learned boundary says yes
        //   3. Fallback: high embedding similarity (>= 0.65) when classifier not trained
        // The classifier adapts per object — keyboard at cls=0.581 won't bypass,
        // but a genuine mouse at cls=0.95 will.
        val classifierConfident = _classifier.isTrained && candidate.embedding != null &&
            _classifier.predict(candidate.embedding!!) >= 0.8f
        val skipTentative = strongMatch || classifierConfident ||
            (!_classifier.isTrained && hasEmbeddings && sim >= TENTATIVE_BYPASS_THRESHOLD)
        if (!skipTentative) {
            val prevBox = tentativeBox
            val candBox = candidate.boundingBox
            if (prevBox != null && computeIou(prevBox, candBox) >= TENTATIVE_IOU_THRESHOLD) {
                tentativeCount++
            } else {
                tentativeCount = 1
            }
            tentativeBox = RectF(candBox)

            if (tentativeCount < TENTATIVE_MIN_FRAMES) {
                val logThis = framesLost % 10 == 1
                if (logThis) dualLog(Log.DEBUG, "TENTATIVE: id=${candidate.id} label=\"${candidate.label}\" sim=${fmtF(sim)} count=$tentativeCount/$TENTATIVE_MIN_FRAMES — waiting")
                return null
            }
            dualLog(Log.DEBUG, "CONFIRMED: id=${candidate.id} label=\"${candidate.label}\" sim=${fmtF(sim)} after $tentativeCount frames")
        }

        tentativeBox = null
        tentativeCount = 0

        val reIdSim = if (lockedReIdEmbedding != null && candidate.reIdEmbedding != null) {
            " reId=${fmtF(cosineSimilarity(lockedReIdEmbedding!!, candidate.reIdEmbedding!!))}"
        } else ""
        val faceSim = if (lockedFaceEmbedding != null && candidate.faceEmbedding != null) {
            " face=${fmtF(cosineSimilarity(lockedFaceEmbedding!!, candidate.faceEmbedding!!))}"
        } else ""
        dualLog(Log.INFO, "REACQUIRE id=${candidate.id} label=\"${candidate.label}\" after $framesLost frames (lockedLabel=\"$lockedLabel\") sim=${fmtF(sim)}$reIdSim$faceSim box=${fmtBox(candidate.boundingBox)}")
        lockedId = candidate.id
        updateFromMatch(candidate)
        return candidate
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

    /** Update last-known velocity from VelocityEstimator (normalized units/frame). */
    fun updateVelocity(vx: Float, vy: Float) {
        lastKnownVelocityX = vx
        lastKnownVelocityY = vy
    }

    /**
     * Prepare for a camera rebind that will interrupt the frame stream.
     * Forces into search mode so the first frame after rebind triggers re-acquisition
     * using the full embedding gallery. Call before unbinding camera use cases.
     */
    fun prepareForRebind() {
        if (!isLocked) return
        framesLost = 1
        android.util.Log.i("Reacq", "REBIND_PREPARE — forced search mode, gallery=${embeddingGallery.size}")
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

        // Person/not-person gate is inside scoreCandidate(). Pre-filter only removes invalid IDs.
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
                    val marginStr = if (candidate.embedding != null && _negativeExamples.isNotEmpty()) {
                        " margin=${fmtF(prototypeMargin(candidate.embedding!!))}"
                    } else ""
                    val clsStr = if (_classifier.isTrained && candidate.embedding != null) {
                        " cls=${fmtF(_classifier.predict(candidate.embedding!!))}"
                    } else ""
                    dualLog(Log.DEBUG, "  scored id=${candidate.id} label=\"${candidate.label}\" score=${fmtF(score)} sim=${sim?.let { fmtF(it) } ?: "n/a"} color=${colorSim?.let { fmtF(it) } ?: "n/a"}$attrStr$reIdStr$faceStr$marginStr$clsStr (min=${fmtF(minScoreThreshold)})")
                }
                Pair(candidate, score)
            } else {
                if (logThis) {
                    dualLog(Log.DEBUG, "  rejected id=${candidate.id} label=\"${candidate.label}\" sim=${sim?.let { fmtF(it) } ?: "n/a"} (hard threshold)")
                }
                null
            }
        }

        if (scored.isEmpty()) return null

        val sortedScored = scored.sortedByDescending { it.second }
        val best = sortedScored[0]

        // --- Lowe's ratio test (SIFT-style) ---
        // If two candidates have similar embedding similarity, the match is ambiguous.
        // Reject and wait for a frame where one candidate is clearly better.
        // This is structural: no threshold on the absolute value, only the ratio.
        if (sortedScored.size >= 2 && hasEmbeddings) {
            val bestSim = if (best.first.embedding != null) bestGallerySimilarity(best.first.embedding!!) else 0f
            val secondSim = if (sortedScored[1].first.embedding != null) bestGallerySimilarity(sortedScored[1].first.embedding!!) else 0f
            if (bestSim > 0f && secondSim > 0f) {
                val ratio = secondSim / bestSim
                if (ratio > RATIO_TEST_THRESHOLD) {
                    if (logThis) dualLog(Log.DEBUG, "  RATIO_REJECT: best=${fmtF(bestSim)} second=${fmtF(secondSim)} ratio=${fmtF(ratio)} > ${fmtF(RATIO_TEST_THRESHOLD)} — ambiguous, waiting")
                    return null
                }
            }
        }

        return best.takeIf { it.second >= minScoreThreshold }?.first
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
        // --- GATE: Require embedding when gallery exists ---
        // If we have reference embeddings but the candidate has none (async not ready),
        // reject it — accepting without identity verification causes wrong-object locks.
        // Always require embeddings. The async pipeline delivers within 2-3 frames.
        // Accepting without embeddings leads to wrong-category reacquisitions
        // (e.g. laptop accepted when tracking chair, scored on position alone).
        if (hasEmbeddings && candidate.embedding == null) {
            return null
        }
        val galleryMature = _embeddingGallery.size >= 8

        // --- GATE: Embedding floor (gallery-relative) ---
        // Instead of a fixed floor, adapt to how consistent the gallery is.
        // A tight gallery (min pair sim=0.8) demands higher similarity from candidates.
        // A diverse gallery (min pair sim=0.4) is more lenient.
        // Floor = minGallerySim * 0.75, clamped to [0.3, 0.5].
        val adaptiveFloor = (_minGallerySim * 0.75f).coerceIn(0.3f, 0.5f)
        val embeddingFloor = if (galleryMature) maxOf(adaptiveFloor, 0.4f) else adaptiveFloor
        if (hasAppearance && appearanceScore < embeddingFloor) {
            addNegativeExample(candidate.embedding!!)
            return null
        }

        // Tiered override: geometric gates use a lower threshold (0.55) because
        // position rejection is about camera movement, not identity confusion.
        // Label gate uses a higher threshold (0.7) to protect against cross-category leakage.
        val geometricOverride = hasAppearance && appearanceScore > GEOMETRIC_OVERRIDE_THRESHOLD
        val labelOverride = hasAppearance && appearanceScore > APPEARANCE_OVERRIDE_THRESHOLD

        // --- GATE A: Position hard filter (with time decay) ---
        // Use velocity-predicted position when available. If the subject was moving
        // right at 3%/frame and we lost it 5 frames ago, look 15% to the right
        // of the last known position instead of at the last known position itself.
        val velocitySpeed = kotlin.math.sqrt(lastKnownVelocityX * lastKnownVelocityX + lastKnownVelocityY * lastKnownVelocityY)
        // Cap prediction to 10 frames — beyond that, velocity is unreliable and prediction
        // pins to screen edge. Revert to last-known position as uncertainty grows.
        val predictionFrames = minOf(framesLost, 10)
        val predictedCenterX = if (velocitySpeed > 0.01f) {
            (refBox.centerX() + lastKnownVelocityX * predictionFrames).coerceIn(0f, 1f)
        } else refBox.centerX()
        val predictedCenterY = if (velocitySpeed > 0.01f) {
            (refBox.centerY() + lastKnownVelocityY * predictionFrames).coerceIn(0f, 1f)
        } else refBox.centerY()
        val dx = candBox.centerX() - predictedCenterX
        val dy = candBox.centerY() - predictedCenterY
        val distance = kotlin.math.sqrt(dx * dx + dy * dy)

        if (distance > posThreshold && !geometricOverride) return null
        if (distance > posThreshold && geometricOverride) {
            dualLog(Log.DEBUG, "  OVERRIDE position: dist=${fmtF(distance)} > thresh=${fmtF(posThreshold)}, but sim=${fmtF(appearanceScore)}")
        }

        // --- GATE A: Size hard filter (with time decay) ---
        val candSize = candBox.width() * candBox.height()
        val sizeRatio = if (lastKnownSize > 0f && candSize > 0f) {
            if (candSize > lastKnownSize) candSize / lastKnownSize else lastKnownSize / candSize
        } else 1f
        val effectiveSizeThreshold = effectiveSizeRatioThreshold()
        // Without embeddings, use a stricter size limit to avoid locking on partial bodies
        // (e.g. a hand when we locked on a full person). Max 3x size ratio without identity.
        val sizeThreshold = if (!hasAppearance && _embeddingGallery.size >= 8) {
            minOf(effectiveSizeThreshold, 3.0f)
        } else effectiveSizeThreshold
        if (sizeRatio > sizeThreshold && !geometricOverride) return null
        if (sizeRatio > sizeThreshold && geometricOverride) {
            dualLog(Log.DEBUG, "  OVERRIDE size: ratio=${fmtF(sizeRatio)} > thresh=${fmtF(effectiveSizeThreshold)}, but sim=${fmtF(appearanceScore)}")
        }

        // --- GATE B: Person/not-person category gate ---
        // Binary gate: a locked person only accepts person candidates, and vice versa.
        // Specific labels don't matter — embedding handles identity within each bucket.
        // This eliminates label flicker problems (bowl/potted plant, deer/sheep/dog).
        val candidateIsPerson = candidate.label == "person"
        if (lockedIsPerson != candidateIsPerson && !labelOverride) {
            // Store as negative example — helps sharpen identity boundary
            if (hasAppearance) addNegativeExample(candidate.embedding!!)
            return null  // REJECT: person/non-person mismatch
        }
        if (lockedIsPerson != candidateIsPerson && labelOverride) {
            dualLog(Log.DEBUG, "  OVERRIDE category: candidate=${if (candidateIsPerson) "person" else "non-person"}, locked=${if (lockedIsPerson) "person" else "non-person"}, sim=${fmtF(appearanceScore)}")
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

        // No label bonus — person/not-person gate handles category, embedding handles identity

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
                   (attrScore * baseAttrW)
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
                   (attrScore * baseAttrW)
        } else if (hasAppearance) {
            // Generic embedding only (non-person, or person without re-ID)
            val centroidScore = centroidSimilarity(candidate.embedding!!).coerceIn(0f, 1f)
            val embScore = (appearanceScore + centroidScore) / 2f

            // Discriminative scoring: classifier > prototype margin > raw cosine.
            // The classifier learns a decision boundary from positives+negatives.
            // Falls back to prototype margin when classifier isn't trained yet.
            val clsScore = if (_classifier.isTrained) _classifier.predict(candidate.embedding!!) else -1f
            val margin = prototypeMargin(candidate.embedding!!)
            val hasDiscriminator = clsScore >= 0f || margin != 0f

            val baseEmbW = if (hasDiscriminator) 0.30f else 0.50f
            val baseClsW = if (clsScore >= 0f) 0.25f else 0f
            val baseMarginW = if (clsScore < 0f && margin != 0f) 0.20f else 0f
            val basePosW = 0.10f * positionConfidence
            val baseSizeW = 0.10f
            val baseColorW = if (hasColor) 0.10f else 0f
            val baseAttrW = if (hasAttrs) 0.05f else 0f
            val unused = (1f - baseEmbW - baseClsW - baseMarginW - basePosW - baseSizeW - baseColorW - baseAttrW).coerceAtLeast(0f)
            val effEmbW = baseEmbW + unused

            val marginScore = ((margin + 1f) / 2f).coerceIn(0f, 1f)
            return (embScore * effEmbW) +
                   (clsScore.coerceAtLeast(0f) * baseClsW) +
                   (marginScore * baseMarginW) +
                   (positionScore * basePosW) +
                   (sizeScore * baseSizeW) +
                   (colorScore * baseColorW) +
                   (attrScore * baseAttrW)
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
                   0f
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
