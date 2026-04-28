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
    var sessionLogger: ((String) -> Unit)? = null,
    /**
     * Pre-computed background-scene MobileNetV3 embeddings used at lock time
     * to (a) cold-start the [OnlineClassifier] and (b) derive a per-lock
     * adaptive embedding floor. Empty list = use static floors (#67 PR1).
     * Loaded by [FrozenNegatives] from `assets/frozen_negatives_mnv3.bin`.
     */
    private val frozenNegativesMobileNet: List<FloatArray> = emptyList(),
    /** Pre-computed background OSNet embeddings; same role for the person-reId path. */
    private val frozenNegativesOsnet: List<FloatArray> = emptyList()
) {

    companion object {
        private const val TAG = "Reacq"
        /** Raw-cosine fallback used when z-score is unavailable (no live or frozen cohort).
         *  In production the frozen offline pool (~1500 entries) is loaded via
         *  [create], so this fallback is only ever hit by unit tests with direct
         *  construction. See [Z_LABEL_OVERRIDE_THRESHOLD] for the calibrated path. */
        const val APPEARANCE_OVERRIDE_THRESHOLD = 0.7f
        /** Raw-cosine fallback for the position/size hard-filter bypass when z-score
         *  is unavailable. See [Z_GEOMETRIC_OVERRIDE_THRESHOLD] for the calibrated path. */
        const val GEOMETRIC_OVERRIDE_THRESHOLD = 0.55f
        /** Raw-cosine fallback for tentative-confirmation bypass when z-score is
         *  unavailable. See [Z_TENTATIVE_BYPASS_THRESHOLD] for the calibrated path. */
        const val TENTATIVE_BYPASS_THRESHOLD = 0.65f

        // Phase 3 (#102): calibrated z-score thresholds. Phase 2 measurements on
        // device showed same-person reacquires score z ≥ 1.5 reliably while
        // wrong-person reacquires cluster in [-0.5, +1.0]. Thresholds picked
        // from that distribution:
        /** Z-score above this bypasses the label gate (cross-category protection). */
        const val Z_LABEL_OVERRIDE_THRESHOLD = 1.5f
        /** Z-score above this bypasses position/size hard filters. Slightly lower
         *  than the label-override threshold because position rejection is about
         *  camera movement, not identity confusion — z ≥ 1 is enough to say
         *  "same object, just moved." */
        const val Z_GEOMETRIC_OVERRIDE_THRESHOLD = 1.0f
        /** Z-score above this bypasses tentative-confirmation. Same band as the
         *  label override — skipping multi-frame consistency requires confidence. */
        const val Z_TENTATIVE_BYPASS_THRESHOLD = 1.5f
        /** Floor on impostor σ when computing z-scores. A homogeneous cohort can
         *  collapse σ → 0 which inflates z arbitrarily (man_desk hit z=9.34 in
         *  Phase 2). Cap σ at this floor so the override decision can't be
         *  driven by a degenerate cohort. */
        const val Z_SIGMA_FLOOR = 0.05f
        /** Minimum embedding similarity to consider a candidate at all.
         *  If the primary embedder says the candidate is a different object (sim < this),
         *  no amount of re-ID, attributes, or color can rescue it. */
        const val MIN_EMBEDDING_SIMILARITY = 0.15f
        /** Floor for OSNet (person re-ID) cosine similarity when gating person candidates.
         *  OSNet on Market-1501 produces same-person sim typically in [0.5, 0.85] and
         *  different-person sim in [0.2, 0.4]. 0.45 sits in the gap. Tuned from a live
         *  capture where same-person reId hit 0.836 while MobileNetV3 sim was 0.558. */
        const val PERSON_REID_FLOOR = 0.45f
        /** Face embedding (MobileFaceNet, ArcFace-style) cosine floor for person identity.
         *  Same-person face sim typically >= 0.5; different-person face sim typically
         *  <= 0.3. 0.4 sits in the gap. When both lock and candidate have face
         *  embeddings, face VETOES OSNet — even a high reId sim gets rejected if face
         *  says different. Catches the case where OSNet's whole-body sim sits in
         *  the wrong-person 0.55-0.70 band but face cleanly separates identity.
         *  Filed as #83. */
        const val FACE_FLOOR = 0.40f
        /** Maximum embeddings to keep in gallery. */
        const val MAX_GALLERY_SIZE = 12
        /** Maximum negative examples to store. */
        const val MAX_NEGATIVE_EXAMPLES = 10
        /** Max paired (face, body) memory entries from non-lock persons observed
         *  during VT-confirmed tracking. Used asymmetrically for #83 phase 2. */
        const val MAX_SCENE_FACE_PAIRS = 16
        /** Floor for face cosine similarity to a stored scene-other to count as
         *  "I recognize this face from earlier — it's not the lock." Above this
         *  AND lock-face match below → veto. Same band as FACE_FLOOR but for
         *  the OTHER side of the comparison. */
        const val SCENE_FACE_MATCH_FLOOR = 0.40f
        /** Margin by which the scene-face match must beat the lock-face match
         *  before the face veto fires. Without a margin a candidate whose face
         *  scores 0.42 against the scene memory and 0.41 against the lock would
         *  be rejected — that's measurement noise, not a real identity signal.
         *  0.05f is roughly the stddev of MobileFaceNet scores on near-duplicate
         *  crops in our captures. */
        const val SCENE_FACE_VETO_MARGIN = 0.05f
        /** Margin by which a candidate's body must match a stored scene-other
         *  better than it matches the lock for the body-only veto path to fire.
         *  Without a margin we'd reject any candidate that's slightly closer to
         *  any stored other than to the lock — too aggressive on borderline
         *  same-person frames. */
        const val SCENE_BODY_VETO_MARGIN = 0.10f
        /**
         * Max frozen negatives used for OnlineClassifier cold-start at lock time.
         * The full frozen-negatives asset (~1500) is used for adaptive-floor
         * statistics (mean+0.5σ benefits from larger sample). For classifier
         * training we deliberately subsample — a 5-positive vs 1500-negative
         * imbalance (300:1) lets the gradient be dominated by negatives, and
         * with `gradB / n` normalization the positive class gets undertrained.
         * 100 negatives keeps the imbalance to 20:1, which the existing L2
         * regularization handles. Also bounds train time at lock to ~10ms.
         * Reviewer flag #3 on PR #79.
         */
        const val CLASSIFIER_COLD_START_NEGATIVES = 100
        /**
         * Tentative bypass decision table:
         * | Condition                          | Bypass tentative? | Rationale                    |
         * |------------------------------------|-------------------|------------------------------|
         * | sim >= 0.7 (APPEARANCE_OVERRIDE)   | YES               | Clearly same object          |
         * | classifier P >= 0.8                | YES               | Learned boundary says yes    |
         * | classifier not trained, sim >= 0.55| YES               | Fallback to geometric level  |
         * | classifier trained, P < 0.8        | NO                | Uncertain — require 3 frames |
         * | sim < 0.55                         | NO                | Weak match — require 3 frames|
         */

        /** Consecutive frames the same detection must win before committing (DeepSORT-style). */
        const val TENTATIVE_MIN_FRAMES = 3
        /** IoU threshold to consider a detection "the same" across consecutive frames. */
        const val TENTATIVE_IOU_THRESHOLD = 0.3f
        /** Lowe's ratio test: reject when secondBestSim / bestSim exceeds this (SIFT-style). */
        const val RATIO_TEST_THRESHOLD = 0.85f

        /** MobileNetV3-Large embedding dimension for frozen-negatives asset validation. */
        const val MOBILENET_EMBED_DIM = 1280
        /** OSNet x1.0 embedding dimension for frozen-negatives asset validation. */
        const val OSNET_EMBED_DIM = 512

        /**
         * Production factory that loads the bundled frozen-negatives assets.
         * Tests should use the default constructor (no negatives → static floors).
         */
        fun create(context: android.content.Context): ReacquisitionEngine {
            val mnv3 = FrozenNegatives.load(context, "frozen_negatives_mnv3.bin", MOBILENET_EMBED_DIM)
            val osnet = FrozenNegatives.load(context, "frozen_negatives_osnet.bin", OSNET_EMBED_DIM)
            return ReacquisitionEngine(
                frozenNegativesMobileNet = mnv3,
                frozenNegativesOsnet = osnet
            )
        }
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

    /**
     * Per-lock adaptive embedding floors derived from the gallery → frozen-negatives
     * similarity distribution at lock time. Replaces the static [MIN_EMBEDDING_SIMILARITY]
     * and [PERSON_REID_FLOOR] when frozen negatives are loaded. Null when no asset is
     * available — the engine then falls back to the static floors.
     */
    private var _adaptiveMobileNetFloor: Float? = null
    private var _adaptiveOsnetFloor: Float? = null

    /**
     * Live impostor stats per modality (#102 Phase 1 + Phase 3).
     * Computed lazily from the *live* gallery × *live* scene-negative cohort.
     *
     * Phase 1: exposed z-scores in [scoreCandidate]'s log line for calibration.
     * Phase 3: drives embedding override gates via [overridePasses]
     * (label/geometric/tentative-bypass). Raw-cosine thresholds remain as the
     * fallback when stats are unavailable (cohort below [MIN_COHORT_FOR_ZNORM]).
     *
     * Invalidated (set to dirty) on lock, addEmbedding, addSceneNegative,
     * addScenePersonPair, addFaceEmbedding. Recomputed on next read.
     */
    private var _mnv3LiveStats: ImpostorStats? = null
    private var _osnetLiveStats: ImpostorStats? = null
    private var _faceLiveStats: ImpostorStats? = null
    private var _liveStatsDirty: Boolean = true

    /** Min cohort size for trustworthy z-norm stats. Below this, [znMnv3Sim] etc. return null. */
    private val MIN_COHORT_FOR_ZNORM = 5

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
    /** Cached negative centroid — recomputed on change, same pattern as positive centroid. */
    private var _negativeCentroid: FloatArray? = null

    private fun addNegativeExample(embedding: FloatArray) {
        if (_negativeExamples.size >= MAX_NEGATIVE_EXAMPLES) _negativeExamples.removeAt(0)
        _negativeExamples.add(embedding.copyOf())
        _negativeCentroid = computeCentroid(_negativeExamples)
        _liveStatsDirty = true
    }

    // ---------------------------------------------------------------------
    // Live impostor stats / Z-norm (#102 Phase 1)
    // ---------------------------------------------------------------------

    private fun recomputeLiveStatsIfNeeded() {
        if (!_liveStatsDirty) return
        // Z-norm requires the impostor cohort to match the test-time
        // distribution. Live scene negatives DO — they're embeddings of
        // other detections seen in the same scene/lighting/pose space as
        // the locked subject. The frozen offline pool (frozenNegativesMobileNet
        // / frozenNegativesOsnet) was tried as a cold-start backfill in
        // Phase 3, but on real scenarios the frozen pool's mean was so far
        // below the test-time impostor mean that z-scores inflated to
        // useless levels (clearly-wrong wife reacquires at znOsnet=3.06,
        // chair correct-reacq at znMnv3=8.5 — both off the calibration
        // we did in Phase 2 with live-only cohort).
        //
        // Keeping the cohort live-only means short-lock scenarios produce
        // no z-score, falling back to raw cosine in the override gate.
        // The cold-start gap (kid_to_wife_panning) is a separate problem
        // the override-only Phase 3 doesn't address — needs scoring-level
        // changes or a session-aware impostor pool, both deferred to Phase 4.
        _mnv3LiveStats = if (_negativeExamples.size >= MIN_COHORT_FOR_ZNORM) {
            computeImpostorStats(_embeddingGallery, _negativeExamples)?.withSigmaFloor()
        } else null

        val osnetCohort = _sceneFacePairs.map { it.body }
        _osnetLiveStats = if (osnetCohort.size >= MIN_COHORT_FOR_ZNORM) {
            computeImpostorStats(lockedReIdEmbedding, osnetCohort)?.withSigmaFloor()
        } else null

        val faceCohort = _sceneFacePairs.map { it.face }
        _faceLiveStats = if (faceCohort.size >= MIN_COHORT_FOR_ZNORM) {
            computeImpostorStats(lockedFaceEmbedding, faceCohort)?.withSigmaFloor()
        } else null
        _liveStatsDirty = false
    }

    /** Apply [Z_SIGMA_FLOOR] so a homogeneous cohort can't inflate z-scores
     *  arbitrarily — see man_desk z=9.34 in #102 Phase 2 measurements. */
    private fun ImpostorStats.withSigmaFloor(): ImpostorStats =
        if (std < Z_SIGMA_FLOOR) ImpostorStats(mean, Z_SIGMA_FLOOR, n) else this

    /**
     * Z-normalized MNV3 similarity for [candidate]: how many σ above the live
     * impostor distribution does this candidate's best-gallery match sit?
     * Returns null when the cohort is below [MIN_COHORT_FOR_ZNORM] or the
     * gallery is empty. Phase 3 (#102): the value drives the embedding
     * override gates via [overridePasses] when available; raw cosine is the
     * fallback path.
     */
    fun znMnv3Sim(candidate: FloatArray): Float? {
        recomputeLiveStatsIfNeeded()
        val stats = _mnv3LiveStats ?: return null
        if (_embeddingGallery.isEmpty()) return null
        // stats.std is already clamped to Z_SIGMA_FLOOR by withSigmaFloor()
        // in recomputeLiveStatsIfNeeded — no extra divide-by-zero guard needed.
        val raw = bestGallerySimilarity(candidate, _embeddingGallery)
        return (raw - stats.mean) / stats.std
    }

    /** Z-normalized OSNet similarity. Null when cohort < [MIN_COHORT_FOR_ZNORM]
     *  or no locked OSNet anchor. Used by [overridePasses] for the person path. */
    fun znOsnetSim(candidate: FloatArray): Float? {
        recomputeLiveStatsIfNeeded()
        val stats = _osnetLiveStats ?: return null
        val anchor = lockedReIdEmbedding ?: return null
        val raw = cosineSimilarity(anchor, candidate)
        return (raw - stats.mean) / stats.std
    }

    /** Z-normalized face similarity. Null when cohort < [MIN_COHORT_FOR_ZNORM]
     *  or no locked face anchor. Available to gate logic but not currently
     *  driving any production gate (face has its own [FACE_FLOOR] path). */
    fun znFaceSim(candidate: FloatArray): Float? {
        recomputeLiveStatsIfNeeded()
        val stats = _faceLiveStats ?: return null
        val anchor = lockedFaceEmbedding ?: return null
        val raw = cosineSimilarity(anchor, candidate)
        return (raw - stats.mean) / stats.std
    }

    /** Read-only access to the most recent live stats (for log/test inspection). */
    val mnv3LiveStats: ImpostorStats? get() { recomputeLiveStatsIfNeeded(); return _mnv3LiveStats }
    val osnetLiveStats: ImpostorStats? get() { recomputeLiveStatsIfNeeded(); return _osnetLiveStats }
    val faceLiveStats: ImpostorStats? get() { recomputeLiveStatsIfNeeded(); return _faceLiveStats }

    /**
     * Pick the appropriate z-score for [candidate] mirroring [effectiveAppearanceSim]'s
     * modality choice (OSNet for person-person; MNV3 otherwise). Returns null when
     * stats aren't available for the chosen modality — caller should fall back to
     * raw cosine in that case.
     */
    private fun effectiveAppearanceZ(candidate: TrackedObject): Float? {
        val candidateIsPerson = candidate.label == "person"
        val hasReId = lockedIsPerson && candidateIsPerson &&
            lockedReIdEmbedding != null && candidate.reIdEmbedding != null
        return when {
            hasReId -> znOsnetSim(candidate.reIdEmbedding!!)
            hasEmbeddings && candidate.embedding != null -> znMnv3Sim(candidate.embedding!!)
            else -> null
        }
    }

    /**
     * Override gate decision: prefer the calibrated z-score; fall back to the raw-
     * cosine threshold only when z-stats are unavailable (no live cohort and no
     * frozen pool). Returns true when the embedding evidence is strong enough to
     * bypass the corresponding hard filter.
     */
    private fun overridePasses(
        candidate: TrackedObject,
        rawSim: Float,
        zThresh: Float,
        rawThresh: Float,
    ): Boolean {
        val z = effectiveAppearanceZ(candidate)
        return if (z != null) z >= zThresh else rawSim >= rawThresh
    }

    /** Add a scene negative — an embedding from a non-locked detection seen during tracking.
     *  Builds the negative prototype for discriminative scoring. */
    fun addSceneNegative(embedding: FloatArray) {
        val centroid = _embeddingCentroid ?: return
        val sim = cosineSimilarity(embedding, centroid)
        // Filter: reject near-positives (sim >= 0.85) — these are likely the target itself
        // from overlapping detections or slightly different crops. Adding them as negatives
        // poisons the classifier boundary. Only genuine scene objects (sim < 0.85) qualify.
        if (sim < 0.85f) {
            addNegativeExample(embedding)
            maybeRetrainClassifier()
            Log.d(TAG, "Scene negative added (sim=${"%.3f".format(sim)}) — total=${_negativeExamples.size}")
        }
    }

    /** Paired face + body embedding from a non-lock person observed during VT-
     *  confirmed tracking. Used asymmetrically: at gate time, candidate compares
     *  against lock AND against this memory. If a stored other matches better
     *  than the lock, candidate is rejected. The face-body LINK lets us recognize
     *  someone later when only one modality is visible. (#83 phase 2) */
    private data class ScenePersonPair(val face: FloatArray, val body: FloatArray)
    private val _sceneFacePairs = mutableListOf<ScenePersonPair>()

    /** Snapshot of stored scene face-body pairs (read-only). Test/debug helper. */
    val sceneFacePairCount: Int get() = _sceneFacePairs.size

    /**
     * Add a paired face+body embedding observed for a non-lock person during
     * VT-confirmed tracking. Caller is responsible for ensuring the pair came
     * from the SAME detection at the same frame (the "link" is what makes this
     * useful asymmetrically later).
     *
     * Filters near-lock matches the same way [addSceneNegative] does — if the
     * body is too similar to the lock gallery (sim >= 0.85) we skip, since
     * the detection is likely the lock itself rather than a genuine other.
     * Same for face: if face matches lock face (sim >= 0.6), skip.
     */
    fun addScenePersonPair(face: FloatArray, body: FloatArray) {
        // Reject if this pair is too close to the lock — it's probably the lock
        // itself from a duplicate detection, and we'd later reject the lock.
        //
        // Compare body to the lock's OSNet body embedding (same dim space).
        // Earlier this used bestGallerySimilarity against _embeddingGallery,
        // which holds MobileNetV3 (1280-dim) embeddings — cosineSimilarity
        // between OSNet (512-dim) and MNV3 falls through the size check and
        // returns 0, so the body half of this filter was silently dead.
        val bodySim = lockedReIdEmbedding?.let { cosineSimilarity(it, body) } ?: 0f
        val faceSim = lockedFaceEmbedding?.let { cosineSimilarity(it, face) } ?: 0f
        if (bodySim >= 0.85f || faceSim >= 0.6f) {
            Log.d(TAG, "Scene pair rejected: bodySim=${"%.3f".format(bodySim)} faceSim=${"%.3f".format(faceSim)} (looks like lock)")
            return
        }
        if (_sceneFacePairs.size >= MAX_SCENE_FACE_PAIRS) {
            _sceneFacePairs.removeAt(0)
        }
        _sceneFacePairs.add(ScenePersonPair(face.copyOf(), body.copyOf()))
        _liveStatsDirty = true
        Log.d(TAG, "Scene pair added: bodySim=${"%.3f".format(bodySim)} faceSim=${"%.3f".format(faceSim)} — total=${_sceneFacePairs.size}")
    }

    /**
     * Best face-cosine of [candidateFace] against any stored scene other.
     * Returns 0 if no stored faces. Used in [scoreCandidate] for the
     * face-recognized-as-known-other veto.
     */
    internal fun bestSceneFaceMatch(candidateFace: FloatArray): Float {
        if (_sceneFacePairs.isEmpty()) return 0f
        var best = 0f
        for (p in _sceneFacePairs) {
            val s = cosineSimilarity(p.face, candidateFace)
            if (s > best) best = s
        }
        return best
    }

    /**
     * Best body-cosine of [candidateBody] against any stored scene-other body.
     * Used for the no-face veto path: if body sim to a stored other exceeds
     * body sim to the lock by [SCENE_BODY_VETO_MARGIN], reject.
     */
    internal fun bestSceneBodyMatch(candidateBody: FloatArray): Float {
        if (_sceneFacePairs.isEmpty()) return 0f
        var best = 0f
        for (p in _sceneFacePairs) {
            val s = cosineSimilarity(p.body, candidateBody)
            if (s > best) best = s
        }
        return best
    }

    /** Prototype margin: sim(candidate, pos_centroid) - sim(candidate, neg_centroid).
     *  Returns margin in [-1, 1]. Positive = closer to locked object than to scene.
     *  Returns 0 when no negatives exist (no discrimination possible). */
    private fun prototypeMargin(candidateEmbedding: FloatArray): Float {
        val negCentroid = _negativeCentroid ?: return 0f
        val posCentroid = _embeddingCentroid ?: return 0f
        val posSim = cosineSimilarity(candidateEmbedding, posCentroid)
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
        _negativeCentroid = null
        _sceneFacePairs.clear()
        _classifier.clear()
        _lastTrainPositives = 0
        _lastTrainNegatives = 0
        lockedColorHistogram = colorHist?.copyOf()
        lockedPersonAttributes = personAttrs
        lockedReIdEmbedding = reIdEmbedding?.copyOf()
        lockedFaceEmbedding = faceEmbedding?.copyOf()
        _mnv3LiveStats = null
        _osnetLiveStats = null
        _faceLiveStats = null
        _liveStatsDirty = true
        lastKnownBox = RectF(boundingBox)
        lastKnownLabel = label
        lastKnownSize = boundingBox.width() * boundingBox.height()
        lastKnownVelocityX = 0f
        lastKnownVelocityY = 0f
        framesLost = 0
        tentativeBox = null
        tentativeCount = 0

        // Derive per-lock adaptive floors from the gallery → frozen-negatives
        // similarity distribution. Floor sits at "mean + 0.5σ" of false-match
        // similarities — i.e. just above the typical random-scene noise level
        // for THIS lock's gallery. Clamped to a sane band.
        _adaptiveMobileNetFloor = if (frozenNegativesMobileNet.isNotEmpty() && _embeddingGallery.isNotEmpty()) {
            computeAdaptiveFloor(_embeddingGallery, frozenNegativesMobileNet)
                .coerceIn(0.20f, 0.55f)
        } else null
        _adaptiveOsnetFloor = if (frozenNegativesOsnet.isNotEmpty() && lockedReIdEmbedding != null) {
            computeAdaptiveFloor(listOf(lockedReIdEmbedding!!), frozenNegativesOsnet)
                .coerceIn(0.30f, 0.60f)
        } else null

        // Cold-start the OnlineClassifier with frozen negatives so it has a
        // decision boundary from frame 1 of search, not after waiting for
        // ≥3 confirmed scene negatives during VT tracking.
        // Subsample to CLASSIFIER_COLD_START_NEGATIVES to bound class imbalance
        // (5 pos : 1500 neg → 5 pos : 100 neg). Deterministic shuffle seeded
        // by trackingId so the same lock always picks the same subset (test
        // reproducibility). The full frozen pool is still used for the adaptive
        // floor — only classifier training is bounded.
        if (frozenNegativesMobileNet.isNotEmpty() && _embeddingGallery.size >= 3) {
            val negSample = if (frozenNegativesMobileNet.size > CLASSIFIER_COLD_START_NEGATIVES) {
                frozenNegativesMobileNet.shuffled(java.util.Random(trackingId.toLong()))
                    .take(CLASSIFIER_COLD_START_NEGATIVES)
            } else frozenNegativesMobileNet
            _classifier.train(_embeddingGallery, negSample)
            if (_classifier.isTrained) {
                _lastTrainPositives = _embeddingGallery.size
                _lastTrainNegatives = negSample.size
            }
        }

        val attrStr = personAttrs?.summary() ?: "n/a"
        val floorStr = buildString {
            _adaptiveMobileNetFloor?.let { append(" mnv3Floor=${fmtF(it)}") }
            _adaptiveOsnetFloor?.let { append(" osnetFloor=${fmtF(it)}") }
        }
        dualLog(Log.INFO, "LOCK id=$trackingId label=\"$label\" box=${fmtBox(boundingBox)} size=${fmtF(lastKnownSize)} gallery=${embeddingGallery.size} colorHist=${colorHist != null} attrs=\"$attrStr\"$floorStr")
    }

    /** Add a face embedding progressively (e.g. when face first appears during tracking). */
    fun addFaceEmbedding(embedding: FloatArray) {
        if (lockedFaceEmbedding == null) {
            lockedFaceEmbedding = embedding.copyOf()
            _liveStatsDirty = true
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
        _liveStatsDirty = true
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
        _adaptiveMobileNetFloor = null
        _adaptiveOsnetFloor = null
        _mnv3LiveStats = null
        _osnetLiveStats = null
        _faceLiveStats = null
        _liveStatsDirty = true
        _negativeExamples.clear()
        _negativeCentroid = null
        _sceneFacePairs.clear()
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
        val strongMatch = overridePasses(
            candidate, sim,
            zThresh = Z_TENTATIVE_BYPASS_THRESHOLD,
            rawThresh = APPEARANCE_OVERRIDE_THRESHOLD,
        )

        // --- Tentative confirmation (DeepSORT-style) ---
        // Don't commit on a single frame. Require the same detection to win
        // for TENTATIVE_MIN_FRAMES consecutive frames.
        //
        // Tentative bypass logic (Phase 3 #102 — z-score preferred, raw fallback):
        //   - Strong z-match (z ≥ 1.5) OR raw fallback sim ≥ 0.7: always bypass
        //   - Classifier trained + confident (P ≥ 0.8): bypass — learned boundary says yes
        //   - Classifier trained + uncertain (P < 0.8): require tentative
        //   - Classifier NOT trained: bypass if z ≥ 1.0 OR raw fallback sim ≥ 0.55
        val clsP = if (_classifier.isTrained && candidate.embedding != null)
            _classifier.predict(candidate.embedding!!) else -1f
        val geometricOverrideForTentative = overridePasses(
            candidate, sim,
            zThresh = Z_GEOMETRIC_OVERRIDE_THRESHOLD,
            rawThresh = GEOMETRIC_OVERRIDE_THRESHOLD,
        )
        val skipTentative = strongMatch ||
            (clsP >= 0.8f) ||
            (clsP < 0f && hasEmbeddings && geometricOverrideForTentative)
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
        // Z-norm diagnostics (#102 Phase 1) — null when cohort below MIN_COHORT_FOR_ZNORM.
        val znStr = buildString {
            candidate.embedding?.let { znMnv3Sim(it) }?.let { append(" znMnv3=${fmtF(it)}") }
            candidate.reIdEmbedding?.let { znOsnetSim(it) }?.let { append(" znOsnet=${fmtF(it)}") }
            candidate.faceEmbedding?.let { znFaceSim(it) }?.let { append(" znFace=${fmtF(it)}") }
        }
        dualLog(Log.INFO, "REACQUIRE id=${candidate.id} label=\"${candidate.label}\" after $framesLost frames (lockedLabel=\"$lockedLabel\") sim=${fmtF(sim)}$reIdSim$faceSim$znStr box=${fmtBox(candidate.boundingBox)}")
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
                    // Z-norm diagnostics (#102 Phase 1) — null when cohort < MIN_COHORT_FOR_ZNORM.
                    val znMnv3 = candidate.embedding?.let { znMnv3Sim(it) }
                    val znOsnet = candidate.reIdEmbedding?.let { znOsnetSim(it) }
                    val znFace = candidate.faceEmbedding?.let { znFaceSim(it) }
                    val znStr = buildString {
                        znMnv3?.let { append(" znMnv3=${fmtF(it)}") }
                        znOsnet?.let { append(" znOsnet=${fmtF(it)}") }
                        znFace?.let { append(" znFace=${fmtF(it)}") }
                    }
                    dualLog(Log.DEBUG, "  scored id=${candidate.id} label=\"${candidate.label}\" score=${fmtF(score)} sim=${sim?.let { fmtF(it) } ?: "n/a"} color=${colorSim?.let { fmtF(it) } ?: "n/a"}$attrStr$reIdStr$faceStr$marginStr$clsStr$znStr (min=${fmtF(minScoreThreshold)})")
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
        // Compares embedding similarity (not final score) intentionally: the ratio test
        // asks "can the embedder tell these apart?" not "do other signals differ?"
        // Two candidates with similar embedding but different color/position are still
        // ambiguous from an identity standpoint — other signals are noisy tiebreakers.
        // Uses the SAME embedding signal that drove the gate (OSNet for person-person,
        // MobileNetV3 elsewhere). Otherwise ambiguity-by-MobileNetV3 could falsely
        // reject pairs that OSNet clearly distinguishes (#67 review).
        if (sortedScored.size >= 2 && hasEmbeddings) {
            val bestSim = effectiveAppearanceSim(best.first)
            val secondSim = effectiveAppearanceSim(sortedScored[1].first)
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
        // For person candidates with OSNet on both sides, gate on OSNet (a real
        // re-ID model) instead of MobileNetV3 (a generic ImageNet classifier
        // that's blind to person-instance identity). MobileNetV3 still drives
        // the ranking signals for non-person candidates and for persons that
        // didn't get OSNet computed.
        val candidateIsPerson = candidate.label == "person"
        val hasReId = lockedIsPerson && candidateIsPerson &&
            lockedReIdEmbedding != null && candidate.reIdEmbedding != null
        val reIdScore = if (hasReId) {
            cosineSimilarity(lockedReIdEmbedding!!, candidate.reIdEmbedding!!).coerceIn(0f, 1f)
        } else 0f

        val hasAppearance = hasEmbeddings && candidate.embedding != null
        val mobileNetScore = if (hasAppearance) {
            bestGallerySimilarity(candidate.embedding!!).coerceIn(0f, 1f)
        } else 0f

        val appearanceScore = if (hasReId) reIdScore else mobileNetScore
        val gateActive = hasReId || hasAppearance

        // --- GATE: Require embedding when gallery exists ---
        // If we have reference embeddings but the candidate has none (async not ready),
        // reject it — accepting without identity verification causes wrong-object locks.
        // Exception: if hasReId, OSNet alone is sufficient for the gate (the
        // candidate has been identified by a real re-ID model).
        if (hasEmbeddings && candidate.embedding == null && !hasReId) {
            return null
        }
        val galleryMature = _embeddingGallery.size >= 8

        // --- GATE: Embedding floor ---
        // Prefer the per-lock adaptive floor derived from gallery → frozen-negatives
        // distribution at lock time (#68). Falls back to static floors when no
        // frozen-negatives asset is loaded.
        val embeddingFloor = if (hasReId) {
            _adaptiveOsnetFloor ?: PERSON_REID_FLOOR
        } else {
            _adaptiveMobileNetFloor ?: run {
                val adaptiveFloor = (_minGallerySim * 0.75f).coerceIn(0.3f, 0.5f)
                if (galleryMature) maxOf(adaptiveFloor, 0.4f) else adaptiveFloor
            }
        }
        // Don't poison the negative pool with rejections during search — when
        // the floor is misconfigured (failure case), the locked person gets
        // repeatedly rejected and added as a negative, training the prototype
        // margin/classifier to actively reject the right answer next time.
        // Negatives are still collected during confirmed VT tracking (addSceneNegative
        // with its 0.85 filter handles that).
        if (gateActive && appearanceScore < embeddingFloor) {
            return null
        }

        // --- GATE: Face identity (#83) ---
        // When both lock and candidate have face embeddings (person-person only),
        // face vetoes OSNet. MobileFaceNet has cleaner same/different-person
        // separation than OSNet's whole-body re-ID — a face sim < FACE_FLOOR is
        // strong evidence of a different person, regardless of body match.
        // Doesn't affect candidates without face data (rejection path goes
        // through OSNet alone in those cases).
        val hasFaceGate = lockedIsPerson && candidateIsPerson &&
            lockedFaceEmbedding != null && candidate.faceEmbedding != null
        val lockFaceSim = if (hasFaceGate) {
            cosineSimilarity(lockedFaceEmbedding!!, candidate.faceEmbedding!!).coerceIn(0f, 1f)
        } else 0f
        if (hasFaceGate) {
            if (lockFaceSim < FACE_FLOOR) {
                dualLog(Log.DEBUG, "  FACE_GATE_REJECT: faceSim=${fmtF(lockFaceSim)} < ${fmtF(FACE_FLOOR)} (reIdSim=${fmtF(reIdScore)})")
                return null
            }
        }

        // --- GATE: Scene face-body memory veto (#83 phase 2) ---
        // Use the paired (face, body) memory of non-lock persons observed during
        // VT-confirmed tracking to reject candidates that look more like a known
        // OTHER person than like the lock. Two paths:
        //
        // (a) Face-known: candidate's face matches a stored other-person face
        //     better than it matches the lock face. Only fires when both
        //     candidate face and lockedFaceEmbedding exist (else lock-face
        //     baseline is missing).
        // (b) Body-known: candidate has no face but its body matches a stored
        //     other-person body better than the lock body by SCENE_BODY_VETO_MARGIN.
        //     Pays off when face was visible earlier (so the body got linked)
        //     but isn't visible now.
        if (lockedIsPerson && candidateIsPerson && _sceneFacePairs.isNotEmpty()) {
            // (a) Face path
            if (hasFaceGate) {
                val sceneFaceMatch = bestSceneFaceMatch(candidate.faceEmbedding!!)
                if (sceneFaceMatch >= SCENE_FACE_MATCH_FLOOR &&
                    sceneFaceMatch > lockFaceSim + SCENE_FACE_VETO_MARGIN) {
                    dualLog(Log.DEBUG, "  SCENE_FACE_VETO: sceneFace=${fmtF(sceneFaceMatch)} > lockFace=${fmtF(lockFaceSim)}+${fmtF(SCENE_FACE_VETO_MARGIN)} (reIdSim=${fmtF(reIdScore)})")
                    return null
                }
            }
            // (b) Body path — only when face data isn't available to drive (a)
            if (!hasFaceGate && candidate.reIdEmbedding != null && lockedReIdEmbedding != null) {
                val sceneBodyMatch = bestSceneBodyMatch(candidate.reIdEmbedding!!)
                val lockBodyMatch = reIdScore  // already computed against locked re-ID
                if (sceneBodyMatch > lockBodyMatch + SCENE_BODY_VETO_MARGIN) {
                    dualLog(Log.DEBUG, "  SCENE_BODY_VETO: sceneBody=${fmtF(sceneBodyMatch)} > lockBody=${fmtF(lockBodyMatch)}+${fmtF(SCENE_BODY_VETO_MARGIN)}")
                    return null
                }
            }
        }

        // Tiered override (Phase 3 #102): z-score preferred, raw cosine as fallback.
        // Geometric gates (position/size) admit lower z (≥1.0) because position
        // rejection is about camera movement, not identity. Label gate uses a
        // stricter z (≥1.5) since cross-category confusion is high-risk.
        // [overridePasses] picks the OSNet z for person-person and the MNV3 z
        // otherwise, mirroring [effectiveAppearanceSim].
        val geometricOverride = gateActive && overridePasses(
            candidate, appearanceScore,
            zThresh = Z_GEOMETRIC_OVERRIDE_THRESHOLD,
            rawThresh = GEOMETRIC_OVERRIDE_THRESHOLD,
        )
        val labelOverride = gateActive && overridePasses(
            candidate, appearanceScore,
            zThresh = Z_LABEL_OVERRIDE_THRESHOLD,
            rawThresh = APPEARANCE_OVERRIDE_THRESHOLD,
        )

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
        // (candidateIsPerson computed earlier; reused here.)
        if (lockedIsPerson != candidateIsPerson && !labelOverride) {
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

        // Re-ID score (hasReId / reIdScore) computed earlier for the OSNet gate;
        // reused here as the dominant ranking signal for person candidates.

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

    /**
     * Best cosine similarity against the **lock-time** portion of the gallery only
     * (the first `LOCK_AUGMENTATION_COUNT` entries — original + rotations + flip).
     * Used by template self-verification (#80): if drift detection used the full
     * gallery, accumulated-during-tracking entries can include drifted VT crops,
     * which then match the (also drifted) VT crop perfectly and mask drift.
     * The lock-time portion is fixed and trustworthy.
     */
    internal fun bestLockGallerySimilarity(candidateEmbedding: FloatArray): Float {
        if (_embeddingGallery.isEmpty()) return 0f
        val lockEntries = _embeddingGallery.take(LOCK_AUGMENTATION_COUNT)
        return bestGallerySimilarity(candidateEmbedding, lockEntries)
    }

    /**
     * Adaptive floor: for each negative, find its closest match in the positive
     * gallery (same operation the gate runs on real candidates). Return
     * `mean + 0.5σ` of that distribution — the floor that typical random-scene
     * stuff would clear by chance. Anything above this is meaningfully more
     * similar than scene noise.
     */
    private fun computeAdaptiveFloor(positives: List<FloatArray>, negatives: List<FloatArray>): Float {
        val s = computeImpostorStats(positives, negatives) ?: return Float.NaN
        return s.mean + 0.5f * s.std
    }

    /**
     * Single-template variant: compares each negative directly to one anchor
     * embedding (no gallery). Used for OSNet / face where the lock side is a
     * single embedding rather than an augmented gallery.
     */
    private fun computeAdaptiveFloor(anchor: FloatArray, negatives: List<FloatArray>): Float {
        val s = computeImpostorStats(anchor, negatives) ?: return Float.NaN
        return s.mean + 0.5f * s.std
    }

    /**
     * Per-modality impostor distribution used both for the adaptive floor
     * (mean + 0.5σ → gate threshold) and for live z-score normalization
     * (#102). The stats describe the empirical noise floor between the
     * locked identity and known-impostor embeddings, so z = (rawSim − mean) / std
     * tells us how many σ above the noise floor a given candidate's match sits.
     */
    data class ImpostorStats(val mean: Float, val std: Float, val n: Int)

    /** Gallery variant: for each negative, scores `bestGallerySimilarity(neg, gallery)`. */
    private fun computeImpostorStats(
        gallery: List<FloatArray>,
        negatives: List<FloatArray>,
    ): ImpostorStats? {
        if (gallery.isEmpty() || negatives.isEmpty()) return null
        val sims = FloatArray(negatives.size) { i ->
            bestGallerySimilarity(negatives[i], gallery)
        }
        return finishStats(sims)
    }

    /** Single-anchor variant: for each negative, scores `cosineSimilarity(anchor, neg)`. */
    private fun computeImpostorStats(
        anchor: FloatArray?,
        negatives: List<FloatArray>,
    ): ImpostorStats? {
        if (anchor == null || negatives.isEmpty()) return null
        val sims = FloatArray(negatives.size) { i ->
            cosineSimilarity(anchor, negatives[i])
        }
        return finishStats(sims)
    }

    private fun finishStats(sims: FloatArray): ImpostorStats? {
        if (sims.isEmpty()) return null
        var sum = 0f
        for (s in sims) sum += s
        val mean = sum / sims.size
        var sqSum = 0f
        for (s in sims) sqSum += (s - mean) * (s - mean)
        val std = kotlin.math.sqrt(sqSum / sims.size)
        return ImpostorStats(mean, std, sims.size)
    }

    /**
     * Effective appearance similarity used by the gate, scoring, and ratio test.
     * Matches the OSNet-vs-MobileNetV3 split in [scoreCandidate]: person-person
     * comparisons (both sides have OSNet) use OSNet cosine; everything else
     * uses MobileNetV3 against the gallery.
     */
    private fun effectiveAppearanceSim(candidate: TrackedObject): Float {
        val candidateIsPerson = candidate.label == "person"
        val hasReId = lockedIsPerson && candidateIsPerson &&
            lockedReIdEmbedding != null && candidate.reIdEmbedding != null
        return when {
            hasReId -> cosineSimilarity(lockedReIdEmbedding!!, candidate.reIdEmbedding!!).coerceIn(0f, 1f)
            hasEmbeddings && candidate.embedding != null ->
                bestGallerySimilarity(candidate.embedding!!).coerceIn(0f, 1f)
            else -> 0f
        }
    }

    /** Log to both logcat and session file. */
    private fun dualLog(level: Int, msg: String) {
        Log.println(level, TAG, msg)
        sessionLogger?.invoke(msg)
    }

    private fun fmtF(f: Float) = "%.3f".format(f)
    private fun fmtBox(b: RectF) = "[%.2f,%.2f,%.2f,%.2f]".format(b.left, b.top, b.right, b.bottom)
}
