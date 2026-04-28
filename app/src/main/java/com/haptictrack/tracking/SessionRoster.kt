package com.haptictrack.tracking

import android.util.Log

/**
 * Per-session roster of distinct persons observed in the camera's view.
 *
 * Generalizes the older `_sceneFacePairs` memory (#83 phase 2): instead of a flat
 * list of paired (face, body) embeddings of non-lock persons, the roster holds
 * one [PersonTracklet] per *distinct identity* — fused by clustering on face/body
 * cosine similarity. Slot 0 is always the locked person; slots 1..N are other
 * persons seen in the session.
 *
 * The roster is the on-line distractor bank for re-acquisition (DaSiamRPN ECCV
 * 2018; DAM4SAM CVPR 2025). At reacquire time, a candidate's identity is judged
 * by RANK among all known identities + a margin to the second-best — not by an
 * absolute similarity to the lock alone. This sidesteps the cohort-poisoning
 * artifact that broke z-norm scoring on the kid_to_wife scenario (#102 Phase 4).
 *
 * Design choices:
 * - **Aggressive observation.** Every detected non-lock person on every frame
 *   feeds [observePerson]. The older memory only fired during VT-confirmed
 *   tracking and missed persons that entered the scene during pan (kid_to_wife).
 * - **Strong-modality merge.** Two slots are fused only when face ≥ FACE_MERGE
 *   OR body ≥ BODY_MERGE — a single modality must be confidently above the
 *   noise floor. Prevents the boy and the wife from collapsing into one slot.
 * - **Lock-near filter.** Observations whose face/body match the lock too
 *   strongly are dropped — they are duplicate detections of the lock itself.
 * - **Bounded gallery.** Each tracklet keeps at most [MAX_GALLERY_PER_SLOT]
 *   embeddings per modality (FIFO). The roster caps at [MAX_SLOTS] tracklets,
 *   evicting LRU when exceeded.
 *
 * Thread-safety: mutation methods ([seedLock], [augmentLock], [observePerson],
 * [clear], [clearNonLock]) are [Synchronized] on the instance — cheap insurance
 * since they're called from the main processing thread and (for [observePerson])
 * potentially the lock-burst executor. Read methods don't take the lock; they
 * iterate `_slots` directly and may briefly observe a partially-mutated state.
 * In practice all reads happen on the same processing thread that drives the
 * mutations, so this is safe.
 */
class SessionRoster {

    companion object {
        private const val TAG = "Roster"

        /** Slot 0 is reserved for the locked person. */
        const val LOCK_SLOT_ID = 0

        /** Maximum tracklets the roster will hold (incl. lock). LRU eviction past this. */
        const val MAX_SLOTS = 16

        /** Maximum face/body embeddings per tracklet (per modality). */
        const val MAX_GALLERY_PER_SLOT = 8

        /** Face cosine ≥ this fuses an observation into an existing tracklet. */
        const val FACE_MERGE_THRESHOLD = 0.55f

        /** Body (OSNet) cosine ≥ this fuses an observation into an existing tracklet.
         *  Higher than face because OSNet's same-person/different-person bands overlap
         *  (0.5–0.85 vs 0.2–0.4 cohort, but on-device noise floor is ~0.62–0.69). */
        const val BODY_MERGE_THRESHOLD = 0.70f

        /** Drop observation if face matches lock above this (likely duplicate of lock). */
        const val LOCK_NEAR_FACE = 0.60f

        /** Drop observation if body matches lock above this (likely duplicate of lock). */
        const val LOCK_NEAR_BODY = 0.85f
    }

    /**
     * One tracklet per distinct person observed. Slot 0 is always the lock.
     */
    data class PersonTracklet(
        val id: Int,
        val isLock: Boolean,
        val faceGallery: MutableList<FloatArray> = mutableListOf(),
        val bodyGallery: MutableList<FloatArray> = mutableListOf(),
        var firstSeenFrame: Int = 0,
        var lastSeenFrame: Int = 0,
        var observationCount: Int = 0,
    ) {
        fun bestFaceSim(face: FloatArray): Float {
            if (faceGallery.isEmpty()) return 0f
            var best = 0f
            for (f in faceGallery) {
                val s = cosineSimilarity(f, face)
                if (s > best) best = s
            }
            return best
        }

        fun bestBodySim(body: FloatArray): Float {
            if (bodyGallery.isEmpty()) return 0f
            var best = 0f
            for (b in bodyGallery) {
                val s = cosineSimilarity(b, body)
                if (s > best) best = s
            }
            return best
        }

        internal fun pushFace(face: FloatArray) {
            if (faceGallery.size >= MAX_GALLERY_PER_SLOT) faceGallery.removeAt(0)
            faceGallery.add(face.copyOf())
        }

        internal fun pushBody(body: FloatArray) {
            if (bodyGallery.size >= MAX_GALLERY_PER_SLOT) bodyGallery.removeAt(0)
            bodyGallery.add(body.copyOf())
        }
    }

    /** Result of [bestMatch]: best slot, its best per-modality similarities, and the runner-up. */
    data class MatchResult(
        val bestSlotId: Int?,
        val bestFaceSim: Float,
        val bestBodySim: Float,
        val secondBestSlotId: Int?,
        val secondBestFaceSim: Float,
        val secondBestBodySim: Float,
    )

    private val _slots = mutableListOf<PersonTracklet>()
    private var _nextId: Int = 0

    /** Snapshot — read-only view for tests and diagnostics. */
    val slots: List<PersonTracklet> get() = _slots
    val size: Int get() = _slots.size
    val lockSlot: PersonTracklet? get() = _slots.firstOrNull { it.isLock }

    /**
     * Initialize slot 0 with the locked person's face/body data. Idempotent —
     * a second call replaces the lock slot. Other slots are preserved.
     */
    @Synchronized
    fun seedLock(face: FloatArray?, body: FloatArray?, frameIdx: Int = 0) {
        // Remove any existing lock slot.
        _slots.removeAll { it.isLock }
        val tracklet = PersonTracklet(
            id = LOCK_SLOT_ID,
            isLock = true,
            firstSeenFrame = frameIdx,
            lastSeenFrame = frameIdx,
            observationCount = 1,
        )
        face?.let { tracklet.pushFace(it) }
        body?.let { tracklet.pushBody(it) }
        _slots.add(0, tracklet)
        _nextId = maxOf(_nextId, 1)
        Log.d(TAG, "seedLock: face=${face != null} body=${body != null}")
    }

    /**
     * Augment the lock slot with another (face, body) sample collected during
     * VT-confirmed tracking. Skips if the lock slot doesn't exist yet.
     */
    @Synchronized
    fun augmentLock(face: FloatArray?, body: FloatArray?, frameIdx: Int) {
        val lock = lockSlot ?: return
        face?.let { lock.pushFace(it) }
        body?.let { lock.pushBody(it) }
        lock.lastSeenFrame = frameIdx
        lock.observationCount++
    }

    /**
     * Observe a non-lock person detection. Either fuses into an existing tracklet
     * or creates a new one. Filters near-lock observations (likely duplicate
     * detections of the locked person).
     *
     * Returns the slot ID this observation was assigned to, or null if it was
     * filtered (lock-near or insufficient signal).
     */
    @Synchronized
    fun observePerson(face: FloatArray?, body: FloatArray?, frameIdx: Int): Int? {
        if (face == null && body == null) return null

        // Lock-near filter: if either modality matches the lock above the
        // duplicate threshold, drop. The lock gallery already covers this.
        val lock = lockSlot
        if (lock != null) {
            val lockFaceSim = if (face != null) lock.bestFaceSim(face) else 0f
            val lockBodySim = if (body != null) lock.bestBodySim(body) else 0f
            if (lockFaceSim >= LOCK_NEAR_FACE || lockBodySim >= LOCK_NEAR_BODY) {
                return null
            }
        }

        // Match against existing non-lock tracklets.
        val (matchSlot, matchSim) = findBestNonLockMatch(face, body)
        if (matchSlot != null) {
            face?.let { matchSlot.pushFace(it) }
            body?.let { matchSlot.pushBody(it) }
            matchSlot.lastSeenFrame = frameIdx
            matchSlot.observationCount++
            return matchSlot.id
        }

        // No match — create a new tracklet.
        return createSlot(face, body, frameIdx)
    }

    private fun findBestNonLockMatch(
        face: FloatArray?,
        body: FloatArray?,
    ): Pair<PersonTracklet?, Float> {
        var best: PersonTracklet? = null
        var bestStrength = 0f
        for (slot in _slots) {
            if (slot.isLock) continue
            val faceSim = if (face != null) slot.bestFaceSim(face) else 0f
            val bodySim = if (body != null) slot.bestBodySim(body) else 0f
            // A single modality crossing its merge threshold is enough to fuse.
            val faceStrong = face != null && slot.faceGallery.isNotEmpty() && faceSim >= FACE_MERGE_THRESHOLD
            val bodyStrong = body != null && slot.bodyGallery.isNotEmpty() && bodySim >= BODY_MERGE_THRESHOLD
            if (!faceStrong && !bodyStrong) continue
            // Merge strength = stronger of the two normalized signals.
            val strength = maxOf(faceSim, bodySim)
            if (strength > bestStrength) {
                bestStrength = strength
                best = slot
            }
        }
        return best to bestStrength
    }

    private fun createSlot(face: FloatArray?, body: FloatArray?, frameIdx: Int): Int {
        evictLruIfFull()
        val id = ++_nextId
        val tracklet = PersonTracklet(
            id = id,
            isLock = false,
            firstSeenFrame = frameIdx,
            lastSeenFrame = frameIdx,
            observationCount = 1,
        )
        face?.let { tracklet.pushFace(it) }
        body?.let { tracklet.pushBody(it) }
        _slots.add(tracklet)
        Log.d(TAG, "newSlot: id=$id face=${face != null} body=${body != null} (size=${_slots.size})")
        return id
    }

    private fun evictLruIfFull() {
        if (_slots.size < MAX_SLOTS) return
        // Find oldest non-lock slot by lastSeenFrame.
        val victim = _slots
            .filter { !it.isLock }
            .minByOrNull { it.lastSeenFrame }
            ?: return
        _slots.remove(victim)
        Log.d(TAG, "evict: id=${victim.id} (LRU, lastSeen=${victim.lastSeenFrame})")
    }

    /** Best face/body cosine of [face]/[body] against the lock slot. (0 if no lock.) */
    fun lockMatch(face: FloatArray?, body: FloatArray?): Pair<Float, Float> {
        val lock = lockSlot ?: return 0f to 0f
        val faceSim = if (face != null) lock.bestFaceSim(face) else 0f
        val bodySim = if (body != null) lock.bestBodySim(body) else 0f
        return faceSim to bodySim
    }

    /** Best face/body cosine of [face]/[body] against any non-lock slot,
     *  taken independently per modality. Useful for cohort statistics; NOT
     *  for reject-gate decisions where the per-modality maxima could come
     *  from different slots and inflate the apparent score. Use
     *  [bestNonLockSlotMatch] for the open-set rejection gate. */
    fun bestNonLockMatch(face: FloatArray?, body: FloatArray?): Pair<Float, Float> {
        var bestFace = 0f
        var bestBody = 0f
        for (slot in _slots) {
            if (slot.isLock) continue
            if (face != null) {
                val s = slot.bestFaceSim(face)
                if (s > bestFace) bestFace = s
            }
            if (body != null) {
                val s = slot.bestBodySim(body)
                if (s > bestBody) bestBody = s
            }
        }
        return bestFace to bestBody
    }

    /** A single non-lock slot's match data — face and body sims come from the
     *  SAME slot, so a downstream reject gate can reason about "this specific
     *  person beat the lock" rather than "max-of-maxima across all non-lock
     *  slots". */
    data class SlotMatch(val slotId: Int, val faceSim: Float, val bodySim: Float)

    /** Best non-lock slot's match against [face]/[body]. Score = max(faceSim,
     *  bodySim) per slot, single-slot winner returned with both modalities
     *  intact. Returns null when no non-lock slots exist. Used for the
     *  open-set rejection gate in [ReacquisitionEngine.scoreCandidate]. */
    fun bestNonLockSlotMatch(face: FloatArray?, body: FloatArray?): SlotMatch? {
        if (face == null && body == null) return null
        var winner: PersonTracklet? = null
        var winnerScore = -1f
        var winnerFace = 0f
        var winnerBody = 0f
        for (slot in _slots) {
            if (slot.isLock) continue
            val faceSim = if (face != null) slot.bestFaceSim(face) else 0f
            val bodySim = if (body != null) slot.bestBodySim(body) else 0f
            val score = maxOf(faceSim, bodySim)
            if (score > winnerScore) {
                winnerScore = score
                winnerFace = faceSim
                winnerBody = bodySim
                winner = slot
            }
        }
        return winner?.let { SlotMatch(it.id, winnerFace, winnerBody) }
    }

    /**
     * Full ranking: returns the best-matching slot and the runner-up. Score is
     * \max(faceSim, bodySim). Useful for open-set rejection ("does this candidate
     * match a non-lock slot better than the lock slot?").
     */
    fun bestMatch(face: FloatArray?, body: FloatArray?): MatchResult {
        if (face == null && body == null) return MatchResult(null, 0f, 0f, null, 0f, 0f)
        var bestSlot: PersonTracklet? = null
        var bestScore = -1f
        var bestFace = 0f
        var bestBody = 0f
        var secondSlot: PersonTracklet? = null
        var secondScore = -1f
        var secondFace = 0f
        var secondBody = 0f
        for (slot in _slots) {
            val faceSim = if (face != null) slot.bestFaceSim(face) else 0f
            val bodySim = if (body != null) slot.bestBodySim(body) else 0f
            val score = maxOf(faceSim, bodySim)
            if (score > bestScore) {
                secondSlot = bestSlot
                secondScore = bestScore
                secondFace = bestFace
                secondBody = bestBody
                bestSlot = slot
                bestScore = score
                bestFace = faceSim
                bestBody = bodySim
            } else if (score > secondScore) {
                secondSlot = slot
                secondScore = score
                secondFace = faceSim
                secondBody = bodySim
            }
        }
        return MatchResult(
            bestSlotId = bestSlot?.id,
            bestFaceSim = bestFace,
            bestBodySim = bestBody,
            secondBestSlotId = secondSlot?.id,
            secondBestFaceSim = secondFace,
            secondBestBodySim = secondBody,
        )
    }

    /** All face embeddings across non-lock slots — used as z-norm impostor cohort. */
    fun allNonLockFaces(): List<FloatArray> {
        val out = mutableListOf<FloatArray>()
        for (slot in _slots) {
            if (slot.isLock) continue
            out.addAll(slot.faceGallery)
        }
        return out
    }

    /** All body embeddings across non-lock slots — used as z-norm impostor cohort. */
    fun allNonLockBodies(): List<FloatArray> {
        val out = mutableListOf<FloatArray>()
        for (slot in _slots) {
            if (slot.isLock) continue
            out.addAll(slot.bodyGallery)
        }
        return out
    }

    /** Count of non-lock tracklets — for diagnostics/tests. */
    val nonLockCount: Int get() = _slots.count { !it.isLock }

    /** Drop everything. Called on lock clear / engine reset. */
    @Synchronized
    fun clear() {
        _slots.clear()
        _nextId = 0
    }

    /** Drop only non-lock slots. Useful when re-locking onto the same target. */
    @Synchronized
    fun clearNonLock() {
        _slots.removeAll { !it.isLock }
    }
}
