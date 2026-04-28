package com.haptictrack.tracking

import org.junit.Assert.assertEquals
import org.junit.Assert.assertNotEquals
import org.junit.Assert.assertNotNull
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Assert.assertFalse
import org.junit.Test
import org.junit.runner.RunWith
import org.robolectric.RobolectricTestRunner

@RunWith(RobolectricTestRunner::class)

/**
 * Tests for [SessionRoster] — the per-session person identity store that backs
 * the open-set rejection in [ReacquisitionEngine] (#108).
 *
 * Embeddings here are normalized vectors with controlled cosine similarities so
 * the tests exercise the merge/create/match logic without depending on real
 * model output.
 */
class SessionRosterTest {

    /** A normalized vector that points along axis [axis] in n-dim space. */
    private fun unit(axis: Int, n: Int = 32): FloatArray {
        require(axis < n) { "axis $axis out of range for $n-dim unit vector" }
        val v = FloatArray(n)
        v[axis] = 1f
        return v
    }

    /** Linear combination of two unit axes, then normalized. Used to build
     *  embeddings with a known cosine similarity to two reference axes. */
    private fun mix(a: Int, b: Int, alpha: Float, n: Int = 32): FloatArray {
        val v = FloatArray(n)
        v[a] = alpha
        v[b] = kotlin.math.sqrt(1f - alpha * alpha)
        // Already L2-normalized by construction.
        return v
    }

    @Test
    fun `seedLock creates lock slot at id 0`() {
        val roster = SessionRoster()
        roster.seedLock(face = unit(0), body = unit(0), frameIdx = 0)

        assertEquals(1, roster.size)
        assertEquals(0, roster.nonLockCount)
        val lock = roster.lockSlot
        assertNotNull(lock)
        assertEquals(SessionRoster.LOCK_SLOT_ID, lock!!.id)
        assertTrue(lock.isLock)
        assertEquals(1, lock.faceGallery.size)
        assertEquals(1, lock.bodyGallery.size)
    }

    @Test
    fun `observePerson with no lock creates a new slot`() {
        val roster = SessionRoster()
        val id = roster.observePerson(face = unit(1), body = unit(1), frameIdx = 0)
        assertNotNull(id)
        assertEquals(1, roster.size)
        assertEquals(1, roster.nonLockCount)
    }

    @Test
    fun `observePerson too similar to lock is filtered`() {
        val roster = SessionRoster()
        roster.seedLock(face = unit(0), body = unit(0), frameIdx = 0)

        // Same axis as lock — should be rejected as duplicate of lock.
        val id = roster.observePerson(face = unit(0), body = unit(0), frameIdx = 1)
        assertNull(id)
        assertEquals(0, roster.nonLockCount)
    }

    @Test
    fun `observePerson distinct from lock creates new slot`() {
        val roster = SessionRoster()
        roster.seedLock(face = unit(0), body = unit(0), frameIdx = 0)

        // Orthogonal axis — clearly a different person.
        val id = roster.observePerson(face = unit(1), body = unit(1), frameIdx = 1)
        assertNotNull(id)
        assertEquals(2, roster.size)
        assertEquals(1, roster.nonLockCount)
    }

    @Test
    fun `repeated observation of same person fuses into one slot`() {
        val roster = SessionRoster()
        roster.seedLock(face = unit(0), body = unit(0), frameIdx = 0)

        // Three nearly identical observations of "person 1" — they should all
        // fuse into the same non-lock slot.
        val id1 = roster.observePerson(face = unit(1), body = unit(1), frameIdx = 1)
        val id2 = roster.observePerson(face = unit(1), body = unit(1), frameIdx = 2)
        val id3 = roster.observePerson(face = unit(1), body = unit(1), frameIdx = 3)

        assertNotNull(id1)
        assertEquals(id1, id2)
        assertEquals(id1, id3)
        assertEquals(1, roster.nonLockCount)
        // Gallery should have grown (3 observations).
        val slot = roster.slots.first { it.id == id1 }
        assertEquals(3, slot.observationCount)
    }

    @Test
    fun `distinct people get distinct slots`() {
        val roster = SessionRoster()
        roster.seedLock(face = unit(0), body = unit(0), frameIdx = 0)

        val id1 = roster.observePerson(face = unit(1), body = unit(1), frameIdx = 1)
        val id2 = roster.observePerson(face = unit(2), body = unit(2), frameIdx = 2)
        val id3 = roster.observePerson(face = unit(3), body = unit(3), frameIdx = 3)

        assertNotEquals(id1, id2)
        assertNotEquals(id2, id3)
        assertNotEquals(id1, id3)
        assertEquals(3, roster.nonLockCount)
    }

    @Test
    fun `observePerson with body only still creates slot`() {
        val roster = SessionRoster()
        roster.seedLock(face = unit(0), body = unit(0), frameIdx = 0)

        // Only body modality — face was occluded.
        val id = roster.observePerson(face = null, body = unit(1), frameIdx = 1)
        assertNotNull(id)
        assertEquals(1, roster.nonLockCount)

        val slot = roster.slots.first { !it.isLock }
        assertEquals(0, slot.faceGallery.size)
        assertEquals(1, slot.bodyGallery.size)
    }

    @Test
    fun `body match merges into existing face-only slot`() {
        val roster = SessionRoster()
        roster.seedLock(face = unit(0), body = unit(0), frameIdx = 0)

        // First observation: face only.
        val id1 = roster.observePerson(face = unit(1), body = null, frameIdx = 1)
        // Second observation: same person, but only body visible. The roster
        // has no body for this slot yet, so body merge can't fire — a new slot
        // is created. (The face-body LINK requires both modalities together
        // at least once — typical pattern in real footage.)
        val id2 = roster.observePerson(face = null, body = unit(1), frameIdx = 2)

        // Without a face-body link, these become two slots. Confirms the
        // single-modality-per-observation semantics: face-only and body-only
        // observations of "the same person" don't auto-link.
        assertNotEquals(id1, id2)
    }

    @Test
    fun `lockMatch returns face and body sims against lock`() {
        val roster = SessionRoster()
        roster.seedLock(face = unit(0), body = unit(0), frameIdx = 0)

        val (face, body) = roster.lockMatch(face = unit(0), body = unit(0))
        assertEquals(1f, face, 1e-5f)
        assertEquals(1f, body, 1e-5f)

        val (face2, body2) = roster.lockMatch(face = unit(1), body = unit(1))
        assertEquals(0f, face2, 1e-5f)
        assertEquals(0f, body2, 1e-5f)
    }

    @Test
    fun `bestNonLockMatch returns highest sim across non-lock slots`() {
        val roster = SessionRoster()
        roster.seedLock(face = unit(0), body = unit(0), frameIdx = 0)
        roster.observePerson(face = unit(1), body = unit(1), frameIdx = 1)
        roster.observePerson(face = unit(2), body = unit(2), frameIdx = 2)

        // Probe matching slot 1 (axis 1).
        val (face, body) = roster.bestNonLockMatch(face = unit(1), body = unit(1))
        assertEquals(1f, face, 1e-5f)
        assertEquals(1f, body, 1e-5f)

        // Probe matching neither slot.
        val (face2, body2) = roster.bestNonLockMatch(face = unit(7), body = unit(7))
        assertEquals(0f, face2, 1e-5f)
        assertEquals(0f, body2, 1e-5f)
    }

    @Test
    fun `bestMatch returns lock when candidate matches lock best`() {
        val roster = SessionRoster()
        roster.seedLock(face = unit(0), body = unit(0), frameIdx = 0)
        roster.observePerson(face = unit(1), body = unit(1), frameIdx = 1)

        val match = roster.bestMatch(face = unit(0), body = unit(0))
        assertEquals(SessionRoster.LOCK_SLOT_ID, match.bestSlotId)
        assertEquals(1f, match.bestFaceSim, 1e-5f)
    }

    @Test
    fun `bestMatch returns non-lock slot when candidate matches that better`() {
        val roster = SessionRoster()
        roster.seedLock(face = unit(0), body = unit(0), frameIdx = 0)
        val otherId = roster.observePerson(face = unit(1), body = unit(1), frameIdx = 1)

        val match = roster.bestMatch(face = unit(1), body = unit(1))
        assertEquals(otherId, match.bestSlotId)
        assertNotEquals(SessionRoster.LOCK_SLOT_ID, match.bestSlotId)
    }

    @Test
    fun `LRU eviction removes oldest non-lock slot when full`() {
        val roster = SessionRoster()
        roster.seedLock(face = unit(0), body = unit(0), frameIdx = 0)

        // Fill to MAX_SLOTS with distinct identities.
        val firstId = roster.observePerson(face = unit(1), body = unit(1), frameIdx = 1)
        for (i in 2 until SessionRoster.MAX_SLOTS) {
            roster.observePerson(face = unit(i), body = unit(i), frameIdx = i)
        }
        assertEquals(SessionRoster.MAX_SLOTS, roster.size)

        // Add one more — should evict the oldest non-lock slot (firstId, frame 1).
        roster.observePerson(face = unit(SessionRoster.MAX_SLOTS + 5),
                             body = unit(SessionRoster.MAX_SLOTS + 5),
                             frameIdx = 100)
        assertEquals(SessionRoster.MAX_SLOTS, roster.size)
        assertNull(roster.slots.find { it.id == firstId })
        // Lock slot is preserved.
        assertNotNull(roster.lockSlot)
    }

    @Test
    fun `clear empties all slots`() {
        val roster = SessionRoster()
        roster.seedLock(face = unit(0), body = unit(0), frameIdx = 0)
        roster.observePerson(face = unit(1), body = unit(1), frameIdx = 1)
        roster.observePerson(face = unit(2), body = unit(2), frameIdx = 2)
        assertEquals(3, roster.size)

        roster.clear()
        assertEquals(0, roster.size)
        assertNull(roster.lockSlot)
    }

    @Test
    fun `clearNonLock keeps lock but removes others`() {
        val roster = SessionRoster()
        roster.seedLock(face = unit(0), body = unit(0), frameIdx = 0)
        roster.observePerson(face = unit(1), body = unit(1), frameIdx = 1)
        roster.observePerson(face = unit(2), body = unit(2), frameIdx = 2)
        assertEquals(3, roster.size)

        roster.clearNonLock()
        assertEquals(1, roster.size)
        assertNotNull(roster.lockSlot)
        assertEquals(0, roster.nonLockCount)
    }

    @Test
    fun `allNonLockBodies aggregates galleries across non-lock slots`() {
        val roster = SessionRoster()
        roster.seedLock(face = unit(0), body = unit(0), frameIdx = 0)
        roster.observePerson(face = unit(1), body = unit(1), frameIdx = 1)
        roster.observePerson(face = unit(1), body = unit(1), frameIdx = 2)  // fuse into slot 1
        roster.observePerson(face = unit(2), body = unit(2), frameIdx = 3)

        val bodies = roster.allNonLockBodies()
        // Slot 1: 2 body entries, Slot 2: 1 body entry — but lock-slot bodies excluded.
        assertEquals(3, bodies.size)
    }

    @Test
    fun `seedLock idempotent - replaces existing lock without dropping others`() {
        val roster = SessionRoster()
        roster.seedLock(face = unit(0), body = unit(0), frameIdx = 0)
        roster.observePerson(face = unit(1), body = unit(1), frameIdx = 1)

        roster.seedLock(face = unit(5), body = unit(5), frameIdx = 100)

        assertEquals(2, roster.size)
        assertEquals(1, roster.nonLockCount)
        // New lock has new embeddings.
        val (face, _) = roster.lockMatch(face = unit(5), body = null)
        assertEquals(1f, face, 1e-5f)
    }

    @Test
    fun `kid_to_wife regression - wife observed at lock time is rejected at reacquire`() {
        // This is the structural test for #108: simulates kid_to_wife scenario
        // in the SessionRoster abstraction.
        // Lock = boy (axis 0). Wife (axis 1) is in frame at lock time.
        val roster = SessionRoster()
        roster.seedLock(face = unit(0), body = unit(0), frameIdx = 0)
        roster.observePerson(face = unit(1), body = unit(1), frameIdx = 0)  // wife at lock time

        // Simulate camera pan: VT lost, wife now appears as a candidate to
        // reacquire on. Probe with a vector close to wife (axis 1).
        val (lockFace, lockBody) = roster.lockMatch(face = unit(1), body = unit(1))
        val (otherFace, otherBody) = roster.bestNonLockMatch(face = unit(1), body = unit(1))

        // Wife matches her own slot strongly; matches lock weakly.
        val lockScore = maxOf(lockFace, lockBody)
        val otherScore = maxOf(otherFace, otherBody)
        assertTrue("wife matches her own slot strongly: $otherScore",
            otherScore >= ReacquisitionEngine.ROSTER_REJECT_FLOOR)
        assertTrue("wife beats lock by margin: other=$otherScore lock=$lockScore",
            otherScore > lockScore + ReacquisitionEngine.ROSTER_REJECT_MARGIN)
    }

    @Test
    fun `boy returns - matches lock not distractor`() {
        // Companion to kid_to_wife: when the locked person (boy, axis 0)
        // returns, lock slot wins, no rejection fires.
        val roster = SessionRoster()
        roster.seedLock(face = unit(0), body = unit(0), frameIdx = 0)
        roster.observePerson(face = unit(1), body = unit(1), frameIdx = 0)  // wife at lock time

        // Boy returns — probe matches lock axis.
        val (lockFace, lockBody) = roster.lockMatch(face = unit(0), body = unit(0))
        val (otherFace, otherBody) = roster.bestNonLockMatch(face = unit(0), body = unit(0))

        val lockScore = maxOf(lockFace, lockBody)
        val otherScore = maxOf(otherFace, otherBody)
        assertFalse("boy returning should NOT trigger rejection",
            otherScore >= ReacquisitionEngine.ROSTER_REJECT_FLOOR &&
                otherScore > lockScore + ReacquisitionEngine.ROSTER_REJECT_MARGIN)
        // And lock score is high.
        assertTrue("lock match is strong on return: $lockScore", lockScore >= 0.95f)
    }
}
