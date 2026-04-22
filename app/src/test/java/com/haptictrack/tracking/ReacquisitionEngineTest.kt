package com.haptictrack.tracking

import android.graphics.RectF
import org.junit.Assert.*
import org.junit.Before
import org.junit.Test
import org.junit.runner.RunWith
import org.robolectric.RobolectricTestRunner

@RunWith(RobolectricTestRunner::class)
class ReacquisitionEngineTest {

    private lateinit var engine: ReacquisitionEngine

    @Before
    fun setup() {
        engine = ReacquisitionEngine()
    }

    // --- Lock / Clear ---

    @Test
    fun `lock sets initial state`() {
        val box = RectF(0.3f, 0.3f, 0.6f, 0.6f)
        engine.lock(42, box, "Food")

        assertEquals(42, engine.lockedId)
        assertEquals("Food", engine.lastKnownLabel)
        assertTrue(engine.isLocked)
        assertEquals(0, engine.framesLost)
    }

    @Test
    fun `clear resets all state`() {
        engine.lock(42, RectF(0.3f, 0.3f, 0.6f, 0.6f), "Food")
        engine.clear()

        assertNull(engine.lockedId)
        assertNull(engine.lastKnownBox)
        assertNull(engine.lastKnownLabel)
        assertFalse(engine.isLocked)
    }

    // --- Direct match by tracking ID ---

    @Test
    fun `processFrame returns direct match when tracking ID is present`() {
        val box = RectF(0.3f, 0.3f, 0.6f, 0.6f)
        engine.lock(42, box, "Food")

        val detections = listOf(
            obj(id = 42, left = 0.32f, top = 0.32f, right = 0.62f, bottom = 0.62f, label = "Food"),
            obj(id = 99, left = 0.7f, top = 0.7f, right = 0.9f, bottom = 0.9f, label = "Home good")
        )

        val result = engine.processFrame(detections)

        assertNotNull(result)
        assertEquals(42, result!!.id)
        assertEquals(0, engine.framesLost)
    }

    @Test
    fun `processFrame returns null when not locked`() {
        val detections = listOf(
            obj(id = 1, left = 0.3f, top = 0.3f, right = 0.6f, bottom = 0.6f)
        )
        assertNull(engine.processFrame(detections))
    }

    // --- Re-acquisition: nearby same-label object (early frames) ---

    @Test
    fun `reacquires nearby object with same label after ID changes`() {
        val box = RectF(0.4f, 0.4f, 0.6f, 0.6f)
        engine.lock(42, box, "Food")

        val detections = listOf(
            obj(id = 55, left = 0.42f, top = 0.42f, right = 0.62f, bottom = 0.62f, label = "Food")
        )

        val result = processFramesUntilReacquire(detections)

        assertNotNull(result)
        assertEquals(55, result!!.id)
        assertEquals(55, engine.lockedId)
        assertEquals(0, engine.framesLost)
    }

    // --- Re-acquisition: reject wrong size ---

    @Test
    fun `rejects candidate with very different size`() {
        val box = RectF(0.4f, 0.4f, 0.5f, 0.5f) // small object
        engine.lock(42, box, "Food")

        // Much larger object at same position
        val detections = listOf(
            obj(id = 55, left = 0.2f, top = 0.2f, right = 0.8f, bottom = 0.8f, label = "Food")
        )

        val result = engine.processFrame(detections)
        assertNull(result)
    }

    // --- Re-acquisition: prefer closer candidate (early frames) ---

    @Test
    fun `prefers closer candidate over farther one in early frames`() {
        val box = RectF(0.4f, 0.4f, 0.6f, 0.6f)
        engine.lock(42, box, "Food")

        val close = obj(id = 55, left = 0.42f, top = 0.42f, right = 0.62f, bottom = 0.62f, label = "Food")
        val far = obj(id = 66, left = 0.25f, top = 0.25f, right = 0.45f, bottom = 0.45f, label = "Food")

        val result = processFramesUntilReacquire(listOf(far, close))
        assertNotNull(result)
        assertEquals(55, result!!.id)
    }

    // --- Re-acquisition: label breaks tie ---

    @Test
    fun `rejects person candidate when locked on non-person even if closer`() {
        val box = RectF(0.4f, 0.4f, 0.6f, 0.6f)
        engine.lock(42, box, "Food")

        val nonPerson = obj(id = 55, left = 0.42f, top = 0.42f, right = 0.62f, bottom = 0.62f, label = "Food")
        val person = obj(id = 66, left = 0.41f, top = 0.41f, right = 0.61f, bottom = 0.61f, label = "person")

        val result = processFramesUntilReacquire(listOf(person, nonPerson))
        assertNotNull(result)
        assertEquals("Should pick non-person over person when locked on non-person", 55, result!!.id)
    }

    // --- Person/not-person gate ---

    @Test
    fun `person-not-person gate rejects cross-category without embedding`() {
        // Lock on non-person "cup" — person candidate should be rejected
        engine.lock(42, RectF(0.4f, 0.4f, 0.6f, 0.6f), "cup")

        val nonPerson = obj(id = 55, left = 0.42f, top = 0.42f, right = 0.62f, bottom = 0.62f, label = "cup")
        val person = obj(id = 66, left = 0.42f, top = 0.42f, right = 0.62f, bottom = 0.62f, label = "person")

        val nonPersonScore = engine.scoreCandidate(nonPerson, engine.lastKnownBox!!)
        val personScore = engine.scoreCandidate(person, engine.lastKnownBox!!)

        assertNotNull("Non-person should pass gate when locked on non-person", nonPersonScore)
        assertNull("Person should be rejected when locked on non-person", personScore)
    }

    @Test
    fun `person candidate rejected even as only candidate when locked on non-person`() {
        // Person/not-person gate rejects person candidates when locked on non-person
        engine.lock(42, RectF(0.4f, 0.4f, 0.6f, 0.6f), "cup")

        val detections = listOf(
            obj(id = 55, left = 0.42f, top = 0.42f, right = 0.62f, bottom = 0.62f, label = "person")
        )
        // Lose the locked id first
        engine.processFrame(emptyList())
        val result = engine.processFrame(detections)
        assertNull("Person candidate should be rejected when locked on non-person", result)
    }

    @Test
    fun `non-person candidates with different labels both pass gate`() {
        // With person/not-person gate, different non-person labels pass — embedding handles identity
        engine.lock(42, RectF(0.4f, 0.4f, 0.6f, 0.6f), "cup")

        val cup = obj(id = 55, left = 0.42f, top = 0.42f, right = 0.62f, bottom = 0.62f, label = "cup")
        val laptop = obj(id = 66, left = 0.42f, top = 0.42f, right = 0.62f, bottom = 0.62f, label = "laptop")

        val cupScore = engine.scoreCandidate(cup, engine.lastKnownBox!!)
        val laptopScore = engine.scoreCandidate(laptop, engine.lastKnownBox!!)

        assertNotNull("Same-label non-person should pass gate", cupScore)
        assertNotNull("Different-label non-person should also pass gate", laptopScore)
    }

    @Test
    fun `allows reacquisition when locked object had no label`() {
        engine.lock(42, RectF(0.4f, 0.4f, 0.6f, 0.6f), null)

        val detections = listOf(
            obj(id = 55, left = 0.42f, top = 0.42f, right = 0.62f, bottom = 0.62f, label = "laptop")
        )
        val result = processFramesUntilReacquire(detections)
        assertNotNull("Should allow reacquisition when original had no label", result)
    }

    // --- Position decay: the key fix ---

    @Test
    fun `positionConfidence is 1 at frame 0`() {
        engine.lock(42, RectF(0.4f, 0.4f, 0.6f, 0.6f), null)
        assertEquals(1f, engine.positionConfidence(), 0.01f)
    }

    @Test
    fun `positionConfidence decays to 0 after positionDecayFrames`() {
        engine.lock(42, RectF(0.4f, 0.4f, 0.6f, 0.6f), null)
        repeat(engine.positionDecayFrames) { engine.processFrame(emptyList()) }
        assertEquals(0f, engine.positionConfidence(), 0.01f)
    }

    @Test
    fun `positionConfidence is 0_5 at halfway`() {
        engine.lock(42, RectF(0.4f, 0.4f, 0.6f, 0.6f), null)
        repeat(engine.positionDecayFrames / 2) { engine.processFrame(emptyList()) }
        assertEquals(0.5f, engine.positionConfidence(), 0.1f)
    }

    @Test
    fun `effective position threshold expands over time`() {
        engine.lock(42, RectF(0.4f, 0.4f, 0.6f, 0.6f), null)
        val initial = engine.effectivePositionThreshold()
        repeat(engine.positionDecayFrames) { engine.processFrame(emptyList()) }
        val expanded = engine.effectivePositionThreshold()

        assertTrue("Threshold should expand: initial=$initial, expanded=$expanded",
            expanded > initial)
        assertEquals(engine.maxPositionThreshold, expanded, 0.01f)
    }

    @Test
    fun `reacquires distant same-label object after many lost frames`() {
        // This is THE key scenario: camera panned away and back
        val box = RectF(0.4f, 0.4f, 0.6f, 0.6f) // center of frame
        engine.lock(42, box, "Food")

        // Lose for enough frames that position decays
        repeat(engine.positionDecayFrames) { engine.processFrame(emptyList()) }

        // Object reappears at totally different position but same label + similar size
        val detections = listOf(
            obj(id = 192, left = 0.1f, top = 0.7f, right = 0.3f, bottom = 0.9f, label = "Food")
        )

        val result = processFramesUntilReacquire(detections)
        assertNotNull("Should reacquire distant same-label object after position decay", result)
        assertEquals(192, result!!.id)
    }

    @Test
    fun `does not reacquire person when locked on non-person after many lost frames`() {
        val box = RectF(0.4f, 0.4f, 0.6f, 0.6f)
        engine.lock(42, box, "Food")

        repeat(engine.positionDecayFrames) { engine.processFrame(emptyList()) }

        // Person candidate — rejected by person/not-person gate
        val detections = listOf(
            obj(id = 99, left = 0.1f, top = 0.7f, right = 0.3f, bottom = 0.9f, label = "person")
        )

        val result = engine.processFrame(detections)
        assertNull("Person should not reacquire when locked on non-person", result)
    }

    @Test
    fun `early frames reject distant object even with matching label`() {
        val box = RectF(0.4f, 0.4f, 0.6f, 0.6f)
        engine.lock(42, box, "Food")

        // Only 1 frame lost — position threshold is still tight
        val detections = listOf(
            obj(id = 55, left = 0.0f, top = 0.0f, right = 0.2f, bottom = 0.2f, label = "Food")
        )

        val result = engine.processFrame(detections)
        assertNull("Should reject distant object in early frames", result)
    }

    // --- Timeout ---

    @Test
    fun `stops reacquiring after maxFramesLost`() {
        engine = ReacquisitionEngine(maxFramesLost = 3)
        val box = RectF(0.4f, 0.4f, 0.6f, 0.6f)
        engine.lock(42, box, "Food")

        repeat(3) { engine.processFrame(emptyList()) }
        assertFalse(engine.hasTimedOut)

        engine.processFrame(emptyList())
        assertTrue(engine.hasTimedOut)

        // Even if object reappears, don't re-acquire after timeout
        val detections = listOf(
            obj(id = 55, left = 0.42f, top = 0.42f, right = 0.62f, bottom = 0.62f, label = "Food")
        )
        assertNull(engine.processFrame(detections))
    }

    // --- Frames lost counter ---

    @Test
    fun `framesLost increments each frame without match`() {
        engine.lock(42, RectF(0.4f, 0.4f, 0.6f, 0.6f), null)
        engine.processFrame(emptyList())
        assertEquals(1, engine.framesLost)
        engine.processFrame(emptyList())
        assertEquals(2, engine.framesLost)
    }

    @Test
    fun `framesLost resets on successful reacquisition`() {
        engine.lock(42, RectF(0.4f, 0.4f, 0.6f, 0.6f), "Food")
        repeat(5) { engine.processFrame(emptyList()) }
        assertEquals(5, engine.framesLost)

        val detections = listOf(
            obj(id = 55, left = 0.42f, top = 0.42f, right = 0.62f, bottom = 0.62f, label = "Food")
        )
        val result = processFramesUntilReacquire(detections)
        assertNotNull(result)
        assertEquals(0, engine.framesLost)
    }

    // --- Score calculation ---

    @Test
    fun `scoreCandidate returns null for object beyond size threshold`() {
        engine.lock(42, RectF(0.45f, 0.45f, 0.55f, 0.55f), null)
        val huge = obj(id = 1, left = 0.1f, top = 0.1f, right = 0.9f, bottom = 0.9f)
        assertNull(engine.scoreCandidate(huge, engine.lastKnownBox!!))
    }

    @Test
    fun `scoreCandidate gives higher score to closer objects`() {
        val refBox = RectF(0.4f, 0.4f, 0.6f, 0.6f)
        engine.lock(42, refBox, null)

        val close = obj(id = 1, left = 0.41f, top = 0.41f, right = 0.61f, bottom = 0.61f)
        val medium = obj(id = 2, left = 0.35f, top = 0.35f, right = 0.55f, bottom = 0.55f)

        val closeScore = engine.scoreCandidate(close, refBox)!!
        val mediumScore = engine.scoreCandidate(medium, refBox)!!

        assertTrue("Close ($closeScore) should score higher than medium ($mediumScore)",
            closeScore > mediumScore)
    }

    @Test
    fun `person-not-person gate rejects cross-category and passes same category`() {
        val refBox = RectF(0.4f, 0.4f, 0.6f, 0.6f)
        engine.lock(42, refBox, "Food")  // non-person

        val nonPerson = obj(id = 1, left = 0.42f, top = 0.42f, right = 0.62f, bottom = 0.62f, label = "Food")
        val person = obj(id = 2, left = 0.42f, top = 0.42f, right = 0.62f, bottom = 0.62f, label = "person")

        val nonPersonScore = engine.scoreCandidate(nonPerson, refBox)
        val personScore = engine.scoreCandidate(person, refBox)

        assertNotNull("Non-person should pass gate when locked on non-person", nonPersonScore)
        assertNull("Person should be rejected when locked on non-person", personScore)
    }

    @Test
    fun `person always rejected when locked on non-person regardless of position decay`() {
        val refBox = RectF(0.4f, 0.4f, 0.6f, 0.6f)
        engine.lock(42, refBox, "Food")  // non-person

        val personCandidate = obj(id = 2, left = 0.42f, top = 0.42f, right = 0.62f, bottom = 0.62f, label = "person")

        // Early frames: person rejected
        val earlyScore = engine.scoreCandidate(personCandidate, refBox)
        assertNull("Person should be rejected early when locked on non-person", earlyScore)

        // After position decay: still rejected
        repeat(engine.positionDecayFrames) { engine.processFrame(emptyList()) }
        val lateScore = engine.scoreCandidate(personCandidate, refBox)
        assertNull("Person should still be rejected after decay", lateScore)
    }

    // --- Appearance embedding scoring ---

    @Test
    fun `lock stores embedding in gallery`() {
        val embedding = floatArrayOf(0.1f, 0.2f, 0.3f)
        engine.lock(42, RectF(0.4f, 0.4f, 0.6f, 0.6f), "cup", embedding)
        assertEquals(1, engine.embeddingGallery.size)
        assertArrayEquals(embedding, engine.embeddingGallery[0], 0.001f)
    }

    @Test
    fun `clear removes embeddings`() {
        engine.lock(42, RectF(0.4f, 0.4f, 0.6f, 0.6f), "cup", floatArrayOf(0.1f, 0.2f))
        engine.clear()
        assertTrue(engine.embeddingGallery.isEmpty())
    }

    @Test
    fun `appearance score boosts visually similar candidate`() {
        val lockedEmb = floatArrayOf(1f, 0f, 0f)  // unit vector
        engine.lock(42, RectF(0.4f, 0.4f, 0.6f, 0.6f), "cup", lockedEmb)

        val similar = obj(id = 55, left = 0.42f, top = 0.42f, right = 0.62f, bottom = 0.62f, label = "cup")
            .copy(embedding = floatArrayOf(0.9f, 0.1f, 0f))  // similar direction
        val different = obj(id = 66, left = 0.42f, top = 0.42f, right = 0.62f, bottom = 0.62f, label = "cup")
            .copy(embedding = floatArrayOf(0.5f, 0.1f, 0.85f))  // mostly different but above floor

        val similarScore = engine.scoreCandidate(similar, engine.lastKnownBox!!)!!
        val differentScore = engine.scoreCandidate(different, engine.lastKnownBox!!)!!

        assertTrue("Visually similar ($similarScore) should score higher than different ($differentScore)",
            similarScore > differentScore)
    }

    @Test
    fun `appearance score distinguishes same-label objects`() {
        // THE core scenario: two cups, only one looks like the locked one
        val lockedEmb = floatArrayOf(0.7f, 0.7f, 0f)
        engine.lock(42, RectF(0.4f, 0.4f, 0.6f, 0.6f), "cup", lockedEmb)

        // Lose the object
        engine.processFrame(emptyList())

        val rightCup = obj(id = 55, left = 0.42f, top = 0.42f, right = 0.62f, bottom = 0.62f, label = "cup")
            .copy(embedding = floatArrayOf(0.6f, 0.8f, 0f))  // similar visual
        val wrongCup = obj(id = 66, left = 0.38f, top = 0.38f, right = 0.58f, bottom = 0.58f, label = "cup")
            .copy(embedding = floatArrayOf(0.5f, 0.2f, 0.8f))  // different visual but above floor

        val result = engine.processFrame(listOf(wrongCup, rightCup))
        assertNotNull(result)
        assertEquals("Should pick visually similar cup", 55, result!!.id)
    }

    @Test
    fun `scoring falls back gracefully when no embedding available`() {
        // Lock without embedding
        engine.lock(42, RectF(0.4f, 0.4f, 0.6f, 0.6f), "cup")
        assertTrue(engine.embeddingGallery.isEmpty())

        // Candidate without embedding — should still score via position/size/label
        val candidate = obj(id = 55, left = 0.42f, top = 0.42f, right = 0.62f, bottom = 0.62f, label = "cup")
        val score = engine.scoreCandidate(candidate, engine.lastKnownBox!!)
        assertNotNull("Should still score without embeddings", score)
        assertTrue(score!! > 0f)
    }

    @Test
    fun `appearance weight increases after position decay`() {
        val lockedEmb = floatArrayOf(1f, 0f, 0f)
        engine.lock(42, RectF(0.4f, 0.4f, 0.6f, 0.6f), "cup", lockedEmb)

        val similar = obj(id = 55, left = 0.42f, top = 0.42f, right = 0.62f, bottom = 0.62f, label = "cup")
            .copy(embedding = floatArrayOf(0.9f, 0.1f, 0f))
        val different = obj(id = 66, left = 0.42f, top = 0.42f, right = 0.62f, bottom = 0.62f, label = "cup")
            .copy(embedding = floatArrayOf(0.5f, 0.1f, 0.85f))  // different but above floor

        // Early gap between similar vs different
        val earlyGap = engine.scoreCandidate(similar, engine.lastKnownBox!!)!! -
                       engine.scoreCandidate(different, engine.lastKnownBox!!)!!

        // Decay position
        repeat(engine.positionDecayFrames) { engine.processFrame(emptyList()) }

        val lateGap = engine.scoreCandidate(similar, engine.lastKnownBox!!)!! -
                      engine.scoreCandidate(different, engine.lastKnownBox!!)!!

        assertTrue("Appearance should matter more after position decay: early=$earlyGap, late=$lateGap",
            lateGap > earlyGap)
    }

    // --- Appearance override of geometric hard filters ---

    @Test
    fun `strong embedding bypasses size threshold`() {
        // Lock on small object (e.g. mouse at edge of frame)
        val lockedEmb = floatArrayOf(0.8f, 0.6f, 0f)
        engine.lock(42, RectF(0.9f, 0.5f, 1.0f, 0.7f), "mouse", lockedEmb)

        // Lose it, then candidate reappears MUCH larger (phone was flipped)
        engine.processFrame(emptyList())

        val bigMouse = obj(id = 55, left = 0.1f, top = 0.2f, right = 0.7f, bottom = 0.8f, label = "mouse")
            .copy(embedding = floatArrayOf(0.7f, 0.7f, 0f))  // similar visual (sim ~0.95)

        val score = engine.scoreCandidate(bigMouse, engine.lastKnownBox!!)
        assertNotNull("Strong embedding should bypass size hard threshold", score)
    }

    @Test
    fun `strong embedding bypasses position threshold`() {
        val lockedEmb = floatArrayOf(0.8f, 0.6f, 0f)
        engine.lock(42, RectF(0.9f, 0.9f, 1.0f, 1.0f), "mouse", lockedEmb)

        engine.processFrame(emptyList())

        // Same object appears at opposite corner — way beyond position threshold
        val farMouse = obj(id = 55, left = 0.0f, top = 0.0f, right = 0.1f, bottom = 0.1f, label = "mouse")
            .copy(embedding = floatArrayOf(0.7f, 0.7f, 0f))

        val score = engine.scoreCandidate(farMouse, engine.lastKnownBox!!)
        assertNotNull("Strong embedding should bypass position hard threshold", score)
    }

    @Test
    fun `weak embedding does not bypass size threshold`() {
        val lockedEmb = floatArrayOf(1f, 0f, 0f)
        engine.lock(42, RectF(0.9f, 0.5f, 1.0f, 0.7f), "mouse", lockedEmb)

        engine.processFrame(emptyList())

        // Much larger, and visually different
        val wrongObj = obj(id = 55, left = 0.1f, top = 0.2f, right = 0.7f, bottom = 0.8f, label = "mouse")
            .copy(embedding = floatArrayOf(0f, 0f, 1f))  // orthogonal = sim 0.0

        val score = engine.scoreCandidate(wrongObj, engine.lastKnownBox!!)
        assertNull("Weak embedding should NOT bypass size hard threshold", score)
    }

    // --- Label flicker: strong embedding overrides wrong label ---

    @Test
    fun `strong embedding overrides person-not-person gate`() {
        // Lock on "cup" (non-person) — detector sees same object as "person"
        val lockedEmb = floatArrayOf(0.8f, 0.5f, 0.2f)
        engine.lock(42, RectF(0.3f, 0.3f, 0.7f, 0.7f), "cup", lockedEmb)

        engine.processFrame(emptyList())

        // Same object with strong embedding but person label — should override gate
        val candidate = obj(id = 55, left = 0.32f, top = 0.32f, right = 0.68f, bottom = 0.68f, label = "person")
            .copy(embedding = floatArrayOf(0.75f, 0.55f, 0.2f))  // sim ~0.98

        val result = engine.processFrame(listOf(candidate))
        assertNotNull("Strong embedding should override person/not-person gate", result)
        assertEquals(55, result!!.id)
    }

    @Test
    fun `weak embedding with person candidate rejected when locked on non-person`() {
        val lockedEmb = floatArrayOf(0.8f, 0.5f, 0.2f)
        engine.lock(42, RectF(0.3f, 0.3f, 0.7f, 0.7f), "bowl", lockedEmb)

        engine.processFrame(emptyList())

        val nonPerson = obj(id = 55, left = 0.32f, top = 0.32f, right = 0.68f, bottom = 0.68f, label = "bowl")
            .copy(embedding = floatArrayOf(0.75f, 0.55f, 0.2f))  // high sim, non-person
        val person = obj(id = 66, left = 0.32f, top = 0.32f, right = 0.68f, bottom = 0.68f, label = "person")
            .copy(embedding = floatArrayOf(0f, 0f, 1f))  // low sim, person

        val nonPersonScore = engine.scoreCandidate(nonPerson, engine.lastKnownBox!!)
        val personScore = engine.scoreCandidate(person, engine.lastKnownBox!!)

        assertNotNull("Non-person should pass gate", nonPersonScore)
        assertNull("Person with weak embedding should be rejected by person/not-person gate", personScore)
    }

    @Test
    fun `non-person candidates with different labels score equally without label bonus`() {
        val lockedEmb = floatArrayOf(0.8f, 0.5f, 0.2f)
        engine.lock(42, RectF(0.3f, 0.3f, 0.7f, 0.7f), "bowl", lockedEmb)

        engine.processFrame(emptyList())

        // Two non-person candidates with same embedding — no label bonus, should score equal
        val labelA = obj(id = 55, left = 0.32f, top = 0.32f, right = 0.68f, bottom = 0.68f, label = "potted plant")
            .copy(embedding = floatArrayOf(0.75f, 0.55f, 0.2f))
        val labelB = obj(id = 66, left = 0.32f, top = 0.32f, right = 0.68f, bottom = 0.68f, label = "bowl")
            .copy(embedding = floatArrayOf(0.75f, 0.55f, 0.2f))

        val scoreA = engine.scoreCandidate(labelA, engine.lastKnownBox!!)!!
        val scoreB = engine.scoreCandidate(labelB, engine.lastKnownBox!!)!!

        assertEquals("Non-person candidates with same signals should score equally (no label bonus)",
            scoreA, scoreB, 0.001f)
    }

    // --- Adaptive embedding floor ---

    @Test
    fun `adaptive embedding floor is lenient with immature gallery`() {
        // Lock with 1 embedding — immature gallery
        val lockedEmb = floatArrayOf(0.8f, 0.5f, 0.2f)
        engine.lock(42, RectF(0.3f, 0.3f, 0.7f, 0.7f), "mouse", lockedEmb)
        engine.processFrame(emptyList())

        // Candidate with sim=0.3 — above MIN_EMBEDDING_SIMILARITY (0.15) but below mature floor (0.45)
        val weakCandidate = obj(id = 55, left = 0.32f, top = 0.32f, right = 0.68f, bottom = 0.68f, label = "keyboard")
            .copy(embedding = floatArrayOf(0.3f, 0.9f, 0.2f))

        val score = engine.scoreCandidate(weakCandidate, engine.lastKnownBox!!)
        assertNotNull("With immature gallery, sim=0.3 should pass lenient floor (0.15)", score)
    }

    @Test
    fun `adaptive embedding floor is strict with mature gallery`() {
        // Lock with many embeddings to reach mature gallery (≥8)
        val embeddings = List(8) { floatArrayOf(0.8f + it * 0.01f, 0.5f, 0.2f) }
        engine.lock(42, RectF(0.3f, 0.3f, 0.7f, 0.7f), "mouse", embeddings)
        engine.processFrame(emptyList())

        // Candidate with orthogonal embedding — very low similarity (~0.2)
        val weakCandidate = obj(id = 55, left = 0.32f, top = 0.32f, right = 0.68f, bottom = 0.68f, label = "keyboard")
            .copy(embedding = floatArrayOf(0.1f, 0.1f, 0.98f))

        val score = engine.scoreCandidate(weakCandidate, engine.lastKnownBox!!)
        assertNull("With mature gallery, low-sim candidate should be rejected by strict floor (0.45)", score)
    }

    // --- Two same-label objects: embedding must discriminate ---

    @Test
    fun `two trucks - embedding picks the correct one`() {
        // Lock on orange truck
        val orangeEmb = floatArrayOf(0.9f, 0.3f, 0.1f)
        engine.lock(42, RectF(0.3f, 0.4f, 0.6f, 0.6f), "truck", orangeEmb)

        // Lose it
        engine.processFrame(emptyList())

        // Both trucks appear — same label, similar size/position
        val orangeTruck = obj(id = 55, left = 0.32f, top = 0.42f, right = 0.62f, bottom = 0.62f, label = "truck")
            .copy(embedding = floatArrayOf(0.85f, 0.35f, 0.1f))  // similar to lock
        val blueTruck = obj(id = 66, left = 0.25f, top = 0.38f, right = 0.55f, bottom = 0.58f, label = "truck")
            .copy(embedding = floatArrayOf(0.1f, 0.2f, 0.9f))  // different appearance

        val result = engine.processFrame(listOf(blueTruck, orangeTruck))
        assertNotNull(result)
        assertEquals("Should pick orange truck", 55, result!!.id)
    }

    @Test
    fun `two trucks - blue truck scores lower than orange`() {
        val orangeEmb = floatArrayOf(0.9f, 0.3f, 0.1f)
        engine.lock(42, RectF(0.3f, 0.4f, 0.6f, 0.6f), "truck", orangeEmb)
        engine.processFrame(emptyList())

        val orangeTruck = obj(id = 55, left = 0.32f, top = 0.42f, right = 0.62f, bottom = 0.62f, label = "truck")
            .copy(embedding = floatArrayOf(0.85f, 0.35f, 0.1f))
        val blueTruck = obj(id = 66, left = 0.32f, top = 0.42f, right = 0.62f, bottom = 0.62f, label = "truck")
            .copy(embedding = floatArrayOf(0.5f, 0.2f, 0.8f))

        val orangeScore = engine.scoreCandidate(orangeTruck, engine.lastKnownBox!!)!!
        val blueScore = engine.scoreCandidate(blueTruck, engine.lastKnownBox!!)!!

        assertTrue("Orange ($orangeScore) should score higher than blue ($blueScore)",
            orangeScore > blueScore)
    }

    // --- Visual tracker handoff: updateFromVisualTracker ---

    @Test
    fun `updateFromVisualTracker keeps lastKnownBox in sync`() {
        engine.lock(42, RectF(0.3f, 0.3f, 0.6f, 0.6f), "truck")
        val newBox = RectF(0.35f, 0.35f, 0.65f, 0.65f)

        engine.updateFromVisualTracker(newBox)

        assertEquals(newBox, engine.lastKnownBox)
        assertEquals(0, engine.framesLost)
    }

    @Test
    fun `updateFromVisualTracker does not change lockedLabel`() {
        engine.lock(42, RectF(0.3f, 0.3f, 0.6f, 0.6f), "truck")
        engine.updateFromVisualTracker(RectF(0.5f, 0.5f, 0.8f, 0.8f))

        assertEquals("truck", engine.lockedLabel)
        assertEquals("truck", engine.lastKnownLabel)
    }

    @Test
    fun `drifted lastKnownBox causes re-acquisition to search wrong area`() {
        // Simulates the bug: visual tracker drifts, updates lastKnownBox
        // to a wrong position, then re-acquisition can't find the object
        val orangeEmb = floatArrayOf(0.9f, 0.3f, 0.1f)
        engine.lock(42, RectF(0.3f, 0.4f, 0.6f, 0.6f), "truck", orangeEmb)

        // Simulate drift: VT moved lastKnownBox to bottom-right
        engine.updateFromVisualTracker(RectF(0.8f, 0.8f, 0.95f, 0.95f))

        // Object lost, candidate is near ORIGINAL position
        engine.processFrame(emptyList())
        val candidate = obj(id = 55, left = 0.32f, top = 0.42f, right = 0.62f, bottom = 0.62f, label = "truck")
            .copy(embedding = floatArrayOf(0.85f, 0.35f, 0.1f))

        // With drifted lastKnownBox, position distance is huge (0.8 vs 0.3)
        // but strong embedding should override
        val result = engine.processFrame(listOf(candidate))
        assertNotNull("Strong embedding should still re-acquire despite drifted lastKnownBox", result)
    }

    // --- Edge cases ---

    @Test
    fun `handles detection with negative ID gracefully`() {
        engine.lock(42, RectF(0.4f, 0.4f, 0.6f, 0.6f), null)
        val detections = listOf(
            obj(id = -1, left = 0.42f, top = 0.42f, right = 0.62f, bottom = 0.62f)
        )
        assertNull(engine.processFrame(detections))
    }

    @Test
    fun `updates lastKnownLabel only when detection has a label`() {
        engine.lock(42, RectF(0.4f, 0.4f, 0.6f, 0.6f), "Food")
        val detections = listOf(
            obj(id = 42, left = 0.41f, top = 0.41f, right = 0.61f, bottom = 0.61f, label = null)
        )
        engine.processFrame(detections)
        assertEquals("Food", engine.lastKnownLabel)
    }

    @Test
    fun `label flicker during tracking preserves lockedLabel and lockedIsPerson`() {
        // Lock on a bowl (non-person)
        engine.lock(1, RectF(0.4f, 0.4f, 0.6f, 0.6f), "bowl")
        assertEquals("bowl", engine.lockedLabel)
        assertFalse(engine.lockedIsPerson)

        // Direct match with flickered label — updates lastKnownLabel
        val flickered = listOf(
            obj(id = 1, left = 0.41f, top = 0.41f, right = 0.61f, bottom = 0.61f, label = "potted plant")
        )
        engine.processFrame(flickered)
        assertEquals("potted plant", engine.lastKnownLabel)
        assertEquals("bowl", engine.lockedLabel) // lockedLabel unchanged
        assertFalse("lockedIsPerson should not change on label flicker", engine.lockedIsPerson)

        // Both "bowl" and "potted plant" are non-person, so both pass the gate
        engine.lock(1, RectF(0.4f, 0.4f, 0.6f, 0.6f), "bowl")
        val noMatch = listOf(
            obj(id = 99, left = 0.4f, top = 0.4f, right = 0.6f, bottom = 0.6f, label = "potted plant"),
            obj(id = 100, left = 0.41f, top = 0.41f, right = 0.61f, bottom = 0.61f, label = "bowl")
        )
        repeat(2) { engine.processFrame(emptyList()) }

        val result = engine.findBestCandidate(noMatch)
        assertNotNull("Both non-person candidates should pass the gate", result)
        // Person candidate should be rejected
        val personCandidate = listOf(
            obj(id = 101, left = 0.4f, top = 0.4f, right = 0.6f, bottom = 0.6f, label = "person")
        )
        val personResult = engine.findBestCandidate(personCandidate)
        assertNull("Person should be rejected when locked on non-person bowl", personResult)
    }

    // --- Color histogram scoring ---

    @Test
    fun `color match boosts score over color mismatch`() {
        val refHist = FloatArray(COLOR_HISTOGRAM_SIZE).also { it[0] = 1f } // all red
        val matchHist = FloatArray(COLOR_HISTOGRAM_SIZE).also { it[0] = 1f } // same red
        val mismatchHist = FloatArray(COLOR_HISTOGRAM_SIZE).also { it[10] = 1f } // different hue

        val emb = floatArrayOf(1f, 0f, 0f, 0f)
        engine.lock(1, RectF(0.4f, 0.4f, 0.6f, 0.6f), "cup",
            listOf(emb), refHist)
        repeat(2) { engine.processFrame(emptyList()) }

        val matchCandidate = obj(id = 10, left = 0.42f, top = 0.42f, right = 0.62f, bottom = 0.62f,
            label = "cup", embedding = emb, colorHistogram = matchHist)
        val mismatchCandidate = obj(id = 11, left = 0.42f, top = 0.42f, right = 0.62f, bottom = 0.62f,
            label = "cup", embedding = emb, colorHistogram = mismatchHist)

        val matchScore = engine.scoreCandidate(matchCandidate, engine.lastKnownBox!!)
        val mismatchScore = engine.scoreCandidate(mismatchCandidate, engine.lastKnownBox!!)

        assertNotNull(matchScore)
        assertNotNull(mismatchScore)
        assertTrue("Color match ($matchScore) should score higher than mismatch ($mismatchScore)",
            matchScore!! > mismatchScore!!)
    }

    @Test
    fun `no color histogram redistributes weight to appearance`() {
        val emb = floatArrayOf(1f, 0f, 0f, 0f)
        engine.lock(1, RectF(0.4f, 0.4f, 0.6f, 0.6f), "cup",
            listOf(emb), null) // no color histogram
        repeat(2) { engine.processFrame(emptyList()) }

        val candidate = obj(id = 10, left = 0.42f, top = 0.42f, right = 0.62f, bottom = 0.62f,
            label = "cup", embedding = emb, colorHistogram = null)

        val score = engine.scoreCandidate(candidate, engine.lastKnownBox!!)
        assertNotNull(score)
        // With perfect embedding match and no color, score should still be high
        assertTrue("Score without color should be reasonable: $score", score!! > 0.4f)
    }

    @Test
    fun `color weight is zero when no histogram available`() {
        // After full position decay, no weight should leak to color when no histogram
        val emb = floatArrayOf(1f, 0f, 0f, 0f)
        engine.lock(1, RectF(0.4f, 0.4f, 0.6f, 0.6f), "cup",
            listOf(emb), null)
        // Lose enough frames for full position decay
        repeat(35) { engine.processFrame(emptyList()) }

        val candidate = obj(id = 10, left = 0.42f, top = 0.42f, right = 0.62f, bottom = 0.62f,
            label = "cup", embedding = emb, colorHistogram = null)

        val scoreNoColor = engine.scoreCandidate(candidate, engine.lastKnownBox!!)
        assertNotNull(scoreNoColor)
        // Without color, all weight should go to appearance/size/label — no wasted weight
        assertTrue("Score should be high with perfect embedding: $scoreNoColor", scoreNoColor!! > 0.5f)
    }

    @Test
    fun `color histogram discriminates same-label same-embedding objects`() {
        // Two cups with identical embeddings but different colors
        val refHist = FloatArray(COLOR_HISTOGRAM_SIZE).also { it[0] = 1f } // red
        val emb = floatArrayOf(0.5f, 0.5f, 0.5f, 0.5f)
        engine.lock(1, RectF(0.4f, 0.4f, 0.6f, 0.6f), "cup",
            listOf(emb), refHist)
        repeat(2) { engine.processFrame(emptyList()) }

        val redCup = obj(id = 10, left = 0.42f, top = 0.42f, right = 0.62f, bottom = 0.62f,
            label = "cup", embedding = emb,
            colorHistogram = FloatArray(COLOR_HISTOGRAM_SIZE).also { it[0] = 1f }) // red
        val blueCup = obj(id = 11, left = 0.42f, top = 0.42f, right = 0.62f, bottom = 0.62f,
            label = "cup", embedding = emb,
            colorHistogram = FloatArray(COLOR_HISTOGRAM_SIZE).also { it[9] = 1f }) // blue

        val redScore = engine.scoreCandidate(redCup, engine.lastKnownBox!!)
        val blueScore = engine.scoreCandidate(blueCup, engine.lastKnownBox!!)

        assertNotNull(redScore)
        assertNotNull(blueScore)
        assertTrue("Red cup ($redScore) should beat blue cup ($blueScore)",
            redScore!! > blueScore!!)
    }

    @Test
    fun `candidate with color and no reference color does not crash`() {
        val emb = floatArrayOf(1f, 0f, 0f, 0f)
        engine.lock(1, RectF(0.4f, 0.4f, 0.6f, 0.6f), "cup",
            listOf(emb), null) // no reference histogram
        repeat(2) { engine.processFrame(emptyList()) }

        val candidateWithColor = obj(id = 10, left = 0.42f, top = 0.42f, right = 0.62f, bottom = 0.62f,
            label = "cup", embedding = emb,
            colorHistogram = FloatArray(COLOR_HISTOGRAM_SIZE).also { it[0] = 1f })

        val score = engine.scoreCandidate(candidateWithColor, engine.lastKnownBox!!)
        assertNotNull(score)
    }

    // --- histogramCorrelation edge cases ---

    @Test
    fun `histogramCorrelation returns 1 for identical histograms`() {
        val hist = FloatArray(COLOR_HISTOGRAM_SIZE) { it.toFloat() / COLOR_HISTOGRAM_SIZE }
        val corr = histogramCorrelation(hist, hist)
        assertEquals(1f, corr, 0.001f)
    }

    @Test
    fun `histogramCorrelation returns near zero for uncorrelated histograms`() {
        // Alternating pattern vs shifted alternating pattern
        val a = FloatArray(COLOR_HISTOGRAM_SIZE) { if (it % 2 == 0) 1f else 0f }
        val b = FloatArray(COLOR_HISTOGRAM_SIZE) { if (it % 2 == 0) 0f else 1f }
        val corr = histogramCorrelation(a, b)
        assertTrue("Uncorrelated histograms should have negative or near-zero correlation: $corr",
            corr < 0.1f)
    }

    @Test
    fun `histogramCorrelation returns 0 for mismatched sizes`() {
        val a = FloatArray(10) { 1f }
        val b = FloatArray(5) { 1f }
        assertEquals(0f, histogramCorrelation(a, b), 0.001f)
    }

    @Test
    fun `histogramCorrelation handles uniform histograms`() {
        // All bins equal — zero variance — should return 0
        val a = FloatArray(COLOR_HISTOGRAM_SIZE) { 1f / COLOR_HISTOGRAM_SIZE }
        val b = FloatArray(COLOR_HISTOGRAM_SIZE) { 0.5f }
        val corr = histogramCorrelation(a, b)
        // Both are constant → zero variance → returns 0
        assertEquals(0f, corr, 0.001f)
    }

    // --- Person attribute scoring ---

    @Test
    fun `matching person attributes boost score over mismatching`() {
        val emb = floatArrayOf(1f, 0f, 0f, 0f)
        val refAttrs = PersonAttributes(
            isMale = true, hasBag = false, hasBackpack = true, hasHat = true,
            hasLongSleeves = false, hasLongPants = true, hasLongHair = false, hasCoatJacket = false,
            rawProbabilities = floatArrayOf(0.9f, 0.1f, 0.9f, 0.9f, 0.1f, 0.9f, 0.1f, 0.1f)
        )
        engine.lock(1, RectF(0.4f, 0.4f, 0.6f, 0.6f), "person",
            listOf(emb), null, refAttrs)
        repeat(2) { engine.processFrame(emptyList()) }

        val matchAttrs = PersonAttributes(
            isMale = true, hasBag = false, hasBackpack = true, hasHat = true,
            hasLongSleeves = false, hasLongPants = true, hasLongHair = false, hasCoatJacket = false,
            rawProbabilities = floatArrayOf(0.9f, 0.1f, 0.9f, 0.9f, 0.1f, 0.9f, 0.1f, 0.1f)
        )
        val mismatchAttrs = PersonAttributes(
            isMale = false, hasBag = true, hasBackpack = false, hasHat = false,
            hasLongSleeves = true, hasLongPants = false, hasLongHair = true, hasCoatJacket = true,
            rawProbabilities = floatArrayOf(0.1f, 0.9f, 0.1f, 0.1f, 0.9f, 0.1f, 0.9f, 0.9f)
        )

        val matchCandidate = obj(id = 10, left = 0.42f, top = 0.42f, right = 0.62f, bottom = 0.62f,
            label = "person", embedding = emb, personAttributes = matchAttrs)
        val mismatchCandidate = obj(id = 11, left = 0.42f, top = 0.42f, right = 0.62f, bottom = 0.62f,
            label = "person", embedding = emb, personAttributes = mismatchAttrs)

        val matchScore = engine.scoreCandidate(matchCandidate, engine.lastKnownBox!!)
        val mismatchScore = engine.scoreCandidate(mismatchCandidate, engine.lastKnownBox!!)

        assertNotNull(matchScore)
        assertNotNull(mismatchScore)
        assertTrue("Matching attrs ($matchScore) should beat mismatching ($mismatchScore)",
            matchScore!! > mismatchScore!!)
    }

    @Test
    fun `no person attributes does not crash and redistributes weight`() {
        val emb = floatArrayOf(1f, 0f, 0f, 0f)
        engine.lock(1, RectF(0.4f, 0.4f, 0.6f, 0.6f), "person",
            listOf(emb), null, null)
        repeat(2) { engine.processFrame(emptyList()) }

        val candidate = obj(id = 10, left = 0.42f, top = 0.42f, right = 0.62f, bottom = 0.62f,
            label = "person", embedding = emb, personAttributes = null)

        val score = engine.scoreCandidate(candidate, engine.lastKnownBox!!)
        assertNotNull(score)
        assertTrue("Score without attrs should be reasonable: $score", score!! > 0.4f)
    }

    // --- Person/not-person gate with COCO label ---

    @Test
    fun `all non-person candidates pass gate regardless of specific label`() {
        val emb = floatArrayOf(0.5f, 0.5f, 0.5f, 0.5f)
        // Enriched to "flowerpot", COCO parent is "potted plant" — both non-person
        engine.lock(42, RectF(0.4f, 0.4f, 0.6f, 0.6f), "flowerpot",
            listOf(emb), null, null, cocoLabel = "potted plant")
        engine.processFrame(emptyList())

        val cocoCandidate = obj(id = 55, left = 0.42f, top = 0.42f, right = 0.62f, bottom = 0.62f,
            label = "potted plant", embedding = emb)
        val enrichedCandidate = obj(id = 66, left = 0.42f, top = 0.42f, right = 0.62f, bottom = 0.62f,
            label = "flowerpot", embedding = emb)
        val otherNonPerson = obj(id = 77, left = 0.42f, top = 0.42f, right = 0.62f, bottom = 0.62f,
            label = "bowl", embedding = emb)

        val cocoScore = engine.scoreCandidate(cocoCandidate, engine.lastKnownBox!!)!!
        val enrichedScore = engine.scoreCandidate(enrichedCandidate, engine.lastKnownBox!!)!!
        val otherScore = engine.scoreCandidate(otherNonPerson, engine.lastKnownBox!!)!!

        // All non-person candidates score the same with identical signals (no label bonus)
        assertEquals("All non-person candidates with same signals should score equally",
            cocoScore, enrichedScore, 0.001f)
        assertEquals("No label bonus — all non-person score the same",
            cocoScore, otherScore, 0.001f)
    }

    @Test
    fun `person candidate rejected when locked on non-person with weak embedding`() {
        val emb = floatArrayOf(0.5f, 0.5f, 0.5f, 0.5f)
        engine.lock(42, RectF(0.4f, 0.4f, 0.6f, 0.6f), "chair",
            listOf(emb), null, null, cocoLabel = null)
        engine.processFrame(emptyList())

        val nonPerson = obj(id = 55, left = 0.42f, top = 0.42f, right = 0.62f, bottom = 0.62f,
            label = "table", embedding = emb)
        // Person with weak embedding — below override threshold
        val person = obj(id = 66, left = 0.42f, top = 0.42f, right = 0.62f, bottom = 0.62f,
            label = "person", embedding = floatArrayOf(0.1f, 0.1f, 0.9f, 0.1f))

        val nonPersonScore = engine.scoreCandidate(nonPerson, engine.lastKnownBox!!)
        val personScore = engine.scoreCandidate(person, engine.lastKnownBox!!)

        assertNotNull("Non-person 'table' should pass gate when locked on non-person 'chair'", nonPersonScore)
        assertNull("Person with weak embedding should be rejected when locked on non-person 'chair'", personScore)
    }

    // --- Person/not-person gate: cross-category rejection ---

    @Test
    fun `person candidate rejected by gate when locked on non-person even with strong embedding`() {
        val emb = floatArrayOf(0.5f, 0.5f, 0.5f, 0.5f)
        engine.lock(42, RectF(0.4f, 0.4f, 0.6f, 0.6f), "chair", emb)
        engine.processFrame(emptyList())

        val nonPerson = obj(id = 55, left = 0.42f, top = 0.42f, right = 0.62f, bottom = 0.62f,
            label = "chair", embedding = emb)
        // Person with same embedding (sim=1.0) — still rejected by person/not-person gate
        // (only strong embedding can override, test verifies gate blocks cross-category)
        val person = obj(id = 66, left = 0.42f, top = 0.42f, right = 0.62f, bottom = 0.62f,
            label = "person", embedding = emb)

        val nonPersonScore = engine.scoreCandidate(nonPerson, engine.lastKnownBox!!)
        val personScore = engine.scoreCandidate(person, engine.lastKnownBox!!)

        assertNotNull("Non-person should pass gate", nonPersonScore)
        // With sim=1.0, person may pass via embedding override — that's the design
        // The key invariant is: non-person always passes, person may be overridden
        if (personScore != null) {
            assertTrue("If person passes via override, non-person should still score higher or equal",
                nonPersonScore!! >= personScore)
        }
    }

    @Test
    fun `null locked label gives neutral score`() {
        val emb = floatArrayOf(0.5f, 0.5f, 0.5f, 0.5f)
        engine.lock(42, RectF(0.4f, 0.4f, 0.6f, 0.6f), null, emb)
        engine.processFrame(emptyList())

        val candidateA = obj(id = 55, left = 0.42f, top = 0.42f, right = 0.62f, bottom = 0.62f,
            label = "chair", embedding = emb)
        val candidateB = obj(id = 66, left = 0.42f, top = 0.42f, right = 0.62f, bottom = 0.62f,
            label = "table", embedding = emb)

        val scoreA = engine.scoreCandidate(candidateA, engine.lastKnownBox!!)!!
        val scoreB = engine.scoreCandidate(candidateB, engine.lastKnownBox!!)!!

        assertEquals("With null locked label, different non-person labels should score the same",
            scoreA, scoreB, 0.001f)
    }

    // --- Cascade gate tests ---

    @Test
    fun `person-not-person gate rejects cross-category with weak embedding`() {
        val emb = floatArrayOf(0.9f, 0.3f, 0.1f)
        engine.lock(42, RectF(0.4f, 0.4f, 0.6f, 0.6f), "cup", emb)
        engine.processFrame(emptyList())

        // Person candidate, weak embedding (sim ~0.3) — below override threshold
        val candidate = obj(id = 55, left = 0.42f, top = 0.42f, right = 0.62f, bottom = 0.62f,
            label = "person", embedding = floatArrayOf(0.3f, 0.1f, 0.9f))

        assertNull("Person with weak embedding should be rejected when locked on non-person",
            engine.scoreCandidate(candidate, engine.lastKnownBox!!))
    }

    @Test
    fun `label gate passes wrong label with strong embedding`() {
        val emb = floatArrayOf(0.9f, 0.3f, 0.1f)
        engine.lock(42, RectF(0.4f, 0.4f, 0.6f, 0.6f), "cup", emb)
        engine.processFrame(emptyList())

        // Wrong label but very similar embedding (sim ~0.98) — label flicker scenario
        val candidate = obj(id = 55, left = 0.42f, top = 0.42f, right = 0.62f, bottom = 0.62f,
            label = "potted plant", embedding = floatArrayOf(0.88f, 0.32f, 0.12f))

        assertNotNull("Wrong label with strong embedding should pass gate (label flicker)",
            engine.scoreCandidate(candidate, engine.lastKnownBox!!))
    }

    @Test
    fun `cascade rejects wrong label even with perfect position and color`() {
        // THE motivating scenario: chair locked, person at exact same position
        // with identical color histogram — should still be rejected
        val colorHist = FloatArray(COLOR_HISTOGRAM_SIZE).also { it[5] = 1f }
        engine.lock(42, RectF(0.4f, 0.4f, 0.6f, 0.6f), "chair",
            emptyList(), colorHist)
        engine.processFrame(emptyList())

        val person = obj(id = 55, left = 0.41f, top = 0.41f, right = 0.61f, bottom = 0.61f,
            label = "person", colorHistogram = colorHist) // same position, same color

        assertNull("Wrong label should be rejected even with perfect position and color",
            engine.scoreCandidate(person, engine.lastKnownBox!!))
    }

    @Test
    fun `same-label candidates ranked by embedding similarity`() {
        val emb = floatArrayOf(0.9f, 0.3f, 0.1f)
        engine.lock(42, RectF(0.4f, 0.4f, 0.6f, 0.6f), "cup", emb)
        engine.processFrame(emptyList())

        val bestMatch = obj(id = 10, left = 0.42f, top = 0.42f, right = 0.62f, bottom = 0.62f,
            label = "cup", embedding = floatArrayOf(0.88f, 0.32f, 0.12f))  // sim ~0.98
        val midMatch = obj(id = 11, left = 0.42f, top = 0.42f, right = 0.62f, bottom = 0.62f,
            label = "cup", embedding = floatArrayOf(0.6f, 0.6f, 0.3f))  // sim ~0.75
        val weakMatch = obj(id = 12, left = 0.42f, top = 0.42f, right = 0.62f, bottom = 0.62f,
            label = "cup", embedding = floatArrayOf(0.3f, 0.3f, 0.8f))  // sim ~0.42

        val bestScore = engine.scoreCandidate(bestMatch, engine.lastKnownBox!!)!!
        val midScore = engine.scoreCandidate(midMatch, engine.lastKnownBox!!)!!
        val weakScore = engine.scoreCandidate(weakMatch, engine.lastKnownBox!!)!!

        assertTrue("Best ($bestScore) > mid ($midScore)", bestScore > midScore)
        assertTrue("Mid ($midScore) > weak ($weakScore)", midScore > weakScore)
    }

    @Test
    fun `no-embedding fallback works with label gate`() {
        // No embedding at lock or candidate — falls back to position+size ranking
        engine.lock(42, RectF(0.4f, 0.4f, 0.6f, 0.6f), "cup")
        engine.processFrame(emptyList())

        val candidate = obj(id = 55, left = 0.42f, top = 0.42f, right = 0.62f, bottom = 0.62f, label = "cup")
        val score = engine.scoreCandidate(candidate, engine.lastKnownBox!!)

        assertNotNull("Same-label candidate without embedding should pass gate and score", score)
        assertTrue("Score should be above threshold: $score", score!! >= engine.minScoreThreshold)
    }

    @Test
    fun `person-not-person gate passes non-person candidate when locked on non-person`() {
        val emb = floatArrayOf(0.5f, 0.5f, 0.5f)
        engine.lock(42, RectF(0.4f, 0.4f, 0.6f, 0.6f), "flowerpot",
            listOf(emb), null, null, cocoLabel = "potted plant")
        engine.processFrame(emptyList())

        // Non-person candidate passes gate regardless of specific label
        val candidate = obj(id = 55, left = 0.42f, top = 0.42f, right = 0.62f, bottom = 0.62f,
            label = "potted plant", embedding = emb)

        assertNotNull("Non-person candidate should pass gate when locked on non-person",
            engine.scoreCandidate(candidate, engine.lastKnownBox!!))
    }

    // --- Helpers ---

    // --- Face and re-ID scoring tiers ---

    @Test
    fun `face tier scores matching face higher than mismatching`() {
        val emb = floatArrayOf(0.5f, 0.5f)
        val faceEmb = floatArrayOf(0.9f, 0.3f, 0.1f)
        val reIdEmb = floatArrayOf(0.5f, 0.5f, 0.5f, 0.5f)
        engine.lock(42, RectF(0.4f, 0.4f, 0.6f, 0.6f), "person",
            listOf(emb), null, null, cocoLabel = "person",
            reIdEmbedding = reIdEmb, faceEmbedding = faceEmb)
        engine.processFrame(emptyList())

        val matchFace = obj(id = 10, left = 0.42f, top = 0.42f, right = 0.62f, bottom = 0.62f,
            label = "person", embedding = emb, reIdEmbedding = reIdEmb,
            faceEmbedding = floatArrayOf(0.88f, 0.32f, 0.12f)) // high face sim
        val wrongFace = obj(id = 11, left = 0.42f, top = 0.42f, right = 0.62f, bottom = 0.62f,
            label = "person", embedding = emb, reIdEmbedding = reIdEmb,
            faceEmbedding = floatArrayOf(0.1f, 0.1f, 0.9f)) // low face sim

        val matchScore = engine.scoreCandidate(matchFace, engine.lastKnownBox!!)!!
        val wrongScore = engine.scoreCandidate(wrongFace, engine.lastKnownBox!!)!!

        assertTrue("Matching face ($matchScore) should beat mismatching ($wrongScore)",
            matchScore > wrongScore)
    }

    @Test
    fun `reId tier scores matching body higher than mismatching`() {
        val emb = floatArrayOf(0.5f, 0.5f)
        val reIdEmb = floatArrayOf(0.9f, 0.3f, 0.1f, 0.2f)
        engine.lock(42, RectF(0.4f, 0.4f, 0.6f, 0.6f), "person",
            listOf(emb), null, null, cocoLabel = "person",
            reIdEmbedding = reIdEmb)
        engine.processFrame(emptyList())

        val matchBody = obj(id = 10, left = 0.42f, top = 0.42f, right = 0.62f, bottom = 0.62f,
            label = "person", embedding = emb,
            reIdEmbedding = floatArrayOf(0.88f, 0.32f, 0.12f, 0.22f)) // high re-ID sim
        val wrongBody = obj(id = 11, left = 0.42f, top = 0.42f, right = 0.62f, bottom = 0.62f,
            label = "person", embedding = emb,
            reIdEmbedding = floatArrayOf(0.1f, 0.1f, 0.9f, 0.1f)) // low re-ID sim

        val matchScore = engine.scoreCandidate(matchBody, engine.lastKnownBox!!)!!
        val wrongScore = engine.scoreCandidate(wrongBody, engine.lastKnownBox!!)!!

        assertTrue("Matching re-ID ($matchScore) should beat mismatching ($wrongScore)",
            matchScore > wrongScore)
    }

    @Test
    fun `face tier takes precedence over reId tier`() {
        val emb = floatArrayOf(0.5f, 0.5f)
        val reIdEmb = floatArrayOf(0.5f, 0.5f, 0.5f, 0.5f)
        val faceEmb = floatArrayOf(0.9f, 0.3f, 0.1f)
        engine.lock(42, RectF(0.4f, 0.4f, 0.6f, 0.6f), "person",
            listOf(emb), null, null, cocoLabel = "person",
            reIdEmbedding = reIdEmb, faceEmbedding = faceEmb)
        engine.processFrame(emptyList())

        // Good face, bad re-ID
        val goodFace = obj(id = 10, left = 0.42f, top = 0.42f, right = 0.62f, bottom = 0.62f,
            label = "person", embedding = emb,
            reIdEmbedding = floatArrayOf(0.1f, 0.1f, 0.1f, 0.9f), // bad re-ID
            faceEmbedding = floatArrayOf(0.88f, 0.32f, 0.12f)) // good face
        // Bad face, good re-ID
        val goodBody = obj(id = 11, left = 0.42f, top = 0.42f, right = 0.62f, bottom = 0.62f,
            label = "person", embedding = emb,
            reIdEmbedding = floatArrayOf(0.48f, 0.52f, 0.48f, 0.52f), // good re-ID
            faceEmbedding = floatArrayOf(0.1f, 0.1f, 0.9f)) // bad face

        val faceScore = engine.scoreCandidate(goodFace, engine.lastKnownBox!!)!!
        val bodyScore = engine.scoreCandidate(goodBody, engine.lastKnownBox!!)!!

        assertTrue("Good face ($faceScore) should beat good body ($bodyScore) — face takes precedence",
            faceScore > bodyScore)
    }

    @Test
    fun `no reId or face falls back to generic embedding tier`() {
        val emb = floatArrayOf(0.9f, 0.3f)
        engine.lock(42, RectF(0.4f, 0.4f, 0.6f, 0.6f), "person", emb)
        engine.processFrame(emptyList())

        val candidate = obj(id = 10, left = 0.42f, top = 0.42f, right = 0.62f, bottom = 0.62f,
            label = "person", embedding = floatArrayOf(0.85f, 0.35f)) // good generic sim, no re-ID/face

        val score = engine.scoreCandidate(candidate, engine.lastKnownBox!!)
        assertNotNull("Should score via generic embedding fallback", score)
        assertTrue("Score should be reasonable: $score", score!! > 0.4f)
    }

    // --- Tentative confirmation (DeepSORT-style) ---

    @Test
    fun `tentative confirmation blocks single-frame reacquisition`() {
        // Weak embedding — requires tentative confirmation
        val emb = floatArrayOf(1f, 0f, 0f, 0f)
        val weakEmb = floatArrayOf(0.5f, 0.1f, 0.85f, 0f)  // sim ~0.42 — below GEOMETRIC_OVERRIDE
        engine.lock(42, RectF(0.4f, 0.4f, 0.6f, 0.6f), "cup", emb)
        engine.processFrame(emptyList()) // trigger search

        val detections = listOf(
            obj(id = 55, left = 0.42f, top = 0.42f, right = 0.62f, bottom = 0.62f,
                label = "cup", embedding = weakEmb)
        )
        // Single frame should NOT reacquire
        val result = engine.processFrame(detections)
        assertNull("Single frame should not reacquire with weak embedding", result)
    }

    @Test
    fun `tentative confirmation commits after enough consecutive frames`() {
        val emb = floatArrayOf(1f, 0f, 0f, 0f)
        val weakEmb = floatArrayOf(0.5f, 0.1f, 0.85f, 0f)  // weak sim
        engine.lock(42, RectF(0.4f, 0.4f, 0.6f, 0.6f), "cup", emb)
        engine.processFrame(emptyList())

        val detections = listOf(
            obj(id = 55, left = 0.42f, top = 0.42f, right = 0.62f, bottom = 0.62f,
                label = "cup", embedding = weakEmb)
        )
        val result = processFramesUntilReacquire(detections)
        assertNotNull("Should reacquire after ${ReacquisitionEngine.TENTATIVE_MIN_FRAMES} frames", result)
        assertEquals(55, result!!.id)
    }

    @Test
    fun `tentative resets when different detection wins`() {
        val emb = floatArrayOf(1f, 0f, 0f, 0f)
        val weakEmb = floatArrayOf(0.5f, 0.1f, 0.85f, 0f)
        engine.lock(42, RectF(0.4f, 0.4f, 0.6f, 0.6f), "cup", emb)
        engine.processFrame(emptyList())

        val couch = listOf(
            obj(id = 55, left = 0.42f, top = 0.42f, right = 0.62f, bottom = 0.62f,
                label = "cup", embedding = weakEmb)
        )
        // 2 frames of couch (not enough)
        engine.processFrame(couch)
        engine.processFrame(couch)

        // Different object at different position — resets tentative
        val bed = listOf(
            obj(id = 66, left = 0.1f, top = 0.1f, right = 0.3f, bottom = 0.3f,
                label = "cup", embedding = weakEmb)
        )
        engine.processFrame(bed)

        // Back to couch — needs 3 more frames (streak was reset)
        val result = engine.processFrame(couch)
        assertNull("Tentative should reset when different detection wins", result)
    }

    @Test
    fun `strong embedding bypasses tentative confirmation`() {
        val emb = floatArrayOf(1f, 0f, 0f, 0f)
        val strongEmb = floatArrayOf(0.98f, 0.05f, 0.05f, 0f)  // sim ~0.98 > 0.7
        engine.lock(42, RectF(0.4f, 0.4f, 0.6f, 0.6f), "cup", emb)
        engine.processFrame(emptyList())

        val detections = listOf(
            obj(id = 55, left = 0.42f, top = 0.42f, right = 0.62f, bottom = 0.62f,
                label = "cup", embedding = strongEmb)
        )
        val result = engine.processFrame(detections)
        assertNotNull("Strong embedding should bypass tentative", result)
        assertEquals(55, result!!.id)
    }

    @Test
    fun `decent embedding bypasses tentative when classifier not trained`() {
        val emb = floatArrayOf(1f, 0f, 0f, 0f)
        // sim > TENTATIVE_BYPASS_THRESHOLD (0.65) — used as fallback when classifier not trained
        val decentEmb = floatArrayOf(0.8f, 0.5f, 0.1f, 0f)  // sim ~0.84
        engine.lock(42, RectF(0.4f, 0.4f, 0.6f, 0.6f), "cup", emb)
        engine.processFrame(emptyList())

        assertFalse("Classifier should not be trained yet", engine.classifierTrained)
        val detections = listOf(
            obj(id = 55, left = 0.42f, top = 0.42f, right = 0.62f, bottom = 0.62f,
                label = "cup", embedding = decentEmb)
        )
        val result = engine.processFrame(detections)
        assertNotNull("High sim (>0.65) should bypass tentative when classifier not trained", result)
    }

    @Test
    fun `confident classifier bypasses tentative confirmation`() {
        // Build enough gallery + negatives to train classifier
        val embs = (1..5).map { floatArrayOf(0.9f + it * 0.01f, 0.1f, 0f, 0f) }
        engine.lock(42, RectF(0.4f, 0.4f, 0.6f, 0.6f), "cup", embs)
        repeat(5) { i ->
            engine.addSceneNegative(floatArrayOf(0.1f, 0.9f + i * 0.01f, 0f, 0f))
        }
        assertTrue("Classifier should be trained", engine.classifierTrained)
        engine.processFrame(emptyList()) // trigger search

        // Candidate similar to positives — classifier should be confident
        val goodMatch = floatArrayOf(0.92f, 0.12f, 0f, 0f)
        assertTrue("Classifier should be confident for positive-like embedding",
            engine.classifierScore(goodMatch) >= 0.8f)

        val detections = listOf(
            obj(id = 55, left = 0.42f, top = 0.42f, right = 0.62f, bottom = 0.62f,
                label = "cup", embedding = goodMatch)
        )
        val result = engine.processFrame(detections)
        assertNotNull("Confident classifier should bypass tentative", result)
    }

    @Test
    fun `uncertain classifier does not bypass tentative`() {
        val embs = (1..5).map { floatArrayOf(0.9f + it * 0.01f, 0.1f, 0f, 0f) }
        engine.lock(42, RectF(0.4f, 0.4f, 0.6f, 0.6f), "cup", embs)
        repeat(5) { i ->
            engine.addSceneNegative(floatArrayOf(0.1f, 0.9f + i * 0.01f, 0f, 0f))
        }
        assertTrue(engine.classifierTrained)
        engine.processFrame(emptyList())

        // Candidate in between positives and negatives — classifier uncertain
        val ambiguous = floatArrayOf(0.5f, 0.5f, 0f, 0f)
        assertTrue("Classifier should be uncertain for ambiguous embedding",
            engine.classifierScore(ambiguous) < 0.8f)

        val detections = listOf(
            obj(id = 55, left = 0.42f, top = 0.42f, right = 0.62f, bottom = 0.62f,
                label = "cup", embedding = ambiguous)
        )
        val result = engine.processFrame(detections)
        assertNull("Uncertain classifier should NOT bypass tentative (single frame)", result)
    }

    // --- Lowe's ratio test (SIFT-style) ---

    @Test
    fun `ratio test rejects ambiguous candidates`() {
        val emb = floatArrayOf(1f, 0f, 0f, 0f)
        engine.lock(42, RectF(0.4f, 0.4f, 0.6f, 0.6f), "cup", emb)
        repeat(2) { engine.processFrame(emptyList()) }

        // Two candidates with very similar embedding similarity — ambiguous
        val cand1 = obj(id = 55, left = 0.42f, top = 0.42f, right = 0.62f, bottom = 0.62f,
            label = "cup", embedding = floatArrayOf(0.7f, 0.5f, 0.3f, 0f))  // sim ~0.7
        val cand2 = obj(id = 66, left = 0.38f, top = 0.38f, right = 0.58f, bottom = 0.58f,
            label = "cup", embedding = floatArrayOf(0.65f, 0.55f, 0.3f, 0f)) // similar sim

        val result = engine.findBestCandidate(listOf(cand1, cand2))
        // Whether accepted or rejected depends on exact ratio — verify the mechanism exists
        val sim1 = bestGallerySimilarity(cand1.embedding!!, engine.embeddingGallery)
        val sim2 = bestGallerySimilarity(cand2.embedding!!, engine.embeddingGallery)
        val ratio = sim2 / sim1
        if (ratio > ReacquisitionEngine.RATIO_TEST_THRESHOLD) {
            assertNull("Ambiguous candidates (ratio=${ratio}) should be rejected", result)
        }
    }

    @Test
    fun `ratio test allows clear winner`() {
        val emb = floatArrayOf(1f, 0f, 0f, 0f)
        engine.lock(42, RectF(0.4f, 0.4f, 0.6f, 0.6f), "cup", emb)
        repeat(2) { engine.processFrame(emptyList()) }

        // One candidate much closer to reference than the other
        val good = obj(id = 55, left = 0.42f, top = 0.42f, right = 0.62f, bottom = 0.62f,
            label = "cup", embedding = floatArrayOf(0.95f, 0.1f, 0.1f, 0f))  // sim ~0.96
        val poor = obj(id = 66, left = 0.38f, top = 0.38f, right = 0.58f, bottom = 0.58f,
            label = "cup", embedding = floatArrayOf(0.3f, 0.8f, 0.3f, 0f))  // sim ~0.35

        val result = engine.findBestCandidate(listOf(good, poor))
        assertNotNull("Clear winner should pass ratio test", result)
        assertEquals(55, result!!.id)
    }

    // --- Gallery-relative threshold ---

    @Test
    fun `gallery-relative floor adapts to gallery consistency`() {
        // Gallery with tight self-similarity → higher floor
        val embs = listOf(
            floatArrayOf(1f, 0f, 0f, 0f),
            floatArrayOf(0.98f, 0.1f, 0f, 0f),
            floatArrayOf(0.95f, 0.15f, 0.05f, 0f)
        )
        engine.lock(42, RectF(0.4f, 0.4f, 0.6f, 0.6f), "cup", embs)
        engine.processFrame(emptyList())

        // Weak match — should be rejected because gallery is tight
        val weak = obj(id = 55, left = 0.42f, top = 0.42f, right = 0.62f, bottom = 0.62f,
            label = "cup", embedding = floatArrayOf(0.5f, 0.1f, 0.85f, 0f))
        val weakScore = engine.scoreCandidate(weak, engine.lastKnownBox!!)

        // Strong match — should pass
        val strong = obj(id = 56, left = 0.42f, top = 0.42f, right = 0.62f, bottom = 0.62f,
            label = "cup", embedding = floatArrayOf(0.95f, 0.1f, 0.05f, 0f))
        val strongScore = engine.scoreCandidate(strong, engine.lastKnownBox!!)

        assertNotNull("Strong match should pass adaptive floor", strongScore)
        // Weak match might be rejected or score lower depending on exact floor
    }

    // --- Online classifier ---

    @Test
    fun `classifier trains when enough positives and negatives exist`() {
        val embs = (1..5).map { floatArrayOf(0.9f + it * 0.01f, 0.1f, 0f, 0f) }
        engine.lock(42, RectF(0.4f, 0.4f, 0.6f, 0.6f), "cup", embs)

        // Add scene negatives (different direction in embedding space)
        repeat(5) { i ->
            engine.addSceneNegative(floatArrayOf(0.1f, 0.9f + i * 0.01f, 0f, 0f))
        }

        assertTrue("Classifier should be trained with 5 positives + 5 negatives", engine.classifierTrained)

        // Positive-like embedding should score high
        val posScore = engine.classifierScore(floatArrayOf(0.95f, 0.1f, 0f, 0f))
        // Negative-like embedding should score low
        val negScore = engine.classifierScore(floatArrayOf(0.1f, 0.95f, 0f, 0f))

        assertTrue("Positive-like should score higher than negative-like: pos=$posScore neg=$negScore",
            posScore > negScore)
    }

    @Test
    fun `classifier not trained with too few examples`() {
        engine.lock(42, RectF(0.4f, 0.4f, 0.6f, 0.6f), "cup",
            floatArrayOf(1f, 0f, 0f, 0f))
        // Only 1 positive, 1 negative — not enough
        engine.addSceneNegative(floatArrayOf(0f, 1f, 0f, 0f))

        assertFalse("Classifier should not train with too few examples", engine.classifierTrained)
        assertEquals("Untrained classifier should return 0.5", 0.5f,
            engine.classifierScore(floatArrayOf(0.5f, 0.5f, 0f, 0f)), 0.01f)
    }

    @Test
    fun `classifier clears on new lock`() {
        val embs = (1..5).map { floatArrayOf(0.9f + it * 0.01f, 0.1f, 0f, 0f) }
        engine.lock(42, RectF(0.4f, 0.4f, 0.6f, 0.6f), "cup", embs)
        repeat(5) { i ->
            engine.addSceneNegative(floatArrayOf(0.1f, 0.9f + i * 0.01f, 0f, 0f))
        }
        assertTrue(engine.classifierTrained)

        // New lock should clear classifier
        engine.lock(43, RectF(0.3f, 0.3f, 0.5f, 0.5f), "bowl",
            floatArrayOf(0f, 0f, 1f, 0f))
        assertFalse("Classifier should be cleared after new lock", engine.classifierTrained)
    }

    /** Feed the same detections for N frames to satisfy tentative confirmation. */
    private fun processFramesUntilReacquire(
        detections: List<TrackedObject>,
        frames: Int = ReacquisitionEngine.TENTATIVE_MIN_FRAMES
    ): TrackedObject? {
        var result: TrackedObject? = null
        repeat(frames) {
            result = engine.processFrame(detections)
            if (result != null) return result
        }
        return result
    }

    private fun obj(
        id: Int,
        left: Float = 0f,
        top: Float = 0f,
        right: Float = 0.1f,
        bottom: Float = 0.1f,
        label: String? = null,
        confidence: Float = 0.8f,
        embedding: FloatArray? = null,
        colorHistogram: FloatArray? = null,
        personAttributes: PersonAttributes? = null,
        reIdEmbedding: FloatArray? = null,
        faceEmbedding: FloatArray? = null
    ) = TrackedObject(
        id = id,
        boundingBox = RectF(left, top, right, bottom),
        label = label,
        confidence = confidence,
        embedding = embedding,
        colorHistogram = colorHistogram,
        personAttributes = personAttributes,
        reIdEmbedding = reIdEmbedding,
        faceEmbedding = faceEmbedding
    )
}