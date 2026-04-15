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

        val result = engine.processFrame(detections)

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

        val result = engine.processFrame(listOf(far, close))
        assertNotNull(result)
        assertEquals(55, result!!.id)
    }

    // --- Re-acquisition: label breaks tie ---

    @Test
    fun `prefers same-label candidate when position is similar`() {
        val box = RectF(0.4f, 0.4f, 0.6f, 0.6f)
        engine.lock(42, box, "Food")

        val withLabel = obj(id = 55, left = 0.42f, top = 0.42f, right = 0.62f, bottom = 0.62f, label = "Food")
        val noLabel = obj(id = 66, left = 0.41f, top = 0.41f, right = 0.61f, bottom = 0.61f, label = "Home good")

        val result = engine.processFrame(listOf(noLabel, withLabel))
        assertNotNull(result)
        assertEquals(55, result!!.id)
    }

    // --- Hard label filter ---

    @Test
    fun `different label scores lower than same label`() {
        engine.lock(42, RectF(0.4f, 0.4f, 0.6f, 0.6f), "cup")

        val sameLabel = obj(id = 55, left = 0.42f, top = 0.42f, right = 0.62f, bottom = 0.62f, label = "cup")
        val diffLabel = obj(id = 66, left = 0.42f, top = 0.42f, right = 0.62f, bottom = 0.62f, label = "laptop")

        val sameScore = engine.scoreCandidate(sameLabel, engine.lastKnownBox!!)!!
        val diffScore = engine.scoreCandidate(diffLabel, engine.lastKnownBox!!)!!

        assertTrue("Same label ($sameScore) should score higher than different ($diffScore)",
            sameScore > diffScore)
    }

    @Test
    fun `different label can still reacquire if only candidate`() {
        // Label is soft — a nearby different-label object should not be hard-rejected
        engine.lock(42, RectF(0.4f, 0.4f, 0.6f, 0.6f), "cup")

        val detections = listOf(
            obj(id = 55, left = 0.42f, top = 0.42f, right = 0.62f, bottom = 0.62f, label = "laptop")
        )
        // Lose the locked id first
        engine.processFrame(emptyList())
        val score = engine.scoreCandidate(detections[0], engine.lastKnownBox!!)
        // Score may be low, but the candidate was not hard-rejected (score != null)
        assertNotNull("Different-label candidate should not be hard-rejected", score)
    }

    @Test
    fun `prefers same-label candidate over different-label nearby`() {
        engine.lock(42, RectF(0.4f, 0.4f, 0.6f, 0.6f), "cup")

        // Two candidates: wrong label nearby, right label further
        val wrong = obj(id = 55, left = 0.42f, top = 0.42f, right = 0.62f, bottom = 0.62f, label = "laptop")
        val right = obj(id = 66, left = 0.35f, top = 0.35f, right = 0.55f, bottom = 0.55f, label = "cup")

        val result = engine.processFrame(listOf(wrong, right))
        assertNotNull(result)
        assertEquals("Should prefer same-label candidate", 66, result!!.id)
        assertEquals("cup", result.label)
    }

    @Test
    fun `allows reacquisition when locked object had no label`() {
        engine.lock(42, RectF(0.4f, 0.4f, 0.6f, 0.6f), null)

        val detections = listOf(
            obj(id = 55, left = 0.42f, top = 0.42f, right = 0.62f, bottom = 0.62f, label = "laptop")
        )
        val result = engine.processFrame(detections)
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

        val result = engine.processFrame(detections)
        assertNotNull("Should reacquire distant same-label object after position decay", result)
        assertEquals(192, result!!.id)
    }

    @Test
    fun `does not reacquire distant wrong-label object after many lost frames`() {
        val box = RectF(0.4f, 0.4f, 0.6f, 0.6f)
        engine.lock(42, box, "Food")

        repeat(engine.positionDecayFrames) { engine.processFrame(emptyList()) }

        // Different label, different position
        val detections = listOf(
            obj(id = 99, left = 0.1f, top = 0.7f, right = 0.3f, bottom = 0.9f, label = "Home good")
        )

        val result = engine.processFrame(detections)
        assertNull("Distant wrong-label object should not reacquire", result)
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
        val result = engine.processFrame(detections)
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
    fun `scoreCandidate boosts same-label match`() {
        val refBox = RectF(0.4f, 0.4f, 0.6f, 0.6f)
        engine.lock(42, refBox, "Food")

        val withLabel = obj(id = 1, left = 0.42f, top = 0.42f, right = 0.62f, bottom = 0.62f, label = "Food")
        val noLabel = obj(id = 2, left = 0.42f, top = 0.42f, right = 0.62f, bottom = 0.62f, label = "Home good")

        val labelScore = engine.scoreCandidate(withLabel, refBox)!!
        val noLabelScore = engine.scoreCandidate(noLabel, refBox)!!

        assertTrue("Same label ($labelScore) should score higher ($noLabelScore)",
            labelScore > noLabelScore)
    }

    @Test
    fun `label weight increases after position decay`() {
        val refBox = RectF(0.4f, 0.4f, 0.6f, 0.6f)
        engine.lock(42, refBox, "Food")

        val candidate = obj(id = 1, left = 0.42f, top = 0.42f, right = 0.62f, bottom = 0.62f, label = "Food")
        val candidateWrong = obj(id = 2, left = 0.42f, top = 0.42f, right = 0.62f, bottom = 0.62f, label = "Home good")

        // Early: label bonus
        val earlyGap = engine.scoreCandidate(candidate, refBox)!! -
                       engine.scoreCandidate(candidateWrong, refBox)!!

        // Lose frames so position decays
        repeat(engine.positionDecayFrames) { engine.processFrame(emptyList()) }

        val lateGap = engine.scoreCandidate(candidate, refBox)!! -
                      engine.scoreCandidate(candidateWrong, refBox)!!

        assertTrue("Label should matter more after decay: earlyGap=$earlyGap, lateGap=$lateGap",
            lateGap > earlyGap)
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
            .copy(embedding = floatArrayOf(0f, 0f, 1f))  // orthogonal

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
            .copy(embedding = floatArrayOf(0f, 0.1f, 0.9f))  // different visual

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
            .copy(embedding = floatArrayOf(0f, 0f, 1f))

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
    fun `strong embedding overrides label mismatch`() {
        // Lock on "bowl" — detector later calls same object "potted plant"
        val lockedEmb = floatArrayOf(0.8f, 0.5f, 0.2f)
        engine.lock(42, RectF(0.3f, 0.3f, 0.7f, 0.7f), "bowl", lockedEmb)

        engine.processFrame(emptyList())

        // Same object, strong embedding, but detector says "potted plant"
        val candidate = obj(id = 55, left = 0.32f, top = 0.32f, right = 0.68f, bottom = 0.68f, label = "potted plant")
            .copy(embedding = floatArrayOf(0.75f, 0.55f, 0.2f))  // sim ~0.98

        val result = engine.processFrame(listOf(candidate))
        assertNotNull("Strong embedding should override label mismatch", result)
        assertEquals(55, result!!.id)
    }

    @Test
    fun `weak embedding with wrong label scores lower than strong with right label`() {
        val lockedEmb = floatArrayOf(0.8f, 0.5f, 0.2f)
        engine.lock(42, RectF(0.3f, 0.3f, 0.7f, 0.7f), "bowl", lockedEmb)

        engine.processFrame(emptyList())

        val strongRight = obj(id = 55, left = 0.32f, top = 0.32f, right = 0.68f, bottom = 0.68f, label = "bowl")
            .copy(embedding = floatArrayOf(0.75f, 0.55f, 0.2f))  // high sim, right label
        val weakWrong = obj(id = 66, left = 0.32f, top = 0.32f, right = 0.68f, bottom = 0.68f, label = "chair")
            .copy(embedding = floatArrayOf(0f, 0f, 1f))  // low sim, wrong label

        val strongScore = engine.scoreCandidate(strongRight, engine.lastKnownBox!!)!!
        val weakScore = engine.scoreCandidate(weakWrong, engine.lastKnownBox!!)!!

        assertTrue("Strong+right ($strongScore) should beat weak+wrong ($weakScore)",
            strongScore > weakScore)
    }

    @Test
    fun `label override prefers correct label when both available`() {
        val lockedEmb = floatArrayOf(0.8f, 0.5f, 0.2f)
        engine.lock(42, RectF(0.3f, 0.3f, 0.7f, 0.7f), "bowl", lockedEmb)

        engine.processFrame(emptyList())

        // Two candidates: same object mislabeled, and correctly labeled
        val mislabeled = obj(id = 55, left = 0.32f, top = 0.32f, right = 0.68f, bottom = 0.68f, label = "potted plant")
            .copy(embedding = floatArrayOf(0.75f, 0.55f, 0.2f))  // high sim
        val correctLabel = obj(id = 66, left = 0.32f, top = 0.32f, right = 0.68f, bottom = 0.68f, label = "bowl")
            .copy(embedding = floatArrayOf(0.75f, 0.55f, 0.2f))  // same high sim

        val result = engine.processFrame(listOf(mislabeled, correctLabel))
        assertNotNull(result)
        // Both should score, correct label should win (gets label score bonus)
        assertEquals("Should prefer correct label", 66, result!!.id)
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
            .copy(embedding = floatArrayOf(0.1f, 0.2f, 0.9f))

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
    fun `label flicker during tracking does not corrupt re-acquisition scoring`() {
        // Lock on a bowl
        engine.lock(1, RectF(0.4f, 0.4f, 0.6f, 0.6f), "bowl")
        assertEquals("bowl", engine.lockedLabel)

        // Direct match with flickered label — updates lastKnownLabel
        val flickered = listOf(
            obj(id = 1, left = 0.41f, top = 0.41f, right = 0.61f, bottom = 0.61f, label = "potted plant")
        )
        engine.processFrame(flickered)
        assertEquals("potted plant", engine.lastKnownLabel)
        assertEquals("bowl", engine.lockedLabel) // lockedLabel unchanged

        // Object lost — re-acquisition should prefer "bowl", not "potted plant"
        engine.lock(1, RectF(0.4f, 0.4f, 0.6f, 0.6f), "bowl")
        // Simulate loss
        val noMatch = listOf(
            obj(id = 99, left = 0.4f, top = 0.4f, right = 0.6f, bottom = 0.6f, label = "potted plant"),
            obj(id = 100, left = 0.41f, top = 0.41f, right = 0.61f, bottom = 0.61f, label = "bowl")
        )
        // Lose the locked id
        repeat(2) { engine.processFrame(emptyList()) }

        val result = engine.findBestCandidate(noMatch)
        assertEquals("bowl", result?.label)
    }

    // --- Helpers ---

    private fun obj(
        id: Int,
        left: Float = 0f,
        top: Float = 0f,
        right: Float = 0.1f,
        bottom: Float = 0.1f,
        label: String? = null,
        confidence: Float = 0.8f
    ) = TrackedObject(
        id = id,
        boundingBox = RectF(left, top, right, bottom),
        label = label,
        confidence = confidence
    )
}