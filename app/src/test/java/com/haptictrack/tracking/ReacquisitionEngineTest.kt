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

    // --- Re-acquisition: nearby same-label object ---

    @Test
    fun `reacquires nearby object with same label after ID changes`() {
        val box = RectF(0.4f, 0.4f, 0.6f, 0.6f)
        engine.lock(42, box, "Food")

        // ID 42 is gone, but a new object appeared nearby with same label
        val detections = listOf(
            obj(id = 55, left = 0.42f, top = 0.42f, right = 0.62f, bottom = 0.62f, label = "Food")
        )

        val result = engine.processFrame(detections)

        assertNotNull(result)
        assertEquals(55, result!!.id)
        assertEquals(55, engine.lockedId) // Updated to new ID
        assertEquals(0, engine.framesLost)
    }

    // --- Re-acquisition: reject distant objects ---

    @Test
    fun `rejects candidate that is too far away`() {
        val box = RectF(0.1f, 0.1f, 0.3f, 0.3f) // top-left area
        engine.lock(42, box, "Food")

        // Object on opposite side of frame
        val detections = listOf(
            obj(id = 55, left = 0.7f, top = 0.7f, right = 0.9f, bottom = 0.9f, label = "Food")
        )

        val result = engine.processFrame(detections)

        assertNull(result)
        assertEquals(1, engine.framesLost)
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

    // --- Re-acquisition: prefer closer candidate ---

    @Test
    fun `prefers closer candidate over farther one`() {
        val box = RectF(0.4f, 0.4f, 0.6f, 0.6f)
        engine.lock(42, box, null)

        // Both same size as original, just at different distances
        val close = obj(id = 55, left = 0.42f, top = 0.42f, right = 0.62f, bottom = 0.62f)
        val far = obj(id = 66, left = 0.3f, top = 0.3f, right = 0.5f, bottom = 0.5f)

        // Verify scoring directly: closer should score higher
        val closeScore = engine.scoreCandidate(close, engine.lastKnownBox!!)
        val farScore = engine.scoreCandidate(far, engine.lastKnownBox!!)

        assertNotNull("Close candidate should be scoreable", closeScore)
        assertNotNull("Far candidate should be scoreable", farScore)
        assertTrue("Close ($closeScore) should score higher than far ($farScore)",
            closeScore!! > farScore!!)

        // And processFrame should pick the closer one
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

    // --- Timeout ---

    @Test
    fun `stops reacquiring after maxFramesLost`() {
        engine = ReacquisitionEngine(maxFramesLost = 3)
        val box = RectF(0.4f, 0.4f, 0.6f, 0.6f)
        engine.lock(42, box, "Food")

        val emptyFrame = emptyList<TrackedObject>()

        // Lose for 3 frames
        assertNull(engine.processFrame(emptyFrame))
        assertNull(engine.processFrame(emptyFrame))
        assertNull(engine.processFrame(emptyFrame))

        assertFalse(engine.hasTimedOut)

        // Frame 4: timed out
        assertNull(engine.processFrame(emptyFrame))
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

        // Lose for 5 frames
        repeat(5) { engine.processFrame(emptyList()) }
        assertEquals(5, engine.framesLost)

        // Reacquire
        val detections = listOf(
            obj(id = 55, left = 0.42f, top = 0.42f, right = 0.62f, bottom = 0.62f, label = "Food")
        )
        val result = engine.processFrame(detections)
        assertNotNull(result)
        assertEquals(0, engine.framesLost)
    }

    // --- Score calculation ---

    @Test
    fun `scoreCandidate returns null for object beyond position threshold`() {
        engine.lock(42, RectF(0.1f, 0.1f, 0.2f, 0.2f), null)
        val far = obj(id = 1, left = 0.8f, top = 0.8f, right = 0.9f, bottom = 0.9f)
        assertNull(engine.scoreCandidate(far, engine.lastKnownBox!!))
    }

    @Test
    fun `scoreCandidate returns null for object beyond size threshold`() {
        engine.lock(42, RectF(0.45f, 0.45f, 0.55f, 0.55f), null) // tiny
        val huge = obj(id = 1, left = 0.1f, top = 0.1f, right = 0.9f, bottom = 0.9f)
        assertNull(engine.scoreCandidate(huge, engine.lastKnownBox!!))
    }

    @Test
    fun `scoreCandidate gives higher score to closer objects`() {
        val refBox = RectF(0.4f, 0.4f, 0.6f, 0.6f)
        engine.lock(42, refBox, null)

        val close = obj(id = 1, left = 0.41f, top = 0.41f, right = 0.61f, bottom = 0.61f)
        val medium = obj(id = 2, left = 0.3f, top = 0.3f, right = 0.5f, bottom = 0.5f)

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

    // --- Edge cases ---

    @Test
    fun `handles detection with negative ID gracefully`() {
        engine.lock(42, RectF(0.4f, 0.4f, 0.6f, 0.6f), null)

        val detections = listOf(
            obj(id = -1, left = 0.42f, top = 0.42f, right = 0.62f, bottom = 0.62f)
        )

        // Should not reacquire to an invalid ID
        assertNull(engine.processFrame(detections))
    }

    @Test
    fun `updates lastKnownLabel only when detection has a label`() {
        engine.lock(42, RectF(0.4f, 0.4f, 0.6f, 0.6f), "Food")

        // Direct match with no label — should keep "Food"
        val detections = listOf(
            obj(id = 42, left = 0.41f, top = 0.41f, right = 0.61f, bottom = 0.61f, label = null)
        )
        engine.processFrame(detections)
        assertEquals("Food", engine.lastKnownLabel)
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