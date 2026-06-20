package com.haptictrack.tracking

import android.graphics.RectF
import org.junit.Assert.*
import org.junit.Test
import org.junit.runner.RunWith
import org.robolectric.RobolectricTestRunner

@RunWith(RobolectricTestRunner::class)
class SentryControllerTest {

    private fun person(id: Int, cx: Float, cy: Float, w: Float = 0.2f, h: Float = 0.5f) =
        TrackedObject(id = id, boundingBox = RectF(cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2), label = "person")

    private class Harness(
        var criteria: SentryCriteria = SentryCriteria(),
        var classifyResult: (TrackedObject) -> FaceAttributes? = { null },
    ) {
        var zoom = 1f; var locked: TrackedObject? = null
        val cues = mutableListOf<SentryCue>()
        val ctrl = SentryController(
            criteria = { criteria },
            setZoomTarget = { zoom = it },
            currentZoom = { zoom },
            minZoom = { 1f }, maxZoom = { 10f },
            classify = { classifyResult(it) },
            lock = { locked = it },
            haptic = { cues.add(it) },
        )
        /** Feed the same centered person for N frames. */
        fun feed(p: TrackedObject, n: Int) = repeat(n) { ctrl.onFrame(listOf(p)) }
    }

    private val CONFIRM = SentryController.CONFIRM_FRAMES

    @Test
    fun `enabling arms scanning`() {
        val h = Harness()
        h.ctrl.setEnabled(true)
        assertEquals(SentryController.State.SCANNING, h.ctrl.state)
        assertTrue(SentryCue.SCANNING in h.cues)
    }

    @Test
    fun `centered person triggers inspection and zoom-in after confirm frames`() {
        val h = Harness()
        h.ctrl.setEnabled(true)
        h.feed(person(1, 0.5f, 0.5f), CONFIRM)
        assertEquals(SentryController.State.INSPECTING, h.ctrl.state)
        assertTrue("should zoom in past scan zoom", h.zoom > SentryController.SCAN_ZOOM)
        assertTrue(SentryCue.INSPECTING in h.cues)
    }

    @Test
    fun `single centered frame does not commit to inspection`() {
        val h = Harness()
        h.ctrl.setEnabled(true)
        h.ctrl.onFrame(listOf(person(1, 0.5f, 0.5f)))  // one frame only
        assertEquals(SentryController.State.SCANNING, h.ctrl.state)
    }

    @Test
    fun `off-center person is not inspected`() {
        val h = Harness()
        h.ctrl.setEnabled(true)
        h.feed(person(1, 0.95f, 0.5f), CONFIRM + 2)  // far right, never centers
        assertEquals(SentryController.State.SCANNING, h.ctrl.state)
    }

    @Test
    fun `matching candidate locks and records`() {
        val h = Harness(criteria = SentryCriteria(GenderFilter.MALE, 30, 50))
        h.classifyResult = { FaceAttributes(isMale = true, age = 40, genderConfidence = 5f) }
        h.ctrl.setEnabled(true)
        h.feed(person(1, 0.5f, 0.5f), CONFIRM + 2)
        assertEquals(SentryController.State.MATCHED, h.ctrl.state)
        assertNotNull("should have locked", h.locked)
        assertTrue(SentryCue.MATCH in h.cues)
    }

    @Test
    fun `non-matching candidate is rejected and cooled down`() {
        val h = Harness(criteria = SentryCriteria(GenderFilter.FEMALE))
        h.classifyResult = { FaceAttributes(isMale = true, age = 40, genderConfidence = 5f) }  // male, filter female
        h.ctrl.setEnabled(true)
        h.feed(person(1, 0.5f, 0.5f), CONFIRM + 2)
        assertEquals(SentryController.State.SCANNING, h.ctrl.state)
        assertNull(h.locked)
        assertTrue(SentryCue.REJECT in h.cues)
        // Same centered person must NOT be re-inspected (IoU cooldown holds it).
        h.feed(person(1, 0.5f, 0.5f), CONFIRM + 2)
        assertEquals(SentryController.State.SCANNING, h.ctrl.state)
    }

    @Test
    fun `no usable face times out to rejection`() {
        val h = Harness()
        h.classifyResult = { null }  // never a face
        h.ctrl.setEnabled(true)
        h.feed(person(1, 0.5f, 0.5f), CONFIRM + SentryController.INSPECT_TIMEOUT + 2)
        assertEquals(SentryController.State.SCANNING, h.ctrl.state)
        assertTrue(SentryCue.REJECT in h.cues)
    }

    @Test
    fun `inspect tolerates brief detection dropouts`() {
        val h = Harness(criteria = SentryCriteria(GenderFilter.MALE))
        h.classifyResult = { FaceAttributes(isMale = true, age = 30, genderConfidence = 5f) }
        h.ctrl.setEnabled(true)
        h.feed(person(1, 0.5f, 0.5f), CONFIRM)        // enter INSPECTING
        h.ctrl.onFrame(emptyList())                    // one dropped frame — must NOT abort
        assertEquals(SentryController.State.INSPECTING, h.ctrl.state)
        h.feed(person(1, 0.5f, 0.5f), 2)               // recovers, classifies, matches
        assertEquals(SentryController.State.MATCHED, h.ctrl.state)
    }

    @Test
    fun `inspect zoom converges and does not oscillate`() {
        // A far subject (small box) should ramp zoom UP monotonically toward the target,
        // never back down — the bug was target computed from a fixed base, causing in/out.
        val h = Harness(criteria = SentryCriteria(GenderFilter.FEMALE))  // never matches male -> stays inspecting
        h.classifyResult = { null }
        h.ctrl.setEnabled(true)
        h.feed(person(1, 0.5f, 0.5f, w = 0.06f, h = 0.12f), CONFIRM)  // small/far
        val zooms = mutableListOf(h.zoom)
        // Simulate the subject growing as we zoom in (box height tracks zoom).
        repeat(6) {
            val z = h.zoom
            val grownH = (0.12f * z).coerceAtMost(0.95f)
            h.ctrl.onFrame(listOf(person(1, 0.5f, 0.5f, w = grownH * 0.4f, h = grownH)))
            zooms.add(h.zoom)
        }
        // Monotonic non-decreasing (allow tiny float noise), and it settles (last two ~equal).
        for (i in 1 until zooms.size) assertTrue("zoom dipped: $zooms", zooms[i] >= zooms[i - 1] - 0.05f)
        assertTrue("should converge", kotlin.math.abs(zooms.last() - zooms[zooms.size - 2]) < 0.3f)
    }

    @Test
    fun `disabling stops the machine`() {
        val h = Harness()
        h.ctrl.setEnabled(true)
        h.feed(person(1, 0.5f, 0.5f), CONFIRM)
        h.ctrl.setEnabled(false)
        assertEquals(SentryController.State.OFF, h.ctrl.state)
        h.ctrl.onFrame(listOf(person(1, 0.5f, 0.5f)))
        assertEquals(SentryController.State.OFF, h.ctrl.state)
    }
}
