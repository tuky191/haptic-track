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
            minZoom = { 1f }, maxZoom = { 10f },
            classify = { classifyResult(it) },
            lock = { locked = it },
            haptic = { cues.add(it) },
        )
    }

    @Test
    fun `enabling arms scanning`() {
        val h = Harness()
        h.ctrl.setEnabled(true)
        assertEquals(SentryController.State.SCANNING, h.ctrl.state)
        assertTrue(SentryCue.SCANNING in h.cues)
    }

    @Test
    fun `centered person triggers inspection and zoom-in`() {
        val h = Harness()
        h.ctrl.setEnabled(true)
        h.ctrl.onFrame(listOf(person(1, 0.5f, 0.5f)))
        assertEquals(SentryController.State.INSPECTING, h.ctrl.state)
        assertTrue("should zoom in past scan zoom", h.zoom > SentryController.SCAN_ZOOM)
        assertTrue(SentryCue.INSPECTING in h.cues)
    }

    @Test
    fun `off-center person is not inspected`() {
        val h = Harness()
        h.ctrl.setEnabled(true)
        h.ctrl.onFrame(listOf(person(1, 0.95f, 0.5f)))  // far right
        assertEquals(SentryController.State.SCANNING, h.ctrl.state)
    }

    @Test
    fun `matching candidate locks and records`() {
        val h = Harness(criteria = SentryCriteria(GenderFilter.MALE, 30, 50))
        h.classifyResult = { FaceAttributes(isMale = true, age = 40, genderConfidence = 5f) }
        h.ctrl.setEnabled(true)
        repeat(SentryController.CLASSIFY_INTERVAL) { h.ctrl.onFrame(listOf(person(1, 0.5f, 0.5f))) }
        assertEquals(SentryController.State.MATCHED, h.ctrl.state)
        assertNotNull("should have locked", h.locked)
        assertTrue(SentryCue.MATCH in h.cues)
    }

    @Test
    fun `non-matching candidate is rejected and cooled down`() {
        val h = Harness(criteria = SentryCriteria(GenderFilter.FEMALE))
        h.classifyResult = { FaceAttributes(isMale = true, age = 40, genderConfidence = 5f) }  // male, filter wants female
        h.ctrl.setEnabled(true)
        repeat(SentryController.CLASSIFY_INTERVAL) { h.ctrl.onFrame(listOf(person(1, 0.5f, 0.5f))) }
        assertEquals(SentryController.State.SCANNING, h.ctrl.state)
        assertNull(h.locked)
        assertTrue(SentryCue.REJECT in h.cues)
        // Same candidate must NOT be re-inspected immediately (cooldown).
        h.ctrl.onFrame(listOf(person(1, 0.5f, 0.5f)))
        assertEquals(SentryController.State.SCANNING, h.ctrl.state)
    }

    @Test
    fun `no usable face times out to rejection`() {
        val h = Harness()
        h.classifyResult = { null }  // never a face
        h.ctrl.setEnabled(true)
        repeat(SentryController.INSPECT_TIMEOUT + 2) { h.ctrl.onFrame(listOf(person(1, 0.5f, 0.5f))) }
        assertEquals(SentryController.State.SCANNING, h.ctrl.state)
        assertTrue(SentryCue.REJECT in h.cues)
    }

    @Test
    fun `disabling stops the machine`() {
        val h = Harness()
        h.ctrl.setEnabled(true)
        h.ctrl.onFrame(listOf(person(1, 0.5f, 0.5f)))
        h.ctrl.setEnabled(false)
        assertEquals(SentryController.State.OFF, h.ctrl.state)
        h.ctrl.onFrame(listOf(person(1, 0.5f, 0.5f)))
        assertEquals(SentryController.State.OFF, h.ctrl.state)
    }
}
