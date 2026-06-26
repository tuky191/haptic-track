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
        var zoomChanges = 0
        val ctrl = SentryController(
            criteria = { criteria },
            setZoomTarget = { if (it != zoom) zoomChanges++; zoom = it },
            currentZoom = { zoom },
            minZoom = { 1f }, maxZoom = { 10f },
            classify = { classifyResult(it) },
            lock = { locked = it },
            haptic = { cues.add(it) },
        )
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
    fun `centered person enters inspection after confirm frames`() {
        val h = Harness()
        h.ctrl.setEnabled(true)
        h.feed(person(1, 0.5f, 0.5f), CONFIRM)
        assertEquals(SentryController.State.INSPECTING, h.ctrl.state)
        assertTrue(SentryCue.INSPECTING in h.cues)
    }

    @Test
    fun `single centered frame does not commit to inspection`() {
        val h = Harness()
        h.ctrl.setEnabled(true)
        h.ctrl.onFrame(listOf(person(1, 0.5f, 0.5f)))
        assertEquals(SentryController.State.SCANNING, h.ctrl.state)
    }

    @Test
    fun `off-center person is not inspected`() {
        val h = Harness()
        h.ctrl.setEnabled(true)
        h.feed(person(1, 0.95f, 0.5f), CONFIRM + 2)
        assertEquals(SentryController.State.SCANNING, h.ctrl.state)
    }

    @Test
    fun `classifiable match locks WITHOUT moving the camera`() {
        val h = Harness(criteria = SentryCriteria(GenderFilter.MALE, 30, 50))
        h.classifyResult = { FaceAttributes(isMale = true, age = 40, genderConfidence = 5f) }
        h.ctrl.setEnabled(true)
        h.feed(person(1, 0.5f, 0.5f), CONFIRM + 2)
        assertEquals(SentryController.State.MATCHED, h.ctrl.state)
        assertNotNull(h.locked)
        assertEquals("must not zoom for a readable subject", SentryController.SCAN_ZOOM, h.zoom, 1e-4f)
    }

    @Test
    fun `classifiable non-match does NOT yo-yo the zoom`() {
        // The reported bug: a readable non-match (e.g. a boy when filtering for women) caused
        // constant zoom in/out. It must classify at the current zoom and never move the camera.
        val h = Harness(criteria = SentryCriteria(GenderFilter.FEMALE))
        h.classifyResult = { FaceAttributes(isMale = true, age = 12, genderConfidence = 5f) }
        h.ctrl.setEnabled(true)
        h.feed(person(1, 0.5f, 0.5f), CONFIRM + 30)  // well past a settle cycle
        assertEquals(SentryController.State.SCANNING, h.ctrl.state)
        assertNull(h.locked)
        assertEquals("readable non-match must not move the camera", 0, h.zoomChanges)
    }

    @Test
    fun `unreadable face triggers zoom-in`() {
        // No face at the current zoom -> the controller should zoom in to enlarge it.
        val h = Harness()
        h.classifyResult = { null }
        h.ctrl.setEnabled(true)
        // Enough frames for the eased ramp to clearly engage (stays under INSPECT_TIMEOUT).
        h.feed(person(1, 0.5f, 0.5f, w = 0.06f, h = 0.12f), CONFIRM + 20)
        assertTrue("should zoom in when no face is found", h.zoom > SentryController.SCAN_ZOOM)
    }

    @Test
    fun `no usable face times out to rejection`() {
        val h = Harness()
        h.classifyResult = { null }
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
        h.feed(person(1, 0.5f, 0.5f), CONFIRM)         // enter INSPECTING
        h.ctrl.onFrame(emptyList())                     // one dropped frame — must NOT abort
        assertEquals(SentryController.State.INSPECTING, h.ctrl.state)
        h.feed(person(1, 0.5f, 0.5f), 2)                // recovers, classifies, matches
        assertEquals(SentryController.State.MATCHED, h.ctrl.state)
    }

    @Test
    fun `matched is terminal until the lock is cleared`() {
        val h = Harness(criteria = SentryCriteria(GenderFilter.MALE))
        h.classifyResult = { FaceAttributes(isMale = true, age = 30, genderConfidence = 5f) }
        h.ctrl.setEnabled(true)
        h.feed(person(1, 0.5f, 0.5f), CONFIRM + 2)
        assertEquals(SentryController.State.MATCHED, h.ctrl.state)
        // Further frames must not re-inspect/re-lock while MATCHED — it owns the lock.
        h.feed(person(2, 0.8f, 0.5f), 10)
        assertEquals(SentryController.State.MATCHED, h.ctrl.state)
    }

    @Test
    fun `onLockCleared re-arms scanning so the sentry resumes after a lost lock`() {
        // This is the controller half of the auto-rearm fix: when the lock is given up,
        // the owner calls onLockCleared() and scanning must resume (and be able to re-lock).
        val h = Harness(criteria = SentryCriteria(GenderFilter.MALE))
        h.classifyResult = { FaceAttributes(isMale = true, age = 30, genderConfidence = 5f) }
        h.ctrl.setEnabled(true)
        h.feed(person(1, 0.5f, 0.5f), CONFIRM + 2)
        assertEquals(SentryController.State.MATCHED, h.ctrl.state)

        h.ctrl.onLockCleared()
        assertEquals(SentryController.State.SCANNING, h.ctrl.state)

        // Scanning genuinely resumed — a fresh candidate is inspected and re-locked.
        h.feed(person(1, 0.5f, 0.5f), CONFIRM + 2)
        assertEquals(SentryController.State.MATCHED, h.ctrl.state)
    }

    @Test
    fun `onLockCleared is a no-op when not matched`() {
        val h = Harness()
        h.ctrl.setEnabled(true)
        h.ctrl.onLockCleared()  // scanning, no lock — must not throw or change state
        assertEquals(SentryController.State.SCANNING, h.ctrl.state)
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
