package com.haptictrack.tracking

import android.graphics.RectF
import org.junit.Assert.*
import org.junit.Before
import org.junit.Test
import org.junit.runner.RunWith
import org.robolectric.RobolectricTestRunner

@RunWith(RobolectricTestRunner::class)
class FrameToFrameTrackerTest {

    private lateinit var tracker: FrameToFrameTracker

    @Before
    fun setup() {
        tracker = FrameToFrameTracker()
    }

    @Test
    fun `first frame assigns fresh IDs`() {
        val detections = listOf(
            raw(0.1f, 0.1f, 0.3f, 0.3f),
            raw(0.5f, 0.5f, 0.8f, 0.8f)
        )
        val result = tracker.assignIds(detections)
        assertEquals(2, result.size)
        assertTrue(result[0].id > 0)
        assertTrue(result[1].id > 0)
        assertNotEquals(result[0].id, result[1].id)
    }

    @Test
    fun `same object in next frame keeps same ID`() {
        val frame1 = listOf(raw(0.1f, 0.1f, 0.4f, 0.4f))
        val ids1 = tracker.assignIds(frame1)

        // Slight movement
        val frame2 = listOf(raw(0.12f, 0.12f, 0.42f, 0.42f))
        val ids2 = tracker.assignIds(frame2)

        assertEquals(ids1[0].id, ids2[0].id)
    }

    @Test
    fun `new object gets new ID`() {
        val frame1 = listOf(raw(0.1f, 0.1f, 0.3f, 0.3f))
        val ids1 = tracker.assignIds(frame1)

        // Original object + new object far away
        val frame2 = listOf(
            raw(0.12f, 0.12f, 0.32f, 0.32f),
            raw(0.7f, 0.7f, 0.9f, 0.9f)
        )
        val ids2 = tracker.assignIds(frame2)

        assertEquals(ids1[0].id, ids2[0].id)
        assertNotEquals(ids1[0].id, ids2[1].id)
    }

    @Test
    fun `object that disappears loses its ID`() {
        val frame1 = listOf(
            raw(0.1f, 0.1f, 0.3f, 0.3f),
            raw(0.5f, 0.5f, 0.8f, 0.8f)
        )
        val ids1 = tracker.assignIds(frame1)

        // Only second object remains
        val frame2 = listOf(raw(0.52f, 0.52f, 0.82f, 0.82f))
        val ids2 = tracker.assignIds(frame2)

        assertEquals(ids1[1].id, ids2[0].id)
    }

    @Test
    fun `non-overlapping objects get different IDs`() {
        val frame1 = listOf(raw(0.0f, 0.0f, 0.2f, 0.2f))
        tracker.assignIds(frame1)

        val frame2 = listOf(raw(0.8f, 0.8f, 1.0f, 1.0f))
        val ids2 = tracker.assignIds(frame2)

        // Should get a new ID since there's no overlap
        assertNotEquals(1, ids2[0].id) // ID 1 was the first object
    }

    @Test
    fun `preserves label and confidence through tracking`() {
        val frame1 = listOf(raw(0.1f, 0.1f, 0.4f, 0.4f, label = "Food", confidence = 0.9f))
        tracker.assignIds(frame1)

        val frame2 = listOf(raw(0.12f, 0.12f, 0.42f, 0.42f, label = "Food", confidence = 0.85f))
        val ids2 = tracker.assignIds(frame2)

        assertEquals("Food", ids2[0].label)
        assertEquals(0.85f, ids2[0].confidence, 0.01f)
    }

    // --- IoU tests ---

    @Test
    fun `computeIou for identical boxes is 1`() {
        val box = RectF(0.1f, 0.1f, 0.5f, 0.5f)
        assertEquals(1f, FrameToFrameTracker.computeIou(box, box), 0.001f)
    }

    @Test
    fun `computeIou for non-overlapping boxes is 0`() {
        val a = RectF(0f, 0f, 0.2f, 0.2f)
        val b = RectF(0.5f, 0.5f, 0.8f, 0.8f)
        assertEquals(0f, FrameToFrameTracker.computeIou(a, b), 0.001f)
    }

    @Test
    fun `computeIou for partial overlap`() {
        val a = RectF(0f, 0f, 0.4f, 0.4f)
        val b = RectF(0.2f, 0.2f, 0.6f, 0.6f)
        val iou = FrameToFrameTracker.computeIou(a, b)
        assertTrue("IoU should be between 0 and 1, got $iou", iou > 0f && iou < 1f)
    }

    @Test
    fun `reset clears state`() {
        tracker.assignIds(listOf(raw(0.1f, 0.1f, 0.3f, 0.3f)))
        tracker.reset()

        val result = tracker.assignIds(listOf(raw(0.1f, 0.1f, 0.3f, 0.3f)))
        assertEquals(1, result[0].id) // IDs restart from 1
    }

    // --- Helpers ---

    private fun raw(
        left: Float, top: Float, right: Float, bottom: Float,
        label: String? = null, confidence: Float = 0.8f
    ) = TrackedObject(
        id = -1,
        boundingBox = RectF(left, top, right, bottom),
        label = label,
        confidence = confidence
    )
}
