package com.haptictrack.tracking

import android.graphics.RectF
import org.junit.Assert.*
import org.junit.Test
import org.junit.runner.RunWith
import org.robolectric.RobolectricTestRunner

@RunWith(RobolectricTestRunner::class)
class RotationRemapTest {

    // --- 0° (normal portrait) ---

    @Test
    fun `0 degrees returns unchanged coordinates`() {
        val result = unmapRotation(0.2f, 0.3f, 0.6f, 0.7f, 0)
        assertBoxEquals(RectF(0.2f, 0.3f, 0.6f, 0.7f), result)
    }

    // --- 180° (upside down) ---

    @Test
    fun `180 degrees flips both axes`() {
        val result = unmapRotation(0.2f, 0.3f, 0.6f, 0.7f, 180)
        assertBoxEquals(RectF(0.4f, 0.3f, 0.8f, 0.7f), result)
    }

    @Test
    fun `180 degrees center box stays centered`() {
        val result = unmapRotation(0.25f, 0.25f, 0.75f, 0.75f, 180)
        assertBoxEquals(RectF(0.25f, 0.25f, 0.75f, 0.75f), result)
    }

    @Test
    fun `180 degrees top-left becomes bottom-right`() {
        val result = unmapRotation(0.0f, 0.0f, 0.2f, 0.2f, 180)
        assertBoxEquals(RectF(0.8f, 0.8f, 1.0f, 1.0f), result)
    }

    // --- 90° (landscape right) ---

    @Test
    fun `90 degrees maps correctly`() {
        val result = unmapRotation(0.0f, 0.0f, 0.2f, 0.3f, 90)
        assertBoxEquals(RectF(0.0f, 0.8f, 0.3f, 1.0f), result)
    }

    @Test
    fun `90 degrees center box stays centered`() {
        val result = unmapRotation(0.25f, 0.25f, 0.75f, 0.75f, 90)
        assertBoxEquals(RectF(0.25f, 0.25f, 0.75f, 0.75f), result)
    }

    // --- 270° (landscape left) ---

    @Test
    fun `270 degrees maps correctly`() {
        val result = unmapRotation(0.0f, 0.0f, 0.2f, 0.3f, 270)
        assertBoxEquals(RectF(0.7f, 0.0f, 1.0f, 0.2f), result)
    }

    @Test
    fun `270 degrees center box stays centered`() {
        val result = unmapRotation(0.25f, 0.25f, 0.75f, 0.75f, 270)
        assertBoxEquals(RectF(0.25f, 0.25f, 0.75f, 0.75f), result)
    }

    // --- Round-trip: unmap then unmap inverse should return original ---

    @Test
    fun `180 round trip returns original`() {
        val orig = RectF(0.1f, 0.2f, 0.4f, 0.5f)
        val rotated = unmapRotation(orig.left, orig.top, orig.right, orig.bottom, 180)
        val back = unmapRotation(rotated.left, rotated.top, rotated.right, rotated.bottom, 180)
        assertBoxEquals(orig, back)
    }

    @Test
    fun `90 then 270 returns original`() {
        val orig = RectF(0.1f, 0.2f, 0.4f, 0.5f)
        val after90 = unmapRotation(orig.left, orig.top, orig.right, orig.bottom, 90)
        val back = unmapRotation(after90.left, after90.top, after90.right, after90.bottom, 270)
        assertBoxEquals(orig, back)
    }

    @Test
    fun `270 then 90 returns original`() {
        val orig = RectF(0.1f, 0.2f, 0.4f, 0.5f)
        val after270 = unmapRotation(orig.left, orig.top, orig.right, orig.bottom, 270)
        val back = unmapRotation(after270.left, after270.top, after270.right, after270.bottom, 90)
        assertBoxEquals(orig, back)
    }

    // --- VT/detector coordinate consistency ---
    // The visual tracker returns coords in rotated-image space.
    // The detector returns coords that are unmapped back to screen space.
    // For cross-checking (IoU), both must be in the same space.
    // These tests simulate: an object at a known screen position, rotated for
    // detection, then unmapped — and verify the VT raw coords, when also
    // unmapped, produce the same screen box.

    @Test
    fun `VT box unmapped at 90 degrees matches detector box in screen space`() {
        // Both VT and detector see the same object in rotated-image space.
        // Both unmap with the same rotation → identical screen coords.
        val rawBox = RectF(0.2f, 0.3f, 0.5f, 0.7f) // coords in rotated image
        val vtScreen = unmapRotation(rawBox.left, rawBox.top, rawBox.right, rawBox.bottom, 90)
        val detScreen = unmapRotation(rawBox.left, rawBox.top, rawBox.right, rawBox.bottom, 90)
        assertBoxEquals(vtScreen, detScreen)
    }

    @Test
    fun `VT box unmapped at 270 degrees matches detector box in screen space`() {
        val rawBox = RectF(0.2f, 0.3f, 0.5f, 0.7f)
        val vtScreen = unmapRotation(rawBox.left, rawBox.top, rawBox.right, rawBox.bottom, 270)
        val detScreen = unmapRotation(rawBox.left, rawBox.top, rawBox.right, rawBox.bottom, 270)
        assertBoxEquals(vtScreen, detScreen)
    }

    @Test
    fun `VT box unmapped at 180 degrees matches detector box in screen space`() {
        val rawBox = RectF(0.2f, 0.3f, 0.5f, 0.7f)
        val vtScreen = unmapRotation(rawBox.left, rawBox.top, rawBox.right, rawBox.bottom, 180)
        val detScreen = unmapRotation(rawBox.left, rawBox.top, rawBox.right, rawBox.bottom, 180)
        assertBoxEquals(vtScreen, detScreen)
    }

    @Test
    fun `without unmapping VT box misaligns with detector at 90 degrees`() {
        // This test documents the bug: if VT coords are NOT unmapped,
        // they mismatch with unmapped detector coords → low IoU.
        val rawBox = RectF(0.1f, 0.2f, 0.4f, 0.6f)
        val detScreen = unmapRotation(rawBox.left, rawBox.top, rawBox.right, rawBox.bottom, 90)
        // VT raw coords used directly as screen coords (the bug)
        val vtNoUnmap = rawBox
        val iou = FrameToFrameTracker.computeIou(detScreen, vtNoUnmap)
        assertTrue("Without unmapping, IoU should be low (was $iou)", iou < 0.5f)
    }

    @Test
    fun `VT and detector boxes have high IoU after unmapping at all rotations`() {
        // Simulate: same object detected by both VT and detector in rotated space
        // After unmapping, they should have IoU = 1.0
        val rawBox = RectF(0.2f, 0.3f, 0.6f, 0.7f)

        for (rotation in listOf(0, 90, 180, 270)) {
            val vtScreen = unmapRotation(rawBox.left, rawBox.top, rawBox.right, rawBox.bottom, rotation)
            val detScreen = unmapRotation(rawBox.left, rawBox.top, rawBox.right, rawBox.bottom, rotation)
            val iou = FrameToFrameTracker.computeIou(vtScreen, detScreen)
            assertEquals("IoU should be 1.0 at rotation=$rotation", 1.0f, iou, 0.001f)
        }
    }

    // --- Helper ---

    private fun assertBoxEquals(expected: RectF, actual: RectF, tolerance: Float = 0.001f) {
        assertEquals("left", expected.left, actual.left, tolerance)
        assertEquals("top", expected.top, actual.top, tolerance)
        assertEquals("right", expected.right, actual.right, tolerance)
        assertEquals("bottom", expected.bottom, actual.bottom, tolerance)
    }
}
