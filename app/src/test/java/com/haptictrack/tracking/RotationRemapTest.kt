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

    // --- Helper ---

    private fun assertBoxEquals(expected: RectF, actual: RectF, tolerance: Float = 0.001f) {
        assertEquals("left", expected.left, actual.left, tolerance)
        assertEquals("top", expected.top, actual.top, tolerance)
        assertEquals("right", expected.right, actual.right, tolerance)
        assertEquals("bottom", expected.bottom, actual.bottom, tolerance)
    }
}
