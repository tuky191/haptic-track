package com.haptictrack.tracking

import android.graphics.RectF
import org.junit.Assert.*
import org.junit.Before
import org.junit.Test
import org.junit.runner.RunWith
import org.robolectric.RobolectricTestRunner

@RunWith(RobolectricTestRunner::class)
class BboxSmootherTest {

    private lateinit var smoother: BboxSmoother

    @Before
    fun setup() {
        smoother = BboxSmoother()
    }

    @Test
    fun `first call initializes to exact input`() {
        val vt = RectF(0.3f, 0.3f, 0.5f, 0.5f)
        val result = smoother.smooth(vt, 0.2f, 0.2f, BboxSmoother.SizeSource.DETECTOR)
        assertEquals(0.4f, result.centerX(), 0.001f)
        assertEquals(0.4f, result.centerY(), 0.001f)
        assertEquals(0.2f, result.width(), 0.001f)
        assertEquals(0.2f, result.height(), 0.001f)
    }

    @Test
    fun `uses VT center not size source center`() {
        // Init
        smoother.smooth(RectF(0.3f, 0.3f, 0.5f, 0.5f), 0.2f, 0.2f, BboxSmoother.SizeSource.DETECTOR)
        // VT moved center, but size stays from detector
        val result = smoother.smooth(RectF(0.5f, 0.5f, 0.7f, 0.7f), 0.2f, 0.2f, BboxSmoother.SizeSource.DETECTOR)
        assertEquals(0.6f, result.centerX(), 0.001f)
        assertEquals(0.6f, result.centerY(), 0.001f)
    }

    @Test
    fun `detector alpha moves dimensions gradually`() {
        // Init with width=0.2
        smoother.smooth(RectF(0.4f, 0.4f, 0.6f, 0.6f), 0.2f, 0.2f, BboxSmoother.SizeSource.DETECTOR)
        // Detector says width=0.3 now — alpha=0.15 should move partially
        val result = smoother.smooth(RectF(0.4f, 0.4f, 0.6f, 0.6f), 0.3f, 0.3f, BboxSmoother.SizeSource.DETECTOR)
        // EMA: 0.2 + 0.15*(0.3-0.2) = 0.215
        assertEquals(0.215f, result.width(), 0.001f)
    }

    @Test
    fun `VT-only alpha barely moves dimensions`() {
        smoother.smooth(RectF(0.4f, 0.4f, 0.6f, 0.6f), 0.2f, 0.2f, BboxSmoother.SizeSource.DETECTOR)
        // VT says 0.4 wide — alpha=0.05 should barely move from 0.2
        val result = smoother.smooth(RectF(0.4f, 0.4f, 0.6f, 0.6f), 0.4f, 0.4f, BboxSmoother.SizeSource.VT_ONLY)
        // EMA: 0.2 + 0.05*(0.4-0.2) = 0.21
        assertEquals(0.21f, result.width(), 0.001f)
    }

    @Test
    fun `segmentation alpha moves dimensions fast`() {
        smoother.smooth(RectF(0.4f, 0.4f, 0.6f, 0.6f), 0.2f, 0.2f, BboxSmoother.SizeSource.DETECTOR)
        // Segmentation says 0.3 wide — alpha=0.35 should move substantially
        val result = smoother.smooth(RectF(0.4f, 0.4f, 0.6f, 0.6f), 0.3f, 0.3f, BboxSmoother.SizeSource.SEGMENTATION)
        // EMA: 0.2 + 0.35*(0.3-0.2) = 0.235
        assertEquals(0.235f, result.width(), 0.001f)
    }

    @Test
    fun `reset clears state`() {
        smoother.smooth(RectF(0.4f, 0.4f, 0.6f, 0.6f), 0.2f, 0.2f, BboxSmoother.SizeSource.DETECTOR)
        smoother.reset()
        // After reset, should re-initialize to new input
        val result = smoother.smooth(RectF(0.4f, 0.4f, 0.6f, 0.6f), 0.4f, 0.4f, BboxSmoother.SizeSource.DETECTOR)
        assertEquals(0.4f, result.width(), 0.001f)
    }

    @Test
    fun `many VT-only frames preserve detector size`() {
        // Init with detector size 0.2
        smoother.smooth(RectF(0.4f, 0.4f, 0.6f, 0.6f), 0.2f, 0.2f, BboxSmoother.SizeSource.DETECTOR)
        // 20 frames of VT saying 0.4 (VT box growing) — should barely drift
        var result = RectF()
        for (i in 1..20) {
            result = smoother.smooth(RectF(0.4f, 0.4f, 0.6f, 0.6f), 0.4f, 0.4f, BboxSmoother.SizeSource.VT_ONLY)
        }
        // After 20 frames: EMA converges but slowly. 0.2*(0.95^20) + 0.4*(1-0.95^20)
        // = 0.2*0.358 + 0.4*0.642 = 0.0716 + 0.2569 = 0.328
        // Still well below 0.4 (VT's reported size)
        assertTrue("Width should stay closer to 0.2 than 0.4: ${result.width()}", result.width() < 0.35f)
    }

    @Test
    fun `detector updates interleaved with VT-only stabilize`() {
        // Simulate: detector every 3rd frame, VT in between
        // Detector consistently says 0.2, VT drifts to 0.3
        smoother.smooth(RectF(0.4f, 0.4f, 0.6f, 0.6f), 0.2f, 0.2f, BboxSmoother.SizeSource.DETECTOR)
        for (i in 1..30) {
            if (i % 3 == 0) {
                smoother.smooth(RectF(0.4f, 0.4f, 0.6f, 0.6f), 0.2f, 0.2f, BboxSmoother.SizeSource.DETECTOR)
            } else {
                smoother.smooth(RectF(0.4f, 0.4f, 0.6f, 0.6f), 0.3f, 0.3f, BboxSmoother.SizeSource.VT_ONLY)
            }
        }
        val result = smoother.smooth(RectF(0.4f, 0.4f, 0.6f, 0.6f), 0.2f, 0.2f, BboxSmoother.SizeSource.DETECTOR)
        // Should stay close to the detector's consistent 0.2 (lower alpha means tighter tracking)
        assertEquals(0.2f, result.width(), 0.03f)
    }

    @Test
    fun `width and height smooth independently`() {
        smoother.smooth(RectF(0.4f, 0.4f, 0.6f, 0.6f), 0.2f, 0.1f, BboxSmoother.SizeSource.DETECTOR)
        val result = smoother.smooth(RectF(0.4f, 0.4f, 0.6f, 0.6f), 0.3f, 0.2f, BboxSmoother.SizeSource.DETECTOR)
        // Width: 0.2 + 0.15*(0.3-0.2) = 0.215
        // Height: 0.1 + 0.15*(0.2-0.1) = 0.115
        assertEquals(0.215f, result.width(), 0.001f)
        assertEquals(0.115f, result.height(), 0.001f)
    }
}
