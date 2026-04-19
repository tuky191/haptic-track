package com.haptictrack.zoom

import android.graphics.RectF
import org.junit.Assert.*
import org.junit.Before
import org.junit.Test
import org.junit.runner.RunWith
import org.robolectric.RobolectricTestRunner

@RunWith(RobolectricTestRunner::class)
class ZoomControllerTest {

    private lateinit var zoom: ZoomController

    @Before
    fun setup() {
        zoom = ZoomController(targetFrameOccupancy = 0.15f, zoomSpeed = 0.05f)
    }

    @Test
    fun `zooms in when subject is too small`() {
        // Small object: 5% of frame
        val box = RectF(0.4f, 0.4f, 0.5f, 0.5f) // 0.1 * 0.1 = 0.01 area
        val result = zoom.calculateZoom(box, minZoom = 1f, maxZoom = 10f)
        assertTrue("Should zoom in from 1.0, got $result", result > 1f)
    }

    @Test
    fun `zooms out when subject is too large`() {
        // Large object: 64% of frame
        val box = RectF(0.1f, 0.1f, 0.9f, 0.9f) // 0.8 * 0.8 = 0.64 area
        val result = zoom.calculateZoom(box, minZoom = 0.5f, maxZoom = 10f)
        assertTrue("Should zoom out from 1.0, got $result", result < 1f)
    }

    @Test
    fun `holds steady when subject is well-framed`() {
        // Object at roughly target occupancy: ~15% area, target is 0.15
        val box = RectF(0.3f, 0.3f, 0.69f, 0.69f) // 0.39 * 0.39 = 0.152 area
        val result = zoom.calculateZoom(box, minZoom = 1f, maxZoom = 10f)
        assertEquals(1f, result, 0.001f)
    }

    @Test
    fun `respects max zoom limit`() {
        val box = RectF(0.49f, 0.49f, 0.51f, 0.51f) // tiny
        // Run many frames to try zooming past max
        var result = 1f
        repeat(500) {
            result = zoom.calculateZoom(box, minZoom = 1f, maxZoom = 3f)
        }
        assertTrue("Should not exceed max zoom 3.0, got $result", result <= 3f)
    }

    @Test
    fun `respects min zoom limit`() {
        val box = RectF(0f, 0f, 1f, 1f) // fills entire frame
        var result = 1f
        repeat(500) {
            result = zoom.calculateZoom(box, minZoom = 0.5f, maxZoom = 10f)
        }
        assertTrue("Should not go below min zoom 0.5, got $result", result >= 0.5f)
    }

    @Test
    fun `zoom changes are gradual`() {
        val box = RectF(0.49f, 0.49f, 0.51f, 0.51f) // tiny, wants to zoom in
        val first = zoom.calculateZoom(box, minZoom = 1f, maxZoom = 10f)
        val second = zoom.calculateZoom(box, minZoom = 1f, maxZoom = 10f)

        val delta = second - first
        assertTrue("Zoom step ($delta) should be at most zoomSpeed", delta <= 0.05f)
    }

    @Test
    fun `reset returns to 1x`() {
        val box = RectF(0.49f, 0.49f, 0.51f, 0.51f)
        repeat(10) { zoom.calculateZoom(box, 1f, 10f) }
        zoom.reset()
        // After reset, a well-framed object should give 1.0
        val box2 = RectF(0.3f, 0.3f, 0.69f, 0.69f) // ~15% area
        assertEquals(1f, zoom.calculateZoom(box2, 1f, 10f), 0.001f)
    }

    // --- Edge awareness ---

    @Test
    fun `holds steady when near edge but not clipped`() {
        // right=0.95: > 1-0.08=0.92 (near edge), < 1-0.02=0.98 (not clipped)
        val box = RectF(0.85f, 0.4f, 0.95f, 0.5f)
        val result = zoom.calculateZoom(box, minZoom = 0.5f, maxZoom = 10f)
        assertEquals("Should hold steady near edge", 1f, result, 0.001f)
    }

    @Test
    fun `zooms out when object is clipped at bottom`() {
        val box = RectF(0.3f, 0.7f, 0.7f, 1.0f) // bottom clipped
        val result = zoom.calculateZoom(box, minZoom = 0.5f, maxZoom = 10f)
        assertTrue("Should zoom out when clipped, got $result", result < 1f)
    }

    @Test
    fun `zooms out when object is clipped at top`() {
        val box = RectF(0.3f, 0.0f, 0.7f, 0.3f) // top clipped
        val result = zoom.calculateZoom(box, minZoom = 0.5f, maxZoom = 10f)
        assertTrue("Should zoom out when top-clipped, got $result", result < 1f)
    }

    @Test
    fun `still zooms in when small and centered`() {
        val box = RectF(0.45f, 0.45f, 0.55f, 0.55f)
        val result = zoom.calculateZoom(box, minZoom = 1f, maxZoom = 10f)
        assertTrue("Should zoom in when centered and small, got $result", result > 1f)
    }

    @Test
    fun `near-edge large object zooms out`() {
        // left=0.05: < 0.08 (near edge), > 0.02 (not clipped). Area > targetArea*1.5
        val box = RectF(0.05f, 0.1f, 0.95f, 0.9f)
        val result = zoom.calculateZoom(box, minZoom = 0.5f, maxZoom = 10f)
        assertTrue("Should zoom out when near edge and large, got $result", result < 1f)
    }

    @Test
    fun `near-edge small object does not zoom in`() {
        // top=0.04: < 0.08 (near edge), > 0.02 (not clipped). Small area.
        val box = RectF(0.4f, 0.04f, 0.6f, 0.15f)
        val result = zoom.calculateZoom(box, minZoom = 0.5f, maxZoom = 10f)
        assertEquals("Should hold steady near edge even if small", 1f, result, 0.001f)
    }

    // --- Edge proximity ---

    @Test
    fun `centered object has low edge proximity`() {
        val box = RectF(0.4f, 0.4f, 0.6f, 0.6f)
        val proximity = zoom.calculateEdgeProximity(box)
        assertTrue("Centered object should have low proximity, got $proximity", proximity < 0.3f)
    }

    @Test
    fun `object at edge has high edge proximity`() {
        val box = RectF(0.8f, 0.4f, 1.0f, 0.6f) // right edge
        val proximity = zoom.calculateEdgeProximity(box)
        assertTrue("Edge object should have high proximity, got $proximity", proximity > 0.7f)
    }

    @Test
    fun `object at corner has maximum edge proximity`() {
        val box = RectF(0.9f, 0.9f, 1.0f, 1.0f)
        val proximity = zoom.calculateEdgeProximity(box)
        assertTrue("Corner object should have ~1.0 proximity, got $proximity", proximity > 0.8f)
    }

    // --- Manual zoom (pinch-to-zoom) ---

    @Test
    fun `manual zoom sets exact ratio`() {
        val result = zoom.setManualZoom(2.5f, minZoom = 1f, maxZoom = 5f)
        assertEquals(2.5f, result, 0.001f)
        assertEquals(2.5f, zoom.getCurrentZoom(), 0.001f)
    }

    @Test
    fun `manual zoom clamps to min and max`() {
        assertEquals(1f, zoom.setManualZoom(0.5f, minZoom = 1f, maxZoom = 5f), 0.001f)
        assertEquals(5f, zoom.setManualZoom(10f, minZoom = 1f, maxZoom = 5f), 0.001f)
    }

    @Test
    fun `manual zoom pauses auto-zoom`() {
        zoom.setManualZoom(3f, minZoom = 1f, maxZoom = 5f)
        assertTrue("Manual override should be active", zoom.manualOverride)

        // Auto-zoom should return current zoom unchanged
        val smallBox = RectF(0.49f, 0.49f, 0.51f, 0.51f) // would normally zoom in
        val result = zoom.calculateZoom(smallBox, minZoom = 1f, maxZoom = 5f)
        assertEquals("Auto-zoom should be paused at manual level", 3f, result, 0.001f)
    }

    @Test
    fun `reset clears manual override`() {
        zoom.setManualZoom(3f, minZoom = 1f, maxZoom = 5f)
        assertTrue(zoom.manualOverride)
        zoom.reset()
        assertFalse("Reset should clear manual override", zoom.manualOverride)
        assertEquals(1f, zoom.getCurrentZoom(), 0.001f)
    }

    // --- Gradual zoom-out on loss ---

    @Test
    fun `gradual zoom-out does not change zoom during delay`() {
        zoom.setManualZoom(3f, minZoom = 1f, maxZoom = 5f)
        val before = zoom.getCurrentZoom()

        // First 4 frames should not zoom out (delay = 5)
        repeat(4) {
            zoom.zoomOutForSearchGradual(minZoom = 1f, maxZoom = 5f)
        }
        assertEquals("Zoom should not change during delay", before, zoom.getCurrentZoom(), 0.001f)
    }

    @Test
    fun `gradual zoom-out starts after delay`() {
        zoom.setManualZoom(3f, minZoom = 1f, maxZoom = 5f)
        val before = zoom.getCurrentZoom()

        // 5 frames = delay expires, 6th should start zoom-out
        repeat(6) {
            zoom.zoomOutForSearchGradual(minZoom = 1f, maxZoom = 5f)
        }
        assertTrue("Zoom should decrease after delay", zoom.getCurrentZoom() < before)
    }

    @Test
    fun `gradual zoom-out preserves zoom during critical reacquisition window`() {
        // Gradual path: zoom stays at 3.0 for the first 4 frames (delay)
        val gradual = ZoomController(targetFrameOccupancy = 0.15f, zoomSpeed = 0.05f)
        gradual.setManualZoom(3f, minZoom = 1f, maxZoom = 5f)

        // Immediate path: zoom drops to ~2.1 on first call
        val immediate = ZoomController(targetFrameOccupancy = 0.15f, zoomSpeed = 0.05f)
        immediate.setManualZoom(3f, minZoom = 1f, maxZoom = 5f)
        immediate.zoomOutForSearch(minZoom = 1f, maxZoom = 5f)

        // During the critical first 4 frames, gradual holds at original zoom
        repeat(4) {
            gradual.zoomOutForSearchGradual(minZoom = 1f, maxZoom = 5f)
            assertTrue("Gradual should preserve zoom during delay (frame ${it+1})",
                gradual.getCurrentZoom() > immediate.getCurrentZoom())
        }
    }

    @Test
    fun `resetLossCounter resets delay`() {
        zoom.setManualZoom(3f, minZoom = 1f, maxZoom = 5f)

        // Accumulate 4 loss frames
        repeat(4) { zoom.zoomOutForSearchGradual(minZoom = 1f, maxZoom = 5f) }
        zoom.resetLossCounter()

        // 4 more frames — still within delay because counter was reset
        val before = zoom.getCurrentZoom()
        repeat(4) { zoom.zoomOutForSearchGradual(minZoom = 1f, maxZoom = 5f) }
        assertEquals("Zoom should not change after reset + 4 frames", before, zoom.getCurrentZoom(), 0.001f)
    }
}