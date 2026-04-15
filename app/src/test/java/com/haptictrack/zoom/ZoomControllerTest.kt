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
        zoom = ZoomController(targetFrameOccupancy = 0.3f, zoomSpeed = 0.05f)
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
        // Object at roughly target occupancy: ~30% area, target is 0.3
        val box = RectF(0.2f, 0.2f, 0.75f, 0.75f) // 0.55 * 0.55 = 0.3025 area
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
        val box2 = RectF(0.2f, 0.2f, 0.75f, 0.75f) // ~30% area
        assertEquals(1f, zoom.calculateZoom(box2, 1f, 10f), 0.001f)
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
}