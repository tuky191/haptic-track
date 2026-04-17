package com.haptictrack.zoom

import android.graphics.RectF

class ZoomController(
    private val targetFrameOccupancy: Float = 0.15f,
    private val zoomSpeed: Float = 0.10f
) {

    private var currentZoom = 1f

    /** When true, [calculateZoom] returns the current zoom unchanged (manual pinch active). */
    var manualOverride: Boolean = false
        private set

    private var manualOverrideExpiry: Long = 0L

    companion object {
        /** Object touching frame edge — actively zoom out. */
        private const val CLIP_THRESHOLD = 0.02f
        /** Object near frame edge — stop zooming in. */
        private const val EDGE_MARGIN = 0.08f
        /** How long manual zoom holds after the last pinch gesture (ms). */
        private const val MANUAL_OVERRIDE_DURATION_MS = 2000L
    }

    /**
     * Set zoom directly from a pinch gesture.
     * Activates manual override that pauses auto-zoom for [MANUAL_OVERRIDE_DURATION_MS].
     */
    fun setManualZoom(ratio: Float, minZoom: Float, maxZoom: Float): Float {
        currentZoom = ratio.coerceIn(minZoom, maxZoom)
        manualOverride = true
        manualOverrideExpiry = System.currentTimeMillis() + MANUAL_OVERRIDE_DURATION_MS
        return currentZoom
    }

    /**
     * Calculate the desired zoom ratio based on the subject's bounding box.
     *
     * @param boundingBox Normalized bounding box (0..1 coordinates)
     * @param minZoom Minimum zoom ratio supported by the camera
     * @param maxZoom Maximum zoom ratio supported by the camera
     * @return Target zoom ratio
     */
    fun calculateZoom(boundingBox: RectF, minZoom: Float, maxZoom: Float): Float {
        // Check if manual override has expired
        if (manualOverride && System.currentTimeMillis() > manualOverrideExpiry) {
            manualOverride = false
        }
        if (manualOverride) return currentZoom

        val boxArea = boundingBox.width() * boundingBox.height()
        val targetArea = targetFrameOccupancy

        // Check if the object is clipped or near the edge of the frame
        val clipped = boundingBox.left <= CLIP_THRESHOLD || boundingBox.top <= CLIP_THRESHOLD ||
                      boundingBox.right >= 1f - CLIP_THRESHOLD || boundingBox.bottom >= 1f - CLIP_THRESHOLD
        val nearEdge = boundingBox.left < EDGE_MARGIN || boundingBox.top < EDGE_MARGIN ||
                       boundingBox.right > 1f - EDGE_MARGIN || boundingBox.bottom > 1f - EDGE_MARGIN

        val zoomAdjustment = when {
            // Object is being cropped — zoom out immediately
            clipped -> -zoomSpeed
            // Object near edge — don't zoom in, hold steady or zoom out if too large
            nearEdge -> if (boxArea > targetArea * 1.5f) -zoomSpeed else 0f
            // Normal: adjust based on area
            boxArea < targetArea * 0.5f -> zoomSpeed
            boxArea > targetArea * 1.5f -> -zoomSpeed
            else -> 0f
        }

        currentZoom = (currentZoom + zoomAdjustment).coerceIn(minZoom, maxZoom)
        return currentZoom
    }

    /**
     * Calculate how close the subject is to the edge of the frame.
     *
     * @param boundingBox Normalized bounding box (0..1 coordinates)
     * @return 0.0 = centered, 1.0 = at edge
     */
    fun calculateEdgeProximity(boundingBox: RectF): Float {
        val centerX = boundingBox.centerX()
        val centerY = boundingBox.centerY()
        val distFromCenterX = kotlin.math.abs(centerX - 0.5f) * 2f
        val distFromCenterY = kotlin.math.abs(centerY - 0.5f) * 2f
        return maxOf(distFromCenterX, distFromCenterY).coerceIn(0f, 1f)
    }

    /**
     * Zoom out partially when the target is lost to widen the field of view.
     * Pulls back 45% of the way between current zoom and minZoom.
     */
    fun zoomOutForSearch(minZoom: Float, maxZoom: Float): Float {
        val pullback = 0.45f
        currentZoom = (currentZoom - (currentZoom - minZoom) * pullback).coerceIn(minZoom, maxZoom)
        return currentZoom
    }

    fun reset() {
        currentZoom = 1f
        manualOverride = false
    }

    /** Current zoom level (for pinch gesture to use as baseline). */
    fun getCurrentZoom(): Float = currentZoom
}
