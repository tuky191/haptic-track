package com.haptictrack.zoom

import android.graphics.RectF
import kotlin.math.sqrt

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
        /** Frames to wait before starting zoom-out on loss (~270ms at 30fps). */
        private const val ZOOM_OUT_DELAY_FRAMES = 8
        /** Exponential smoothing factor per frame (0.05 = slow/smooth, 0.20 = fast). */
        private const val ZOOM_SMOOTH_ALPHA = 0.05f
        /** Faster smoothing for zoom-out when clipped (avoid losing the subject). */
        private const val ZOOM_SMOOTH_ALPHA_CLIP = 0.15f
        /** Smoothing factor for bbox area input (filters detector jitter). */
        private const val AREA_SMOOTH_ALPHA = 0.15f
        /** Dead zone: hold zoom if area ratio is within this range of target. */
        private const val DEAD_ZONE_LOW = 0.70f   // area is 70% of target → slightly too small
        private const val DEAD_ZONE_HIGH = 1.40f   // area is 140% of target → slightly too large
    }

    /** Counts frames since tracking was lost, for gradual zoom-out delay. */
    private var lossFrameCount = 0
    /** Zoom level when the object was last locked — floor for search zoom-out.
     *  Prevents zooming all the way to 1x which makes small objects undetectable. */
    private var lockedZoom = 1f
    /** Smoothed bbox area — filters detector noise before computing ideal zoom. */
    private var smoothedArea = -1f

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
     * Computes an ideal zoom from the area ratio (sqrt because zoom scales
     * linearly while area scales quadratically), then exponentially smooths
     * toward it. Edge/clip guards override the ideal when the subject is
     * near or past the frame boundary.
     */
    fun calculateZoom(boundingBox: RectF, minZoom: Float, maxZoom: Float): Float {
        if (manualOverride && System.currentTimeMillis() > manualOverrideExpiry) {
            manualOverride = false
        }
        if (manualOverride) return currentZoom

        val rawArea = boundingBox.width() * boundingBox.height()

        // Smooth the bbox area to filter detector jitter before computing ideal zoom.
        if (smoothedArea < 0f) smoothedArea = rawArea
        smoothedArea += AREA_SMOOTH_ALPHA * (rawArea - smoothedArea)
        val boxArea = smoothedArea

        val clipped = boundingBox.left <= CLIP_THRESHOLD || boundingBox.top <= CLIP_THRESHOLD ||
                      boundingBox.right >= 1f - CLIP_THRESHOLD || boundingBox.bottom >= 1f - CLIP_THRESHOLD
        val nearEdge = boundingBox.left < EDGE_MARGIN || boundingBox.top < EDGE_MARGIN ||
                       boundingBox.right > 1f - EDGE_MARGIN || boundingBox.bottom > 1f - EDGE_MARGIN

        // Dead zone: if the smoothed area is within ±30% of target, hold steady.
        val areaRatio = if (boxArea > 1e-6f) targetFrameOccupancy / boxArea else 1f
        val idealZoom = if (areaRatio in DEAD_ZONE_LOW..DEAD_ZONE_HIGH) {
            currentZoom
        } else if (boxArea > 1e-6f) {
            (currentZoom * sqrt(areaRatio)).coerceIn(minZoom, maxZoom)
        } else {
            currentZoom
        }

        val targetZoom = when {
            clipped -> {
                val emergencyTarget = (currentZoom - zoomSpeed).coerceAtLeast(minZoom)
                minOf(idealZoom, emergencyTarget)
            }
            nearEdge -> minOf(idealZoom, currentZoom)
            else -> idealZoom
        }

        val alpha = if (clipped) ZOOM_SMOOTH_ALPHA_CLIP else ZOOM_SMOOTH_ALPHA
        currentZoom += alpha * (targetZoom - currentZoom)
        currentZoom = currentZoom.coerceIn(minZoom, maxZoom)
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

    /**
     * Gradual zoom-out for search, called each LOST frame.
     * Delays [ZOOM_OUT_DELAY_FRAMES] frames before starting, then applies
     * a gentle 8% pullback per frame. This gives reacquisition a chance to
     * find the subject at the original zoom before widening the field of view.
     */
    fun zoomOutForSearchGradual(minZoom: Float, maxZoom: Float): Float {
        lossFrameCount++
        if (lossFrameCount >= ZOOM_OUT_DELAY_FRAMES) {
            val searchFloor = maxOf(minZoom, lockedZoom * 0.5f)
            val pullback = 0.08f
            currentZoom = (currentZoom - (currentZoom - searchFloor) * pullback).coerceIn(searchFloor, maxZoom)
        }
        return currentZoom
    }

    /** Reset the loss frame counter and save current zoom as locked baseline. */
    fun resetLossCounter() {
        lossFrameCount = 0
        lockedZoom = currentZoom
    }

    fun reset() {
        currentZoom = 1f
        manualOverride = false
        lossFrameCount = 0
        smoothedArea = -1f
    }

    /** Current zoom level (for pinch gesture to use as baseline). */
    fun getCurrentZoom(): Float = currentZoom
}
