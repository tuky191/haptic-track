package com.haptictrack.zoom

import android.graphics.RectF

class ZoomController(
    private val targetFrameOccupancy: Float = 0.3f,
    private val zoomSpeed: Float = 0.05f
) {

    private var currentZoom = 1f

    /**
     * Calculate the desired zoom ratio based on the subject's bounding box.
     *
     * @param boundingBox Normalized bounding box (0..1 coordinates)
     * @param minZoom Minimum zoom ratio supported by the camera
     * @param maxZoom Maximum zoom ratio supported by the camera
     * @return Target zoom ratio
     */
    fun calculateZoom(boundingBox: RectF, minZoom: Float, maxZoom: Float): Float {
        val boxWidth = boundingBox.width()
        val boxHeight = boundingBox.height()
        val boxArea = boxWidth * boxHeight
        val targetArea = targetFrameOccupancy * targetFrameOccupancy

        val zoomAdjustment = if (boxArea < targetArea * 0.5f) {
            // Subject too small — zoom in
            zoomSpeed
        } else if (boxArea > targetArea * 1.5f) {
            // Subject too large — zoom out
            -zoomSpeed
        } else {
            0f
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

    fun reset() {
        currentZoom = 1f
    }
}
