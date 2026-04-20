package com.haptictrack.tracking

import android.graphics.PointF
import android.graphics.RectF
import kotlin.math.sqrt

/**
 * Estimates object velocity from a stream of bounding box center positions.
 *
 * Uses exponential moving average for smooth velocity estimation and supports
 * linear position prediction for N frames ahead. Velocity is in normalized
 * coordinates per frame (e.g. 0.03 = 3% of frame width per frame).
 *
 * Used by [ObjectTracker] to:
 * - Adapt VT drift detection (faster fallback when subject moves fast)
 * - Predict search position during reacquisition
 * - Improve FrameToFrameTracker ID matching for fast-moving objects
 */
class VelocityEstimator(
    private val smoothingAlpha: Float = 0.3f
) {

    companion object {
        /** 3% of frame per frame at 30fps ≈ object crosses full frame in ~1s. */
        const val HIGH_VELOCITY_THRESHOLD = 0.03f
        /** 6% of frame per frame ≈ child running / pet sprinting. */
        const val VERY_HIGH_VELOCITY_THRESHOLD = 0.06f
    }

    /** Smoothed velocity in normalized X units per frame. */
    var velocityX: Float = 0f
        private set

    /** Smoothed velocity in normalized Y units per frame. */
    var velocityY: Float = 0f
        private set

    /** Magnitude of the velocity vector. */
    val speed: Float get() = sqrt(velocityX * velocityX + velocityY * velocityY)

    private var lastCenterX: Float = Float.NaN
    private var lastCenterY: Float = Float.NaN
    private var hasHistory: Boolean = false

    /**
     * Feed a new position. Call once per processed frame with the tracked
     * object's bounding box center in normalized [0,1] coordinates.
     */
    fun update(centerX: Float, centerY: Float) {
        if (hasHistory) {
            val dx = centerX - lastCenterX
            val dy = centerY - lastCenterY
            velocityX = smoothingAlpha * dx + (1f - smoothingAlpha) * velocityX
            velocityY = smoothingAlpha * dy + (1f - smoothingAlpha) * velocityY
        }
        lastCenterX = centerX
        lastCenterY = centerY
        hasHistory = true
    }

    /** Predict where the center will be [framesAhead] frames from now. */
    fun predictPosition(framesAhead: Int): PointF {
        return PointF(
            (lastCenterX + velocityX * framesAhead).coerceIn(0f, 1f),
            (lastCenterY + velocityY * framesAhead).coerceIn(0f, 1f)
        )
    }

    /** Predict where the bounding box will be [framesAhead] frames from now. */
    fun predictBox(lastBox: RectF, framesAhead: Int): RectF {
        val dx = velocityX * framesAhead
        val dy = velocityY * framesAhead
        return RectF(
            (lastBox.left + dx).coerceIn(0f, 1f),
            (lastBox.top + dy).coerceIn(0f, 1f),
            (lastBox.right + dx).coerceIn(0f, 1f),
            (lastBox.bottom + dy).coerceIn(0f, 1f)
        )
    }

    /** True if the subject is moving moderately fast. */
    fun isHighVelocity(): Boolean = speed > HIGH_VELOCITY_THRESHOLD

    /** True if the subject is moving very fast (running/sprinting). */
    fun isVeryHighVelocity(): Boolean = speed > VERY_HIGH_VELOCITY_THRESHOLD

    /** Clear all state. Call on lock clear, rebind, or new lock. */
    fun reset() {
        velocityX = 0f
        velocityY = 0f
        lastCenterX = Float.NaN
        lastCenterY = Float.NaN
        hasHistory = false
    }
}
