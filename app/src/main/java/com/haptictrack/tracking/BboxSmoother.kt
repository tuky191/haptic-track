package com.haptictrack.tracking

import android.graphics.RectF

/**
 * Produces a smoothed bounding box for zoom/display by combining VitTracker's
 * accurate center with EMA-filtered dimensions from multiple size sources.
 *
 * VitTracker tracks position well (pixel correlation) but its regression head
 * lets width/height drift. The detector and segmenter give better size estimates
 * at lower cadence. This smoother fuses them: VT center every frame, dimensions
 * weighted by source quality.
 */
class BboxSmoother {

    enum class SizeSource(val alpha: Float) {
        SEGMENTATION(0.35f),
        DETECTOR(0.15f),
        VT_ONLY(0.05f),
    }

    private var smoothedWidth = 0f
    private var smoothedHeight = 0f
    private var initialized = false

    fun reset() {
        initialized = false
        smoothedWidth = 0f
        smoothedHeight = 0f
    }

    /**
     * True when [w]×[h] is close enough to the current smoothed size that
     * keeping the EMA state is better than reinitializing. When false, the
     * caller should [reset] before the next [smooth] call.
     */
    fun isCompatible(w: Float, h: Float, maxRatio: Float = 2.0f): Boolean {
        if (!initialized || smoothedWidth <= 0f || smoothedHeight <= 0f) return false
        val wr = if (w > smoothedWidth) w / smoothedWidth else smoothedWidth / w
        val hr = if (h > smoothedHeight) h / smoothedHeight else smoothedHeight / h
        return wr <= maxRatio && hr <= maxRatio
    }

    /**
     * Returns a bbox with [vtBox]'s center and EMA-smoothed dimensions.
     * [sizeWidth]/[sizeHeight] come from the best available source this frame.
     */
    fun smooth(vtBox: RectF, sizeWidth: Float, sizeHeight: Float, source: SizeSource): RectF {
        val alpha = source.alpha

        if (!initialized) {
            smoothedWidth = sizeWidth
            smoothedHeight = sizeHeight
            initialized = true
        } else {
            smoothedWidth += alpha * (sizeWidth - smoothedWidth)
            smoothedHeight += alpha * (sizeHeight - smoothedHeight)
        }

        val cx = vtBox.centerX()
        val cy = vtBox.centerY()
        return RectF(
            cx - smoothedWidth / 2f,
            cy - smoothedHeight / 2f,
            cx + smoothedWidth / 2f,
            cy + smoothedHeight / 2f
        )
    }
}
