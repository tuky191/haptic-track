package com.haptictrack.tracking

import android.graphics.PointF
import android.graphics.RectF
import kotlin.math.abs

/**
 * Pure per-frame framing assessment. Given the locked subject geometry + camera roll + the chosen
 * framing target, it computes a composition bullseye, the desired subject size, a drift vector
 * (for the haptic channel), and the single highest-priority correction cue (for the voice channel).
 * assess() is stateless; throttle()/reset() carry the spoken-cue cadence state.
 */
class GuidanceEngine {
    companion object {
        const val LEVEL_TOL_DEG = 8f
        const val DRIFT_TOL = 0.10f
        const val OCC_LOW = 0.7f
        const val OCC_HIGH = 1.4f
        const val YAW_PROFILE_DEG = 35f
        const val EDGE_MARGIN = 0.04f
        const val MIN_GAP_MS = 1800L
        private const val HEADROOM_TOP = 0.18f
        private const val ZOOM_LIMIT_EPS = 0.05f
    }

    private var lastSpokenCue: Cue = Cue.NONE
    // Seeded one gap in the past so the first cue always clears MIN_GAP_MS. (Do NOT use
    // Long.MIN_VALUE — `frameTimeMs - Long.MIN_VALUE` overflows and silences the first cue.)
    private var lastSpokenMs: Long = -MIN_GAP_MS

    /** Decide whether [cue] should actually be spoken this frame; see Task 4 rules. */
    fun throttle(cue: Cue, frameTimeMs: Long): Cue {
        if (cue == Cue.NONE) return Cue.NONE
        if (cue == Cue.HOLD && lastSpokenCue == Cue.HOLD) return Cue.NONE  // hold spoken once
        if (frameTimeMs - lastSpokenMs < MIN_GAP_MS) return Cue.NONE       // global min gap
        lastSpokenCue = cue
        lastSpokenMs = frameTimeMs
        return cue
    }

    fun reset() {
        lastSpokenCue = Cue.NONE
        lastSpokenMs = -MIN_GAP_MS
    }

    fun assess(input: FramingInput): FramingAssessment {
        val subject = input.subject
        if (input.status != TrackingStatus.LOCKED || subject == null) {
            return FramingAssessment(Cue.NONE, PointF(0.5f, 0.5f), 0.15f, 0f, 0f, satisfied = false)
        }
        val region = regionFor(input.target, subject, input.face)
        val desiredOcc = occupancyFor(input.target)
        val bullseye = bullseyeFor(region, input.face?.yawDeg)
        val rcx = region.centerX(); val rcy = region.centerY()
        val driftX = ((rcx - bullseye.x) * 2f).coerceIn(-1f, 1f)
        val driftY = ((rcy - bullseye.y) * 2f).coerceIn(-1f, 1f)

        val cue = deriveCue(input, region, desiredOcc, driftX, driftY)
        return FramingAssessment(cue, bullseye, desiredOcc, driftX, driftY, satisfied = cue == Cue.HOLD)
    }

    private fun regionFor(target: FramingTarget, subject: RectF, face: FaceFraming?): RectF = when (target) {
        FramingTarget.FULL_BODY -> RectF(subject)
        FramingTarget.UPPER_BODY -> RectF(subject.left, subject.top, subject.right,
            subject.top + subject.height() * 0.55f)
        FramingTarget.FACE_HEAD -> face?.faceBox?.let { RectF(it) }
            ?: RectF(subject.left, subject.top, subject.right, subject.top + subject.height() * 0.25f)
    }

    private fun occupancyFor(target: FramingTarget): Float = when (target) {
        FramingTarget.FULL_BODY -> 0.45f
        FramingTarget.UPPER_BODY -> 0.40f
        FramingTarget.FACE_HEAD -> 0.22f
    }

    private fun bullseyeFor(region: RectF, yawDeg: Float?): PointF {
        val x = when {
            yawDeg == null || abs(yawDeg) < 12f -> 0.5f
            yawDeg > 0f -> 0.33f   // facing image-right -> seat on left third (lead room ahead)
            else -> 0.67f
        }
        val y = (HEADROOM_TOP + region.height() / 2f).coerceIn(0.2f, 0.8f)
        return PointF(x, y)
    }

    private fun deriveCue(
        input: FramingInput, region: RectF, desiredOcc: Float, driftX: Float, driftY: Float,
    ): Cue {
        if (abs(input.rollDeg) > LEVEL_TOL_DEG) return Cue.LEVEL
        if (region.left < EDGE_MARGIN || region.top < EDGE_MARGIN ||
            region.right > 1f - EDGE_MARGIN || region.bottom > 1f - EDGE_MARGIN) return Cue.CUT_OFF

        val area = (region.width() * region.height()).coerceAtLeast(1e-6f)
        val occRatio = desiredOcc / area  // >1 = too small, <1 = too big
        val atMaxZoom = input.zoomRatio >= input.maxZoom - ZOOM_LIMIT_EPS
        val atMinZoom = input.zoomRatio <= input.minZoom + ZOOM_LIMIT_EPS
        if (occRatio > OCC_HIGH && atMaxZoom) return Cue.STEP_CLOSER
        if (occRatio < OCC_LOW && atMinZoom) return Cue.STEP_BACK

        if (abs(driftX) > DRIFT_TOL) return if (driftX > 0f) Cue.MOVE_LEFT else Cue.MOVE_RIGHT
        if (abs(driftY) > DRIFT_TOL) return if (driftY > 0f) Cue.TILT_UP else Cue.TILT_DOWN

        val yaw = input.face?.yawDeg
        if (yaw != null && abs(yaw) > YAW_PROFILE_DEG && input.target != FramingTarget.FULL_BODY) return Cue.FACING_AWAY
        return Cue.HOLD
    }
}
