package com.haptictrack.tracking

import android.graphics.PointF
import android.graphics.RectF

/** What region of the subject the coach frames around. */
enum class FramingTarget { FULL_BODY, UPPER_BODY, FACE_HEAD }

/** Which guidance channel(s) are active. */
enum class GuidanceMode { OFF, HAPTIC, VOICE, BOTH }

/** The single correction the coach surfaces this frame (NONE = stay silent). */
enum class Cue { NONE, LEVEL, MOVE_LEFT, MOVE_RIGHT, TILT_UP, TILT_DOWN, STEP_CLOSER, STEP_BACK, FACING_AWAY, HOLD }

fun FramingTarget.next(): FramingTarget =
    FramingTarget.entries[(ordinal + 1) % FramingTarget.entries.size]

fun GuidanceMode.next(): GuidanceMode =
    GuidanceMode.entries[(ordinal + 1) % GuidanceMode.entries.size]

/** Per-frame face info for the locked subject, in normalized screen coords. */
data class FaceFraming(val faceBox: RectF, val yawDeg: Float?)

/** Everything GuidanceEngine.assess needs for one frame. All boxes normalized screen coords. */
data class FramingInput(
    val status: TrackingStatus,
    val subject: RectF?,        // locked subject bbox; null when not locked
    val face: FaceFraming?,     // locked subject's face; null if none detected
    val rollDeg: Float,         // camera roll vs gravity (deg); + = tilted clockwise
    val zoomRatio: Float,
    val minZoom: Float,
    val maxZoom: Float,
    val target: FramingTarget,
    val frameTimeMs: Long,
)

/** Result of one assessment. driftX/driftY are regionCenter→bullseye in [-1,1] (for haptics). */
data class FramingAssessment(
    val cue: Cue,
    val bullseye: PointF,
    val desiredOccupancy: Float,
    val driftX: Float,
    val driftY: Float,
    val satisfied: Boolean,
)

/**
 * Coarse head yaw from BlazeFace keypoints. When the head turns, the nose tip shifts toward the
 * eye on the side being turned toward, relative to the eye midpoint. We normalize that horizontal
 * offset by the inter-eye distance (a scale-invariant proxy for sin(yaw)) and map to degrees.
 * + = facing image-right, - = facing image-left. Clamped to ±60° (coarse; good for a binary cue).
 */
fun estimateYawDeg(rightEye: PointF, leftEye: PointF, nose: PointF): Float? {
    val eyeMidX = (rightEye.x + leftEye.x) / 2f
    val eyeDist = kotlin.math.abs(leftEye.x - rightEye.x)
    if (eyeDist < 1e-4f) return null
    // offset>0 when nose is toward the (larger-x) left-eye side = facing image-right.
    val offset = (nose.x - eyeMidX) / eyeDist            // ~[-0.5,0.5] frontal..profile
    val deg = (offset / 0.5f) * 60f                       // scale: half-eye-span ≈ 60°
    return deg.coerceIn(-60f, 60f)
}

/** Short spoken phrase for a cue; null = say nothing. Keep phrases ≤3 words for low BT latency. */
fun cuePhrase(cue: Cue): String? = when (cue) {
    Cue.NONE -> null
    Cue.LEVEL -> "level the camera"
    Cue.MOVE_LEFT -> "move left"
    Cue.MOVE_RIGHT -> "move right"
    Cue.TILT_UP -> "tilt up"
    Cue.TILT_DOWN -> "tilt down"
    Cue.STEP_CLOSER -> "step closer"
    Cue.STEP_BACK -> "step back"
    Cue.FACING_AWAY -> "facing away"
    Cue.HOLD -> "good, hold"
}
