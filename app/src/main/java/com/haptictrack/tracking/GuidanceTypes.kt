package com.haptictrack.tracking

import android.graphics.PointF
import android.graphics.RectF

/** What region of the subject the coach frames around. */
enum class FramingTarget { FULL_BODY, UPPER_BODY, FACE_HEAD }

/** Which guidance channel(s) are active. */
enum class GuidanceMode { OFF, HAPTIC, VOICE, BOTH }

/** The single correction the coach surfaces this frame (NONE = stay silent). */
enum class Cue { NONE, LEVEL, CUT_OFF, MOVE_LEFT, MOVE_RIGHT, TILT_UP, TILT_DOWN, STEP_CLOSER, STEP_BACK, FACING_AWAY, HOLD }

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
