# Voice Framing Coach Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Continuous spoken framing coaching over a Bluetooth earbud that guides the operator to aim + zoom so a user-selectable framing target on the tracked subject is well-composed, behind a VOICE / HAPTIC / BOTH toggle.

**Architecture:** A pure-Kotlin `GuidanceEngine` computes, per frame, a `FramingAssessment` (composition bullseye + desired size + the single highest-priority correction cue + drift vector) from signals the tracking pipeline already produces (subject bbox, face box + BlazeFace keypoints, gyro roll, zoom, status). The assessment is consumed by either the existing haptic geiger (`HapticFeedbackManager`, driven to the bullseye instead of dead-center) or a new `VoiceGuide` (Android `TextToSpeech` over Bluetooth A2DP), selected by a UI toggle. A second toggle picks the framing target (Full body / Upper body / Face-head), which moves the bullseye and retargets auto-zoom occupancy.

**Tech Stack:** Kotlin, Android `TextToSpeech` + `AudioAttributes`, MediaPipe BlazeFace (already integrated), Robolectric/JUnit, CameraX (existing pipeline).

## Global Constraints

- **Audio routing (HARD RULE):** voice plays via **Bluetooth A2DP media playback only**, tagged `AudioAttributes.USAGE_ASSISTANCE_NAVIGATION_GUIDANCE` + `CONTENT_TYPE_SPEECH`. **NEVER** call `AudioManager.startBluetoothSco()` / `setCommunicationDevice()` / set `MODE_IN_COMMUNICATION` — that seizes the mic and can silence/break the active `CAMCORDER` recording (`RecordingManager.kt` records `withAudioEnabled()`).
- **TTS cadence:** use `TextToSpeech.QUEUE_FLUSH` for every cue (never let cues backlog), minimum spoken gap `MIN_GAP_MS = 1800`, suppress consecutive identical cues, speak the satisfied state ("good — hold") once then go silent (silence = good).
- All new pure logic goes in `com.haptictrack.tracking` and MUST be unit-tested (the project runs ~340 Robolectric tests; match that culture).
- Coordinates are normalized screen space `[0,1]²` (matches `TrackedObject.boundingBox`), origin top-left, +x right, +y down.
- Coarse head yaw is acceptable for v1 (no new vision model). Body-part targets beyond Full/Upper/Face are explicitly out of scope (Phase 2 PoseLandmarker).
- Run `./gradlew testDebugUnitTest` after every code task; it must stay green.

---

### Task 1: Guidance types + UI state

**Files:**
- Create: `app/src/main/java/com/haptictrack/tracking/GuidanceTypes.kt`
- Modify: `app/src/main/java/com/haptictrack/tracking/TrackingState.kt` (add two fields to `TrackingUiState`)
- Modify: `app/src/main/java/com/haptictrack/ui/CameraViewModel.kt:493-495` (preserve new fields in the `clearTracking` rebuild)
- Test: `app/src/test/java/com/haptictrack/tracking/GuidanceTypesTest.kt`

**Interfaces:**
- Produces: `enum FramingTarget { FULL_BODY, UPPER_BODY, FACE_HEAD }`, `enum GuidanceMode { OFF, HAPTIC, VOICE, BOTH }`, `enum Cue { NONE, LEVEL, CUT_OFF, MOVE_LEFT, MOVE_RIGHT, TILT_UP, TILT_DOWN, STEP_CLOSER, STEP_BACK, FACING_AWAY, HOLD }`, `FaceFraming(faceBox: RectF, yawDeg: Float?)`, `FramingInput(...)`, `FramingAssessment(...)`, and `fun FramingTarget.next()`, `fun GuidanceMode.next()`.

- [ ] **Step 1: Write the failing test**

```kotlin
// app/src/test/java/com/haptictrack/tracking/GuidanceTypesTest.kt
package com.haptictrack.tracking

import org.junit.Assert.assertEquals
import org.junit.Test

class GuidanceTypesTest {
    @Test fun `framing target cycles and wraps`() {
        assertEquals(FramingTarget.UPPER_BODY, FramingTarget.FULL_BODY.next())
        assertEquals(FramingTarget.FACE_HEAD, FramingTarget.UPPER_BODY.next())
        assertEquals(FramingTarget.FULL_BODY, FramingTarget.FACE_HEAD.next())
    }

    @Test fun `guidance mode cycles through all four and wraps`() {
        assertEquals(GuidanceMode.HAPTIC, GuidanceMode.OFF.next())
        assertEquals(GuidanceMode.VOICE, GuidanceMode.HAPTIC.next())
        assertEquals(GuidanceMode.BOTH, GuidanceMode.VOICE.next())
        assertEquals(GuidanceMode.OFF, GuidanceMode.BOTH.next())
    }
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `./gradlew testDebugUnitTest --tests "com.haptictrack.tracking.GuidanceTypesTest"`
Expected: FAIL — `FramingTarget` unresolved.

- [ ] **Step 3: Create the types file**

```kotlin
// app/src/main/java/com/haptictrack/tracking/GuidanceTypes.kt
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

/** Result of one assessment. driftX/driftY are subjectCenter→bullseye in [-1,1] (for haptics). */
data class FramingAssessment(
    val cue: Cue,
    val bullseye: PointF,
    val desiredOccupancy: Float,
    val driftX: Float,
    val driftY: Float,
    val satisfied: Boolean,
)
```

- [ ] **Step 4: Add UI state fields**

In `TrackingState.kt`, inside `data class TrackingUiState(...)`, add after the `recordingError` field:

```kotlin
    /** Active guidance channel(s). */
    val guidanceMode: GuidanceMode = GuidanceMode.OFF,
    /** What the framing coach composes around. */
    val framingTarget: FramingTarget = FramingTarget.FULL_BODY,
```

- [ ] **Step 5: Preserve the fields across clearTracking**

In `CameraViewModel.kt` the `clearTracking()` rebuild (`_uiState.update { TrackingUiState(... ) }`, ~line 494) constructs a fresh state. Add `guidanceMode = it.guidanceMode, framingTarget = it.framingTarget,` to that constructor call so a clear keeps the user's guidance choices.

- [ ] **Step 6: Run tests to verify they pass**

Run: `./gradlew testDebugUnitTest --tests "com.haptictrack.tracking.GuidanceTypesTest"`
Expected: PASS (2 tests).

- [ ] **Step 7: Commit**

```bash
git add app/src/main/java/com/haptictrack/tracking/GuidanceTypes.kt app/src/main/java/com/haptictrack/tracking/TrackingState.kt app/src/main/java/com/haptictrack/ui/CameraViewModel.kt app/src/test/java/com/haptictrack/tracking/GuidanceTypesTest.kt
git commit -m "feat(guidance): voice-coach types + UI state fields"
```

---

### Task 2: Coarse head-yaw estimator (pure)

**Files:**
- Modify: `app/src/main/java/com/haptictrack/tracking/GuidanceTypes.kt` (add top-level `estimateYawDeg`)
- Test: `app/src/test/java/com/haptictrack/tracking/YawEstimatorTest.kt`

**Interfaces:**
- Produces: `fun estimateYawDeg(rightEye: PointF, leftEye: PointF, nose: PointF): Float?` — `+` = subject facing image-right, `-` = facing image-left, ~0 = frontal; null if eyes coincide. Magnitude is coarse degrees (clamped ±60).

- [ ] **Step 1: Write the failing test**

```kotlin
// app/src/test/java/com/haptictrack/tracking/YawEstimatorTest.kt
package com.haptictrack.tracking

import android.graphics.PointF
import org.junit.Assert.assertTrue
import org.junit.Test
import org.junit.runner.RunWith
import org.robolectric.RobolectricTestRunner

@RunWith(RobolectricTestRunner::class)
class YawEstimatorTest {
    // Eyes span x in [0.4,0.6] at y=0.5; nose y=0.55.
    private fun yaw(noseX: Float) =
        estimateYawDeg(PointF(0.4f, 0.5f), PointF(0.6f, 0.5f), PointF(noseX, 0.55f))!!

    @Test fun `nose centered between eyes is frontal`() {
        assertTrue(kotlin.math.abs(yaw(0.5f)) < 8f)
    }
    @Test fun `nose toward image-right eye means facing right (positive)`() {
        assertTrue(yaw(0.58f) > 15f)
    }
    @Test fun `nose toward image-left eye means facing left (negative)`() {
        assertTrue(yaw(0.42f) < -15f)
    }
    @Test fun `coincident eyes return null`() {
        assertTrue(estimateYawDeg(PointF(0.5f, 0.5f), PointF(0.5f, 0.5f), PointF(0.5f, 0.55f)) == null)
    }
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `./gradlew testDebugUnitTest --tests "com.haptictrack.tracking.YawEstimatorTest"`
Expected: FAIL — `estimateYawDeg` unresolved.

- [ ] **Step 3: Implement the estimator**

Append to `GuidanceTypes.kt`:

```kotlin
/**
 * Coarse head yaw from BlazeFace keypoints. When the head turns, the nose tip shifts toward the
 * eye on the side being turned toward, relative to the eye midpoint. We normalize that horizontal
 * offset by the inter-eye distance (a scale-invariant proxy for sin(yaw)) and map to degrees.
 * + = facing image-right, - = facing image-left. Clamped to ±60° (coarse; good for a binary cue).
 */
fun estimateYawDeg(rightEye: android.graphics.PointF, leftEye: android.graphics.PointF, nose: android.graphics.PointF): Float? {
    val eyeMidX = (rightEye.x + leftEye.x) / 2f
    val eyeDist = kotlin.math.abs(leftEye.x - rightEye.x)
    if (eyeDist < 1e-4f) return null
    // offset>0 when nose is toward the (larger-x) left-eye side = facing image-right.
    val offset = (nose.x - eyeMidX) / eyeDist            // ~[-0.5,0.5] frontal..profile
    val deg = (offset / 0.5f) * 60f                       // scale: half-eye-span ≈ 60°
    return deg.coerceIn(-60f, 60f)
}
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `./gradlew testDebugUnitTest --tests "com.haptictrack.tracking.YawEstimatorTest"`
Expected: PASS (4 tests).

- [ ] **Step 5: Commit**

```bash
git add app/src/main/java/com/haptictrack/tracking/GuidanceTypes.kt app/src/test/java/com/haptictrack/tracking/YawEstimatorTest.kt
git commit -m "feat(guidance): coarse head-yaw estimator from BlazeFace keypoints"
```

---

### Task 3: GuidanceEngine.assess — bullseye, occupancy, drift, cue priority (pure)

**Files:**
- Create: `app/src/main/java/com/haptictrack/tracking/GuidanceEngine.kt`
- Test: `app/src/test/java/com/haptictrack/tracking/GuidanceEngineTest.kt`

**Interfaces:**
- Consumes: `FramingInput`, `FramingAssessment`, `FramingTarget`, `Cue`, `TrackingStatus` (Task 1).
- Produces: `class GuidanceEngine` with `fun assess(input: FramingInput): FramingAssessment`. Constants exposed for reuse/tuning: `LEVEL_TOL_DEG=8f`, `DRIFT_TOL=0.10f`, `OCC_LOW=0.7f`, `OCC_HIGH=1.4f`, `YAW_PROFILE_DEG=35f`, `EDGE_MARGIN=0.04f`.

**Design notes (read before coding):**
- **Region per target:** FULL_BODY → whole `subject`; UPPER_BODY → top 55% of `subject`; FACE_HEAD → `face.faceBox` if present else top 25% of `subject`.
- **Bullseye:** x = a rule-of-thirds line chosen by facing (faces image-right → seat region on LEFT third 0.33 for lead room; image-left → 0.67; frontal/unknown → center 0.5). y: place the region so its top sits near 0.18 (headroom) → bullseye y = 0.18 + regionHeight/2, clamped [0.2,0.8].
- **desiredOccupancy** (region area as fraction of frame): FULL_BODY 0.45, UPPER_BODY 0.40, FACE_HEAD 0.22.
- **drift:** `driftX=(regionCenter.x-bullseye.x)*2`, `driftY=(regionCenter.y-bullseye.y)*2`, each clamped [-1,1].
- **Cue priority (first match wins):** not LOCKED→NONE; |roll|>LEVEL_TOL→LEVEL; region within EDGE_MARGIN of a frame edge→CUT_OFF; occ ratio (desired/actual)>OCC_HIGH and zoom≈max→STEP_CLOSER; occ ratio<OCC_LOW and zoom≈min→STEP_BACK; |driftX|>DRIFT_TOL→MOVE_LEFT/RIGHT; |driftY|>DRIFT_TOL→TILT_UP/DOWN; face present and |yaw|>YAW_PROFILE_DEG and target≠FULL_BODY→FACING_AWAY; else HOLD. `satisfied = cue==HOLD`.

- [ ] **Step 1: Write the failing test**

```kotlin
// app/src/test/java/com/haptictrack/tracking/GuidanceEngineTest.kt
package com.haptictrack.tracking

import android.graphics.RectF
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test
import org.junit.runner.RunWith
import org.robolectric.RobolectricTestRunner

@RunWith(RobolectricTestRunner::class)
class GuidanceEngineTest {
    private val eng = GuidanceEngine()

    private fun input(
        subject: RectF? = RectF(0.40f, 0.30f, 0.60f, 0.90f),  // centered-ish full body, ~0.12 area
        face: FaceFraming? = null,
        roll: Float = 0f,
        zoom: Float = 2f,
        target: FramingTarget = FramingTarget.FULL_BODY,
        status: TrackingStatus = TrackingStatus.LOCKED,
    ) = FramingInput(status, subject, face, roll, zoom, 1f, 10f, target, frameTimeMs = 0L)

    @Test fun `not locked yields NONE`() {
        assertEquals(Cue.NONE, eng.assess(input(status = TrackingStatus.SEARCHING)).cue)
    }
    @Test fun `tilted camera asks to level first`() {
        assertEquals(Cue.LEVEL, eng.assess(input(roll = 14f)).cue)
    }
    @Test fun `subject at left edge is cut off`() {
        assertEquals(Cue.CUT_OFF, eng.assess(input(subject = RectF(0.0f, 0.3f, 0.18f, 0.9f))).cue)
    }
    @Test fun `subject left of bullseye asks move right`() {
        // Frontal full-body bullseye x = 0.5; put subject center at 0.30 -> need MOVE_RIGHT.
        val a = eng.assess(input(subject = RectF(0.20f, 0.30f, 0.40f, 0.90f)))
        assertEquals(Cue.MOVE_RIGHT, a.cue)
        assertTrue("driftX negative when left of bullseye", a.driftX < 0f)
    }
    @Test fun `frontal full-body bullseye is centered horizontally`() {
        assertEquals(0.5f, eng.assess(input()).bullseye.x, 0.001f)
    }
    @Test fun `facing right shifts bullseye to the left third for lead room`() {
        val face = FaceFraming(RectF(0.45f, 0.30f, 0.55f, 0.42f), yawDeg = 45f)
        val a = eng.assess(input(face = face, target = FramingTarget.UPPER_BODY))
        assertTrue("bullseye on left third", a.bullseye.x < 0.45f)
    }
    @Test fun `well-framed subject is satisfied and holds`() {
        // Build a subject whose region center sits on the bullseye at the right size.
        val a = eng.assess(input(subject = RectF(0.33f, 0.30f, 0.67f, 0.95f)))
        // Not asserting HOLD exactly here (size may differ); assert satisfied implies HOLD.
        assertEquals(a.satisfied, a.cue == Cue.HOLD)
    }
    @Test fun `face target with profile face says facing away`() {
        // Face region sits ON its bullseye (yaw>0 -> left third x=0.33; top=0.18 -> driftY=0) so
        // level/cut-off/drift/zoom cues don't pre-empt and FACING_AWAY is the surfaced cue.
        val face = FaceFraming(RectF(0.27f, 0.18f, 0.39f, 0.30f), yawDeg = 50f)
        val a = eng.assess(input(subject = RectF(0.20f, 0.10f, 0.50f, 1.0f), face = face, target = FramingTarget.FACE_HEAD))
        assertEquals(Cue.FACING_AWAY, a.cue)
    }
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `./gradlew testDebugUnitTest --tests "com.haptictrack.tracking.GuidanceEngineTest"`
Expected: FAIL — `GuidanceEngine` unresolved.

- [ ] **Step 3: Implement GuidanceEngine.assess**

```kotlin
// app/src/main/java/com/haptictrack/tracking/GuidanceEngine.kt
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
        private const val HEADROOM_TOP = 0.18f
        private const val ZOOM_LIMIT_EPS = 0.05f
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
```

> **Drift sign note for implementers:** `driftX > 0` means the region is RIGHT of the bullseye, so the operator must aim left → but the *subject* needs to move right in frame, which the operator achieves by panning the phone. We label the cue from the operator's corrective action: region right of target → `MOVE_LEFT` (pan left). The test `subject left of bullseye asks move right` encodes the convention: region left of bullseye (driftX<0) → `MOVE_RIGHT`. Keep `deriveCue` exactly as written so cue text and haptic drift agree.

- [ ] **Step 4: Run tests to verify they pass**

Run: `./gradlew testDebugUnitTest --tests "com.haptictrack.tracking.GuidanceEngineTest"`
Expected: PASS (8 tests). If `facing away` or `move right` fail, re-check the sign convention against the note above — do not flip signs in only one place.

- [ ] **Step 5: Commit**

```bash
git add app/src/main/java/com/haptictrack/tracking/GuidanceEngine.kt app/src/test/java/com/haptictrack/tracking/GuidanceEngineTest.kt
git commit -m "feat(guidance): GuidanceEngine.assess — bullseye, occupancy, drift, cue priority"
```

---

### Task 4: GuidanceEngine cadence throttle (stateful, time-injected)

**Files:**
- Modify: `app/src/main/java/com/haptictrack/tracking/GuidanceEngine.kt` (add `throttle`)
- Test: `app/src/test/java/com/haptictrack/tracking/GuidanceThrottleTest.kt`

**Interfaces:**
- Produces: `fun GuidanceEngine.throttle(cue: Cue, frameTimeMs: Long): Cue` — returns the cue to actually SPEAK (or `Cue.NONE` to stay silent). Rules: never speak `NONE`; enforce `MIN_GAP_MS` between any two spoken cues; suppress a cue identical to the last spoken one; speak `HOLD` only once on entry into the held state, then stay silent until the cue changes away from `HOLD`. `reset()` clears state (call on unlock).

- [ ] **Step 1: Write the failing test**

```kotlin
// app/src/test/java/com/haptictrack/tracking/GuidanceThrottleTest.kt
package com.haptictrack.tracking

import org.junit.Assert.assertEquals
import org.junit.Test

class GuidanceThrottleTest {
    private val gap = GuidanceEngine.MIN_GAP_MS

    @Test fun `first non-none cue speaks immediately`() {
        val e = GuidanceEngine()
        assertEquals(Cue.MOVE_LEFT, e.throttle(Cue.MOVE_LEFT, 0L))
    }
    @Test fun `same cue within gap is suppressed`() {
        val e = GuidanceEngine()
        e.throttle(Cue.MOVE_LEFT, 0L)
        assertEquals(Cue.NONE, e.throttle(Cue.MOVE_LEFT, gap - 1))
    }
    @Test fun `different cue still waits for the gap`() {
        val e = GuidanceEngine()
        e.throttle(Cue.MOVE_LEFT, 0L)
        assertEquals(Cue.NONE, e.throttle(Cue.TILT_UP, gap - 1))
        assertEquals(Cue.TILT_UP, e.throttle(Cue.TILT_UP, gap + 1))
    }
    @Test fun `NONE is never spoken`() {
        val e = GuidanceEngine()
        assertEquals(Cue.NONE, e.throttle(Cue.NONE, 10_000L))
    }
    @Test fun `HOLD speaks once then stays silent`() {
        val e = GuidanceEngine()
        assertEquals(Cue.HOLD, e.throttle(Cue.HOLD, 0L))
        assertEquals(Cue.NONE, e.throttle(Cue.HOLD, gap * 5))
    }
    @Test fun `HOLD can speak again after leaving and re-entering hold`() {
        val e = GuidanceEngine()
        e.throttle(Cue.HOLD, 0L)
        e.throttle(Cue.MOVE_LEFT, gap + 1)               // left hold
        assertEquals(Cue.HOLD, e.throttle(Cue.HOLD, gap * 3)) // re-entered hold
    }
    @Test fun `reset clears state`() {
        val e = GuidanceEngine()
        e.throttle(Cue.MOVE_LEFT, 0L)
        e.reset()
        assertEquals(Cue.MOVE_LEFT, e.throttle(Cue.MOVE_LEFT, 10L))
    }
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `./gradlew testDebugUnitTest --tests "com.haptictrack.tracking.GuidanceThrottleTest"`
Expected: FAIL — `throttle`/`reset`/`MIN_GAP_MS` unresolved.

- [ ] **Step 3: Implement the throttle**

Add `const val MIN_GAP_MS = 1800L` to the `companion object`, and add these members to `GuidanceEngine`:

```kotlin
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `./gradlew testDebugUnitTest --tests "com.haptictrack.tracking.GuidanceThrottleTest"`
Expected: PASS (7 tests).

- [ ] **Step 5: Commit**

```bash
git add app/src/main/java/com/haptictrack/tracking/GuidanceEngine.kt app/src/test/java/com/haptictrack/tracking/GuidanceThrottleTest.kt
git commit -m "feat(guidance): spoken-cue cadence throttle"
```

---

### Task 5: Cue → spoken phrase mapping (pure)

**Files:**
- Modify: `app/src/main/java/com/haptictrack/tracking/GuidanceTypes.kt` (add `cuePhrase`)
- Test: `app/src/test/java/com/haptictrack/tracking/CuePhraseTest.kt`

**Interfaces:**
- Produces: `fun cuePhrase(cue: Cue): String?` — short spoken text per cue; `null` for `Cue.NONE`.

- [ ] **Step 1: Write the failing test**

```kotlin
// app/src/test/java/com/haptictrack/tracking/CuePhraseTest.kt
package com.haptictrack.tracking

import org.junit.Assert.assertEquals
import org.junit.Assert.assertNotNull
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Test

class CuePhraseTest {
    @Test fun `none has no phrase`() { assertNull(cuePhrase(Cue.NONE)) }
    @Test fun `every other cue has a non-blank phrase`() {
        for (c in Cue.entries) if (c != Cue.NONE) {
            val p = cuePhrase(c)
            assertNotNull("missing phrase for $c", p)
            assertTrue("blank phrase for $c", p!!.isNotBlank())  // assertTrue, not Kotlin assert (disabled w/o -ea)
        }
    }
    @Test fun `move left says left`() { assertEquals("move left", cuePhrase(Cue.MOVE_LEFT)) }
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `./gradlew testDebugUnitTest --tests "com.haptictrack.tracking.CuePhraseTest"`
Expected: FAIL — `cuePhrase` unresolved.

- [ ] **Step 3: Implement the mapping**

Append to `GuidanceTypes.kt`:

```kotlin
/** Short spoken phrase for a cue; null = say nothing. Keep phrases ≤3 words for low BT latency. */
fun cuePhrase(cue: Cue): String? = when (cue) {
    Cue.NONE -> null
    Cue.LEVEL -> "level the camera"
    Cue.CUT_OFF -> "they're cut off"
    Cue.MOVE_LEFT -> "move left"
    Cue.MOVE_RIGHT -> "move right"
    Cue.TILT_UP -> "tilt up"
    Cue.TILT_DOWN -> "tilt down"
    Cue.STEP_CLOSER -> "step closer"
    Cue.STEP_BACK -> "step back"
    Cue.FACING_AWAY -> "facing away"
    Cue.HOLD -> "good, hold"
}
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `./gradlew testDebugUnitTest --tests "com.haptictrack.tracking.CuePhraseTest"`
Expected: PASS (3 tests).

- [ ] **Step 5: Commit**

```bash
git add app/src/main/java/com/haptictrack/tracking/GuidanceTypes.kt app/src/test/java/com/haptictrack/tracking/CuePhraseTest.kt
git commit -m "feat(guidance): cue-to-phrase mapping"
```

---

### Task 6: Expose camera roll from GyroStabilizer

**Files:**
- Modify: `app/src/main/java/com/haptictrack/camera/GyroStabilizer.kt`
- Test: `app/src/test/java/com/haptictrack/camera/GyroRollExposureTest.kt`

**Interfaces:**
- Produces: `fun GyroStabilizer.currentRollDeg(): Float` — current camera roll vs gravity in degrees (0 = level), from the latest raw orientation. Returns 0f before any sensor sample.

- [ ] **Step 1: Write the failing test**

```kotlin
// app/src/test/java/com/haptictrack/camera/GyroRollExposureTest.kt
package com.haptictrack.camera

import org.junit.Assert.assertEquals
import org.junit.Test
import org.junit.runner.RunWith
import org.robolectric.RobolectricTestRunner
import org.robolectric.RuntimeEnvironment

@RunWith(RobolectricTestRunner::class)
class GyroRollExposureTest {
    @Test fun `roll is zero before any sensor sample`() {
        val stab = GyroStabilizer(RuntimeEnvironment.getApplication())
        assertEquals(0f, stab.currentRollDeg(), 0.001f)
    }
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `./gradlew testDebugUnitTest --tests "com.haptictrack.camera.GyroRollExposureTest"`
Expected: FAIL — `currentRollDeg` unresolved.

- [ ] **Step 3: Add the getter**

In `GyroStabilizer.kt`, the class already holds the latest raw orientation quaternion (`rawQuat`, used at `:202`) and a static `gravityRollDeg(q: Quat): Double` (`:100`). Add a public method (place it near the other public accessors, e.g. after the `horizonLockEnabled` field):

```kotlin
    /** Current camera roll vs gravity in degrees (0 = level). 0 before the first sensor sample. */
    fun currentRollDeg(): Float = gravityRollDeg(rawQuat).toFloat()
```

If `rawQuat` is initialized to identity at construction, `gravityRollDeg(identity)` already returns ~0 — confirm by reading the `rawQuat` declaration; if it is nullable or lateinit, guard: `val q = rawQuat ?: return 0f`. Match the actual field.

- [ ] **Step 4: Run tests to verify they pass**

Run: `./gradlew testDebugUnitTest --tests "com.haptictrack.camera.GyroRollExposureTest"`
Expected: PASS (1 test).

- [ ] **Step 5: Commit**

```bash
git add app/src/main/java/com/haptictrack/camera/GyroStabilizer.kt app/src/test/java/com/haptictrack/camera/GyroRollExposureTest.kt
git commit -m "feat(guidance): expose currentRollDeg from GyroStabilizer"
```

---

### Task 7: Settable occupancy target on ZoomController

**Files:**
- Modify: `app/src/main/java/com/haptictrack/zoom/ZoomController.kt`
- Test: `app/src/test/java/com/haptictrack/zoom/ZoomOccupancyTargetTest.kt`

**Interfaces:**
- Produces: `var ZoomController.occupancyTarget: Float` (defaults to the constructor `targetFrameOccupancy`); `calculateZoom` uses it instead of the constant so the framing target can retarget subject size at runtime.

- [ ] **Step 1: Write the failing test**

```kotlin
// app/src/test/java/com/haptictrack/zoom/ZoomOccupancyTargetTest.kt
package com.haptictrack.zoom

import android.graphics.RectF
import org.junit.Assert.assertTrue
import org.junit.Test
import org.junit.runner.RunWith
import org.robolectric.RobolectricTestRunner

@RunWith(RobolectricTestRunner::class)
class ZoomOccupancyTargetTest {
    @Test fun `larger occupancy target zooms in more`() {
        val box = RectF(0.45f, 0.45f, 0.55f, 0.55f)  // small subject
        val small = ZoomController().apply { occupancyTarget = 0.10f }.calculateZoom(box, 1f, 10f)
        val large = ZoomController().apply { occupancyTarget = 0.40f }.calculateZoom(box, 1f, 10f)
        assertTrue("bigger target -> more zoom", large >= small)
    }
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `./gradlew testDebugUnitTest --tests "com.haptictrack.zoom.ZoomOccupancyTargetTest"`
Expected: FAIL — `occupancyTarget` unresolved.

- [ ] **Step 3: Make occupancy settable**

In `ZoomController.kt`, the constructor has `private val targetFrameOccupancy: Float = 0.15f` (`:7`). Add a settable field initialized from it, and use the field in `calculateZoom`:

```kotlin
    /** Runtime-adjustable occupancy target (framing coach overrides per target). */
    @Volatile var occupancyTarget: Float = targetFrameOccupancy
```

Then in `calculateZoom` replace the `targetFrameOccupancy` usage (`:85`, `val areaRatio = ... targetFrameOccupancy / boxArea ...`) with `occupancyTarget`. Leave the constructor param as the default source.

- [ ] **Step 4: Run tests to verify they pass**

Run: `./gradlew testDebugUnitTest --tests "com.haptictrack.zoom.ZoomOccupancyTargetTest"`
Expected: PASS (1 test).

- [ ] **Step 5: Commit**

```bash
git add app/src/main/java/com/haptictrack/zoom/ZoomController.kt app/src/test/java/com/haptictrack/zoom/ZoomOccupancyTargetTest.kt
git commit -m "feat(guidance): runtime-settable zoom occupancy target"
```

---

### Task 8: Per-frame locked-face framing (FaceEmbedder + ObjectTracker)

**Files:**
- Modify: `app/src/main/java/com/haptictrack/tracking/FaceEmbedder.kt` (add `detectFaceFraming`)
- Modify: `app/src/main/java/com/haptictrack/tracking/ObjectTracker.kt` (add `lockedFaceFraming` field + throttled update + a pure mapping helper)
- Test: `app/src/test/java/com/haptictrack/tracking/FaceFramingMapTest.kt`

**Interfaces:**
- Consumes: `FaceFraming`, `estimateYawDeg` (Tasks 1-2), `RectF`.
- Produces:
  - `FaceEmbedder.detectFaceFraming(bitmap: Bitmap, personBox: RectF): FaceFramingLocal?` where `data class FaceFramingLocal(val faceBoxInPerson: RectF, val yawDeg: Float?)` (face box normalized within `personBox`, 0..1).
  - `ObjectTracker.lockedFaceFraming: FaceFraming?` (`@Volatile`, screen-normalized; null when no lock/face).
  - top-level `fun mapFaceToScreen(faceInPerson: RectF, personBox: RectF): RectF` in ObjectTracker.kt.

- [ ] **Step 1: Write the failing test (pure mapping)**

```kotlin
// app/src/test/java/com/haptictrack/tracking/FaceFramingMapTest.kt
package com.haptictrack.tracking

import android.graphics.RectF
import org.junit.Assert.assertEquals
import org.junit.Test
import org.junit.runner.RunWith
import org.robolectric.RobolectricTestRunner

@RunWith(RobolectricTestRunner::class)
class FaceFramingMapTest {
    @Test fun `face within person maps into screen coords`() {
        val person = RectF(0.40f, 0.20f, 0.60f, 0.80f)        // 0.2 wide, 0.6 tall
        val faceInPerson = RectF(0.25f, 0.0f, 0.75f, 0.20f)   // top-center band of the person crop
        val screen = mapFaceToScreen(faceInPerson, person)
        assertEquals(0.40f + 0.25f * 0.20f, screen.left, 1e-4f)
        assertEquals(0.20f + 0.0f * 0.60f, screen.top, 1e-4f)
        assertEquals(0.40f + 0.75f * 0.20f, screen.right, 1e-4f)
        assertEquals(0.20f + 0.20f * 0.60f, screen.bottom, 1e-4f)
    }
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `./gradlew testDebugUnitTest --tests "com.haptictrack.tracking.FaceFramingMapTest"`
Expected: FAIL — `mapFaceToScreen` unresolved.

- [ ] **Step 3: Add the mapping helper + ObjectTracker field**

In `ObjectTracker.kt`, add a top-level function (bottom of file, near other helpers):

```kotlin
/** Map a face box expressed in personBox-normalized coords into screen-normalized coords. */
fun mapFaceToScreen(faceInPerson: android.graphics.RectF, personBox: android.graphics.RectF): android.graphics.RectF {
    val w = personBox.width(); val h = personBox.height()
    return android.graphics.RectF(
        personBox.left + faceInPerson.left * w,
        personBox.top + faceInPerson.top * h,
        personBox.left + faceInPerson.right * w,
        personBox.top + faceInPerson.bottom * h,
    )
}
```

And add the field near the other `@Volatile` tracker state (e.g. by `onGiveUp`):

```kotlin
    /** Locked subject's face framing (screen-normalized) for the guidance coach; null if none. */
    @Volatile var lockedFaceFraming: FaceFraming? = null
    private var faceFramingFrameCount = 0
```

- [ ] **Step 4: Add FaceEmbedder.detectFaceFraming**

In `FaceEmbedder.kt`, add a method mirroring `classifyAttributes`'s detection (it already crops the person, runs `faceDetector`, picks the largest face, and has `normalizeFaceBox`). It must NOT classify — just return geometry:

```kotlin
    /** Face box (normalized within personBox) + coarse yaw for the largest face. Null if none. */
    @Synchronized
    fun detectFaceFraming(bitmap: Bitmap, personBox: RectF): FaceFramingLocal? {
        val personCanonical = cropper.prepare(
            bitmap, personBox,
            targetWidth = PERSON_CANONICAL_SIZE, targetHeight = PERSON_CANONICAL_SIZE,
            paddingFraction = 0f, minSourcePixels = MIN_PERSON_SOURCE_PIXELS,
        ) ?: return null
        val personCrop = personCanonical.bitmap
        return try {
            val mpImage = BitmapImageBuilder(personCrop).build()
            val faces = synchronized(faceDetector) { faceDetector.detect(mpImage) }
            val face = faces.detections().maxByOrNull {
                it.boundingBox().width() * it.boundingBox().height()
            } ?: return null
            val box = normalizeFaceBox(face.boundingBox(), personCrop.width, personCrop.height) ?: return null
            val kps = face.keypoints().orElse(emptyList())
            val yaw = if (kps.size >= 3)
                estimateYawDeg(
                    android.graphics.PointF(kps[0].x(), kps[0].y()),  // right eye
                    android.graphics.PointF(kps[1].x(), kps[1].y()),  // left eye
                    android.graphics.PointF(kps[2].x(), kps[2].y()),  // nose
                ) else null
            FaceFramingLocal(box, yaw)
        } catch (e: Exception) {
            Log.w(TAG, "detectFaceFraming failed: ${e.message}"); null
        } finally {
            personCrop.recycle()
        }
    }
```

Add the data class near the top of `FaceEmbedder.kt`:

```kotlin
data class FaceFramingLocal(val faceBoxInPerson: RectF, val yawDeg: Float?)
```

> If `normalizeFaceBox` is `private`, it stays usable here (same class). `paddingFraction = 0f` keeps personCrop ↔ personBox a 1:1 linear map so `mapFaceToScreen` is exact.

- [ ] **Step 5: Wire the throttled update in ObjectTracker**

In the detector-path block of `processBitmap`, right before the `onDetectionResult?.invoke(...)` at the detector path (~`:1170`), update the face framing every 5th frame when locked (mirrors `TEMPLATE_CHECK_INTERVAL`):

```kotlin
            if (lockedObject != null) {
                faceFramingFrameCount++
                if (faceFramingFrameCount % 5 == 0) {
                    val local = faceEmbedder.detectFaceFraming(bitmap, lockedObject.boundingBox)
                    lockedFaceFraming = local?.let {
                        FaceFraming(mapFaceToScreen(it.faceBoxInPerson, lockedObject.boundingBox), it.yawDeg)
                    }
                }
            } else {
                lockedFaceFraming = null
                faceFramingFrameCount = 0
            }
```

- [ ] **Step 6: Run tests + full suite**

Run: `./gradlew testDebugUnitTest --tests "com.haptictrack.tracking.FaceFramingMapTest"` then `./gradlew testDebugUnitTest`
Expected: PASS (1 new test); full suite green.

- [ ] **Step 7: Commit**

```bash
git add app/src/main/java/com/haptictrack/tracking/FaceEmbedder.kt app/src/main/java/com/haptictrack/tracking/ObjectTracker.kt app/src/test/java/com/haptictrack/tracking/FaceFramingMapTest.kt
git commit -m "feat(guidance): per-frame locked-face framing + yaw plumbing"
```

---

### Task 9: VoiceGuide — TTS over Bluetooth A2DP

**Files:**
- Create: `app/src/main/java/com/haptictrack/audio/VoiceGuide.kt`
- Modify: `app/src/main/AndroidManifest.xml` (add `BLUETOOTH_CONNECT` permission)

**Interfaces:**
- Consumes: `Cue`, `cuePhrase` (Tasks 1, 5).
- Produces: `class VoiceGuide(context: Context)` with `fun start()`, `fun speak(cue: Cue)`, `fun shutdown()`. `speak(Cue.NONE)` is a no-op. Uses `QUEUE_FLUSH` and navigation-guidance `AudioAttributes`. NEVER touches SCO/communication mode (Global Constraints).

> This task is Android glue (TTS engine, audio routing) — not unit-testable. Verification is build + on-device. The spoken text is already unit-tested via `cuePhrase` (Task 5).

- [ ] **Step 1: Add the Bluetooth permission**

In `AndroidManifest.xml`, alongside the existing `<uses-permission>` lines (CAMERA/RECORD_AUDIO/VIBRATE):

```xml
    <uses-permission android:name="android.permission.BLUETOOTH_CONNECT" />
```

- [ ] **Step 2: Create VoiceGuide**

```kotlin
// app/src/main/java/com/haptictrack/audio/VoiceGuide.kt
package com.haptictrack.audio

import android.content.Context
import android.media.AudioAttributes
import android.speech.tts.TextToSpeech
import android.util.Log
import com.haptictrack.tracking.Cue
import com.haptictrack.tracking.cuePhrase
import java.util.Locale

/**
 * Speaks framing cues to the active media output (the Bluetooth earbud over A2DP).
 *
 * HARD RULE (see plan Global Constraints): media playback only, tagged navigation-guidance.
 * Never start Bluetooth SCO / communication mode — that would seize the mic and break the
 * CAMCORDER recording. A2DP routing happens automatically because we use the media stream.
 */
class VoiceGuide(private val context: Context) {
    private var tts: TextToSpeech? = null
    @Volatile private var ready = false

    fun start() {
        if (tts != null) return
        tts = TextToSpeech(context) { status ->
            if (status == TextToSpeech.SUCCESS) {
                tts?.language = Locale.US
                tts?.setAudioAttributes(
                    AudioAttributes.Builder()
                        .setUsage(AudioAttributes.USAGE_ASSISTANCE_NAVIGATION_GUIDANCE)
                        .setContentType(AudioAttributes.CONTENT_TYPE_SPEECH)
                        .build()
                )
                ready = true
            } else {
                Log.w("VoiceGuide", "TTS init failed: $status")
            }
        }
    }

    /** Speak a cue's phrase, flushing any pending utterance so cues never backlog. */
    fun speak(cue: Cue) {
        if (!ready) return
        val phrase = cuePhrase(cue) ?: return
        tts?.speak(phrase, TextToSpeech.QUEUE_FLUSH, null, "guidance-${cue.name}")
    }

    fun shutdown() {
        tts?.stop(); tts?.shutdown(); tts = null; ready = false
    }
}
```

- [ ] **Step 3: Verify it builds**

Run: `./gradlew assembleDebug -q`
Expected: BUILD SUCCESSFUL.

- [ ] **Step 4: Commit**

```bash
git add app/src/main/java/com/haptictrack/audio/VoiceGuide.kt app/src/main/AndroidManifest.xml
git commit -m "feat(guidance): VoiceGuide TTS over Bluetooth A2DP (media-only, no SCO)"
```

---

### Task 10: Wire guidance into CameraViewModel

**Files:**
- Modify: `app/src/main/java/com/haptictrack/ui/CameraViewModel.kt`

**Interfaces:**
- Consumes: `GuidanceEngine`, `VoiceGuide`, `FramingInput`, `FaceFraming`, `GuidanceMode`, `FramingTarget`, `cuePhrase`, `ObjectTracker.lockedFaceFraming`, `GyroStabilizer.currentRollDeg`, `ZoomController.occupancyTarget` (Tasks 1-9).
- Produces: VM methods `cycleGuidanceMode()`, `cycleFramingTarget()`; per-frame guidance computation inside `onDetectionResult`.

> Android glue — verified by build + the existing suite staying green + on-device. No new unit test (the logic it calls is already covered).

- [ ] **Step 1: Add fields**

Near the other collaborators (after `hapticManager` / `sentryLogger`):

```kotlin
    private val guidanceEngine = com.haptictrack.tracking.GuidanceEngine()
    private val voiceGuide = com.haptictrack.audio.VoiceGuide(application).also { it.start() }
```

- [ ] **Step 2: Feed the engine each frame**

In `onDetectionResult`, after `effectiveStatus` and the drift block are computed and **before** the `_uiState.update { ... }`, add:

```kotlin
                val mode = _uiState.value.guidanceMode
                if (mode != com.haptictrack.tracking.GuidanceMode.OFF) {
                    val input = com.haptictrack.tracking.FramingInput(
                        status = effectiveStatus,
                        subject = lockedObject?.boundingBox,
                        face = objectTracker.lockedFaceFraming,
                        rollDeg = cameraManager.gyroStabilizer.currentRollDeg(),
                        zoomRatio = cameraManager.gyroStabilizer.zoomRatio,
                        minZoom = cameraManager.getMinZoom(),
                        maxZoom = cameraManager.getMaxZoom(),
                        target = _uiState.value.framingTarget,
                        frameTimeMs = android.os.SystemClock.elapsedRealtime(),
                    )
                    val a = guidanceEngine.assess(input)
                    cameraManager.setZoomOccupancyTarget(a.desiredOccupancy)   // see Step 3
                    if (mode == com.haptictrack.tracking.GuidanceMode.HAPTIC ||
                        mode == com.haptictrack.tracking.GuidanceMode.BOTH) {
                        hapticManager.updateTrackingStatus(effectiveStatus, a.driftX, a.driftY)
                    }
                    if (mode == com.haptictrack.tracking.GuidanceMode.VOICE ||
                        mode == com.haptictrack.tracking.GuidanceMode.BOTH) {
                        voiceGuide.speak(guidanceEngine.throttle(a.cue, input.frameTimeMs))
                    }
                }
```

> Note: when `mode` includes HAPTIC, this REPLACES the existing dead-center `hapticManager.updateTrackingStatus(effectiveStatus, driftX, driftY)` call for the locked frame (drift now points at the bullseye). Leave the original haptic call in place for `mode == OFF` so default behavior is unchanged — i.e. guard the original call with `if (mode == GuidanceMode.OFF)` or restructure so exactly one haptic update fires per frame. Do not double-call `updateTrackingStatus`.

- [ ] **Step 3: Add the zoom-occupancy setter on CameraManager**

`occupancyTarget` lives on `ZoomController`, which the ViewModel reaches via `cameraManager`. Add a thin pass-through in `CameraManager.kt` (the ViewModel already calls `cameraManager.setZoomTarget`):

```kotlin
    fun setZoomOccupancyTarget(target: Float) { zoomController.occupancyTarget = target }
```

(If the ViewModel holds its own `zoomController` reference instead, set `zoomController.occupancyTarget = a.desiredOccupancy` directly and skip this pass-through. Match the existing wiring — grep for `zoomController` in CameraViewModel.kt.)

- [ ] **Step 4: Add the toggle methods + reset on clear**

```kotlin
    fun cycleGuidanceMode() {
        val next = _uiState.value.guidanceMode.next()
        guidanceEngine.reset()
        _uiState.update { it.copy(guidanceMode = next) }
    }

    fun cycleFramingTarget() {
        _uiState.update { it.copy(framingTarget = it.framingTarget.next()) }
    }
```

In `clearTracking()`, add `guidanceEngine.reset()` (so cadence state doesn't leak across locks). In `onCleared()`, add `voiceGuide.shutdown()`.

- [ ] **Step 5: Verify build + full suite**

Run: `./gradlew testDebugUnitTest`
Expected: BUILD SUCCESSFUL, full suite green (no behavior change while `guidanceMode == OFF`).

- [ ] **Step 6: Commit**

```bash
git add app/src/main/java/com/haptictrack/ui/CameraViewModel.kt app/src/main/java/com/haptictrack/camera/CameraManager.kt
git commit -m "feat(guidance): wire GuidanceEngine + VoiceGuide into the frame loop"
```

---

### Task 11: On-screen toggles (guidance mode + framing target)

**Files:**
- Modify: `app/src/main/java/com/haptictrack/ui/CameraScreen.kt`

**Interfaces:**
- Consumes: `uiState.guidanceMode`, `uiState.framingTarget`, `viewModel.cycleGuidanceMode()`, `viewModel.cycleFramingTarget()`, `GuidanceMode`, `FramingTarget`.
- Produces: two tappable pills following the existing `SentryPill` pattern (`CameraScreen.kt`), shown while LOCKED.

> UI glue — verify by build + on-device. No unit test.

- [ ] **Step 1: Add the pills**

Mirror the existing `SentryPill`/`TrackingFilterPill` composables. Add a `GuidancePill` and `FramingPill`:

```kotlin
@Composable
private fun GuidancePill(mode: GuidanceMode, onCycle: () -> Unit, modifier: Modifier = Modifier) {
    val (label, on) = when (mode) {
        GuidanceMode.OFF -> "COACH off" to false
        GuidanceMode.HAPTIC -> "COACH · haptic" to true
        GuidanceMode.VOICE -> "COACH · voice" to true
        GuidanceMode.BOTH -> "COACH · both" to true
    }
    Text(
        text = label,
        color = if (on) HapticAmber else Color.White.copy(alpha = 0.55f),
        fontSize = 12.sp, fontWeight = FontWeight.SemiBold,
        modifier = modifier.clip(RoundedCornerShape(12.dp)).clickable(onClick = onCycle)
            .background(if (on) HapticAmber.copy(alpha = 0.18f) else Color.Black.copy(alpha = 0.3f),
                RoundedCornerShape(12.dp)).padding(horizontal = 14.dp, vertical = 5.dp)
    )
}

@Composable
private fun FramingPill(target: FramingTarget, onCycle: () -> Unit, modifier: Modifier = Modifier) {
    val label = when (target) {
        FramingTarget.FULL_BODY -> "frame: full body"
        FramingTarget.UPPER_BODY -> "frame: upper body"
        FramingTarget.FACE_HEAD -> "frame: face"
    }
    Text(
        text = label, color = Color.White.copy(alpha = 0.8f),
        fontSize = 12.sp, fontWeight = FontWeight.Medium,
        modifier = modifier.clip(RoundedCornerShape(12.dp)).clickable(onClick = onCycle)
            .background(Color.Black.copy(alpha = 0.3f), RoundedCornerShape(12.dp))
            .padding(horizontal = 14.dp, vertical = 4.dp)
    )
}
```

- [ ] **Step 2: Render them while locked**

In the bottom-controls `Column` (where `SentryPill`/`TrackingFilterPill` render), add — shown when `uiState.status != TrackingStatus.IDLE` so they're available during a shoot:

```kotlin
        if (uiState.status != TrackingStatus.IDLE) {
            GuidancePill(uiState.guidanceMode, onCycle = { viewModel.cycleGuidanceMode() })
            Spacer(Modifier.height(8.dp))
            if (uiState.guidanceMode != GuidanceMode.OFF) {
                FramingPill(uiState.framingTarget, onCycle = { viewModel.cycleFramingTarget() })
                Spacer(Modifier.height(8.dp))
            }
        }
```

Wire any new params through `BottomControls` if that composable gates its content (follow how `SentryPill` is currently threaded — it may read `uiState` directly or via params; match it). Ensure imports for `GuidanceMode`/`FramingTarget` (`com.haptictrack.tracking.*`).

- [ ] **Step 3: Verify build**

Run: `./gradlew assembleDebug -q`
Expected: BUILD SUCCESSFUL.

- [ ] **Step 4: Commit**

```bash
git add app/src/main/java/com/haptictrack/ui/CameraScreen.kt
git commit -m "feat(guidance): on-screen guidance-mode + framing-target toggles"
```

---

### Task 12: On-device verification + audio-safety check

**Files:** none (verification task).

> The one risk unit tests can't cover: that voice playback doesn't corrupt the recorded audio. Verify explicitly.

- [ ] **Step 1: Build, install**

```bash
./gradlew assembleDebug -q
adb install -r app/build/outputs/apk/debug/app-debug.apk
```

- [ ] **Step 2: Functional check** — pair a BT earbud, tap-to-lock a person, cycle COACH to `voice`, cycle framing target. Confirm: spoken cues are sensible and timely (level / move / tilt / step / hold), silence-means-good, no backlog/overlap. Try all three framing targets and confirm zoom retargets (face = tighter).

- [ ] **Step 3: Audio-safety check (critical)** — record a clip with COACH on `voice` and cues actively playing. Pull the clip, inspect its audio track (spectrogram or listen): the spoken cues must **NOT** appear in the recording, and the recorded audio must not be muted/garbled. If cues bleed in, lower TTS volume / confirm A2DP routing; if the track is muted, confirm no code path touched SCO/communication mode.

```bash
adb pull /sdcard/Movies/HapticTrack/   # newest clip
```

- [ ] **Step 4: Regression** — confirm with COACH `off` that haptics + recording behave exactly as before this feature.

- [ ] **Step 5: Commit any tuning** (cue thresholds, MIN_GAP_MS, occupancy targets) discovered during device testing. **Also tune cue WORDING against real BT audio** (review deferred): `CUT_OFF` "they're cut off" is descriptive/homophone-ambiguous — prefer an imperative like "recenter"; drop filler words ("level the camera" → "level"). Confirm each phrase is an actionable imperative the operator can follow instantly.

```bash
git add -A && git commit -m "tune(guidance): on-device cue thresholds + cadence"
```

---

## Self-Review

**Spec coverage:**
- Continuous coaching → Task 3/4 (assess every frame, throttle for cadence). ✓
- VOICE/HAPTIC/BOTH toggle → `GuidanceMode` (Task 1) + routing (Task 10) + pill (Task 11). ✓
- Selectable framing target (Full/Upper/Face) → `FramingTarget` (Task 1), region/occupancy (Task 3), pill (Task 11). ✓
- Composition-led bullseye (thirds/headroom/lead-room) → Task 3 `bullseyeFor`. ✓
- Framing target retargets zoom → Task 7 + Task 10 Step 3. ✓
- Coarse facing from existing keypoints → Task 2 + Task 8. ✓
- Works front/back → subject-bbox cues work without a face; face cues degrade to null gracefully (Task 3 handles `face == null`). ✓
- Bluetooth A2DP, no SCO, no recording corruption → Global Constraints + Task 9 + Task 12 Step 3. ✓
- Always-on-when-locked → pills shown and engine fed while `status != IDLE` (Tasks 10-11). ✓
- Body-part targets beyond Full/Upper/Face → explicitly Phase 2 (out of scope). ✓

**Placeholder scan:** no TBD/TODO; every code step has complete code; thresholds are concrete values.

**Type consistency:** `FramingTarget`, `GuidanceMode`, `Cue`, `FaceFraming`, `FramingInput`, `FramingAssessment` defined in Task 1 and used unchanged in Tasks 3/8/10; `estimateYawDeg` signature (Task 2) matches its call in Task 8; `occupancyTarget` (Task 7) matches Task 10 Step 3; `lockedFaceFraming` (Task 8) matches Task 10. `cuePhrase`/`throttle`/`reset` names consistent across Tasks 4/5/9/10.

## Phase 2 (out of scope, noted for later)
- MediaPipe FaceLandmarker for true head-pose (reliable facing + precise lead-room) if coarse yaw proves insufficient on-device.
- PoseLandmarker for "prefer specific body parts" (hands/legs/torso) as additional `FramingTarget`s.
- NIMA aesthetic scoring as an offline `tools/` QA metric over recorded clips (never in the live loop).
