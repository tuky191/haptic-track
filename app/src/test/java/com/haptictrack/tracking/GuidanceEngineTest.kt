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
    @Test fun `well-framed subject holds`() {
        // Full body, no face -> bullseye (0.5, 0.18+h/2); subject top=0.18, centered x=0.5 -> drift 0.
        // zoom=2 is off both rails so the (too-small) occupancy doesn't surface STEP_CLOSER.
        val a = eng.assess(input(subject = RectF(0.35f, 0.18f, 0.65f, 0.78f)))
        assertEquals(Cue.HOLD, a.cue)
        assertTrue(a.satisfied)
    }
    @Test fun `too small at max zoom says step closer`() {
        val a = eng.assess(input(subject = RectF(0.45f, 0.45f, 0.55f, 0.55f), zoom = 10f))
        assertEquals(Cue.STEP_CLOSER, a.cue)
    }
    @Test fun `too large at min zoom says step back`() {
        val a = eng.assess(input(subject = RectF(0.05f, 0.05f, 0.95f, 0.95f), zoom = 1f))
        assertEquals(Cue.STEP_BACK, a.cue)
    }
    @Test fun `face target with profile face says facing away`() {
        // Face region sits ON its bullseye (yaw>0 -> left third x=0.33; top=0.18 -> driftY=0) so
        // level/cut-off/drift/zoom cues don't pre-empt and FACING_AWAY is the surfaced cue.
        val face = FaceFraming(RectF(0.27f, 0.18f, 0.39f, 0.30f), yawDeg = 50f)
        val a = eng.assess(input(subject = RectF(0.20f, 0.10f, 0.50f, 1.0f), face = face, target = FramingTarget.FACE_HEAD))
        assertEquals(Cue.FACING_AWAY, a.cue)
    }
}
