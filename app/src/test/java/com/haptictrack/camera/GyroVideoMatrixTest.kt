package com.haptictrack.camera

import android.hardware.Sensor
import android.hardware.SensorEvent
import org.junit.Assert.*
import org.junit.Test
import org.junit.runner.RunWith
import org.robolectric.RobolectricTestRunner
import org.robolectric.RuntimeEnvironment
import org.robolectric.shadows.ShadowSensor
import org.robolectric.util.ReflectionHelpers
import org.robolectric.util.ReflectionHelpers.ClassParameter
import kotlin.math.cos
import kotlin.math.hypot
import kotlin.math.sin

/**
 * getVideoMatrix (lookahead video path) must mirror the causal path's geometry:
 *  - corrections scale with the camera zoom ratio (the stream is zoom-cropped)
 *  - the OIS-split fast reference is evaluated AT the frame timestamp, not "now".
 *    Frames render ~400ms after capture; using the causal smoothFastQuat mixed
 *    a 400ms-newer orientation into the correction, injecting garbage rotation.
 */
@RunWith(RobolectricTestRunner::class)
class GyroVideoMatrixTest {

    private val sensor: Sensor = ShadowSensor.newInstance(Sensor.TYPE_GAME_ROTATION_VECTOR)

    private fun event(x: Float, y: Float, z: Float, w: Float, tNs: Long): SensorEvent {
        val e = ReflectionHelpers.callConstructor(
            SensorEvent::class.java,
            ClassParameter.from(Int::class.javaPrimitiveType, 4)
        )
        e.values[0] = x; e.values[1] = y; e.values[2] = z; e.values[3] = w
        e.sensor = sensor
        e.timestamp = tNs
        return e
    }

    private fun feedIdentity(stab: GyroStabilizer, fromNs: Long, toNs: Long, stepNs: Long): Long {
        var t = fromNs
        while (t <= toNs) {
            stab.onSensorChanged(event(0f, 0f, 0f, 1f, t))
            t += stepNs
        }
        return t
    }

    private fun feedTilt(stab: GyroStabilizer, halfAngle: Float, fromNs: Long, toNs: Long, stepNs: Long) {
        var t = fromNs
        while (t <= toNs) {
            stab.onSensorChanged(event(sin(halfAngle), 0f, 0f, cos(halfAngle), t))
            t += stepNs
        }
    }

    /** History: identity then sustained tilt; frame at the end. Returns video matrix. */
    private fun videoMatrix(zoom: Float, halfAngle: Float): FloatArray {
        val stab = GyroStabilizer(RuntimeEnvironment.getApplication())
        stab.adaptiveSmoothing = false
        stab.setZoomImmediate(zoom)
        val step = 5_000_000L
        feedIdentity(stab, 1_000_000_000L, 1_100_000_000L, step)
        feedTilt(stab, halfAngle, 1_105_000_000L, 1_200_000_000L, step)
        return stab.getVideoMatrix(1_200_000_000L)
    }

    @Test
    fun `video correction translation scales with camera zoom`() {
        fun corr(zoom: Float): Double {
            val m = videoMatrix(zoom, halfAngle = 0.01f)
            val base = videoMatrix(zoom, halfAngle = 0f)
            return hypot((m[6] - base[6]).toDouble(), (m[7] - base[7]).toDouble())
        }
        val t1 = corr(1f)
        val t3 = corr(3f)
        assertTrue("expected nonzero video correction at 1x, got $t1", t1 > 1e-5)
        assertEquals("video correction must scale ~3x at 3x zoom", 3.0, t3 / t1, 0.15)
    }

    @Test
    fun `fast reference ignores rotation that happens after the frame`() {
        // OIS split active. The device is still through the frame epoch, then a
        // quick rotation happens 550ms AFTER the frame (i.e. between capture and
        // lookahead render). The correction for the frame must not contain it.
        val stab = GyroStabilizer(RuntimeEnvironment.getApplication())
        stab.adaptiveSmoothing = false
        stab.oisCompensation = 0.4
        val step = 5_000_000L
        feedIdentity(stab, 1_000_000_000L, 1_750_000_000L, step)
        feedTilt(stab, halfAngle = 0.15f, fromNs = 1_755_000_000L, toNs = 1_800_000_000L, stepNs = step)

        val frameTs = 1_200_000_000L
        val m = stab.getVideoMatrix(frameTs)

        // Baseline: identical timeline with no rotation at all → pure static crop.
        val stabBase = GyroStabilizer(RuntimeEnvironment.getApplication())
        stabBase.adaptiveSmoothing = false
        stabBase.oisCompensation = 0.4
        feedIdentity(stabBase, 1_000_000_000L, 1_800_000_000L, step)
        val base = stabBase.getVideoMatrix(frameTs)

        val corr = hypot((m[6] - base[6]).toDouble(), (m[7] - base[7]).toDouble())
        // Old behavior: corrRef = causal smoothFastQuat ≈ 0.3 rad rotated → corr ≈ 0.15 UV.
        // New behavior: zero-phase reference at the frame time ≈ identity → corr ≈ 0.
        assertTrue("correction for a still frame must not include future rotation (got $corr UV)", corr < 0.03)
    }

    /** Translation samples: cumX oscillates at 5Hz, cumY constant. Quats stay identity. */
    private fun translationOffsets(cumXAt: (Double) -> Double): Pair<Float, Float> {
        val stab = GyroStabilizer(RuntimeEnvironment.getApplication())
        stab.adaptiveSmoothing = false
        stab.translationCorrectionEnabled = true  // apply is off by default (unvalidated sensor)
        feedIdentity(stab, 1_000_000_000L, 1_800_000_000L, 5_000_000L)
        var t = 1_000_000_000L
        while (t <= 1_800_000_000L) {
            val sec = (t - 1_000_000_000L) / 1e9
            stab.recordTranslationSample(t, cumXAt(sec), 0.0)
            t += 25_000_000L  // 40fps so a sample lands exactly on the frame timestamp
        }
        val m = stab.getVideoMatrix(1_450_000_000L)
        val base = GyroStabilizer(RuntimeEnvironment.getApplication())
        base.adaptiveSmoothing = false
        feedIdentity(base, 1_000_000_000L, 1_800_000_000L, 5_000_000L)
        val b = base.getVideoMatrix(1_450_000_000L)
        return (m[6] - b[6]) to (m[7] - b[7])
    }

    @Test
    fun `video translation correction cancels oscillation at the frame time`() {
        // 5Hz hand-shake translation, frame at a positive peak (t=0.45s → sin=1).
        val amp = 0.02
        val (dx, _) = translationOffsets { sec -> amp * kotlin.math.sin(2 * Math.PI * 5.0 * sec) }
        // Zero-phase mean over the symmetric window ≈ 0 → correction ≈ +amp on m6.
        assertEquals("translation correction must offset the frame by ~the shake amplitude",
            amp, dx.toDouble(), 0.005)
    }

    @Test
    fun `video translation correction rejects constant offset`() {
        val (dx, dy) = translationOffsets { 0.015 }
        assertEquals(0.0, dx.toDouble(), 1e-4)
        assertEquals(0.0, dy.toDouble(), 1e-4)
    }
}
