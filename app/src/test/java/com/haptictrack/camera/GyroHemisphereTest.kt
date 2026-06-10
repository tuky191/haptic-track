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
 * Quaternions double-cover rotations: q and −q are the same orientation, and the
 * sensor may deliver either sign. Session 20260610_173942 spent 100% of its frames
 * with dot(smoothed, raw) < 0 — the leash deviation read ~360°, fired every sample,
 * and snapped smoothed onto raw: stabilization silently off for the whole session.
 * Corrections must be identical regardless of the incoming hemisphere.
 */
@RunWith(RobolectricTestRunner::class)
class GyroHemisphereTest {

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

    /** Identity then sustained tilt, with every quat multiplied by [sign]. */
    private fun correctionTranslation(sign: Float): Double {
        fun run(halfAngle: Float): FloatArray {
            val stab = GyroStabilizer(RuntimeEnvironment.getApplication())
            stab.adaptiveSmoothing = false
            var t = 1_000_000_000L
            while (t <= 1_100_000_000L) {
                stab.onSensorChanged(event(0f, 0f, 0f, sign * 1f, t)); t += 5_000_000L
            }
            while (t <= 1_300_000_000L) {
                stab.onSensorChanged(event(sign * sin(halfAngle), 0f, 0f, sign * cos(halfAngle), t))
                t += 5_000_000L
            }
            return stab.getMatrix()
        }
        val m = run(0.01f)
        val base = run(0f)
        return hypot((m[6] - base[6]).toDouble(), (m[7] - base[7]).toDouble())
    }

    @Test
    fun `corrections are identical for both quaternion hemispheres`() {
        val pos = correctionTranslation(sign = +1f)
        val neg = correctionTranslation(sign = -1f)
        assertTrue("expected nonzero correction, got $pos", pos > 1e-5)
        assertEquals("negative-hemisphere quats must produce the same correction", pos, neg, pos * 0.02)
    }

    @Test
    fun `mid-stream hemisphere flip does not latch the leash`() {
        // Same orientation stream, but the sensor flips quat sign halfway through.
        val stab = GyroStabilizer(RuntimeEnvironment.getApplication())
        stab.adaptiveSmoothing = false
        var t = 1_000_000_000L
        while (t <= 1_100_000_000L) {
            stab.onSensorChanged(event(0f, 0f, 0f, 1f, t)); t += 5_000_000L
        }
        while (t <= 1_300_000_000L) {  // identical rotation, negated representation
            stab.onSensorChanged(event(-sin(0.01f), 0f, 0f, -cos(0.01f), t)); t += 5_000_000L
        }
        val m = stab.getMatrix()

        val ref = GyroStabilizer(RuntimeEnvironment.getApplication())
        ref.adaptiveSmoothing = false
        t = 1_000_000_000L
        while (t <= 1_100_000_000L) {
            ref.onSensorChanged(event(0f, 0f, 0f, 1f, t)); t += 5_000_000L
        }
        while (t <= 1_300_000_000L) {
            ref.onSensorChanged(event(sin(0.01f), 0f, 0f, cos(0.01f), t)); t += 5_000_000L
        }
        val r = ref.getMatrix()

        assertEquals("flip mid-stream must not change the correction (m6)", r[6], m[6], 1e-4f)
        assertEquals("flip mid-stream must not change the correction (m7)", r[7], m[7], 1e-4f)
    }
}
