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
import kotlin.math.atan2
import kotlin.math.cos
import kotlin.math.sin

/**
 * Horizon lock: roll-only stabilization toward gravity. OIS is lens-shift and
 * physically cannot correct roll; the walking capture (20260610_195144)
 * measured ±5° roll drift that reaches the screen at ~100% survival.
 */
@RunWith(RobolectricTestRunner::class)
class HorizonLockTest {

    private val sensor: Sensor = ShadowSensor.newInstance(Sensor.TYPE_GAME_ROTATION_VECTOR)

    // Device-to-world quat for: portrait upright (pitch 90° about world X),
    // then rolled by [rollDeg] about the device z (optical) axis.
    private fun portraitQuat(rollDeg: Double): DoubleArray {
        val p = Math.toRadians(90.0) / 2
        val r = Math.toRadians(rollDeg) / 2
        // q = q_pitchX * q_rollZ  (device-frame roll post-multiplies)
        val pw = cos(p); val px = sin(p)
        val rw = cos(r); val rz = sin(r)
        return doubleArrayOf(
            pw * rw,            // w
            px * rw,            // x
            -px * rz,           // y  (Hamilton product cross term)
            pw * rz             // z
        )
    }

    private fun event(q: DoubleArray, tNs: Long): SensorEvent {
        val e = ReflectionHelpers.callConstructor(
            SensorEvent::class.java,
            ClassParameter.from(Int::class.javaPrimitiveType, 4)
        )
        // sensor value layout: x, y, z, w
        e.values[0] = q[1].toFloat(); e.values[1] = q[2].toFloat()
        e.values[2] = q[3].toFloat(); e.values[3] = q[0].toFloat()
        e.sensor = sensor
        e.timestamp = tNs
        return e
    }

    @Test
    fun `gravity roll extracts the device roll in portrait`() {
        for (deg in listOf(0.0, 5.0, -8.0, 20.0)) {
            val q = portraitQuat(deg)
            val roll = GyroStabilizer.gravityRollDeg(
                GyroStabilizer.Quat(q[0], q[1], q[2], q[3]))
            assertEquals("roll for $deg", deg, roll, 0.01)
        }
    }

    @Test
    fun `lock target clamps fades and releases`() {
        assertEquals(-5.0, GyroStabilizer.lockTargetDeg(5.0), 1e-9)     // in range: full counter
        assertEquals(8.0, GyroStabilizer.lockTargetDeg(-8.0), 1e-9)
        assertEquals(-10.0, GyroStabilizer.lockTargetDeg(12.0), 1e-9)   // clamped, below fade start
        assertEquals(-5.0, GyroStabilizer.lockTargetDeg(20.0), 1e-9)    // fade: (25-20)/(25-15)=0.5 of clamp
        assertEquals(0.0, GyroStabilizer.lockTargetDeg(30.0), 1e-9)     // released: intentional tilt
        assertEquals(0.0, GyroStabilizer.lockTargetDeg(90.0), 1e-9)     // landscape: free
    }

    private fun feed(stab: GyroStabilizer, rollDeg: Double, fromNs: Long, toNs: Long) {
        var t = fromNs
        while (t <= toNs) {
            stab.onSensorChanged(event(portraitQuat(rollDeg), t))
            t += 5_000_000L
        }
    }

    /** Rotation angle (deg) of the applied matrix, aspect-corrected back to pixel space. */
    private fun matrixRollDeg(m: FloatArray): Double {
        // column-major; in-UV entries m[0]=a, m[1]=d, m[3]=b, m[4]=e with
        // d = iz*sin*aspectInv... recover angle from pixel-space rotation:
        val a = m[0].toDouble()
        val d = m[1].toDouble() * GyroStabilizer.LOCK_STREAM_ASPECT
        return Math.toDegrees(atan2(d, a))
    }

    @Test
    fun `lock counter-rotates by the device roll with EIS off`() {
        val stab = GyroStabilizer(RuntimeEnvironment.getApplication())
        stab.enabled = false
        stab.horizonLockEnabled = true
        feed(stab, rollDeg = 6.0, fromNs = 1_000_000_000L, toNs = 2_000_000_000L)
        val m = stab.getMatrix()
        assertEquals("counter-rotation must equal -roll", -6.0, matrixRollDeg(m), 0.3)
        // crop applied: uniform scale 1/1.15 in pixel space
        val scale = Math.hypot(m[0].toDouble(), m[1].toDouble() * GyroStabilizer.LOCK_STREAM_ASPECT)
        assertEquals(1.0 / 1.15, scale, 0.01)
    }

    @Test
    fun `lock releases on intentional tilt but keeps the crop stable`() {
        val stab = GyroStabilizer(RuntimeEnvironment.getApplication())
        stab.enabled = false
        stab.horizonLockEnabled = true
        feed(stab, rollDeg = 40.0, fromNs = 1_000_000_000L, toNs = 2_000_000_000L)
        val m = stab.getMatrix()
        assertEquals("released beyond 25 deg", 0.0, matrixRollDeg(m), 0.3)
        val scale = Math.hypot(m[0].toDouble(), m[1].toDouble() * GyroStabilizer.LOCK_STREAM_ASPECT)
        assertEquals("crop must stay constant while feature is on", 1.0 / 1.15, scale, 0.01)
    }

    @Test
    fun `lock off and EIS off yields identity`() {
        val stab = GyroStabilizer(RuntimeEnvironment.getApplication())
        stab.enabled = false
        feed(stab, rollDeg = 6.0, fromNs = 1_000_000_000L, toNs = 1_200_000_000L)
        val m = stab.getMatrix()
        assertEquals(1f, m[0]); assertEquals(1f, m[4]); assertEquals(0f, m[6]); assertEquals(0f, m[7])
    }

    @Test
    fun `video path locks at the frame timestamp`() {
        val stab = GyroStabilizer(RuntimeEnvironment.getApplication())
        stab.enabled = false
        stab.horizonLockEnabled = true
        feed(stab, rollDeg = 6.0, fromNs = 1_000_000_000L, toNs = 2_000_000_000L)
        val m = stab.getVideoMatrix(1_500_000_000L)
        assertEquals(-6.0, matrixRollDeg(m), 0.3)
    }

    @Test
    fun `applied correction slews instead of jumping`() {
        val stab = GyroStabilizer(RuntimeEnvironment.getApplication())
        stab.enabled = false
        stab.horizonLockEnabled = true
        feed(stab, rollDeg = 0.0, fromNs = 1_000_000_000L, toNs = 1_500_000_000L)
        // roll steps to 8 deg; after only 30ms the applied correction must be
        // mid-slew (rate-limited), not snapped to -8.
        feed(stab, rollDeg = 8.0, fromNs = 1_505_000_000L, toNs = 1_530_000_000L)
        val mid = matrixRollDeg(stab.getMatrix())
        assertTrue("expected partial slew, got $mid", mid < -0.5 && mid > -7.5)
        feed(stab, rollDeg = 8.0, fromNs = 1_535_000_000L, toNs = 3_000_000_000L)
        assertEquals(-8.0, matrixRollDeg(stab.getMatrix()), 0.3)
    }
}
