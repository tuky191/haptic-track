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
 * The crop margin grants a fixed correction budget in view UV; at zoom Z the
 * angular budget shrinks by 1/Z while a causal smoother demands deviation
 * ~ angular-rate x TC. Session 20260610_175612 ran with mean correction (2.15 deg)
 * riding the budget ceiling (2.66 deg at zoom 2.72) and the leash clamping 26% of
 * frames — smoothing demanded ~5x more than physics allowed. The effective TC must
 * be capped so the demanded deviation fits the budget WITHOUT the leash chopping it.
 */
@RunWith(RobolectricTestRunner::class)
class GyroBudgetTcTest {

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

    @Test
    fun `correction stays within crop margin at zoom without the leash`() {
        // 20 deg/s sustained rotation at 3x zoom for 1s, leash OFF: only the
        // budget-aware TC can keep the correction inside the crop margin.
        val stab = GyroStabilizer(RuntimeEnvironment.getApplication()).apply { enabled = true }
        stab.adaptiveSmoothing = false
        stab.leashEnabled = false
        stab.setZoomImmediate(3f)

        var t = 1_000_000_000L
        var angleDeg = 0.0
        while (t <= 2_000_000_000L) {
            val half = Math.toRadians(angleDeg / 2.0)
            stab.onSensorChanged(event(sin(half).toFloat(), 0f, 0f, cos(half).toFloat(), t))
            angleDeg += 0.1  // 0.1 deg per 5ms = 20 deg/s
            t += 5_000_000L
        }
        val m = stab.getMatrix()

        val base = GyroStabilizer(RuntimeEnvironment.getApplication()).apply { enabled = true }
        base.adaptiveSmoothing = false
        base.leashEnabled = false
        base.setZoomImmediate(3f)
        t = 1_000_000_000L
        while (t <= 2_000_000_000L) {
            base.onSensorChanged(event(0f, 0f, 0f, 1f, t))
            t += 5_000_000L
        }
        val b = base.getMatrix()

        val corr = hypot((m[6] - b[6]).toDouble(), (m[7] - b[7]).toDouble())
        val cropMargin = 0.5 * (1.0 - 1.0 / 1.3)  // 0.1154 UV at default cropZoom
        assertTrue(
            "correction $corr UV must fit the crop margin $cropMargin via TC capping (leash disabled)",
            corr < cropMargin * 1.15
        )
        assertTrue("correction should be substantial, not zero (got $corr)", corr > 0.02)
    }
}
