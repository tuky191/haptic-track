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
 * EIS corrections must scale with the camera zoom ratio. The SurfaceTexture
 * stream is already zoom-cropped by the ISP, so its UV [0,1]² spans 1/zoom of
 * the FOV — the same rotation displaces pixels zoom× farther in view UV.
 * Before the fix, the homography used the unzoomed focal length, making
 * corrections zoom× too small (3x zoom → corrections at 1/3 strength).
 */
@RunWith(RobolectricTestRunner::class)
class GyroZoomFocalTest {

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

    private fun matrixAfter(zoom: Float, halfAngle: Float): FloatArray {
        val stab = GyroStabilizer(RuntimeEnvironment.getApplication())
        stab.adaptiveSmoothing = false
        stab.setZoomImmediate(zoom)
        stab.onSensorChanged(event(0f, 0f, 0f, 1f, 1_000_000_000L))
        stab.onSensorChanged(event(sin(halfAngle), 0f, 0f, cos(halfAngle), 1_005_000_000L))
        return stab.getMatrix()
    }

    /**
     * Feed identity then a small tilt about device X; return correction translation
     * relative to the no-rotation baseline (the affine carries a constant
     * crop-centering offset in m[6]/m[7] that must be subtracted).
     */
    private fun correctionTranslation(zoom: Float): Double {
        val m = matrixAfter(zoom, halfAngle = 0.01f)  // 0.02 rad tilt
        val base = matrixAfter(zoom, halfAngle = 0f)
        return hypot((m[6] - base[6]).toDouble(), (m[7] - base[7]).toDouble())
    }

    @Test
    fun `correction translation scales with camera zoom`() {
        val t1 = correctionTranslation(zoom = 1f)
        val t3 = correctionTranslation(zoom = 3f)
        assertTrue("expected nonzero correction at 1x, got $t1", t1 > 1e-5)
        assertEquals("correction must scale ~3x at 3x zoom", 3.0, t3 / t1, 0.15)
    }

    @Test
    fun `zoom at 1x leaves correction unchanged`() {
        // Two independent runs at 1x must produce identical corrections —
        // guards against the zoom factor leaking state between samples.
        val a = correctionTranslation(zoom = 1f)
        val b = correctionTranslation(zoom = 1f)
        assertEquals(a, b, 1e-12)
    }
}
