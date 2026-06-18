package com.haptictrack.camera

import org.junit.Assert.*
import org.junit.Test
import org.junit.runner.RunWith
import org.robolectric.RobolectricTestRunner
import org.robolectric.RuntimeEnvironment

/**
 * Zoom interpolation must run independently of the EIS enabled state (#174).
 * Before the fix, zoom dispatch lived below the `if (!enabled) return` in
 * onSensorChanged — auto-zoom targets were silently dropped with EIS off.
 */
@RunWith(RobolectricTestRunner::class)
class GyroZoomDispatchTest {

    private fun stabilizer(enabled: Boolean): GyroStabilizer {
        val stab = GyroStabilizer(RuntimeEnvironment.getApplication())
        stab.enabled = enabled
        return stab
    }

    @Test
    fun `zoom dispatches when EIS is disabled`() {
        val stab = stabilizer(enabled = false)
        var dispatched = -1f
        stab.onZoomApply = { dispatched = it }
        stab.setZoomTarget(5f)

        stab.interpolateZoom(1_000_000_000L)  // first sample — records timestamp only
        stab.interpolateZoom(1_020_000_000L)  // 20ms later — interpolates + dispatches

        assertTrue("expected a zoom dispatch, got none", dispatched > 0f)
        assertTrue("zoom should ramp toward target, got $dispatched", dispatched > 1f && dispatched < 5f)
    }

    @Test
    fun `zoom converges to target over repeated samples`() {
        val stab = stabilizer(enabled = false)
        var dispatched = -1f
        stab.onZoomApply = { dispatched = it }
        stab.setZoomTarget(4f)

        var t = 1_000_000_000L
        repeat(200) {  // 200 samples at 5ms = 1s of sensor time
            stab.interpolateZoom(t)
            t += 5_000_000L
        }
        assertEquals(4f, dispatched, 0.05f)
    }

    @Test
    fun `long sensor gap snaps zoom without interpolating stale dt`() {
        val stab = stabilizer(enabled = false)
        var dispatched = -1f
        stab.onZoomApply = { dispatched = it }
        stab.setZoomTarget(3f)

        stab.interpolateZoom(1_000_000_000L)
        stab.interpolateZoom(2_000_000_000L)  // 1s gap > threshold — snap, no dispatch

        assertEquals(-1f, dispatched, 0f)     // gap path doesn't dispatch
        stab.interpolateZoom(2_020_000_000L)  // next regular sample dispatches the snapped value
        assertEquals(3f, dispatched, 0.01f)
    }

    @Test
    fun `dispatch is throttled below 16ms`() {
        val stab = stabilizer(enabled = false)
        var dispatchCount = 0
        stab.onZoomApply = { dispatchCount++ }
        stab.setZoomTarget(2f)

        stab.interpolateZoom(1_000_000_000L)
        stab.interpolateZoom(1_020_000_000L)  // dispatch #1
        stab.interpolateZoom(1_025_000_000L)  // 5ms later — throttled
        stab.interpolateZoom(1_030_000_000L)  // 10ms later — throttled

        assertEquals(1, dispatchCount)
    }
}
