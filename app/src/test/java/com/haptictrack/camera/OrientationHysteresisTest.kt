package com.haptictrack.camera

import org.junit.Assert.*
import org.junit.Test

class OrientationHysteresisTest {

    @Test
    fun `stays at 0 when near boundary`() {
        // At 50° — past the 45° midpoint but within 45+20=65° sticky range
        assertEquals(0, snapWithHysteresis(50, current = 0))
        assertEquals(0, snapWithHysteresis(60, current = 0))
        assertEquals(0, snapWithHysteresis(64, current = 0))
    }

    @Test
    fun `switches from 0 to 90 when well past boundary`() {
        // At 70° — past the 65° sticky range, into the 90° entry zone
        assertEquals(90, snapWithHysteresis(70, current = 0))
        assertEquals(90, snapWithHysteresis(90, current = 0))
    }

    @Test
    fun `stays at 90 when near boundary with 0`() {
        // At 30° — below 45° midpoint but within 45-20=25° sticky range
        assertEquals(90, snapWithHysteresis(30, current = 90))
        assertEquals(90, snapWithHysteresis(26, current = 90))
    }

    @Test
    fun `switches from 90 to 0 when well past boundary`() {
        // At 20° — below the 25° sticky range
        assertEquals(0, snapWithHysteresis(20, current = 90))
        assertEquals(0, snapWithHysteresis(10, current = 90))
    }

    @Test
    fun `stays at 0 when near 270 boundary`() {
        // At 300° — within 315-20=295° sticky range
        assertEquals(0, snapWithHysteresis(300, current = 0))
        assertEquals(0, snapWithHysteresis(296, current = 0))
    }

    @Test
    fun `switches from 0 to 270 when well past boundary`() {
        assertEquals(270, snapWithHysteresis(290, current = 0))
    }

    @Test
    fun `dead zone keeps current state`() {
        // Exactly at the boundary region between sticky zones
        // At 65° — right at the edge, still sticky for 0
        assertEquals(0, snapWithHysteresis(65, current = 0))
        // At 25° — right at the edge, still sticky for 90
        assertEquals(90, snapWithHysteresis(25, current = 90))
    }

    @Test
    fun `180 stays when near boundaries`() {
        // 180° sticky range: 135-20=115 to 225+20=245
        assertEquals(180, snapWithHysteresis(120, current = 180))
        assertEquals(180, snapWithHysteresis(240, current = 180))
    }

    @Test
    fun `180 switches when far past boundaries`() {
        assertEquals(90, snapWithHysteresis(110, current = 180))
        assertEquals(270, snapWithHysteresis(250, current = 180))
    }

    @Test
    fun `270 stays when near boundaries`() {
        // 270° sticky range: 225-20=205 to 315+20=335
        assertEquals(270, snapWithHysteresis(210, current = 270))
        assertEquals(270, snapWithHysteresis(330, current = 270))
    }

    @Test
    fun `exact cardinal directions are stable`() {
        assertEquals(0, snapWithHysteresis(0, current = 0))
        assertEquals(90, snapWithHysteresis(90, current = 90))
        assertEquals(180, snapWithHysteresis(180, current = 180))
        assertEquals(270, snapWithHysteresis(270, current = 270))
    }

    @Test
    fun `rapid oscillation around 45 does not flicker`() {
        // Simulate accelerometer jitter around 45°
        var current = 0
        val readings = listOf(44, 46, 43, 47, 44, 48, 42, 46, 45, 44)
        for (reading in readings) {
            current = snapWithHysteresis(reading, current)
        }
        // Should stay at 0 the whole time — none of these reach 65°
        assertEquals(0, current)
    }

    @Test
    fun `deliberate rotation transitions correctly`() {
        // Simulate a smooth rotation from 0° to 90°
        var current = 0
        val readings = listOf(10, 20, 30, 40, 50, 60, 70, 80, 90)
        for (reading in readings) {
            current = snapWithHysteresis(reading, current)
        }
        assertEquals(90, current)
    }
}
