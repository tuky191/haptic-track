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
