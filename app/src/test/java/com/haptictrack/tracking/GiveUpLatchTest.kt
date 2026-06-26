package com.haptictrack.tracking

import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test

class GiveUpLatchTest {

    @Test
    fun `does not fire while locked`() {
        val latch = GiveUpLatch()
        assertFalse(latch.update(hasLock = true, hasTimedOut = false))
        assertFalse(latch.update(hasLock = true, hasTimedOut = false))
    }

    @Test
    fun `does not fire while searching but not yet timed out`() {
        val latch = GiveUpLatch()
        repeat(5) { assertFalse(latch.update(hasLock = false, hasTimedOut = false)) }
    }

    @Test
    fun `fires exactly once on timeout`() {
        val latch = GiveUpLatch()
        assertFalse(latch.update(hasLock = true, hasTimedOut = false))   // locked
        assertFalse(latch.update(hasLock = false, hasTimedOut = false))  // searching
        assertTrue(latch.update(hasLock = false, hasTimedOut = true))    // give up — fire
        // subsequent timed-out frames must NOT re-fire
        repeat(10) { assertFalse(latch.update(hasLock = false, hasTimedOut = true)) }
    }

    @Test
    fun `re-arms after a new lock and fires again on the next timeout`() {
        val latch = GiveUpLatch()
        assertTrue(latch.update(hasLock = false, hasTimedOut = true))    // first give-up
        assertFalse(latch.update(hasLock = false, hasTimedOut = true))   // still given up
        assertFalse(latch.update(hasLock = true, hasTimedOut = false))   // re-locked -> re-arm
        assertFalse(latch.update(hasLock = false, hasTimedOut = false))  // searching again
        assertTrue(latch.update(hasLock = false, hasTimedOut = true))    // second give-up fires
    }
}
