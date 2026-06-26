package com.haptictrack.tracking

/**
 * One-shot edge detector for reacquisition give-up.
 *
 * Fires exactly once when a lock has timed out with no live object, and re-arms when a live
 * lock returns — so the owner can reset to IDLE (and re-arm the sentry) without a dead lock
 * pinning the status in LOST forever, and without re-firing every subsequent frame.
 *
 * Extracted from [ObjectTracker] purely so this trigger logic is unit-testable (the tracker
 * itself loads GPU models on construction and can't be instantiated in a unit test).
 */
class GiveUpLatch {
    private var fired = false

    /**
     * @param hasLock     whether there is a live locked object this frame
     * @param hasTimedOut whether reacquisition has exceeded maxFramesLost
     * @return true exactly once per give-up episode (re-armed by a subsequent live lock)
     */
    fun update(hasLock: Boolean, hasTimedOut: Boolean): Boolean {
        if (hasLock) { fired = false; return false }
        if (hasTimedOut && !fired) { fired = true; return true }
        return false
    }
}
