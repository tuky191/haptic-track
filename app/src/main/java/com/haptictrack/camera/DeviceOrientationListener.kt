package com.haptictrack.camera

import android.content.Context
import android.view.OrientationEventListener
import android.view.Surface

/**
 * Detects the physical orientation of the device using the accelerometer.
 *
 * Since the activity is locked to portrait, CameraX doesn't know when
 * the phone is held upside down. This listener provides the actual
 * device rotation so we can correct the camera image before detection.
 */
class DeviceOrientationListener(context: Context) {

    companion object {
        /**
         * Hysteresis margin in degrees. To leave the current orientation, the
         * sensor must read past the midpoint (45°) by this many degrees.
         * Creates a dead zone of 2×HYSTERESIS degrees at each boundary.
         */
        const val HYSTERESIS = 20
    }

    /** Current device rotation in degrees: 0, 90, 180, or 270. */
    @Volatile
    var deviceRotation: Int = 0
        private set

    /** Surface rotation constant matching the device orientation. */
    val surfaceRotation: Int
        get() = when (deviceRotation) {
            0 -> Surface.ROTATION_0
            90 -> Surface.ROTATION_90
            180 -> Surface.ROTATION_180
            270 -> Surface.ROTATION_270
            else -> Surface.ROTATION_0
        }

    private val listener = object : OrientationEventListener(context) {
        override fun onOrientationChanged(orientation: Int) {
            if (orientation == ORIENTATION_UNKNOWN) return
            deviceRotation = snapWithHysteresis(orientation, deviceRotation)
        }
    }

    fun start() {
        if (listener.canDetectOrientation()) {
            listener.enable()
        }
    }

    fun stop() {
        listener.disable()
    }
}

/**
 * Snap raw orientation to 0/90/180/270 with directional hysteresis.
 *
 * Once in a state, the sensor must move [DeviceOrientationListener.HYSTERESIS]
 * degrees past the midpoint before switching. This prevents flickering when
 * the phone is held near a boundary (e.g. ~45°).
 *
 * Package-level for testability.
 */
internal fun snapWithHysteresis(orientation: Int, current: Int): Int {
    val h = DeviceOrientationListener.HYSTERESIS
    // Check if we're still within the sticky range of the current orientation.
    // The sticky range is wider than the entry range by HYSTERESIS on each side.
    val inCurrent = when (current) {
        0   -> orientation in 0..(45 + h) || orientation in (315 - h)..359
        90  -> orientation in (45 - h)..(135 + h)
        180 -> orientation in (135 - h)..(225 + h)
        270 -> orientation in (225 - h)..(315 + h)
        else -> false
    }
    if (inCurrent) return current

    // Outside the sticky range — commit to whichever quadrant center is nearest
    return when (orientation) {
        in 0..(45 - h), in (315 + h)..359 -> 0
        in (45 + h)..(135 - h) -> 90
        in (135 + h)..(225 - h) -> 180
        in (225 + h)..(315 - h) -> 270
        else -> current  // in dead zone between quadrants, keep current
    }
}
