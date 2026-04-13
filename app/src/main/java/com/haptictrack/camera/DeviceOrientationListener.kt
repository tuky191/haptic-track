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

            // Snap to nearest 90° bucket with hysteresis
            deviceRotation = when (orientation) {
                in 0..44, in 316..359 -> 0
                in 45..134 -> 90
                in 135..224 -> 180
                in 225..315 -> 270
                else -> 0
            }
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
