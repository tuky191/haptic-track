package com.haptictrack.haptics

import android.content.Context
import android.os.VibrationEffect
import android.os.Vibrator
import android.os.VibratorManager
import com.haptictrack.tracking.TrackingStatus

class HapticFeedbackManager(context: Context) {

    private val vibrator: Vibrator = run {
        val manager = context.getSystemService(Context.VIBRATOR_MANAGER_SERVICE) as VibratorManager
        manager.defaultVibrator
    }

    private var currentStatus: TrackingStatus? = null

    fun updateTrackingStatus(status: TrackingStatus, edgeProximity: Float = 0f) {
        if (status == currentStatus && status != TrackingStatus.LOCKED) return
        currentStatus = status

        when (status) {
            TrackingStatus.LOCKED -> vibrateLockedPulse(edgeProximity)
            TrackingStatus.SEARCHING -> vibrateSearching()
            TrackingStatus.LOST -> stopVibration()
            TrackingStatus.IDLE -> stopVibration()
        }
    }

    private fun vibrateLockedPulse(edgeProximity: Float) {
        // edgeProximity: 0.0 = centered, 1.0 = at edge of frame
        val amplitude = (60 + (edgeProximity * 195)).toInt().coerceIn(1, 255)
        val effect = VibrationEffect.createOneShot(100, amplitude)
        vibrator.vibrate(effect)
    }

    private fun vibrateSearching() {
        val timings = longArrayOf(0, 50, 100, 50, 100, 50)
        val amplitudes = intArrayOf(0, 80, 0, 80, 0, 80)
        val effect = VibrationEffect.createWaveform(timings, amplitudes, -1)
        vibrator.vibrate(effect)
    }

    private fun stopVibration() {
        vibrator.cancel()
    }

    fun shutdown() {
        vibrator.cancel()
    }
}
