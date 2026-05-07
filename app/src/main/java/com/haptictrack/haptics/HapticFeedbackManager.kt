package com.haptictrack.haptics

import android.content.Context
import android.os.VibrationEffect
import android.os.Vibrator
import android.os.VibratorManager
import com.haptictrack.tracking.TrackingStatus
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.Job
import kotlinx.coroutines.delay
import kotlinx.coroutines.launch
import kotlin.math.abs

class HapticFeedbackManager(context: Context) {

    private val vibrator: Vibrator = run {
        val manager = context.getSystemService(Context.VIBRATOR_MANAGER_SERVICE) as VibratorManager
        manager.defaultVibrator
    }

    private val scope = CoroutineScope(Dispatchers.Default)
    private var heartbeatJob: Job? = null

    @Volatile private var currentStatus: TrackingStatus = TrackingStatus.IDLE
    @Volatile private var driftX: Float = 0f
    @Volatile private var driftY: Float = 0f

    fun updateTrackingStatus(status: TrackingStatus, driftX: Float = 0f, driftY: Float = 0f) {
        this.driftX = driftX
        this.driftY = driftY

        if (status == currentStatus) return
        currentStatus = status

        heartbeatJob?.cancel()
        vibrator.cancel()

        when (status) {
            TrackingStatus.LOCKED -> heartbeatJob = scope.launch { lockedHeartbeat() }
            TrackingStatus.LOST -> heartbeatJob = scope.launch { lostHeartbeat() }
            TrackingStatus.SEARCHING -> heartbeatJob = scope.launch { lostHeartbeat() }
            TrackingStatus.IDLE -> {}
        }
    }

    private suspend fun lockedHeartbeat() {
        while (true) {
            val dx = abs(driftX)
            val dy = abs(driftY)
            val edgeProximity = maxOf(dx, dy).coerceIn(0f, 1f)
            val horizontalDominant = dx >= dy

            // Remap: 0.05 dead zone, full urgency by 0.5 (zoomed subjects fill frame)
            val urgency = ((edgeProximity - 0.05f) / 0.45f).coerceIn(0f, 1f)

            val intervalMs = lerp(900f, 180f, urgency).toLong()
            val amplitude = lerp(20f, 180f, urgency).toInt().coerceIn(1, 255)
            val pulseDuration = lerp(35f, 70f, urgency).toLong()

            if (!horizontalDominant && urgency > 0.05f) {
                // Double pulse — vertical drift dominates
                val tapAmp = (amplitude * 0.7f).toInt().coerceIn(1, 255)
                vibrator.vibrate(VibrationEffect.createOneShot(pulseDuration, tapAmp))
                delay(pulseDuration + 60)
                vibrator.vibrate(VibrationEffect.createOneShot(pulseDuration, amplitude))
                delay(intervalMs)
            } else {
                // Single pulse — centered or horizontal drift
                vibrator.vibrate(VibrationEffect.createOneShot(pulseDuration, amplitude))
                delay(intervalMs)
            }
        }
    }

    private suspend fun lostHeartbeat() {
        while (true) {
            // Distinct double-tap at slow rate
            vibrator.vibrate(VibrationEffect.createOneShot(30, 50))
            delay(100)
            vibrator.vibrate(VibrationEffect.createOneShot(30, 50))
            delay(1400)
        }
    }

    private fun lerp(a: Float, b: Float, t: Float): Float = a + (b - a) * t

    fun shutdown() {
        heartbeatJob?.cancel()
        vibrator.cancel()
    }
}
