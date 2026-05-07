package com.haptictrack.haptics

import android.content.Context
import android.os.VibrationEffect
import android.os.Vibrator
import android.os.VibratorManager
import com.haptictrack.tracking.TrackingStatus
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.Job
import kotlinx.coroutines.SupervisorJob
import kotlinx.coroutines.cancel
import kotlinx.coroutines.delay
import kotlinx.coroutines.isActive
import kotlinx.coroutines.launch
import kotlin.math.abs

class HapticFeedbackManager(context: Context) {

    companion object {
        private const val DEAD_ZONE = 0.05f
        private const val URGENCY_RANGE = 0.45f
        private const val INTERVAL_MIN_MS = 180f
        private const val INTERVAL_MAX_MS = 900f
        private const val AMP_MIN = 20f
        private const val AMP_MAX = 180f
        private const val PULSE_MIN_MS = 35f
        private const val PULSE_MAX_MS = 70f
        private const val DOUBLE_PULSE_GAP_MS = 50L
    }

    private val vibrator: Vibrator = run {
        val manager = context.getSystemService(Context.VIBRATOR_MANAGER_SERVICE) as VibratorManager
        manager.defaultVibrator
    }

    private val scope = CoroutineScope(Dispatchers.Default + SupervisorJob())
    private var heartbeatJob: Job? = null

    @Volatile private var currentStatus: TrackingStatus = TrackingStatus.IDLE
    @Volatile private var driftX: Float = 0f
    @Volatile private var driftY: Float = 0f

    @Synchronized
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

    private suspend fun CoroutineScope.lockedHeartbeat() {
        while (isActive) {
            val dx = abs(driftX)
            val dy = abs(driftY)
            val edgeProximity = maxOf(dx, dy).coerceIn(0f, 1f)
            val horizontalDominant = dx >= dy

            val urgency = ((edgeProximity - DEAD_ZONE) / URGENCY_RANGE).coerceIn(0f, 1f)

            val intervalMs = lerp(INTERVAL_MAX_MS, INTERVAL_MIN_MS, urgency).toLong()
            val amplitude = lerp(AMP_MIN, AMP_MAX, urgency).toInt().coerceIn(1, 255)
            val pulseDuration = lerp(PULSE_MIN_MS, PULSE_MAX_MS, urgency).toLong()

            if (!horizontalDominant && urgency > 0.05f) {
                val tapAmp = (amplitude * 0.7f).toInt().coerceIn(1, 255)
                vibrator.vibrate(VibrationEffect.createOneShot(pulseDuration, tapAmp))
                delay(pulseDuration + DOUBLE_PULSE_GAP_MS)
                vibrator.vibrate(VibrationEffect.createOneShot(pulseDuration, amplitude))
                // Subtract double-pulse overhead so total cycle matches single-pulse
                delay(maxOf(intervalMs - pulseDuration - DOUBLE_PULSE_GAP_MS, 30L))
            } else {
                vibrator.vibrate(VibrationEffect.createOneShot(pulseDuration, amplitude))
                delay(intervalMs)
            }
        }
    }

    private suspend fun CoroutineScope.lostHeartbeat() {
        while (isActive) {
            vibrator.vibrate(VibrationEffect.createOneShot(30, 50))
            delay(100)
            vibrator.vibrate(VibrationEffect.createOneShot(30, 50))
            delay(1400)
        }
    }

    private fun lerp(a: Float, b: Float, t: Float): Float = a + (b - a) * t

    fun shutdown() {
        scope.cancel()
        vibrator.cancel()
    }
}
