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

        // Geiger: fastest at center, slowest at edge
        private const val INTERVAL_CENTER_MS = 120f
        private const val INTERVAL_EDGE_MS = 800f
        private const val PULSE_MS = 25L

        private const val DOUBLE_PULSE_GAP_MS = 45L
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

    /** Vibration amplitude 0.0–1.0. Controls how strong each click feels, not frequency. */
    @Volatile var strength: Float = 0.5f

    @Synchronized
    fun updateTrackingStatus(status: TrackingStatus, driftX: Float = 0f, driftY: Float = 0f) {
        this.driftX = driftX
        this.driftY = driftY

        if (status == currentStatus) return
        currentStatus = status

        heartbeatJob?.cancel()
        vibrator.cancel()

        when (status) {
            TrackingStatus.LOCKED -> heartbeatJob = scope.launch { geigerLoop() }
            TrackingStatus.LOST -> {}
            TrackingStatus.SEARCHING -> {}
            TrackingStatus.IDLE -> {}
        }
    }

    private suspend fun CoroutineScope.geigerLoop() {
        while (isActive) {
            val dx = abs(driftX)
            val dy = abs(driftY)
            val edgeProximity = maxOf(dx, dy).coerceIn(0f, 1f)
            val horizontalDominant = dx >= dy

            // 1.0 at center, 0.0 at edge
            val centering = 1f - ((edgeProximity - DEAD_ZONE) / URGENCY_RANGE).coerceIn(0f, 1f)

            // Geiger: fast clicks when centered, slow when drifting
            val intervalMs = lerp(INTERVAL_EDGE_MS, INTERVAL_CENTER_MS, centering).toLong()
            val amp = (strength * 255f).toInt().coerceIn(1, 255)

            if (!horizontalDominant && centering < 0.95f) {
                // Double click — vertical drift dominates
                vibrator.vibrate(VibrationEffect.createOneShot(PULSE_MS, (amp * 0.7f).toInt().coerceIn(1, 255)))
                delay(PULSE_MS + DOUBLE_PULSE_GAP_MS)
                vibrator.vibrate(VibrationEffect.createOneShot(PULSE_MS, amp))
                delay(maxOf(intervalMs - PULSE_MS - DOUBLE_PULSE_GAP_MS, 30L))
            } else {
                // Single click — centered or horizontal drift
                vibrator.vibrate(VibrationEffect.createOneShot(PULSE_MS, amp))
                delay(intervalMs)
            }
        }
    }

    private fun lerp(a: Float, b: Float, t: Float): Float = a + (b - a) * t

    fun shutdown() {
        scope.cancel()
        vibrator.cancel()
    }
}
