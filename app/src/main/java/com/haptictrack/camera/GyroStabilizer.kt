package com.haptictrack.camera

import android.content.Context
import android.hardware.Sensor
import android.hardware.SensorEvent
import android.hardware.SensorEventListener
import android.hardware.SensorManager
import android.hardware.camera2.CameraCharacteristics
import android.hardware.camera2.CameraManager as Camera2Manager
import android.util.Log
import java.io.File
import java.io.FileWriter
import java.io.PrintWriter
import java.util.concurrent.atomic.AtomicReference
import kotlin.math.*

/**
 * Gyroscope-based software EIS that stacks on top of the ISP's hardware stabilization.
 *
 * Algorithm (adapted from Gyroflow):
 * 1. Read device orientation from TYPE_GAME_ROTATION_VECTOR (fused gyro+accel, no mag)
 * 2. Apply causal exponential SLERP smoothing to the orientation quaternion
 * 3. Compute correction rotation = raw⁻¹ × smoothed (the shake to undo)
 * 4. Convert to a 3×3 homography H = K × R × K⁻¹ in texture UV space
 * 5. The GL shader applies H to texture coordinates before sampling
 *
 * The timeConstant sets the base smoothing strength (higher = more stable,
 * lower = more responsive). Adaptive pan detection monitors angular velocity
 * and reduces the effective TC during intentional pans for faster response,
 * restoring it during shake for maximum stability.
 */
class GyroStabilizer(context: Context) : SensorEventListener {

    companion object {
        private const val TAG = "GyroStab"
        private const val DEFAULT_TIME_CONSTANT = 0.50
        private const val DEFAULT_HFOV_DEGREES = 75.0
        private const val TEL_INTERVAL = 200
        private const val SENSOR_GAP_THRESHOLD_NS = 100_000_000L

        // Adaptive smoothing: pan detection → dynamic TC
        private const val PAN_VELOCITY_THRESHOLD_DEG = 15.0
        private const val PAN_ONSET_SEC = 0.20
        private const val PAN_TC_FACTOR = 0.30
        private const val VELOCITY_SMOOTHING_TC = 0.05
        private const val TC_RAMP_SPEED = 5.0
    }

    private val sensorManager = context.getSystemService(Context.SENSOR_SERVICE) as SensorManager
    private val rotationSensor = sensorManager.getDefaultSensor(Sensor.TYPE_GAME_ROTATION_VECTOR)

    /** Smoothing time constant in seconds. Lower = more stable. */
    var timeConstant: Double = DEFAULT_TIME_CONSTANT

    /** Focal length in UV-normalized coordinates, derived from camera intrinsics. */
    private var fxUv: Double = hfovToFocalUv(DEFAULT_HFOV_DEGREES)
    private var fyUv: Double = fxUv

    /** Quaternion that rotates correction from device space to camera sensor space. */
    private var deviceToSensorQuat = sensorOrientationToQuat(90)

    /** Sensor orientation in degrees — needed to convert sensor-UV homography to portrait UV. */
    private var sensorOrientation: Int = 90

    /** Crop zoom applied to absorb warp margins (1.0 = no crop, 1.05 = 5% crop). */
    var cropZoom: Float = 1.15f

    /** Adaptive pan detection: reduces TC during pans, increases during shake. */
    var adaptiveSmoothing: Boolean = true

    /** OIS compensation: scales correction magnitude when optical stabilization is active.
     *  OIS handles ~20% of shake; without compensation the gyro overcorrects. */
    var oisCompensation: Double = 1.0

    /** Current stabilization matrix in column-major order for GL (mat3). Identity when disabled. */
    private val currentMatrix = AtomicReference(IDENTITY_MATRIX.clone())

    /** Whether stabilization is active. */
    @Volatile
    private var _enabled: Boolean = true
    var enabled: Boolean
        get() = _enabled
        set(value) {
            if (_enabled != value) {
                _enabled = value
                Log.i(TAG, if (value) "ON tc=${"%.3f".format(timeConstant)}" else "OFF")
                if (!value) {
                    currentMatrix.set(IDENTITY_MATRIX.clone())
                    initialized = false
                }
            }
        }

    private val IDENTITY_QUAT = Quat(1.0, 0.0, 0.0, 0.0)
    @Volatile private var rawQuat = Quat(1.0, 0.0, 0.0, 0.0)
    @Volatile private var smoothedQuat = Quat(1.0, 0.0, 0.0, 0.0)
    @Volatile private var initialized = false
    private var lastTimestampNs = 0L
    private var sampleRate = 200.0

    // Adaptive smoothing state
    private var prevRawQuat = Quat(1.0, 0.0, 0.0, 0.0)
    private var smoothedAngularVelocityDeg = 0.0
    private var highVelocityDurationSec = 0.0
    @Volatile private var effectiveTc = DEFAULT_TIME_CONSTANT
    @Volatile private var lastLeashActive = false
    @Volatile private var lastCorrAngleDeg = 0.0

    // Session log file (gyro.log in the tracking session directory)
    @Volatile
    private var sessionWriter: PrintWriter? = null

    // Bench capture: full-rate gyro CSV + frame timestamp CSV for off-device analysis
    @Volatile
    private var benchGyroWriter: PrintWriter? = null
    @Volatile
    private var benchFrameWriter: PrintWriter? = null
    @Volatile
    private var benchCorrWriter: PrintWriter? = null

    // Telemetry accumulators (reset every TEL_INTERVAL sensor events)
    private var telFrames = 0
    private var telSumAlpha = 0.0
    private var telSumCorrDeg = 0.0
    private var telPeakCorrDeg = 0.0
    private var telPeakExcursion = 0f
    private var telWorstGapMs = 0.0
    private var telSumVel = 0.0
    private var telPeakVel = 0.0
    private var telSumEffTc = 0.0
    private var telPanFrames = 0

    fun start() {
        if (rotationSensor == null) {
            Log.w(TAG, "TYPE_GAME_ROTATION_VECTOR not available — stabilization disabled")
            return
        }
        sensorManager.registerListener(this, rotationSensor, SensorManager.SENSOR_DELAY_FASTEST)
        Log.i(TAG, "Started (timeConstant=${timeConstant}s, fx=${"%.3f".format(fxUv)}, crop=$cropZoom, oisComp=${"%.2f".format(oisCompensation)})")
    }

    fun stop() {
        sensorManager.unregisterListener(this)
        Log.i(TAG, "Stopped (was ${if (_enabled) "ON" else "OFF"}, hz=${"%.0f".format(sampleRate)})")
        endSessionLog()
        initialized = false
        currentMatrix.set(IDENTITY_MATRIX.clone())
        resetTelemetry()
    }

    fun startSessionLog(dir: File) {
        endSessionLog()
        try {
            sessionWriter = PrintWriter(FileWriter(File(dir, "gyro.log"), true), true)
            sessionWriter?.println("# tc=${"%.3f".format(timeConstant)} crop=$cropZoom fx=${"%.3f".format(fxUv)} fy=${"%.3f".format(fyUv)} hz=${"%.0f".format(sampleRate)}")
        } catch (e: Exception) {
            Log.w(TAG, "Failed to create gyro.log: ${e.message}")
        }
    }

    fun endSessionLog() {
        sessionWriter?.close()
        sessionWriter = null
    }

    fun startBenchCapture(dir: File) {
        endBenchCapture()
        try {
            benchGyroWriter = PrintWriter(FileWriter(File(dir, "gyro_raw.csv")), false).also {
                it.println("timestamp_ns,w,x,y,z")
            }
            benchFrameWriter = PrintWriter(FileWriter(File(dir, "frames.csv")), false).also {
                it.println("frame_idx,timestamp_ns")
            }
            benchCorrWriter = PrintWriter(FileWriter(File(dir, "corrections.csv")), false).also {
                it.println("frame_idx,timestamp_ns,raw_w,raw_x,raw_y,raw_z,smooth_w,smooth_x,smooth_y,smooth_z,eff_tc,corr_deg,leash,m0,m1,m2,m3,m4,m5,m6,m7,m8")
            }
            PrintWriter(FileWriter(File(dir, "bench_params.csv"))).use { pw ->
                pw.println("timeConstant,cropZoom,fxUv,fyUv,clampMarginFraction,oisCompensation")
                pw.println("$timeConstant,$cropZoom,$fxUv,$fyUv,0.6,$oisCompensation")
            }
            Log.i(TAG, "Bench capture started → ${dir.absolutePath}")
        } catch (e: Exception) {
            Log.w(TAG, "Failed to start bench capture: ${e.message}")
        }
    }

    fun endBenchCapture() {
        benchGyroWriter?.flush()
        benchGyroWriter?.close()
        benchGyroWriter = null
        benchFrameWriter?.flush()
        benchFrameWriter?.close()
        benchFrameWriter = null
        benchCorrWriter?.flush()
        benchCorrWriter?.close()
        benchCorrWriter = null
    }

    fun logFrameTimestamp(frameIdx: Long, timestampNs: Long) {
        benchFrameWriter?.println("$frameIdx,$timestampNs")
        val cw = benchCorrWriter ?: return
        val rq = rawQuat
        val sq = smoothedQuat
        val mat = currentMatrix.get()
        val leash = if (lastLeashActive) 1 else 0
        cw.println("$frameIdx,$timestampNs," +
            "${rq.w},${rq.x},${rq.y},${rq.z}," +
            "${sq.w},${sq.x},${sq.y},${sq.z}," +
            "$effectiveTc,$lastCorrAngleDeg,$leash," +
            "${mat[0]},${mat[1]},${mat[2]},${mat[3]},${mat[4]},${mat[5]},${mat[6]},${mat[7]},${mat[8]}")
    }

    /** Get the current stabilization matrix (column-major mat3, 9 floats). Thread-safe. */
    fun getMatrix(): FloatArray = currentMatrix.get()

    /** Read camera intrinsics from Camera2 characteristics. */
    fun readCameraIntrinsics(context: Context, frontFacing: Boolean = false) {
        val targetFacing = if (frontFacing) CameraCharacteristics.LENS_FACING_FRONT
                           else CameraCharacteristics.LENS_FACING_BACK
        try {
            val cam2 = context.getSystemService(Context.CAMERA_SERVICE) as Camera2Manager
            for (cameraId in cam2.cameraIdList) {
                val chars = cam2.getCameraCharacteristics(cameraId)
                val facing = chars.get(CameraCharacteristics.LENS_FACING)
                if (facing != targetFacing) continue

                val activeArray = chars.get(CameraCharacteristics.SENSOR_INFO_ACTIVE_ARRAY_SIZE)
                val orientation = chars.get(CameraCharacteristics.SENSOR_ORIENTATION) ?: 90
                sensorOrientation = orientation
                deviceToSensorQuat = sensorOrientationToQuat(orientation)

                // Try LENS_INTRINSIC_CALIBRATION first — calibrated pixel focal lengths
                val intrinsicCal = chars.get(CameraCharacteristics.LENS_INTRINSIC_CALIBRATION)
                if (intrinsicCal != null && activeArray != null) {
                    val fxPx = intrinsicCal[0]  // focal length in pixels (x)
                    val fyPx = intrinsicCal[1]  // focal length in pixels (y)
                    val arrayW = activeArray.width().toDouble()
                    val arrayH = activeArray.height().toDouble()
                    // Bench regression (eis_bench_ois_off_2) measured 1.27x uniform
                    // scale error on both axes — HAL applies additional crop beyond
                    // what active array dimensions describe.
                    val empiricalScale = 1.27
                    fxUv = fxPx / arrayW * empiricalScale
                    fyUv = fyPx / arrayH * empiricalScale
                    Log.i(TAG, "Intrinsics (calibrated): fxPx=${"%.1f".format(fxPx)} fyPx=${"%.1f".format(fyPx)} " +
                        "array=${arrayW.toInt()}x${arrayH.toInt()} scale=$empiricalScale " +
                        "→ fx=${"%.3f".format(fxUv)} fy=${"%.3f".format(fyUv)}")
                } else {
                    // Fallback: physical focal length / sensor size
                    val focalLengths = chars.get(CameraCharacteristics.LENS_INFO_AVAILABLE_FOCAL_LENGTHS)
                    val sensorSize = chars.get(CameraCharacteristics.SENSOR_INFO_PHYSICAL_SIZE)
                    if (focalLengths != null && focalLengths.isNotEmpty() && sensorSize != null) {
                        val focalMm = focalLengths[0].toDouble()
                        val sensorW = sensorSize.width.toDouble()
                        val sensorH = sensorSize.height.toDouble()
                        val empiricalScale = 1.27
                        fxUv = focalMm / sensorW * empiricalScale
                        fyUv = focalMm / sensorH * empiricalScale
                        Log.i(TAG, "Intrinsics (physical): focal=${"%.2f".format(focalMm)}mm " +
                            "sensor=${"%.2f".format(sensorW)}x${"%.2f".format(sensorH)}mm " +
                            "scale=$empiricalScale " +
                            "→ fx=${"%.3f".format(fxUv)} fy=${"%.3f".format(fyUv)}" +
                            (if (activeArray != null) " array=${activeArray.width()}x${activeArray.height()}" else ""))
                    }
                }
                Log.i(TAG, "Sensor orientation: ${orientation}° → d2s quat=(${deviceToSensorQuat.w}, ${deviceToSensorQuat.x}, ${deviceToSensorQuat.y}, ${deviceToSensorQuat.z})")
                break
            }
        } catch (e: Exception) {
            Log.w(TAG, "Failed to read intrinsics: ${e.message}")
        }
    }

    // --- SensorEventListener ---

    override fun onSensorChanged(event: SensorEvent) {
        if (event.sensor.type != Sensor.TYPE_GAME_ROTATION_VECTOR) return

        val quaternion = FloatArray(4)
        SensorManager.getQuaternionFromVector(quaternion, event.values)
        rawQuat = Quat(quaternion[0].toDouble(), quaternion[1].toDouble(),
                       quaternion[2].toDouble(), quaternion[3].toDouble()).normalized()

        val nowNs = event.timestamp
        try { benchGyroWriter?.println("$nowNs,${rawQuat.w},${rawQuat.x},${rawQuat.y},${rawQuat.z}") } catch (_: Exception) {}

        if (!enabled) {
            currentMatrix.set(IDENTITY_MATRIX.clone())
            return
        }

        if (!initialized) {
            smoothedQuat = rawQuat
            initialized = true
            lastTimestampNs = nowNs
            prevRawQuat = rawQuat
            resetAdaptiveState()
            return
        }

        val dtNs = nowNs - lastTimestampNs
        lastTimestampNs = nowNs
        if (dtNs <= 0) return
        if (dtNs > SENSOR_GAP_THRESHOLD_NS) {
            val warn = "SENSOR_GAP dt=${"%.0f".format(dtNs / 1_000_000.0)}ms — reset smoothed"
            Log.w(TAG, warn)
            try { sessionWriter?.println("${System.currentTimeMillis()} WARN $warn") } catch (_: Exception) {}
            smoothedQuat = rawQuat
            prevRawQuat = rawQuat
            resetAdaptiveState()
            return
        }

        val dtSec = dtNs / 1_000_000_000.0
        sampleRate = 0.95 * sampleRate + 0.05 * (1.0 / dtSec)

        // --- Adaptive smoothing: angular velocity → pan detection → effective TC ---
        val deltaQuat = prevRawQuat.conjugate() * rawQuat
        val deltaAngle = 2.0 * acos(abs(deltaQuat.w).coerceIn(0.0, 1.0))
        val angVelDeg = Math.toDegrees(deltaAngle) / dtSec

        val velAlpha = 1.0 - exp(-dtSec / VELOCITY_SMOOTHING_TC)
        smoothedAngularVelocityDeg += velAlpha * (angVelDeg - smoothedAngularVelocityDeg)

        if (smoothedAngularVelocityDeg > PAN_VELOCITY_THRESHOLD_DEG) {
            highVelocityDurationSec += dtSec
        } else {
            highVelocityDurationSec = 0.0
        }

        val isPanning = highVelocityDurationSec >= PAN_ONSET_SEC
        if (adaptiveSmoothing) {
            val targetTc = if (isPanning) timeConstant * PAN_TC_FACTOR else timeConstant
            val tcAlpha = 1.0 - exp(-dtSec * TC_RAMP_SPEED)
            effectiveTc += tcAlpha * (targetTc - effectiveTc)
        } else {
            effectiveTc = timeConstant
        }
        prevRawQuat = rawQuat

        // Exponential SLERP smoothing with adaptive time constant
        val alpha = 1.0 - exp(-(1.0 / sampleRate) / effectiveTc)
        smoothedQuat = slerp(smoothedQuat, rawQuat, alpha)

        // Leash: limit how far smoothed can deviate from raw. Without this, the
        // smoothed path drifts far during handheld walking (34% of samples exceed
        // the crop margin). The hard clamp that used to follow would truncate 87%
        // of the correction during peaks, creating visible wobble artifacts.
        // The leash pulls smoothed toward raw so corrections always fit the margin.
        val cropMargin = 0.5 * (1.0 - 1.0 / cropZoom)
        val maxCorrAngle = cropMargin / maxOf(fxUv, fyUv) / oisCompensation
        val devQuat = smoothedQuat.conjugate() * rawQuat
        val devAngle = 2.0 * acos(devQuat.w.coerceIn(-1.0, 1.0))
        lastLeashActive = devAngle > maxCorrAngle && devAngle > 1e-6
        if (lastLeashActive) {
            val catchUp = 1.0 - maxCorrAngle / devAngle
            smoothedQuat = slerp(smoothedQuat, rawQuat, catchUp)
        }

        // Correction: for each output pixel (in the smoothed frame), find where to
        // sample in the raw (shaky) input texture. That's smooth⁻¹ × raw.
        var correctionDevice = smoothedQuat.conjugate() * rawQuat

        // Scale correction when OIS is active — OIS handles part of the shake,
        // so full correction overcorrects. slerp(identity, corr, factor) scales the
        // rotation angle by factor.
        if (oisCompensation < 1.0) {
            correctionDevice = slerp(IDENTITY_QUAT, correctionDevice, oisCompensation)
        }

        // Rotate correction from device coordinate space into camera sensor space.
        val correction = deviceToSensorQuat * correctionDevice * deviceToSensorQuat.conjugate()

        // Build homography H = K × R × K⁻¹ in UV [0,1]² space
        val r = correction.toRotationMatrix()
        val h = computeHomographyUV(r, fxUv, fyUv, cropZoom.toDouble())

        val hPortrait = sensorToPortraitGL(h, sensorOrientation)
        val rawExcursion = maxCornerExcursion(hPortrait)
        currentMatrix.set(hPortrait)

        // --- Telemetry ---
        val corrAngleDeg = 2.0 * acos(correction.w.coerceIn(-1.0, 1.0)) * (180.0 / PI)
        lastCorrAngleDeg = corrAngleDeg

        telFrames++
        telSumAlpha += alpha
        telSumCorrDeg += corrAngleDeg
        if (corrAngleDeg > telPeakCorrDeg) telPeakCorrDeg = corrAngleDeg
        if (rawExcursion > telPeakExcursion) telPeakExcursion = rawExcursion
        val dtMs = dtSec * 1000.0
        if (dtMs > telWorstGapMs) telWorstGapMs = dtMs
        telSumVel += smoothedAngularVelocityDeg
        if (smoothedAngularVelocityDeg > telPeakVel) telPeakVel = smoothedAngularVelocityDeg
        telSumEffTc += effectiveTc
        if (isPanning) telPanFrames++

        if (telFrames >= TEL_INTERVAL) {
            val line = "hz=${"%.0f".format(sampleRate)} " +
                "alpha=${"%.4f".format(telSumAlpha / telFrames)} " +
                "corrDeg=${"%.2f".format(telSumCorrDeg / telFrames)}/${"%.2f".format(telPeakCorrDeg)} " +
                "excur=${"%.4f".format(telPeakExcursion)}/margin${"%.4f".format(cropMargin)} " +
                "gap=${"%.1f".format(telWorstGapMs)}ms " +
                "tc=${"%.3f".format(timeConstant)}→${"%.3f".format(telSumEffTc / telFrames)} " +
                "vel=${"%.1f".format(telSumVel / telFrames)}/${"%.1f".format(telPeakVel)}°/s " +
                "pan=${"%.0f".format(100.0 * telPanFrames / telFrames)}% " +
                "crop=${"%.2f".format(cropZoom)}"
            Log.d(TAG, line)
            try { sessionWriter?.println("${System.currentTimeMillis()} $line") } catch (_: Exception) {}
            resetTelemetry()
        }
    }

    override fun onAccuracyChanged(sensor: Sensor?, accuracy: Int) {}

    // --- Quaternion ---

    data class Quat(val w: Double, val x: Double, val y: Double, val z: Double) {
        fun conjugate() = Quat(w, -x, -y, -z)

        fun norm() = sqrt(w * w + x * x + y * y + z * z)

        fun normalized(): Quat {
            val n = norm()
            return if (n > 1e-10) Quat(w / n, x / n, y / n, z / n) else this
        }

        operator fun times(q: Quat) = Quat(
            w * q.w - x * q.x - y * q.y - z * q.z,
            w * q.x + x * q.w + y * q.z - z * q.y,
            w * q.y - x * q.z + y * q.w + z * q.x,
            w * q.z + x * q.y - y * q.x + z * q.w
        )

        fun dot(q: Quat) = w * q.w + x * q.x + y * q.y + z * q.z

        /** Convert to 3×3 rotation matrix (row-major double array). */
        fun toRotationMatrix(): DoubleArray {
            val ww = w * w; val xx = x * x; val yy = y * y; val zz = z * z
            val wx = w * x; val wy = w * y; val wz = w * z
            val xy = x * y; val xz = x * z; val yz = y * z
            return doubleArrayOf(
                1 - 2 * (yy + zz),     2 * (xy - wz),     2 * (xz + wy),
                    2 * (xy + wz), 1 - 2 * (xx + zz),     2 * (yz - wx),
                    2 * (xz - wy),     2 * (yz + wx), 1 - 2 * (xx + yy)
            )
        }
    }

    // --- Math utilities ---

    // Smoothing uses the package-level slerp() below.


    /**
     * Compute stabilization as crop zoom + center translation (affine, no perspective).
     *
     * Computes the full homography H = K × R × K⁻¹ to find the center pixel
     * displacement, then builds an affine matrix with constant scale (1/zoom)
     * and variable translation. This eliminates the perspective/scale variation
     * from the full homography that causes visible "breathing" (objects appearing
     * to stretch and compress as the correction angle changes).
     */
    private fun computeHomographyUV(
        r: DoubleArray, fx: Double, fy: Double, zoom: Double
    ): FloatArray {
        // Full H = K × R × K⁻¹ for center displacement extraction
        val kr = doubleArrayOf(
            fx * r[0] + 0.5 * r[6],  fx * r[1] + 0.5 * r[7],  fx * r[2] + 0.5 * r[8],
            fy * r[3] + 0.5 * r[6],  fy * r[4] + 0.5 * r[7],  fy * r[5] + 0.5 * r[8],
                           r[6],                r[7],                r[8]
        )
        val ifx = 1.0 / fx; val ify = 1.0 / fy
        val h02 = kr[2] - kr[0] * 0.5 * ifx - kr[1] * 0.5 * ify
        val h12 = kr[5] - kr[3] * 0.5 * ifx - kr[4] * 0.5 * ify
        val h22 = kr[8] - kr[6] * 0.5 * ifx - kr[7] * 0.5 * ify

        // Center displacement: H × [0.5, 0.5, 1] with perspective division
        val w  = kr[6] * ifx * 0.5 + kr[7] * ify * 0.5 + h22
        val cu = (kr[0] * ifx * 0.5 + kr[1] * ify * 0.5 + h02) / w
        val cv = (kr[3] * ifx * 0.5 + kr[4] * ify * 0.5 + h12) / w
        val du = cu - 0.5
        val dv = cv - 0.5

        // Affine: constant crop zoom + translation (no scale/perspective variation)
        val iz = 1.0 / zoom
        val tx = 0.5 * (1.0 - iz) + iz * du
        val ty = 0.5 * (1.0 - iz) + iz * dv

        // Row-major [[iz,0,tx],[0,iz,ty],[0,0,1]] → column-major for GL
        return floatArrayOf(
            iz.toFloat(), 0f, 0f,
            0f, iz.toFloat(), 0f,
            tx.toFloat(), ty.toFloat(), 1f
        )
    }

    private fun maxCornerExcursion(colMajorMat3: FloatArray): Float {
        val m = colMajorMat3
        var maxExc = 0f
        for (cu in 0..1) {
            for (cv in 0..1) {
                val u = cu.toFloat(); val v = cv.toFloat()
                val tu = m[0] * u + m[3] * v + m[6]
                val tv = m[1] * u + m[4] * v + m[7]
                val exc = maxOf(-tu, tu - 1f, -tv, tv - 1f, 0f)
                if (exc > maxExc) maxExc = exc
            }
        }
        return maxExc
    }

    private fun resetTelemetry() {
        telFrames = 0; telSumAlpha = 0.0; telSumCorrDeg = 0.0
        telPeakCorrDeg = 0.0; telPeakExcursion = 0f
        telWorstGapMs = 0.0
        telSumVel = 0.0; telPeakVel = 0.0
        telSumEffTc = 0.0; telPanFrames = 0
    }

    private fun resetAdaptiveState() {
        smoothedAngularVelocityDeg = 0.0
        highVelocityDurationSec = 0.0
        effectiveTc = timeConstant
    }

    private fun hfovToFocalUv(hfovDegrees: Double): Double {
        val hfovRad = Math.toRadians(hfovDegrees)
        return 1.0 / (2.0 * tan(hfovRad / 2.0))
    }

    private fun sensorOrientationToQuat(degrees: Int): Quat {
        val angle = Math.toRadians(degrees.toDouble())
        return Quat(cos(angle / 2), 0.0, 0.0, sin(angle / 2))
    }

    /**
     * Convert a sensor-UV homography (GL column-major) to portrait UV.
     *
     * The GL shader applies the matrix to quad UV coordinates which are in portrait
     * orientation, but computeHomographyUV produces the matrix in sensor UV space.
     * H_portrait = T⁻¹ × H_sensor × T where T maps portrait UV → sensor UV.
     */
    private fun sensorToPortraitGL(colMajor: FloatArray, orientation: Int): FloatArray {
        if (orientation != 90 && orientation != 270) return colMajor

        // GL column-major → row-major element naming
        val h00 = colMajor[0]; val h10 = colMajor[1]; val h20 = colMajor[2]
        val h01 = colMajor[3]; val h11 = colMajor[4]; val h21 = colMajor[5]
        val h02 = colMajor[6]; val h12 = colMajor[7]; val h22 = colMajor[8]

        val p00: Float; val p01: Float; val p02: Float
        val p10: Float; val p11: Float; val p12: Float
        val p20: Float; val p21: Float; val p22: Float

        if (orientation == 90) {
            // portrait_u = sensor_v, portrait_v = 1 - sensor_u
            // T = [0,-1,1; 1,0,0; 0,0,1]  T⁻¹ = [0,1,0; -1,0,1; 0,0,1]
            p00 = h11;        p01 = -h10;       p02 = h10 + h12
            p10 = h21 - h01;  p11 = h00 - h20;  p12 = h20 + h22 - h00 - h02
            p20 = h21;        p21 = -h20;        p22 = h20 + h22
        } else {
            // 270°: portrait_u = 1 - sensor_v, portrait_v = sensor_u
            // T = [0,1,0; -1,0,1; 0,0,1]  T⁻¹ = [0,-1,1; 1,0,0; 0,0,1]
            p00 = h11 - h21;  p01 = h20 - h10;  p02 = h21 + h22 - h11 - h12
            p10 = -h01;       p11 = h00;         p12 = h01 + h02
            p20 = -h21;       p21 = h20;         p22 = h21 + h22
        }

        return floatArrayOf(
            p00, p10, p20,
            p01, p11, p21,
            p02, p12, p22
        )
    }
}

private val IDENTITY_MATRIX = floatArrayOf(
    1f, 0f, 0f,
    0f, 1f, 0f,
    0f, 0f, 1f
)

internal fun slerp(a: GyroStabilizer.Quat, b: GyroStabilizer.Quat, t: Double): GyroStabilizer.Quat {
    var dot = a.dot(b)
    val b2 = if (dot < 0) { dot = -dot; GyroStabilizer.Quat(-b.w, -b.x, -b.y, -b.z) } else b

    return if (dot > 0.9995) {
        GyroStabilizer.Quat(
            a.w + t * (b2.w - a.w),
            a.x + t * (b2.x - a.x),
            a.y + t * (b2.y - a.y),
            a.z + t * (b2.z - a.z)
        ).normalized()
    } else {
        val theta = kotlin.math.acos(dot.coerceIn(-1.0, 1.0))
        val sinTheta = kotlin.math.sin(theta)
        val wa = kotlin.math.sin((1 - t) * theta) / sinTheta
        val wb = kotlin.math.sin(t * theta) / sinTheta
        GyroStabilizer.Quat(
            wa * a.w + wb * b2.w,
            wa * a.x + wb * b2.x,
            wa * a.y + wb * b2.y,
            wa * a.z + wb * b2.z
        ).normalized()
    }
}
