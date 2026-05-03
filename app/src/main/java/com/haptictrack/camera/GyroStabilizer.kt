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
 * The timeConstant controls smoothing strength:
 *   lower = more aggressive smoothing (more stable, but laggier on intentional pans)
 *   higher = more responsive (less smoothing, preserves fast pans)
 */
class GyroStabilizer(context: Context) : SensorEventListener {

    companion object {
        private const val TAG = "GyroStab"
        private const val DEFAULT_TIME_CONSTANT = 0.17
        private const val DEFAULT_HFOV_DEGREES = 75.0
        private const val TEL_INTERVAL = 200
        private const val OOB_WARN_COOLDOWN_NS = 2_000_000_000L
        private const val SENSOR_GAP_THRESHOLD_NS = 100_000_000L
        private const val CLAMP_MARGIN_FRACTION = 0.6
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

    /** Crop zoom applied to absorb warp margins (1.0 = no crop, 1.05 = 5% crop). */
    var cropZoom: Float = 1.125f

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
                if (!value) currentMatrix.set(IDENTITY_MATRIX.clone())
                resetTelemetry()
            }
        }

    private var rawQuat = Quat(1.0, 0.0, 0.0, 0.0)
    private var smoothedQuat = Quat(1.0, 0.0, 0.0, 0.0)
    private var initialized = false
    private var lastTimestampNs = 0L
    private var sampleRate = 200.0

    // Session log file (gyro.log in the tracking session directory)
    @Volatile
    private var sessionWriter: PrintWriter? = null

    // Telemetry accumulators (reset every TEL_INTERVAL sensor events)
    private var telFrames = 0
    private var telSumAlpha = 0.0
    private var telSumCorrDeg = 0.0
    private var telPeakCorrDeg = 0.0
    private var telPeakExcursion = 0f
    private var telSumClamp = 0.0
    private var telWorstGapMs = 0.0
    private var telLastWarnNs = 0L

    fun start() {
        if (rotationSensor == null) {
            Log.w(TAG, "TYPE_GAME_ROTATION_VECTOR not available — stabilization disabled")
            return
        }
        sensorManager.registerListener(this, rotationSensor, SensorManager.SENSOR_DELAY_FASTEST)
        Log.i(TAG, "Started (timeConstant=${timeConstant}s, fx=${"%.3f".format(fxUv)}, crop=$cropZoom)")
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

    /** Get the current stabilization matrix (column-major mat3, 9 floats). Thread-safe. */
    fun getMatrix(): FloatArray = currentMatrix.get()

    /** Update camera intrinsics. Call after camera bind when focal length is known. */
    fun setCameraIntrinsics(focalLengthMm: Float, sensorWidthMm: Float, sensorHeightMm: Float) {
        if (sensorWidthMm > 0 && sensorHeightMm > 0 && focalLengthMm > 0) {
            fxUv = (focalLengthMm / sensorWidthMm).toDouble()
            fyUv = (focalLengthMm / sensorHeightMm).toDouble()
            Log.i(TAG, "Intrinsics: focal=${focalLengthMm}mm, sensor=${sensorWidthMm}x${sensorHeightMm}mm → fx=${"%.3f".format(fxUv)} fy=${"%.3f".format(fyUv)}")
        }
    }

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

                val focalLengths = chars.get(CameraCharacteristics.LENS_INFO_AVAILABLE_FOCAL_LENGTHS)
                val sensorSize = chars.get(CameraCharacteristics.SENSOR_INFO_PHYSICAL_SIZE)
                if (focalLengths != null && focalLengths.isNotEmpty() && sensorSize != null) {
                    setCameraIntrinsics(focalLengths[0], sensorSize.width, sensorSize.height)
                }
                val orientation = chars.get(CameraCharacteristics.SENSOR_ORIENTATION) ?: 90
                deviceToSensorQuat = sensorOrientationToQuat(orientation)
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
        if (!enabled) {
            currentMatrix.set(IDENTITY_MATRIX.clone())
            return
        }

        val quaternion = FloatArray(4)
        SensorManager.getQuaternionFromVector(quaternion, event.values)
        // Android returns [w, x, y, z]
        rawQuat = Quat(quaternion[0].toDouble(), quaternion[1].toDouble(),
                       quaternion[2].toDouble(), quaternion[3].toDouble()).normalized()

        val nowNs = event.timestamp
        if (!initialized) {
            smoothedQuat = rawQuat
            initialized = true
            lastTimestampNs = nowNs
            return
        }

        val dtNs = nowNs - lastTimestampNs
        lastTimestampNs = nowNs
        if (dtNs <= 0) return
        if (dtNs > SENSOR_GAP_THRESHOLD_NS) {
            val warn = "SENSOR_GAP dt=${"%.0f".format(dtNs / 1_000_000.0)}ms — reset smoothed"
            Log.w(TAG, warn)
            sessionWriter?.println("${System.currentTimeMillis()} WARN $warn")
            smoothedQuat = rawQuat
            return
        }

        val dtSec = dtNs / 1_000_000_000.0
        sampleRate = 0.95 * sampleRate + 0.05 * (1.0 / dtSec)

        // Exponential SLERP smoothing (causal, forward-only — Gyroflow's plain algorithm)
        val alpha = 1.0 - exp(-(1.0 / sampleRate) / timeConstant)
        smoothedQuat = slerp(smoothedQuat, rawQuat, alpha)

        // Correction: undo device shake so the output matches the smoothed orientation.
        // R = raw⁻¹ × smoothed transforms from the raw (shaky) sensor frame to the
        // smoothed (stable) frame, telling the shader where to sample in the texture.
        val correctionDevice = rawQuat.conjugate() * smoothedQuat

        // Rotate correction from device coordinate space into camera sensor space.
        // The sensor is physically rotated (SENSOR_ORIENTATION, typically 90°) from
        // the device, so the homography's rotation axes must match the sensor frame.
        val correction = deviceToSensorQuat * correctionDevice * deviceToSensorQuat.conjugate()

        // Build homography H = K × R × K⁻¹ in UV [0,1]² space
        val r = correction.toRotationMatrix()
        var h = computeHomographyUV(r, fxUv, fyUv, cropZoom.toDouble())

        // Clamp correction so edge excursion stays well inside the crop margin.
        // Using 60% of the margin keeps perspective distortion invisible at the edges;
        // excess shake passes through rather than creating warp artifacts.
        val rawExcursion = maxCornerExcursion(h)
        val cropMargin = (0.5 * (1.0 - 1.0 / cropZoom)).toFloat()
        val usableMargin = cropMargin * CLAMP_MARGIN_FRACTION
        var clampRatio = 1.0
        if (rawExcursion > usableMargin) {
            clampRatio = (usableMargin / rawExcursion).toDouble()
            val clamped = slerp(Quat(1.0, 0.0, 0.0, 0.0), correction, clampRatio)
            h = computeHomographyUV(clamped.toRotationMatrix(), fxUv, fyUv, cropZoom.toDouble())
        }
        currentMatrix.set(h)

        // --- Telemetry ---
        val corrAngleDeg = 2.0 * acos(correction.w.coerceIn(-1.0, 1.0)) * (180.0 / PI)

        telFrames++
        telSumAlpha += alpha
        telSumCorrDeg += corrAngleDeg
        if (corrAngleDeg > telPeakCorrDeg) telPeakCorrDeg = corrAngleDeg
        if (rawExcursion > telPeakExcursion) telPeakExcursion = rawExcursion
        telSumClamp += clampRatio
        val dtMs = dtSec * 1000.0
        if (dtMs > telWorstGapMs) telWorstGapMs = dtMs

        if (telFrames >= TEL_INTERVAL) {
            val line = "hz=${"%.0f".format(sampleRate)} " +
                "alpha=${"%.4f".format(telSumAlpha / telFrames)} " +
                "corrDeg=${"%.2f".format(telSumCorrDeg / telFrames)}/${"%.2f".format(telPeakCorrDeg)} " +
                "clamp=${"%.0f".format(100.0 * telSumClamp / telFrames)}% " +
                "excur=${"%.4f".format(telPeakExcursion)}/margin${"%.4f".format(cropMargin)} " +
                "gap=${"%.1f".format(telWorstGapMs)}ms " +
                "tc=${"%.3f".format(timeConstant)} crop=${"%.2f".format(cropZoom)}"
            Log.d(TAG, line)
            sessionWriter?.println("${System.currentTimeMillis()} $line")
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
     * Compute H = K × R × K⁻¹ in UV [0,1]² coordinates, with crop zoom.
     *
     * K_uv = [[fx, 0, 0.5], [0, fy, 0.5], [0, 0, 1]]
     * K_uv⁻¹ = [[1/fx, 0, -0.5/fx], [0, 1/fy, -0.5/fy], [0, 0, 1]]
     *
     * The crop zoom scales texture coordinates INWARD so the viewport maps to the
     * centre of the texture, leaving margin at the edges for stabilization correction.
     * S = [[1/z, 0, 0.5*(1-1/z)], [0, 1/z, 0.5*(1-1/z)], [0, 0, 1]]
     * Final: H = S × K × R × K⁻¹
     */
    private fun computeHomographyUV(
        r: DoubleArray, fx: Double, fy: Double, zoom: Double
    ): FloatArray {
        // K × R (3×3 row-major)
        val kr = doubleArrayOf(
            fx * r[0] + 0.5 * r[6],  fx * r[1] + 0.5 * r[7],  fx * r[2] + 0.5 * r[8],
            fy * r[3] + 0.5 * r[6],  fy * r[4] + 0.5 * r[7],  fy * r[5] + 0.5 * r[8],
                           r[6],                r[7],                r[8]
        )
        // (K × R) × K⁻¹
        val ifx = 1.0 / fx; val ify = 1.0 / fy
        val h = doubleArrayOf(
            kr[0] * ifx,              kr[1] * ify,              kr[2] - kr[0] * 0.5 * ifx - kr[1] * 0.5 * ify,
            kr[3] * ifx,              kr[4] * ify,              kr[5] - kr[3] * 0.5 * ifx - kr[4] * 0.5 * ify,
            kr[6] * ifx,              kr[7] * ify,              kr[8] - kr[6] * 0.5 * ifx - kr[7] * 0.5 * ify
        )
        // Apply crop: scale inward around center (1/zoom keeps tex coords within [0,1])
        val iz = 1.0 / zoom
        val tx = 0.5 * (1.0 - iz)
        val result = floatArrayOf(
            (iz * h[0]).toFloat(),             (iz * h[1]).toFloat(),             (iz * h[2] + tx).toFloat(),
            (iz * h[3]).toFloat(),             (iz * h[4]).toFloat(),             (iz * h[5] + tx).toFloat(),
            h[6].toFloat(),                      h[7].toFloat(),                      h[8].toFloat()
        )
        // Convert from row-major to column-major for GL
        return floatArrayOf(
            result[0], result[3], result[6],
            result[1], result[4], result[7],
            result[2], result[5], result[8]
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
        telSumClamp = 0.0; telWorstGapMs = 0.0
    }

    private fun hfovToFocalUv(hfovDegrees: Double): Double {
        val hfovRad = Math.toRadians(hfovDegrees)
        return 1.0 / (2.0 * tan(hfovRad / 2.0))
    }

    private fun sensorOrientationToQuat(degrees: Int): Quat {
        val angle = -Math.toRadians(degrees.toDouble())
        return Quat(cos(angle / 2), 0.0, 0.0, sin(angle / 2))
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
