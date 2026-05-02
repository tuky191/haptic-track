package com.haptictrack.camera

import android.content.Context
import android.hardware.Sensor
import android.hardware.SensorEvent
import android.hardware.SensorEventListener
import android.hardware.SensorManager
import android.hardware.camera2.CameraCharacteristics
import android.hardware.camera2.CameraManager as Camera2Manager
import android.util.Log
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
        private const val DEFAULT_TIME_CONSTANT = 0.12
        private const val DEFAULT_HFOV_DEGREES = 75.0
    }

    private val sensorManager = context.getSystemService(Context.SENSOR_SERVICE) as SensorManager
    private val rotationSensor = sensorManager.getDefaultSensor(Sensor.TYPE_GAME_ROTATION_VECTOR)

    /** Smoothing time constant in seconds. Lower = more stable. */
    var timeConstant: Double = DEFAULT_TIME_CONSTANT

    /** Focal length in UV-normalized coordinates, derived from camera intrinsics. */
    private var fxUv: Double = hfovToFocalUv(DEFAULT_HFOV_DEGREES)
    private var fyUv: Double = fxUv

    /** Crop zoom applied to absorb warp margins (1.0 = no crop, 1.05 = 5% crop). */
    var cropZoom: Float = 1.05f

    /** Current stabilization matrix in column-major order for GL (mat3). Identity when disabled. */
    private val currentMatrix = AtomicReference(IDENTITY_MATRIX.clone())

    /** Whether stabilization is active. */
    @Volatile
    var enabled: Boolean = true

    private var rawQuat = Quat(1.0, 0.0, 0.0, 0.0)
    private var smoothedQuat = Quat(1.0, 0.0, 0.0, 0.0)
    private var initialized = false
    private var lastTimestampNs = 0L
    private var sampleRate = 200.0

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
        initialized = false
        currentMatrix.set(IDENTITY_MATRIX.clone())
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
        if (dtNs <= 0 || dtNs > 500_000_000L) return // skip bad timestamps

        val dtSec = dtNs / 1_000_000_000.0
        sampleRate = 0.95 * sampleRate + 0.05 * (1.0 / dtSec)

        // Exponential SLERP smoothing (causal, forward-only — Gyroflow's plain algorithm)
        val alpha = 1.0 - exp(-(1.0 / sampleRate) / timeConstant)
        smoothedQuat = slerp(smoothedQuat, rawQuat, alpha)

        // Correction: rotation that maps output (smoothed) position to input (raw) position
        // For each output pixel, "where did this content actually land in the raw frame?"
        val correction = rawQuat * smoothedQuat.conjugate()

        // Build homography H = K × R × K⁻¹ in UV [0,1]² space
        val r = correction.toRotationMatrix()
        val h = computeHomographyUV(r, fxUv, fyUv, cropZoom.toDouble())
        currentMatrix.set(h)
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
     * The crop zoom is applied as a scale around the image center:
     * S = [[z, 0, 0.5*(1-z)], [0, z, 0.5*(1-z)], [0, 0, 1]]
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
        // Apply zoom: scale around center
        val tx = 0.5 * (1.0 - zoom)
        val result = floatArrayOf(
            (zoom * h[0]).toFloat(),             (zoom * h[1]).toFloat(),             (zoom * h[2] + tx).toFloat(),
            (zoom * h[3]).toFloat(),             (zoom * h[4]).toFloat(),             (zoom * h[5] + tx).toFloat(),
            h[6].toFloat(),                      h[7].toFloat(),                      h[8].toFloat()
        )
        // Convert from row-major to column-major for GL
        return floatArrayOf(
            result[0], result[3], result[6],
            result[1], result[4], result[7],
            result[2], result[5], result[8]
        )
    }

    private fun hfovToFocalUv(hfovDegrees: Double): Double {
        val hfovRad = Math.toRadians(hfovDegrees)
        return 1.0 / (2.0 * tan(hfovRad / 2.0))
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
