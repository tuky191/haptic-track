package com.haptictrack.camera

import android.content.Context
import android.hardware.Sensor
import android.hardware.SensorEvent
import android.hardware.SensorEventListener
import android.hardware.SensorManager
import android.hardware.camera2.CameraCharacteristics
import android.hardware.camera2.CameraManager as Camera2Manager
import android.os.Build
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
        private const val DEFAULT_TIME_CONSTANT = 0.70
        private const val DEFAULT_HFOV_DEGREES = 75.0
        private const val TEL_INTERVAL = 200
        private const val SENSOR_GAP_THRESHOLD_NS = 100_000_000L
        private const val GAUSSIAN_SIGMA_MS = 400.0  // σ for video Gaussian kernel (bench: optimal at 400ms)

        // Adaptive smoothing: pan detection → dynamic TC
        private const val PAN_VELOCITY_THRESHOLD_DEG = 15.0
        private const val PAN_ONSET_SEC = 0.20
        private const val PAN_TC_FACTOR = 0.30
        private const val VELOCITY_SMOOTHING_TC = 0.05
        private const val TC_RAMP_SPEED = 5.0
        private const val OIS_FAST_TC = 0.03      // fast-tracker TC when OIS is off (full correction)
        private const val OIS_FAST_TC_CALM = 0.06 // fast TC during calm holding — OIS handles more, filter more out
        private const val OIS_FAST_TC_SWAY = 0.02 // fast TC during sway — OIS handles less, let more through
        private const val OIS_ADAPT_VEL_LOW = 3.0  // °/s below which OIS handles nearly everything
        private const val OIS_ADAPT_VEL_HIGH = 15.0 // °/s above which OIS struggles with sway
        // Video path: zero-phase fast reference. Maps the causal fast TC to a Gaussian σ
        // with matching -3dB cutoff: f_c = 1/(2πTC) for one pole, ≈0.187/σ for Gaussian.
        private const val FAST_SIGMA_TC_SCALE = 1.175
        private const val LOCAL_VEL_WINDOW_NS = 60_000_000L  // ±60ms for frame-local velocity
        // Budget-aware TC: fraction of the leash budget the smoother may demand
        // (deviation ≈ rate × TC must fit cropMargin/fEff at the current zoom).
        private const val TC_BUDGET_SAFETY = 0.8

        /**
         * Per-device correction for intrinsics that don't match the HAL's actual
         * output crop. Xiaomi 13 Pro reports focal lengths 1.27× too short
         * (bench regression eis_bench_ois_off_2 measured the uniform scale error
         * on both axes). S26 Ultra intrinsics are genuinely calibrated (fx≠fy,
         * FOV sanity-checks at ~72°) — applying 1.27× there over-corrects every
         * frame by 27%. #158 tracks replacing this table with runtime auto-calibration.
         */
        fun empiricalFocalScale(manufacturer: String): Double =
            if (manufacturer.equals("Xiaomi", ignoreCase = true)) 1.27 else 1.0

        /**
         * Gyro EIS earns its keep only where hardware stabilization is weak
         * (Xiaomi 13 Pro: measured 2.14× improvement). On the S26 Ultra the
         * OIS+HAL already removes 90–96% of rotational shake in every band and
         * software EIS measured 2–2.9× WORSE than off (controlled matrix,
         * sessions 20260610_1907xx). Default off except on known-weak devices;
         * the UI toggle still allows manual override.
         */
        fun defaultEnabled(manufacturer: String): Boolean =
            manufacturer.equals("Xiaomi", ignoreCase = true)

        // Horizon lock: roll-only stabilization toward gravity — the one axis
        // lens-shift OIS physically cannot correct. Walking capture
        // 20260610_195144 measured ±5° roll drift reaching the screen at ~100%.
        const val LOCK_RANGE_DEG = 10.0          // max counter-rotation
        const val LOCK_RELEASE_START_DEG = 15.0  // fade begins (intentional tilt)
        const val LOCK_RELEASE_END_DEG = 25.0    // fully released
        const val LOCK_STREAM_ASPECT = 16.0 / 9.0  // portrait stream h/w in quad UV
        const val LOCK_CROP = 1.15               // EIS-off crop; supports ~±5° (covers the measured ±5° walk). Larger roll is clamped by maxLockAngleDeg, not cropped to black.
        private const val LOCK_SLEW_DEG_PER_S = 60.0  // safety bound on correction rate
        // The lock is a slow LEVELER, not a rigid clamp: the horizon drifts at <1Hz,
        // everything faster is shake. Unsmoothed, the correction carried 9.2° RMS of
        // 3-8Hz oscillation (A/B session 20260611_073437) — chasing fusion noise and
        // aliasing through the 30fps render as visible wobble.
        private const val LOCK_TC = 0.4          // causal low-pass; video path uses the σ=400ms Gaussian

        /** Roll of the device about the optical axis vs gravity, degrees (0 = portrait level). */
        fun gravityRollDeg(q: Quat): Double {
            val gx = 2.0 * (q.x * q.z - q.w * q.y)
            val gy = 2.0 * (q.y * q.z + q.w * q.x)
            return Math.toDegrees(atan2(gx, gy))
        }

        /**
         * Max roll (deg) a given crop can counter-rotate without exposing black
         * corners. A centered rotation θ of a portrait frame (h/w = aspect) needs
         * zoom z ≥ cosθ + aspect·sinθ; inverting gives the largest θ for a crop.
         * (Verified against maxCornerExcursion: crop 1.15 → 4.9°, 1.30 → 10.2°.)
         */
        fun maxLockAngleDeg(crop: Double): Double {
            val a = LOCK_STREAM_ASPECT
            val r = sqrt(1.0 + a * a)
            if (crop >= r) return 90.0
            return Math.toDegrees(atan(a) - acos((crop / r).coerceIn(-1.0, 1.0))).coerceAtLeast(0.0)
        }

        /**
         * Lock correction for a given roll: faded to zero toward release end, then
         * clamped to [maxAngleDeg] — the crop-supported limit, so the counter-rotation
         * never demands more margin than exists (prevents black corners).
         */
        fun lockTargetDeg(rollDeg: Double, maxAngleDeg: Double = LOCK_RANGE_DEG): Double {
            val a = abs(rollDeg)
            val fade = ((LOCK_RELEASE_END_DEG - a) /
                (LOCK_RELEASE_END_DEG - LOCK_RELEASE_START_DEG)).coerceIn(0.0, 1.0)
            val limit = minOf(LOCK_RANGE_DEG, maxAngleDeg)
            return -rollDeg.coerceIn(-limit, limit) * fade
        }
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

    /** Crop zoom applied to absorb warp margins (1.0 = no crop, 1.10 = 10% crop). */
    var cropZoom: Float = 1.30f

    /** Adaptive pan detection: reduces TC during pans, increases during shake. */
    var adaptiveSmoothing: Boolean = true

    /** Leash: limits smoothed-to-raw deviation to prevent corrections exceeding crop margin. */
    var leashEnabled: Boolean = true

    /** OIS compensation: scales correction magnitude when optical stabilization is active.
     *  OIS handles ~20% of shake; without compensation the gyro overcorrects. */
    var oisCompensation: Double = 1.0

    /** Current camera zoom ratio — used to scale TC (more smoothing at zoom). */
    @Volatile var zoomRatio: Float = 1f

    /** Horizon lock: counter-rotate roll toward gravity. Works with EIS on or off. On by default. */
    @Volatile var horizonLockEnabled: Boolean = true
    @Volatile private var lockAppliedDeg = 0.0
    private var lastLockNs = 0L

    /** Zoom interpolation: target set at 10-12fps by tracker, interpolated at 200Hz. */
    @Volatile private var zoomTarget: Float = 1f   // desired zoom from auto-zoom controller
    @Volatile private var zoomApplied: Float = 1f  // interpolated zoom dispatched to CameraX
    private var lastZoomDispatchNs: Long = 0L
    private var lastZoomInterpNs: Long = 0L        // independent of EIS state (#174)
    private val ZOOM_INTERP_RATE = 25.0   // interpolation speed (~40ms TC)
    private val ZOOM_DISPATCH_NS = 16_000_000L  // CameraX dispatch throttle (~60Hz)

    /** Callback to apply interpolated zoom to CameraX. Called at ~60Hz from sensor thread. */
    var onZoomApply: ((Float) -> Unit)? = null

    fun setZoomTarget(target: Float) {
        zoomTarget = target
    }

    /** Snap zoom immediately (for pinch gesture). Sets all three zoom fields to avoid race. */
    fun setZoomImmediate(ratio: Float) {
        zoomTarget = ratio
        zoomApplied = ratio
        zoomRatio = ratio
    }

    /** Low-pass the applied lock correction toward the gravity target (slew-bounded). */
    private fun updateLock(nowNs: Long) {
        if (!horizonLockEnabled) {
            lockAppliedDeg = 0.0
            lastLockNs = nowNs
            return
        }
        // Clamp to the angle the crop this frame renders through can support:
        // EIS-off path uses LOCK_CROP; EIS-on path composes on the slider cropZoom.
        val renderCrop = if (_enabled) cropZoom.toDouble() else LOCK_CROP
        val target = lockTargetDeg(gravityRollDeg(rawQuat), maxLockAngleDeg(renderCrop))
        val last = lastLockNs
        lastLockNs = nowNs
        val dtNs = nowNs - last
        if (last == 0L || dtNs <= 0 || dtNs > SENSOR_GAP_THRESHOLD_NS) {
            lockAppliedDeg = target
            return
        }
        val dtSec = dtNs / 1e9
        val alpha = 1.0 - exp(-dtSec / LOCK_TC)
        val maxStep = LOCK_SLEW_DEG_PER_S * dtSec
        lockAppliedDeg += (alpha * (target - lockAppliedDeg)).coerceIn(-maxStep, maxStep)
    }

    /**
     * Counter-rotation affine about the view center in portrait UV, with crop.
     * Rotation happens in PIXEL space — UV axes have different physical lengths
     * (portrait h/w = LOCK_STREAM_ASPECT), so the off-diagonal terms carry the
     * aspect, else rotation shears the image.
     *
     * [corrDeg] is the rotation the IMAGE needs, applied directly to texcoords.
     * Sign settled empirically on the causal video path (session
     * 20260610_220747: regression of video roll on gyro+applied warp fit
     * "v≈dg−da" best with the negated matrix → un-negated is correct). The
     * earlier "doubling" read (210744) was meter noise — the lookahead path
     * was dropping uStabMatrix entirely, so no sign was observable through it.
     */
    private fun lockMatrix(corrDeg: Double, crop: Double): FloatArray {
        val th = Math.toRadians(corrDeg)
        val iz = 1.0 / crop
        val c = iz * cos(th)
        val s = iz * sin(th)
        val a = c
        val b = -s * LOCK_STREAM_ASPECT
        val d = s / LOCK_STREAM_ASPECT
        val e = c
        val tx = 0.5 - 0.5 * a - 0.5 * b
        val ty = 0.5 - 0.5 * d - 0.5 * e
        // column-major mat3
        return floatArrayOf(
            a.toFloat(), d.toFloat(), 0f,
            b.toFloat(), e.toFloat(), 0f,
            tx.toFloat(), ty.toFloat(), 1f
        )
    }

    /** Column-major mat3 multiply: out = a × b. */
    private fun mat3Mul(a: FloatArray, b: FloatArray): FloatArray {
        val o = FloatArray(9)
        for (col in 0..2) {
            for (row in 0..2) {
                o[col * 3 + row] =
                    a[row] * b[col * 3] + a[3 + row] * b[col * 3 + 1] + a[6 + row] * b[col * 3 + 2]
            }
        }
        return o
    }

    /**
     * Zoom interpolation: ramp toward target at sensor rate, dispatch to CameraX at ~60Hz.
     * Runs on every sensor sample regardless of EIS state — auto-zoom must work with
     * stabilization off (#174). Owns its timestamp so EIS smoothing resets don't touch it.
     */
    internal fun interpolateZoom(nowNs: Long) {
        val last = lastZoomInterpNs
        lastZoomInterpNs = nowNs
        val dtNs = nowNs - last
        if (last == 0L || dtNs <= 0) return
        if (dtNs > SENSOR_GAP_THRESHOLD_NS) {
            zoomApplied = zoomTarget
            zoomRatio = zoomApplied
            return
        }
        val dtSec = dtNs / 1_000_000_000.0
        val zoomAlpha = (1.0 - exp(-dtSec * ZOOM_INTERP_RATE)).toFloat()
        zoomApplied += zoomAlpha * (zoomTarget - zoomApplied)
        zoomRatio = zoomApplied
        if (nowNs - lastZoomDispatchNs >= ZOOM_DISPATCH_NS) {
            lastZoomDispatchNs = nowNs
            onZoomApply?.invoke(zoomApplied)
        }
    }

    /** Gaussian kernel smoothing for video frames (400ms output latency, ~95MB FBO buffer). */
    /**
     * Lookahead (Gaussian, zero-phase) video path. DISABLED: renderFromFBO
     * produces an identity warp on S26 — recordings through it never contained
     * uStabMatrix (verified: EIS crop absent comparing braced EIS-on/off
     * sessions 1907xx; lock warp absent in walks 210744/211537 while the
     * causal session 220747 demonstrably renders it). Root cause TBD — until
     * then the causal single-pass path is the one that actually stabilizes.
     */
    var rtsLookahead: Boolean = false

    /** Current stabilization matrix in column-major order for GL (mat3). Identity when disabled. */
    private val currentMatrix = AtomicReference(IDENTITY_MATRIX.clone())

    /** Whether stabilization is active. Device-gated default — see [defaultEnabled]. */
    @Volatile
    private var _enabled: Boolean = defaultEnabled(Build.MANUFACTURER)
    var enabled: Boolean
        get() = _enabled
        set(value) {
            if (_enabled != value) {
                _enabled = value
                Log.i(TAG, if (value) "ON tc=${"%.3f".format(timeConstant)}" else "OFF")
                if (!value) {
                    currentMatrix.set(IDENTITY_MATRIX.clone())
                    initialized = false
                    resetTranslationState()
                    clearTrackingState()
                    zoomApplied = zoomTarget
                }
            }
        }

    private val IDENTITY_QUAT = Quat(1.0, 0.0, 0.0, 0.0)
    @Volatile private var rawQuat = Quat(1.0, 0.0, 0.0, 0.0)
    @Volatile private var smoothedQuat = Quat(1.0, 0.0, 0.0, 0.0)
    @Volatile private var smoothFastQuat = Quat(1.0, 0.0, 0.0, 0.0)
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

    // Ring buffer for raw quaternion history (Gaussian kernel smoothing for video)
    private val QUAT_HISTORY_SIZE = 1024  // ~5s at 200Hz (covers ±3σ for σ=400ms with margin)
    private val historyW = DoubleArray(QUAT_HISTORY_SIZE)
    private val historyX = DoubleArray(QUAT_HISTORY_SIZE)
    private val historyY = DoubleArray(QUAT_HISTORY_SIZE)
    private val historyZ = DoubleArray(QUAT_HISTORY_SIZE)
    private val historyTs = LongArray(QUAT_HISTORY_SIZE)
    private val historyZoom = FloatArray(QUAT_HISTORY_SIZE)  // zoom at sample time (video fxEff)
    private var historyHead = 0
    private var historyCount = 0
    private val historyLock = Any()

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

    // Translation correction: optical flow → position-domain smoothing with leash.
    // Measures post-gyro residual displacement from stabilized analysis frames,
    // smooths the cumulative path, applies the difference as a crop offset.
    /**
     * Whether the measured translation track is APPLIED as a correction. Off by
     * default: the raw-FBO optical flow measured 7× the physically possible
     * displacement on S26 (session 20260610_182744) and the injected correction
     * anti-correlated with stability. Measurement + telemetry always run so
     * EIS-off captures double as sensor-validation data.
     */
    var translationCorrectionEnabled: Boolean = false
        set(value) {
            if (field != value) {
                field = value
                if (!value) resetTranslationState()
            }
        }
    private val TRANS_TC = 0.15          // smoothing time constant (seconds) — lower = more responsive to shake
    private val TRANS_MARGIN_FRAC = 0.5  // fraction of crop margin reserved for translation
    private val TRANS_DEAD_ZONE_PX = 0.3 // phase correlation noise floor (pixels at raw FBO res)
    @Volatile private var transCumX = 0.0    // cumulative translation path (UV)
    @Volatile private var transCumY = 0.0
    @Volatile private var transSmoothX = 0.0 // smoothed translation path (UV)
    @Volatile private var transSmoothY = 0.0
    @Volatile private var transTargetUvX = 0f  // target correction from GL thread
    @Volatile private var transTargetUvY = 0f
    @Volatile private var transAppliedUvX = 0f // smoothed correction applied at 200Hz
    @Volatile private var transAppliedUvY = 0f
    @Volatile private var prevGrayMat: org.opencv.core.Mat? = null
    @Volatile private var prevGyroUvX = 0.0  // previous frame's rotation-only center offset
    @Volatile private var prevGyroUvY = 0.0
    @Volatile private var rotOnlyUvX = 0.0   // latest rotation-only center offset (no translation)
    @Volatile private var rotOnlyUvY = 0.0

    // Translation path history for the video (lookahead) path — zero-phase smoothing
    // of the cumulative translation track, mirroring the rotation history ring.
    private val TRANS_HISTORY_SIZE = 256  // ~8.5s at 30fps
    private val transHistTs = LongArray(TRANS_HISTORY_SIZE)
    private val transHistX = DoubleArray(TRANS_HISTORY_SIZE)
    private val transHistY = DoubleArray(TRANS_HISTORY_SIZE)
    private var transHistHead = 0
    private var transHistCount = 0
    private val transHistLock = Any()

    // Subject centering: nudge the crop toward the locked subject's bbox center.
    // Smooths the bbox center heavily to avoid detector jitter, then drifts the
    // crop offset so the subject moves toward frame center.
    private val CENTER_BBOX_TC = 0.50     // bbox center smoothing (seconds) — heavy to filter detector noise
    private val CENTER_MARGIN_FRAC = 0.5  // fraction of crop margin available for centering
    @Volatile private var trackSmoothX = 0.5  // smoothed bbox center
    @Volatile private var trackSmoothY = 0.5
    @Volatile private var trackTargetUvX = 0f // target centering offset
    @Volatile private var trackTargetUvY = 0f
    @Volatile private var trackAppliedUvX = 0f
    @Volatile private var trackAppliedUvY = 0f
    @Volatile private var trackingActive = false

    /**
     * Feed a raw (pre-stabilization) frame for optical-flow translation correction.
     * Called from the GL thread at camera rate (~30fps). The bitmap is reused by
     * the caller — all OpenCV work must complete before returning.
     *
     * Displacement = rotation + translation. We subtract the gyro-predicted rotation
     * displacement to isolate the translation component.
     */
    fun onRawFrame(bitmap: android.graphics.Bitmap, frameTimestampNs: Long) {
        if (!_enabled) return  // measurement runs regardless of the apply flag

        val mat = org.opencv.core.Mat()
        val gray = org.opencv.core.Mat()
        val smallFloat = org.opencv.core.Mat()
        try {
            org.opencv.android.Utils.bitmapToMat(bitmap, mat)
            org.opencv.imgproc.Imgproc.cvtColor(mat, gray, org.opencv.imgproc.Imgproc.COLOR_RGBA2GRAY)
            gray.convertTo(smallFloat, org.opencv.core.CvType.CV_64F)
        } finally {
            mat.release()
            gray.release()
        }

        // Read rotation-only center offset — written by onSensorChanged BEFORE
        // translation correction is added, so no feedback loop.
        val gyroUvX = rotOnlyUvX
        val gyroUvY = rotOnlyUvY

        val prev = prevGrayMat
        if (prev != null && prev.size() == smallFloat.size()) {
            val result = org.opencv.imgproc.Imgproc.phaseCorrelate(prev, smallFloat)

            val rawDx = result.x
            val rawDy = result.y
            val dxDs = if (abs(rawDx) < TRANS_DEAD_ZONE_PX) 0.0 else rawDx
            val dyDs = if (abs(rawDy) < TRANS_DEAD_ZONE_PX) 0.0 else rawDy

            val portW = bitmap.width.toDouble()
            val portH = bitmap.height.toDouble()
            val rawDispUvX = dxDs / portW
            val rawDispUvY = dyDs / portH

            // Gyro-predicted rotation displacement (change since last frame)
            val gyroDispUvX = gyroUvX - prevGyroUvX
            val gyroDispUvY = gyroUvY - prevGyroUvY

            // Residual = raw - gyro prediction = translation component
            val transDispUvX = rawDispUvX - gyroDispUvX
            val transDispUvY = rawDispUvY - gyroDispUvY

            transCumX += transDispUvX
            transCumY += transDispUvY

            val alpha = 1.0 - exp(-1.0 / (30.0 * TRANS_TC))
            transSmoothX += alpha * (transCumX - transSmoothX)
            transSmoothY += alpha * (transCumY - transSmoothY)

            // Leash: limit deviation to fraction of crop margin
            val cropMarginUv = 0.5 * (1.0 - 1.0 / cropZoom)
            val maxCorrUv = cropMarginUv * TRANS_MARGIN_FRAC
            val devX = transCumX - transSmoothX
            val devY = transCumY - transSmoothY
            if (abs(devX) > maxCorrUv) {
                transSmoothX = transCumX - sign(devX) * maxCorrUv
            }
            if (abs(devY) > maxCorrUv) {
                transSmoothY = transCumY - sign(devY) * maxCorrUv
            }

            // Periodic rebase to prevent unbounded integral drift
            if (abs(transSmoothX) > 1.0 || abs(transSmoothY) > 1.0) {
                transCumX -= transSmoothX
                transCumY -= transSmoothY
                transSmoothX = 0.0
                transSmoothY = 0.0
            }

            transTargetUvX = (transCumX - transSmoothX).toFloat()
            transTargetUvY = (transCumY - transSmoothY).toFloat()
            recordTranslationSample(frameTimestampNs, transCumX, transCumY)

            prev.release()
        } else {
            prev?.release()
        }
        prevGyroUvX = gyroUvX
        prevGyroUvY = gyroUvY
        prevGrayMat = smallFloat
    }

    /**
     * Feed tracking bbox center for position-domain stabilization at zoom.
     * Called from the tracking callback at ~10-12fps when a subject is locked.
     * @param centerX bbox center X in normalized [0,1] frame coordinates
     * @param centerY bbox center Y in normalized [0,1] frame coordinates
     * @param bboxArea bbox area in normalized [0,1] coordinates (width*height)
     */
    fun onTrackingUpdate(centerX: Float, centerY: Float, bboxArea: Float) {
        if (!_enabled) {
            if (trackingActive) clearTrackingState()
            return
        }
        if (!trackingActive) {
            trackSmoothX = centerX.toDouble()
            trackSmoothY = centerY.toDouble()
            trackingActive = true
            return
        }
        // Smooth the bbox center heavily to filter detector noise
        val alpha = 1.0 - exp(-1.0 / (12.0 * CENTER_BBOX_TC))
        trackSmoothX += alpha * (centerX - trackSmoothX)
        trackSmoothY += alpha * (centerY - trackSmoothY)

        // Scale centering by how small the subject is — large bbox = noisy center, less useful
        val areaScale = (1.0 - bboxArea).coerceIn(0.0, 1.0)

        // Centering offset: how far from center, clamped by available crop margin
        val cropMarginUv = 0.5 * (1.0 - 1.0 / cropZoom)
        val maxCenterUv = cropMarginUv * CENTER_MARGIN_FRAC
        val offsetX = ((0.5 - trackSmoothX) * areaScale).coerceIn(-maxCenterUv, maxCenterUv)
        val offsetY = ((0.5 - trackSmoothY) * areaScale).coerceIn(-maxCenterUv, maxCenterUv)

        trackTargetUvX = offsetX.toFloat()
        trackTargetUvY = offsetY.toFloat()
    }

    fun clearTracking() {
        clearTrackingState()
    }

    private fun clearTrackingState() {
        trackingActive = false
        trackSmoothX = 0.5; trackSmoothY = 0.5
        trackTargetUvX = 0f; trackTargetUvY = 0f
        trackAppliedUvX = 0f; trackAppliedUvY = 0f
    }

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
        resetTranslationState()
        zoomApplied = zoomTarget
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
                it.println("frame_idx,timestamp_ns,raw_w,raw_x,raw_y,raw_z,smooth_w,smooth_x,smooth_y,smooth_z,eff_tc,corr_deg,leash,m0,m1,m2,m3,m4,m5,m6,m7,m8,zoom,trans_cum_x,trans_cum_y,trans_vid_x,trans_vid_y")
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
        val videoTrans = videoTranslationCorrection(timestampNs)
        cw.println("$frameIdx,$timestampNs," +
            "${rq.w},${rq.x},${rq.y},${rq.z}," +
            "${sq.w},${sq.x},${sq.y},${sq.z}," +
            "$effectiveTc,$lastCorrAngleDeg,$leash," +
            "${mat[0]},${mat[1]},${mat[2]},${mat[3]},${mat[4]},${mat[5]},${mat[6]},${mat[7]},${mat[8]},$zoomRatio," +
            "$transCumX,$transCumY,${videoTrans.first},${videoTrans.second}")
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
                    val empiricalScale = empiricalFocalScale(Build.MANUFACTURER)
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
                        val empiricalScale = empiricalFocalScale(Build.MANUFACTURER)
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

                // Check OIS data availability — Camera2 can expose lens displacement samples
                val oisModes = chars.get(CameraCharacteristics.STATISTICS_INFO_AVAILABLE_OIS_DATA_MODES)
                if (oisModes != null && oisModes.contains(android.hardware.camera2.CameraMetadata.STATISTICS_OIS_DATA_MODE_ON)) {
                    Log.i(TAG, "OIS_DATA available — lens displacement samples can be read from CaptureResult")
                } else {
                    Log.i(TAG, "OIS_DATA not available on this device (modes=${oisModes?.toList()})")
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

        val quaternion = FloatArray(4)
        SensorManager.getQuaternionFromVector(quaternion, event.values)
        rawQuat = Quat(quaternion[0].toDouble(), quaternion[1].toDouble(),
                       quaternion[2].toDouble(), quaternion[3].toDouble()).normalized()

        // Quaternions double-cover rotations (q ≡ −q) and the sensor may flip sign
        // mid-stream. Keep raw in the smoothing state's hemisphere — otherwise the
        // leash deviation reads ~360°, fires every sample, and snaps smoothed onto
        // raw, silently disabling stabilization (session_20260610_173942: 100% of
        // frames latched).
        if (initialized && rawQuat.dot(smoothedQuat) < 0.0) {
            rawQuat = Quat(-rawQuat.w, -rawQuat.x, -rawQuat.y, -rawQuat.z)
        }

        val nowNs = event.timestamp
        try { benchGyroWriter?.println("$nowNs,${rawQuat.w},${rawQuat.x},${rawQuat.y},${rawQuat.z}") } catch (_: Exception) {}

        // Zoom dispatch must run before the enabled gate — auto-zoom works with EIS off (#174).
        // Before the history write so the recorded per-sample zoom is fresh.
        interpolateZoom(nowNs)
        updateLock(nowNs)

        synchronized(historyLock) {
            historyW[historyHead] = rawQuat.w
            historyX[historyHead] = rawQuat.x
            historyY[historyHead] = rawQuat.y
            historyZ[historyHead] = rawQuat.z
            historyTs[historyHead] = nowNs
            historyZoom[historyHead] = zoomRatio
            historyHead = (historyHead + 1) % QUAT_HISTORY_SIZE
            if (historyCount < QUAT_HISTORY_SIZE) historyCount++
        }

        if (!enabled) {
            currentMatrix.set(
                if (horizonLockEnabled) lockMatrix(lockAppliedDeg, LOCK_CROP)
                else IDENTITY_MATRIX.clone()
            )
            return
        }

        if (!initialized) {
            smoothedQuat = rawQuat
            smoothFastQuat = rawQuat
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
            smoothFastQuat = rawQuat
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

        // Camera zoom magnifies on-screen motion: the SurfaceTexture stream is
        // already zoom-cropped by the ISP, so its UV [0,1]² spans 1/zoom of the
        // FOV and the same rotation displaces pixels zoom× farther. Scale the
        // focal lengths so corrections (and the leash budget) match the stream.
        val zoomFocal = zoomRatio.toDouble().coerceAtLeast(0.1)
        val fxEff = fxUv * zoomFocal
        val fyEff = fyUv * zoomFocal
        val cropMargin = 0.5 * (1.0 - 1.0 / cropZoom)

        // Budget-aware TC: the crop margin grants maxCorrAngle of correction range,
        // and a causal smoother deviates by ~rate×TC under sustained rotation. Cap
        // TC so the demanded deviation fits the budget — otherwise the leash chops
        // corrections nonlinearly every sample and stabilization degrades to the
        // margin width (measured: 26% leash duty at zoom 2.7 with the old
        // tc×sqrt(zoom) boost, mean correction pinned at the ceiling).
        val budgetDeg = Math.toDegrees(cropMargin / maxOf(fxEff, fyEff))
        val tcBudget = TC_BUDGET_SAFETY * budgetDeg / smoothedAngularVelocityDeg.coerceAtLeast(1.0)
        val cappedTc = min(timeConstant, tcBudget)
        if (adaptiveSmoothing) {
            val targetTc = if (isPanning) cappedTc * PAN_TC_FACTOR else cappedTc
            val tcAlpha = 1.0 - exp(-dtSec * TC_RAMP_SPEED)
            effectiveTc += tcAlpha * (targetTc - effectiveTc)
        } else {
            effectiveTc = cappedTc
        }
        prevRawQuat = rawQuat

        // Exponential SLERP smoothing with adaptive time constant
        val alpha = 1.0 - exp(-(1.0 / sampleRate) / effectiveTc)
        smoothedQuat = slerp(smoothedQuat, rawQuat, alpha)

        // Fast-tracking filter: follows the device closely, filtering out
        // high-frequency vibration that OIS handles optically. The correction
        // smoothHeavy⁻¹ × smoothFast contains only sway OIS misses.
        // Adaptive: during calm holding OIS handles more → raise TC to filter more.
        // During walking sway OIS struggles → lower TC to let more correction through.
        val vel = smoothedAngularVelocityDeg
        val adaptFastTc = if (oisCompensation < 1.0) {
            val t = ((vel - OIS_ADAPT_VEL_LOW) / (OIS_ADAPT_VEL_HIGH - OIS_ADAPT_VEL_LOW)).coerceIn(0.0, 1.0)
            OIS_FAST_TC_CALM + t * (OIS_FAST_TC_SWAY - OIS_FAST_TC_CALM)
        } else OIS_FAST_TC
        val fastAlpha = 1.0 - exp(-(1.0 / sampleRate) / adaptFastTc)
        smoothFastQuat = slerp(smoothFastQuat, rawQuat, fastAlpha)

        // Leash: limit how far smoothed can deviate from raw (safety net — the
        // budget-aware TC above should keep deviations inside the margin).
        val maxCorrAngle = cropMargin / maxOf(fxEff, fyEff)
        val devQuat = smoothedQuat.conjugate() * rawQuat
        val devAngle = 2.0 * acos(abs(devQuat.w).coerceIn(0.0, 1.0))
        lastLeashActive = leashEnabled && devAngle > maxCorrAngle && devAngle > 1e-6
        if (lastLeashActive) {
            val catchUp = 1.0 - maxCorrAngle / devAngle
            smoothedQuat = slerp(smoothedQuat, rawQuat, catchUp)
        }

        // Correction: smoothHeavy⁻¹ × reference.
        // When OIS is active (oisCompensation < 1.0), use smoothFast as the reference
        // so we only correct the low-frequency sway that OIS can't handle mechanically.
        // When OIS is off, use raw for full correction.
        val corrReference = if (oisCompensation < 1.0) smoothFastQuat else rawQuat
        val correctionDevice = smoothedQuat.conjugate() * corrReference

        // Rotate correction from device coordinate space into camera sensor space.
        val correction = deviceToSensorQuat * correctionDevice * deviceToSensorQuat.conjugate()

        // Build homography H = K × R × K⁻¹ in UV [0,1]² space
        val r = correction.toRotationMatrix()
        val h = computeHomographyUV(r, fxEff, fyEff, cropZoom.toDouble())

        val hPortrait = sensorToPortraitGL(h, sensorOrientation)
        // Capture rotation-only center offset BEFORE adding translation correction.
        // onRawFrame reads these to subtract gyro-predicted rotation from optical flow,
        // isolating the translation component. Reading from currentMatrix would include
        // the previous frame's translation correction → feedback loop.
        rotOnlyUvX = hPortrait[6].toDouble()
        rotOnlyUvY = hPortrait[7].toDouble()
        if (translationCorrectionEnabled) {
            val transAlpha = (1.0 - exp(-dtSec * 60.0)).toFloat() // ~17ms ramp
            transAppliedUvX += transAlpha * (transTargetUvX - transAppliedUvX)
            transAppliedUvY += transAlpha * (transTargetUvY - transAppliedUvY)
            hPortrait[6] += transAppliedUvX
            hPortrait[7] -= transAppliedUvY
        }
        // Subject centering not applied — bbox center at 10-12fps is too noisy.
        // onTrackingUpdate() still computes offsets for future use (VT-based center).
        val rawExcursion = maxCornerExcursion(hPortrait)
        // Horizon lock composes on top — the EIS crop (1.3) already covers ±10°
        // of rotation margin, so no extra crop when both are active.
        currentMatrix.set(
            if (horizonLockEnabled) mat3Mul(lockMatrix(lockAppliedDeg, 1.0), hPortrait)
            else hPortrait
        )

        // --- Telemetry ---
        val corrAngleDeg = 2.0 * acos(abs(correction.w).coerceIn(0.0, 1.0)) * (180.0 / PI)
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
                "crop=${"%.2f".format(cropZoom)} zoom=${"%.1f".format(zoomRatio)}" +
                if (trackingActive) " center=${"%.3f".format(trackAppliedUvX)}/${"%.3f".format(trackAppliedUvY)}" else ""
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
     * Compute a Gaussian-kernel-smoothed stabilization matrix for a video frame.
     *
     * Uses a symmetric Gaussian kernel (σ = GAUSSIAN_SIGMA_MS) centered on the
     * frame timestamp. The weighted quaternion average is computed iteratively
     * in the tangent space (log-map / exp-map). This produces better smoothing
     * than the RTS forward-backward SLERP because:
     * - No initialization transient (shift-invariant kernel)
     * - Proper frequency response (Gaussian → Gaussian in frequency domain)
     * - Intrinsically handles pans (symmetric weighting = zero phase lag)
     *
     * Bench result: 1.69x improvement vs 1.44x causal SLERP (σ=400ms, crop=1.20).
     *
     * Called by StabilizationProcessor for delayed video frames — by the time
     * the frame is rendered, we have LOOKAHEAD_FRAMES worth of future gyro data.
     */
    fun getVideoMatrix(frameTimestampNs: Long): FloatArray {
        if (!_enabled) {
            if (!horizonLockEnabled) return IDENTITY_MATRIX.clone()
            // Lock-only path: roll from the Gaussian-smoothed orientation around the
            // frame timestamp (zero-phase). Instantaneous roll wobbles at 3-8Hz
            // (fusion noise + real shake) — the lock levels slow drift only.
            val window = snapshotWindow(frameTimestampNs) ?: return getMatrix()
            if (window.count < 10) return getMatrix()
            val ti = findClosestIndex(window.timestamps, window.count, frameTimestampNs)
            val meanQ = gaussianMeanQuat(window, frameTimestampNs, GAUSSIAN_SIGMA_MS * 1_000_000.0, ti)
            return lockMatrix(lockTargetDeg(gravityRollDeg(meanQ), maxLockAngleDeg(LOCK_CROP)), LOCK_CROP)
        }

        val window = snapshotWindow(frameTimestampNs) ?: return getMatrix()
        if (window.count < 10) return getMatrix()

        val targetIdx = findClosestIndex(window.timestamps, window.count, frameTimestampNs)
        var smoothed = gaussianMeanQuat(window, frameTimestampNs, GAUSSIAN_SIGMA_MS * 1_000_000.0, targetIdx)
        val rawAtTarget = Quat(window.w[targetIdx], window.x[targetIdx], window.y[targetIdx], window.z[targetIdx])

        // Camera zoom at FRAME time (not "now" — zoom may have ramped during the
        // lookahead buffer). Same geometry as the causal path: the zoom-cropped
        // stream magnifies on-screen motion, so focal lengths scale with zoom.
        val zoomFocal = window.zoom[targetIdx].toDouble().coerceAtLeast(0.1)
        val fxEff = fxUv * zoomFocal
        val fyEff = fyUv * zoomFocal

        // Leash (same as causal path) — prevent corrections exceeding crop margin
        val cropMargin = 0.5 * (1.0 - 1.0 / cropZoom)
        val maxCorrAngle = cropMargin / maxOf(fxEff, fyEff)
        val devQuat = smoothed.conjugate() * rawAtTarget
        val devAngle = 2.0 * acos(abs(devQuat.w).coerceIn(0.0, 1.0))
        if (leashEnabled && devAngle > maxCorrAngle && devAngle > 1e-6) {
            val catchUp = 1.0 - maxCorrAngle / devAngle
            smoothed = slerp(smoothed, rawAtTarget, catchUp)
        }

        // Band-split when OIS active: the fast reference must be evaluated AT the
        // frame timestamp. The causal smoothFastQuat is ~400ms newer than the frame
        // (lookahead buffering) — using it mixed the rotation since capture into the
        // correction, injecting high-frequency garbage. With history on both sides of
        // the frame we can do better than causal: a narrow symmetric Gaussian gives a
        // ZERO-PHASE fast reference (the causal single-pole lags ~45°+ near cutoff,
        // which made 3-8Hz corrections arrive anti-phase and amplify shake).
        val corrRef = if (oisCompensation < 1.0) {
            val vel = localAngVelDeg(window, targetIdx)
            val t = ((vel - OIS_ADAPT_VEL_LOW) / (OIS_ADAPT_VEL_HIGH - OIS_ADAPT_VEL_LOW)).coerceIn(0.0, 1.0)
            val fastTc = OIS_FAST_TC_CALM + t * (OIS_FAST_TC_SWAY - OIS_FAST_TC_CALM)
            gaussianMeanQuat(window, frameTimestampNs, FAST_SIGMA_TC_SCALE * fastTc * 1e9, targetIdx)
        } else rawAtTarget
        val correctionDevice = smoothed.conjugate() * corrRef
        val correction = deviceToSensorQuat * correctionDevice * deviceToSensorQuat.conjugate()
        val r = correction.toRotationMatrix()
        val h = computeHomographyUV(r, fxEff, fyEff, cropZoom.toDouble())
        val hPortrait = sensorToPortraitGL(h, sensorOrientation)

        // Translation correction, zero-phase: the causal path applies a lagged
        // high-pass of the optical-flow translation track; with lookahead we
        // subtract a symmetric Gaussian mean instead (same signs as causal path).
        if (translationCorrectionEnabled) {
            val (tx, ty) = videoTranslationCorrection(frameTimestampNs)
            hPortrait[6] += tx
            hPortrait[7] -= ty
        }
        // Horizon lock composes on top (no extra crop — rides the EIS cropZoom
        // margin, so the lock angle is clamped to what cropZoom supports).
        // Roll from the heavy-smoothed orientation, not the instantaneous one.
        return if (horizonLockEnabled) {
            val maxA = maxLockAngleDeg(cropZoom.toDouble())
            mat3Mul(lockMatrix(lockTargetDeg(gravityRollDeg(smoothed), maxA), 1.0), hPortrait)
        } else hPortrait
    }

    /** Record one optical-flow translation sample (cumulative track) for the video path. */
    internal fun recordTranslationSample(timestampNs: Long, cumX: Double, cumY: Double) {
        synchronized(transHistLock) {
            transHistTs[transHistHead] = timestampNs
            transHistX[transHistHead] = cumX
            transHistY[transHistHead] = cumY
            transHistHead = (transHistHead + 1) % TRANS_HISTORY_SIZE
            if (transHistCount < TRANS_HISTORY_SIZE) transHistCount++
        }
    }

    /**
     * Zero-phase translation correction at the frame timestamp: cumulative track
     * at the frame minus a Gaussian-weighted mean of the track around it.
     * Leashed to the same margin fraction as the causal path.
     */
    private fun videoTranslationCorrection(frameTimestampNs: Long): Pair<Float, Float> {
        val sigmaNs = FAST_SIGMA_TC_SCALE * TRANS_TC * 1e9
        val windowNs = (3.0 * sigmaNs).toLong()
        var nearIdx = -1; var nearDist = Long.MAX_VALUE
        var sumW = 0.0; var meanX = 0.0; var meanY = 0.0
        var cumAtFrameX = 0.0; var cumAtFrameY = 0.0
        synchronized(transHistLock) {
            if (transHistCount < 5) return 0f to 0f
            for (i in 0 until transHistCount) {
                val idx = (transHistHead - transHistCount + i + TRANS_HISTORY_SIZE) % TRANS_HISTORY_SIZE
                val dt = transHistTs[idx] - frameTimestampNs
                if (abs(dt) > windowNs) continue
                val w = exp(-0.5 * (dt / sigmaNs) * (dt / sigmaNs))
                sumW += w
                meanX += w * transHistX[idx]
                meanY += w * transHistY[idx]
                if (abs(dt) < nearDist) {
                    nearDist = abs(dt); nearIdx = idx
                    cumAtFrameX = transHistX[idx]; cumAtFrameY = transHistY[idx]
                }
            }
        }
        // No sample close to the frame (stale history) → no correction
        if (nearIdx < 0 || nearDist > 100_000_000L || sumW < 1e-12) return 0f to 0f
        var tx = cumAtFrameX - meanX / sumW
        var ty = cumAtFrameY - meanY / sumW
        val maxCorrUv = 0.5 * (1.0 - 1.0 / cropZoom) * TRANS_MARGIN_FRAC
        tx = tx.coerceIn(-maxCorrUv, maxCorrUv)
        ty = ty.coerceIn(-maxCorrUv, maxCorrUv)
        return tx.toFloat() to ty.toFloat()
    }

    /** Weighted quaternion average via iterative tangent-space refinement. */
    private fun gaussianMeanQuat(window: QuatWindow, centerNs: Long, sigmaNs: Double, initIdx: Int): Quat {
        var weightSum = 0.0
        val weights = DoubleArray(window.count)
        for (i in 0 until window.count) {
            val dt = (window.timestamps[i] - centerNs).toDouble()
            val w = exp(-0.5 * (dt / sigmaNs) * (dt / sigmaNs))
            weights[i] = w
            weightSum += w
        }
        if (weightSum < 1e-12) return Quat(window.w[initIdx], window.x[initIdx], window.y[initIdx], window.z[initIdx])
        for (i in 0 until window.count) weights[i] /= weightSum

        var meanQ = Quat(window.w[initIdx], window.x[initIdx], window.y[initIdx], window.z[initIdx])
        for (iter in 0..2) {
            var tx = 0.0; var ty = 0.0; var tz = 0.0
            val meanConj = meanQ.conjugate()
            for (i in 0 until window.count) {
                val q = Quat(window.w[i], window.x[i], window.y[i], window.z[i])
                var delta = meanConj * q
                if (delta.w < 0.0) delta = Quat(-delta.w, -delta.x, -delta.y, -delta.z)
                val angle = 2.0 * acos(delta.w.coerceIn(-1.0, 1.0))
                if (angle < 1e-8) continue
                val sinHalf = sin(angle / 2.0)
                val scale = weights[i] * angle / sinHalf
                tx += scale * delta.x
                ty += scale * delta.y
                tz += scale * delta.z
            }
            val tangentAngle = sqrt(tx * tx + ty * ty + tz * tz)
            if (tangentAngle > 1e-8) {
                val ax = tx / tangentAngle
                val ay = ty / tangentAngle
                val az = tz / tangentAngle
                val half = tangentAngle / 2.0
                val sinH = sin(half)
                val stepQ = Quat(cos(half), sinH * ax, sinH * ay, sinH * az)
                meanQ = (meanQ * stepQ).normalized()
            }
        }
        return meanQ
    }

    /** Angular velocity (deg/s) around a window sample, from neighbors within ±LOCAL_VEL_WINDOW_NS. */
    private fun localAngVelDeg(window: QuatWindow, targetIdx: Int): Double {
        val t0 = window.timestamps[targetIdx]
        var lo = targetIdx
        var hi = targetIdx
        while (lo > 0 && t0 - window.timestamps[lo - 1] <= LOCAL_VEL_WINDOW_NS) lo--
        while (hi < window.count - 1 && window.timestamps[hi + 1] - t0 <= LOCAL_VEL_WINDOW_NS) hi++
        if (hi <= lo) return 0.0
        val qa = Quat(window.w[lo], window.x[lo], window.y[lo], window.z[lo])
        val qb = Quat(window.w[hi], window.x[hi], window.y[hi], window.z[hi])
        val d = qa.conjugate() * qb
        val angle = 2.0 * acos(abs(d.w).coerceIn(0.0, 1.0))
        val dtSec = (window.timestamps[hi] - window.timestamps[lo]) / 1e9
        return Math.toDegrees(angle) / dtSec
    }

    private data class QuatWindow(
        val w: DoubleArray, val x: DoubleArray, val y: DoubleArray, val z: DoubleArray,
        val timestamps: LongArray, val zoom: FloatArray, val count: Int
    )

    private fun snapshotWindow(targetNs: Long): QuatWindow? {
        synchronized(historyLock) {
            if (historyCount < 10) return null

            // Window covers ±3σ of the Gaussian kernel
            val windowNs = (3.0 * GAUSSIAN_SIGMA_MS * 1_000_000).toLong()
            val startNs = targetNs - windowNs
            val endNs = targetNs + windowNs

            val indices = mutableListOf<Int>()
            for (i in 0 until historyCount) {
                val idx = (historyHead - historyCount + i + QUAT_HISTORY_SIZE) % QUAT_HISTORY_SIZE
                if (historyTs[idx] in startNs..endNs) indices.add(idx)
            }

            if (indices.size < 10) return null

            val n = indices.size
            return QuatWindow(
                w = DoubleArray(n) { historyW[indices[it]] },
                x = DoubleArray(n) { historyX[indices[it]] },
                y = DoubleArray(n) { historyY[indices[it]] },
                z = DoubleArray(n) { historyZ[indices[it]] },
                timestamps = LongArray(n) { historyTs[indices[it]] },
                zoom = FloatArray(n) { historyZoom[indices[it]] },
                count = n
            )
        }
    }

    private fun findClosestIndex(timestamps: LongArray, count: Int, targetNs: Long): Int {
        var bestIdx = 0
        var bestDist = Long.MAX_VALUE
        for (i in 0 until count) {
            val dist = abs(timestamps[i] - targetNs)
            if (dist < bestDist) {
                bestDist = dist
                bestIdx = i
            }
        }
        return bestIdx
    }

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

    private fun resetTranslationState() {
        prevGrayMat?.release()
        prevGrayMat = null
        transCumX = 0.0; transCumY = 0.0
        transSmoothX = 0.0; transSmoothY = 0.0
        transTargetUvX = 0f; transTargetUvY = 0f
        transAppliedUvX = 0f; transAppliedUvY = 0f
        prevGyroUvX = 0.0; prevGyroUvY = 0.0
        rotOnlyUvX = 0.0; rotOnlyUvY = 0.0
        synchronized(transHistLock) {
            transHistHead = 0; transHistCount = 0
        }
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
