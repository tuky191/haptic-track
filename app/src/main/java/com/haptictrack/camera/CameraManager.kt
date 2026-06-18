package com.haptictrack.camera

import android.content.Context
import android.hardware.camera2.CameraCharacteristics
import android.hardware.camera2.CameraMetadata
import android.hardware.camera2.CaptureRequest
import android.hardware.camera2.CameraManager as Camera2Manager
import android.util.Log
import androidx.camera.camera2.interop.Camera2Interop
import androidx.camera.core.CameraControl
import androidx.camera.core.CameraInfo
import androidx.camera.core.CameraSelector
import androidx.camera.core.Preview
import androidx.camera.core.UseCaseGroup
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.camera.video.FallbackStrategy
import androidx.camera.video.Quality
import androidx.camera.video.QualitySelector
import androidx.camera.video.Recorder
import androidx.camera.video.VideoCapture
import androidx.camera.view.PreviewView
import androidx.lifecycle.LifecycleOwner
import android.os.Handler
import android.os.Looper
import java.util.concurrent.Executors

class CameraManager(private val context: Context) {

    companion object {
        private const val TAG = "CameraManager"
        /** Fallback if we can't detect optical zoom range. */
        private const val DEFAULT_OPTICAL_MAX = 1f
    }

    private val mainHandler = Handler(Looper.getMainLooper())
    private val mainExecutor = java.util.concurrent.Executor { runnable -> mainHandler.post(runnable) }

    private var cameraProvider: ProcessCameraProvider? = null
    private var cameraControl: CameraControl? = null
    private var cameraInfo: CameraInfo? = null
    private var lifecycleOwnerRef: LifecycleOwner? = null
    private var previewViewRef: PreviewView? = null

    /** Detected optical zoom limit from physical camera focal lengths. */
    private var opticalZoomMax: Float = DEFAULT_OPTICAL_MAX

    /** Software gyro-based EIS, stacks on top of ISP stabilization. */
    val gyroStabilizer = GyroStabilizer(context)

    /** Whether to request ISP-level preview stabilization on next bind. */
    var ispStabilizationEnabled: Boolean = false

    /** ISP tracker probe (#ISP-tracker experiment): result logging gate. */
    @Volatile private var ispTrackerActive = false

    /**
     * ISP tracker probe — PERMANENTLY OFF on S26 Ultra. The vendor channel is
     * open (config keys accepted, result keys returned) and the source-verified
     * CamX T2T protocol was used (one-shot register, square active-array ROI),
     * but Samsung's HAL SEGFAULTS natively in CamX::TrackerNode::TrackerThreadCb
     * on any registration (2026-06-11, camera.qcom.core.so +2524) — their
     * 3rd-party topology likely never wires the node's FD buffer port. Not
     * fixable from an app. com.qti.stats.tracker.so ships on the device; keys
     * and protocol kept for reference should another device wire the node.
     */
    var ispTrackerProbeEnabled: Boolean = false

    /** Current lens facing — back by default. */
    var isFrontCamera: Boolean = false
        private set

    /** Reads frames from Preview surface via OpenGL. Always active. */
    private var frameReader: SurfaceTextureFrameReader? = null

    /** GPU stabilization processor for VideoCapture (gyro EIS on recorded footage). */
    private var stabProcessor: StabilizationProcessor? = null

    /**
     * Callback for analysis frames from SurfaceTexture (processing thread, ~10-12fps).
     * Consumer must call [releaseAnalysisBitmap] when done with the bitmap so it can
     * be returned to the pool.
     */
    var onAnalysisFrame: ((android.graphics.Bitmap) -> Unit)? = null

    /** Return a processing bitmap to the frame reader's pool. */
    fun releaseAnalysisBitmap(bitmap: android.graphics.Bitmap) {
        frameReader?.releaseAnalysisBitmap(bitmap)
    }

    /** Callback for viewfinder display frames from SurfaceTexture (GL thread, ~29fps). */
    var onViewfinderFrame: ((android.graphics.Bitmap) -> Unit)? = null

    lateinit var preview: Preview
        private set

    private val videoExecutor = Executors.newSingleThreadExecutor()
    var videoCapture = createVideoCapture()
        private set

    /**
     * Recording preset. false = 4K30 (UHD — the hardware's 4K fps cap).
     * true = FHD 1080p60 with vendor VDIS: 60fps is only available at
     * 1080p-class sizes, and Samsung grants third-party VDIS real corrective
     * margin below 4K (vdisPreviewMargin=0 at UHD → inert). Takes effect on the
     * next rebind() (bindUseCases recreates VideoCapture from this flag).
     */
    var fhd60VdisPreset: Boolean = false

    private fun createVideoCapture(): VideoCapture<Recorder> {
        val qualities = if (fhd60VdisPreset) listOf(Quality.FHD, Quality.HD)
                        else listOf(Quality.UHD, Quality.FHD, Quality.HD)
        val recorder = Recorder.Builder()
            .setQualitySelector(
                QualitySelector.fromOrderedList(
                    qualities,
                    FallbackStrategy.higherQualityOrLowerThan(Quality.FHD)
                )
            )
            .setExecutor(videoExecutor)
            .build()
        return VideoCapture.withOutput(recorder)
    }

    init {
        opticalZoomMax = detectOpticalZoomMax()
        checkOisDataSupport()
        gyroStabilizer.readCameraIntrinsics(context)
    }

    fun startCamera(lifecycleOwner: LifecycleOwner, previewView: PreviewView) {
        lifecycleOwnerRef = lifecycleOwner
        previewViewRef = previewView
        gyroStabilizer.start()
        val providerFuture = ProcessCameraProvider.getInstance(context)
        providerFuture.addListener({
            cameraProvider = providerFuture.get()
            bindUseCases(lifecycleOwner, previewView)
        }, mainExecutor)
    }

    fun switchCamera() {
        isFrontCamera = !isFrontCamera
        val owner = lifecycleOwnerRef ?: return
        val view = previewViewRef ?: return
        if (isFrontCamera) {
            opticalZoomMax = DEFAULT_OPTICAL_MAX  // front cameras have no optical zoom
        } else {
            opticalZoomMax = detectOpticalZoomMax()
        }
        bindUseCases(owner, view)
        Log.i(TAG, "Switched to ${if (isFrontCamera) "front" else "back"} camera")
    }

    fun rebind() {
        val owner = lifecycleOwnerRef ?: return
        val view = previewViewRef ?: return
        bindUseCases(owner, view)
    }

    /**
     * Rebind camera use cases. Always uses 2-stream (Preview + VideoCapture) with
     * SurfaceTextureFrameReader providing both analysis and viewfinder frames.
     * No mode switching — recording is just toggling VideoCapture on/off.
     */
    private fun bindUseCases(lifecycleOwner: LifecycleOwner, previewView: PreviewView) {
        val provider = cameraProvider ?: return
        provider.unbindAll()

        // Stop any existing frame reader
        frameReader?.stop()
        frameReader = null

        // Recreate VideoCapture for fresh recorder per session
        videoCapture = createVideoCapture()

        // Release previous stabilization processor
        stabProcessor?.release()
        stabProcessor = null

        val selector = if (isFrontCamera) CameraSelector.DEFAULT_FRONT_CAMERA
                       else CameraSelector.DEFAULT_BACK_CAMERA

        val previewBuilder = Preview.Builder()
        val useVdis = (ispStabilizationEnabled || fhd60VdisPreset) && !gyroStabilizer.enabled
        @Suppress("UnsafeOptInUsageError")
        Camera2Interop.Extender(previewBuilder).apply {
            setCaptureRequestOption(
                CaptureRequest.LENS_OPTICAL_STABILIZATION_MODE,
                CaptureRequest.LENS_OPTICAL_STABILIZATION_MODE_ON
            )
            if (useVdis) {
                // Vendor VDIS (CONTROL_VIDEO_STABILIZATION_MODE_ON): Samsung's own
                // digital stabilization — zero-phase, RS-aware, translation-handling.
                // Applies to the whole repeating request (both streams). The S26
                // advertises modes [OFF, ON, PREVIEW_STABILIZATION]; whether the
                // HAL honors ON at 4K is what this build tests. The old CameraX
                // setPreviewStabilizationEnabled used mode 2, which Snapdragon
                // ISPs cap at 1080p.
                setCaptureRequestOption(
                    CaptureRequest.CONTROL_VIDEO_STABILIZATION_MODE,
                    CaptureRequest.CONTROL_VIDEO_STABILIZATION_MODE_ON
                )
            }
            if (fhd60VdisPreset) {
                // 60fps pin — legal at FHD (minFrameDuration 16.6ms); forces
                // exposures <= 1/60s, so indoor footage gets a bit darker.
                setCaptureRequestOption(
                    CaptureRequest.CONTROL_AE_TARGET_FPS_RANGE,
                    android.util.Range(60, 60)
                )
            }
        }
        Log.i(TAG, if (useVdis) "VDIS ON (vendor video stabilization via interop)"
               else if (ispStabilizationEnabled) "VDIS OFF (gyro EIS takes over)"
               else "VDIS OFF (user toggle)")
        if (gyroStabilizer.enabled) {
            gyroStabilizer.oisCompensation = 0.40
            Log.i(TAG, "OIS + gyro EIS: oisCompensation=${gyroStabilizer.oisCompensation}")
        } else {
            gyroStabilizer.oisCompensation = 1.0
            Log.i(TAG, "OIS only, no gyro correction")
        }
        gyroStabilizer.readCameraIntrinsics(context, frontFacing = isFrontCamera)
        Log.i(TAG, "Gyro EIS ${if (gyroStabilizer.enabled) "ON" else "OFF"}")
        preview = previewBuilder.build()

        // Always route Preview to SurfaceTextureFrameReader for fast off-thread frame capture.
        // 2-stream binding (preview + video) — FHD when stabilized, 4K otherwise.
        preview.surfaceProvider = Preview.SurfaceProvider { request ->
            val inputSize = request.resolution // camera's native buffer size (landscape, e.g. 1600x1200)
            // Output in portrait at analysis resolution.
            // Camera outputs landscape; transform matrix rotates 90°.
            // So output width = min dim scaled, height = max dim scaled.
            val analysisShort = 640 // short edge of output
            val aspect = inputSize.width.toFloat() / inputSize.height
            val analysisLong = (analysisShort * aspect).toInt()
            // Portrait: width=short, height=long
            val outW = analysisShort
            val outH = analysisLong
            Log.i(TAG, "SurfaceTexture: input=${inputSize}, output=${outW}x${outH}")

            val reader = SurfaceTextureFrameReader(
                inputWidth = inputSize.width,
                inputHeight = inputSize.height,
                outputWidth = outW,
                outputHeight = outH,
                onFrame = { bitmap -> onAnalysisFrame?.invoke(bitmap) },
                onViewfinderFrame = { bitmap -> onViewfinderFrame?.invoke(bitmap) },
                stabMatrixProvider = { gyroStabilizer.getMatrix() },
                onRawFrame = { bitmap, ts -> gyroStabilizer.onRawFrame(bitmap, ts) }
            )
            val readerSurface = reader.start()
            frameReader = reader

            request.provideSurface(readerSurface, Executors.newSingleThreadExecutor()) { result ->
                Log.d(TAG, "Preview surface result: ${result.resultCode}")
            }
        }
        val useCaseGroupBuilder = UseCaseGroup.Builder()
            .addUseCase(preview)
            .addUseCase(videoCapture)

        val processor = StabilizationProcessor(
            stabMatrixProvider = { gyroStabilizer.getMatrix() },
            videoMatrixProvider = if (gyroStabilizer.rtsLookahead) ({ ts -> gyroStabilizer.getVideoMatrix(ts) }) else null,
            frameTimestampLogger = { idx, ts -> gyroStabilizer.logFrameTimestamp(idx, ts) }
        )
        stabProcessor = processor
        useCaseGroupBuilder.addEffect(StabilizationEffect(processor))
        Log.i(TAG, "StabilizationProcessor added (EIS ${if (gyroStabilizer.enabled) "ON" else "OFF — identity pass-through"}, rts=${gyroStabilizer.rtsLookahead})")

        val camera = provider.bindToLifecycle(lifecycleOwner, selector, useCaseGroupBuilder.build())

        cameraControl = camera.cameraControl
        cameraInfo = camera.cameraInfo

        gyroStabilizer.onZoomApply = { ratio ->
            val clamped = ratio.coerceIn(getMinZoom(), getMaxZoom())
            cameraControl?.setZoomRatio(clamped)
        }

        val previewRes = preview.resolutionInfo?.resolution
        Log.i(TAG, "Bound use cases — preview: $previewRes, frameReader: ${frameReader != null}, gyroVideo: ${stabProcessor != null}")
    }

    fun setTranslationCorrectionEnabled(enabled: Boolean) {
        gyroStabilizer.translationCorrectionEnabled = enabled
        frameReader?.rawFrameEnabled = enabled
    }

    fun setZoomTarget(ratio: Float) {
        val clamped = ratio.coerceIn(getMinZoom(), getMaxZoom())
        gyroStabilizer.setZoomTarget(clamped)
    }

    fun setZoomImmediate(ratio: Float) {
        val clamped = ratio.coerceIn(getMinZoom(), getMaxZoom())
        cameraControl?.setZoomRatio(clamped)
        gyroStabilizer.setZoomImmediate(clamped)
    }

    /**
     * Register a region with the Qualcomm ISP object tracker (Touch-to-Track) —
     * probe experiment. [normBox] is in portrait-normalized view coords; converted
     * to sensor active-array coords (orientation 90, zoom-visible region).
     * ROI layout assumed [x, y, w, h]; status/score/ROI are logged per frame by the
     * IspTracker capture callback for offline comparison against VisualTracker.
     */
    @Suppress("UnsafeOptInUsageError")
    fun ispTrackerRegister(normBox: android.graphics.RectF, zoomRatio: Float) {
        if (!ispTrackerProbeEnabled) return
        val cc = cameraControl ?: return
        try {
            val aw = 4080f; val ah = 3060f  // S26 active array (probe-only; TODO read from characteristics)
            val z = maxOf(zoomRatio, 1f)
            val visW = aw / z; val visH = ah / z
            val cx0 = (aw - visW) / 2f; val cy0 = (ah - visH) / 2f
            // portrait norm -> sensor (SENSOR_ORIENTATION 90): sx = py, sy = 1 - px
            fun sx(py: Float) = cx0 + py * visW
            fun sy(px: Float) = cy0 + (1f - px) * visH
            val x1 = sx(normBox.top); val x2 = sx(normBox.bottom)
            val y1 = sy(normBox.right); val y2 = sy(normBox.left)
            val l = minOf(x1, x2); val t = minOf(y1, y2)
            val w = kotlin.math.abs(x2 - x1); val h = kotlin.math.abs(y2 - y1)
            // The HAL forces the ROI square (width := height) — send it square,
            // centered on the original box (camxtrackernode.cpp).
            val side = minOf(w, h)
            val roi = intArrayOf(
                (l + w / 2f - side / 2f).toInt(), (t + h / 2f - side / 2f).toInt(),
                side.toInt(), side.toInt()
            )
            val c2 = androidx.camera.camera2.interop.Camera2CameraControl.from(cc)
            c2.addCaptureRequestOptions(
                androidx.camera.camera2.interop.CaptureRequestOptions.Builder()
                    .setCaptureRequestOption(CaptureRequest.Key(
                        "org.quic.camera2.objectTrackingConfig.Enable", Byte::class.javaObjectType), 1.toByte())
                    .setCaptureRequestOption(CaptureRequest.Key(
                        "org.quic.camera2.objectTrackingConfig.RegisterROI", IntArray::class.java), roi)
                    .setCaptureRequestOption(CaptureRequest.Key(
                        "org.quic.camera2.objectTrackingConfig.CmdTrigger", Int::class.javaObjectType), 1)
                    .build()
            )
            ispTrackerActive = true
            Log.i("IspTracker", "REGISTER roi=${roi.joinToString()} (zoom=$z, normBox=$normBox)")
            // One-shot emulation: demote the trigger to Track (0) after a few frames.
            // A sticky Reg re-registers whenever zoom changes the crop-translated ROI
            // (HAL dedup compares post-translation) — that storm froze round 1.
            android.os.Handler(android.os.Looper.getMainLooper()).postDelayed({
                if (!ispTrackerActive) return@postDelayed
                try {
                    c2.addCaptureRequestOptions(
                        androidx.camera.camera2.interop.CaptureRequestOptions.Builder()
                            .setCaptureRequestOption(CaptureRequest.Key(
                                "org.quic.camera2.objectTrackingConfig.CmdTrigger", Int::class.javaObjectType), 0)
                            .build()
                    )
                    Log.i("IspTracker", "TRIGGER demoted to Track")
                } catch (e: Throwable) {
                    Log.w("IspTracker", "trigger demote failed: ${e.message}")
                }
            }, 150)
        } catch (e: Throwable) {
            Log.w("IspTracker", "register failed: ${e.message}")
        }
    }

    @Suppress("UnsafeOptInUsageError")
    fun ispTrackerCancel() {
        if (!ispTrackerProbeEnabled) return
        val cc = cameraControl ?: return
        ispTrackerActive = false
        try {
            val opts = androidx.camera.camera2.interop.CaptureRequestOptions.Builder()
                .setCaptureRequestOption(CaptureRequest.Key(
                    "org.quic.camera2.objectTrackingConfig.Enable", Byte::class.javaObjectType), 0.toByte())
                .setCaptureRequestOption(CaptureRequest.Key(
                    "org.quic.camera2.objectTrackingConfig.CmdTrigger", Int::class.javaObjectType), 2)
                .build()
            androidx.camera.camera2.interop.Camera2CameraControl.from(cc).addCaptureRequestOptions(opts)
            Log.i("IspTracker", "CANCEL")
        } catch (e: Throwable) {
            Log.w("IspTracker", "cancel failed: ${e.message}")
        }
    }

    fun getMinZoom(): Float = cameraInfo?.zoomState?.value?.minZoomRatio ?: 1f

    /**
     * Maximum zoom capped at the optical range.
     * Digital zoom degrades image quality for detection and embedding.
     */
    fun getMaxZoom(): Float {
        val hardwareMax = cameraInfo?.zoomState?.value?.maxZoomRatio ?: 1f
        return minOf(hardwareMax, opticalZoomMax)
    }

    /**
     * Detect the optical zoom range by querying Camera2 for all back-facing
     * physical camera focal lengths. The ratio of the longest to the shortest
     * focal length gives the optical zoom range.
     */
    private fun checkOisDataSupport() {
        try {
            val cam2 = context.getSystemService(Context.CAMERA_SERVICE) as Camera2Manager
            for (cameraId in cam2.cameraIdList) {
                val chars = cam2.getCameraCharacteristics(cameraId)
                val facing = chars.get(CameraCharacteristics.LENS_FACING)
                if (facing != CameraCharacteristics.LENS_FACING_BACK) continue
                val modes = chars.get(CameraCharacteristics.STATISTICS_INFO_AVAILABLE_OIS_DATA_MODES)
                val hasOisData = modes != null && modes.contains(CameraMetadata.STATISTICS_OIS_DATA_MODE_ON)
                Log.i(TAG, "Camera $cameraId OIS data modes: ${modes?.toList()}, supported=$hasOisData")
            }
        } catch (e: Exception) {
            Log.w(TAG, "Failed to check OIS data support: ${e.message}")
        }
    }

    private fun detectOpticalZoomMax(): Float {
        return try {
            val cam2 = context.getSystemService(Context.CAMERA_SERVICE) as Camera2Manager
            val focalLengths = mutableListOf<Float>()

            for (cameraId in cam2.cameraIdList) {
                val chars = cam2.getCameraCharacteristics(cameraId)
                val facing = chars.get(CameraCharacteristics.LENS_FACING)
                if (facing != CameraCharacteristics.LENS_FACING_BACK) continue

                val lengths = chars.get(CameraCharacteristics.LENS_INFO_AVAILABLE_FOCAL_LENGTHS)
                if (lengths != null) focalLengths.addAll(lengths.toList())

                // Also check physical cameras in a logical multi-camera
                val physicalIds = chars.physicalCameraIds
                for (physId in physicalIds) {
                    val physChars = cam2.getCameraCharacteristics(physId)
                    val physLengths = physChars.get(CameraCharacteristics.LENS_INFO_AVAILABLE_FOCAL_LENGTHS)
                    if (physLengths != null) focalLengths.addAll(physLengths.toList())
                }
            }

            if (focalLengths.isEmpty()) {
                Log.w(TAG, "No focal lengths found, using default optical max=$DEFAULT_OPTICAL_MAX")
                return DEFAULT_OPTICAL_MAX
            }

            val minFocal = focalLengths.min()
            val maxFocal = focalLengths.max()
            val ratio = if (minFocal > 0f) maxFocal / minFocal else DEFAULT_OPTICAL_MAX

            Log.i(TAG, "Optical zoom: focal lengths=${focalLengths.sorted()}, ratio=${String.format("%.1f", ratio)}x")
            ratio
        } catch (e: Exception) {
            Log.w(TAG, "Failed to detect optical zoom: ${e.message}")
            DEFAULT_OPTICAL_MAX
        }
    }

    fun shutdown() {
        gyroStabilizer.stop()
        frameReader?.stop()
        frameReader = null
        stabProcessor?.release()
        stabProcessor = null
        cameraProvider?.unbindAll()
    }
}
