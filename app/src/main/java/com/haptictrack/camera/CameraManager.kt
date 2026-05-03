package com.haptictrack.camera

import android.content.Context
import android.hardware.camera2.CameraCharacteristics
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
    var ispStabilizationEnabled: Boolean = true

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

    var preview = Preview.Builder().build()
        private set

    private val videoExecutor = Executors.newSingleThreadExecutor()
    var videoCapture = createVideoCapture()
        private set

    private fun createVideoCapture(): VideoCapture<Recorder> {
        val recorder = Recorder.Builder()
            .setQualitySelector(
                QualitySelector.fromOrderedList(
                    listOf(Quality.UHD, Quality.FHD, Quality.HD),
                    FallbackStrategy.higherQualityOrLowerThan(Quality.FHD)
                )
            )
            .setExecutor(videoExecutor)
            .build()
        return VideoCapture.withOutput(recorder)
    }

    init {
        opticalZoomMax = detectOpticalZoomMax()
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
        if (ispStabilizationEnabled && !gyroStabilizer.enabled) {
            try {
                val caps = Preview.getPreviewCapabilities(provider.getCameraInfo(selector))
                if (caps.isStabilizationSupported) {
                    previewBuilder.setPreviewStabilizationEnabled(true)
                    Log.i(TAG, "ISP stabilization ON")
                } else {
                    Log.i(TAG, "ISP stabilization requested but not supported on this device")
                }
            } catch (e: Exception) {
                Log.w(TAG, "Failed to query stabilization caps: ${e.message}")
            }
        } else if (ispStabilizationEnabled && gyroStabilizer.enabled) {
            Log.i(TAG, "ISP stabilization OFF (gyro EIS takes over)")
        } else {
            Log.i(TAG, "ISP stabilization OFF (user toggle)")
        }
        @Suppress("UnsafeOptInUsageError")
        if (gyroStabilizer.enabled) {
            Camera2Interop.Extender(previewBuilder)
                .setCaptureRequestOption(
                    CaptureRequest.LENS_OPTICAL_STABILIZATION_MODE,
                    CaptureRequest.LENS_OPTICAL_STABILIZATION_MODE_OFF
                )
            Log.i(TAG, "OIS disabled (gyro EIS handles stabilization)")
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
                stabMatrixProvider = { gyroStabilizer.getMatrix() }
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

        if (gyroStabilizer.enabled) {
            val processor = StabilizationProcessor(
                stabMatrixProvider = { gyroStabilizer.getMatrix() },
                frameTimestampLogger = { idx, ts -> gyroStabilizer.logFrameTimestamp(idx, ts) }
            )
            stabProcessor = processor
            useCaseGroupBuilder.addEffect(StabilizationEffect(processor))
            Log.i(TAG, "Video stabilization effect added to VideoCapture pipeline")
        }

        val camera = provider.bindToLifecycle(lifecycleOwner, selector, useCaseGroupBuilder.build())

        cameraControl = camera.cameraControl
        cameraInfo = camera.cameraInfo

        val previewRes = preview.resolutionInfo?.resolution
        Log.i(TAG, "Bound use cases — preview: $previewRes, frameReader: ${frameReader != null}, gyroVideo: ${stabProcessor != null}")
    }

    fun setZoomRatio(ratio: Float) {
        cameraControl?.setZoomRatio(ratio.coerceIn(getMinZoom(), getMaxZoom()))
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
