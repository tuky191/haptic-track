package com.haptictrack.camera

import android.content.Context
import android.hardware.camera2.CameraCharacteristics
import android.hardware.camera2.CameraManager as Camera2Manager
import android.util.Log
import androidx.camera.core.CameraControl
import androidx.camera.core.CameraInfo
import androidx.camera.core.CameraSelector
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.Preview
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.camera.core.resolutionselector.ResolutionSelector
import androidx.camera.core.resolutionselector.ResolutionStrategy
import androidx.camera.video.FallbackStrategy
import androidx.camera.video.Quality
import androidx.camera.video.QualitySelector
import androidx.camera.video.Recorder
import androidx.camera.video.VideoCapture
import androidx.camera.view.PreviewView
import androidx.lifecycle.LifecycleOwner
import android.os.Handler
import android.os.Looper
import android.util.Size
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

    /** Current lens facing — back by default. */
    var isFrontCamera: Boolean = false
        private set

    /** When true, ImageAnalysis is unbound to free a stream for higher video resolution. */
    private var recordingMode: Boolean = false

    /** Whether ImageAnalysis is currently active (false during recording). */
    var analysisActive: Boolean = true
        private set

    /** Reads frames from Preview surface via OpenGL during recording mode. */
    private var frameReader: SurfaceTextureFrameReader? = null

    /** Callback for frames read from the Preview surface during recording (processing thread). */
    var onRecordingFrame: ((android.graphics.Bitmap) -> Unit)? = null

    /** Callback for viewfinder display frames during recording (GL thread, ~29fps). */
    var onViewfinderFrame: ((android.graphics.Bitmap) -> Unit)? = null

    val preview = Preview.Builder().build()

    val imageAnalysis: ImageAnalysis = ImageAnalysis.Builder()
        .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
        .setResolutionSelector(
            ResolutionSelector.Builder()
                .setResolutionStrategy(
                    ResolutionStrategy(Size(640, 480), ResolutionStrategy.FALLBACK_RULE_CLOSEST_LOWER_THEN_HIGHER)
                )
                .build()
        )
        .build()

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
    }

    fun startCamera(lifecycleOwner: LifecycleOwner, previewView: PreviewView) {
        lifecycleOwnerRef = lifecycleOwner
        previewViewRef = previewView
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

    /**
     * Switch to recording mode (2 streams: preview + video → higher video resolution).
     * Analysis frames must be obtained via PreviewView.getBitmap().
     */
    fun enterRecordingMode() {
        if (recordingMode) return
        recordingMode = true
        val owner = lifecycleOwnerRef ?: return
        val view = previewViewRef ?: return
        bindUseCases(owner, view)
    }

    /**
     * Switch back to tracking mode (3 streams: preview + analysis + video).
     * ImageAnalysis provides full-quality frames for detection.
     */
    fun exitRecordingMode() {
        if (!recordingMode) return
        recordingMode = false
        val owner = lifecycleOwnerRef ?: return
        val view = previewViewRef ?: return
        bindUseCases(owner, view)
    }

    private fun bindUseCases(lifecycleOwner: LifecycleOwner, previewView: PreviewView) {
        val provider = cameraProvider ?: return
        provider.unbindAll()

        // Stop any existing frame reader
        frameReader?.stop()
        frameReader = null

        // Recreate VideoCapture to avoid resolution mismatch when switching modes
        videoCapture = createVideoCapture()

        val selector = if (isFrontCamera) CameraSelector.DEFAULT_FRONT_CAMERA
                       else CameraSelector.DEFAULT_BACK_CAMERA

        val camera = if (recordingMode) {
            // 2 streams: preview + video → 4K video
            // Preview goes to SurfaceTextureFrameReader for fast off-thread frame capture
            // instead of PreviewView.getBitmap() (slow, main-thread, 7-10fps)
            preview.surfaceProvider = Preview.SurfaceProvider { request ->
                val inputSize = request.resolution // camera's native buffer size (landscape, e.g. 1600x1200)
                // Output in portrait at analysis resolution.
                // Camera outputs landscape; transform matrix rotates 90°.
                // So output width = min dim scaled, height = max dim scaled.
                val analysisShort = 480 // short edge of output
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
                    onFrame = { bitmap -> onRecordingFrame?.invoke(bitmap) },
                    onViewfinderFrame = { bitmap -> onViewfinderFrame?.invoke(bitmap) }
                )
                val readerSurface = reader.start()
                frameReader = reader

                request.provideSurface(readerSurface, java.util.concurrent.Executors.newSingleThreadExecutor()) { result ->
                    Log.d(TAG, "Preview surface result: ${result.resultCode}")
                }
            }
            provider.bindToLifecycle(lifecycleOwner, selector, preview, videoCapture)
        } else {
            // 3 streams: preview + analysis + video → full-quality analysis
            preview.surfaceProvider = previewView.surfaceProvider
            provider.bindToLifecycle(lifecycleOwner, selector, preview, imageAnalysis, videoCapture)
        }
        analysisActive = !recordingMode

        cameraControl = camera.cameraControl
        cameraInfo = camera.cameraInfo

        val previewRes = preview.resolutionInfo?.resolution
        val analysisRes = if (!recordingMode) imageAnalysis.resolutionInfo?.resolution else null
        Log.i(TAG, "Bound use cases — preview: $previewRes, analysis: $analysisRes, frameReader: ${frameReader != null}, mode: ${if (recordingMode) "recording (2-stream)" else "tracking (3-stream)"}")
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
        frameReader?.stop()
        frameReader = null
        cameraProvider?.unbindAll()
    }
}
