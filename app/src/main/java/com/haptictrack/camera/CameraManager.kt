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

    /** Detected optical zoom limit from physical camera focal lengths. */
    private var opticalZoomMax: Float = DEFAULT_OPTICAL_MAX

    val preview = Preview.Builder().build()

    val imageAnalysis: ImageAnalysis = ImageAnalysis.Builder()
        .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
        .build()

    private val recorder = Recorder.Builder()
        .setQualitySelector(QualitySelector.from(Quality.HIGHEST))
        .setExecutor(Executors.newSingleThreadExecutor())
        .build()

    val videoCapture = VideoCapture.withOutput(recorder)

    init {
        opticalZoomMax = detectOpticalZoomMax()
    }

    fun startCamera(lifecycleOwner: LifecycleOwner, previewView: PreviewView) {
        val providerFuture = ProcessCameraProvider.getInstance(context)
        providerFuture.addListener({
            cameraProvider = providerFuture.get()
            bindUseCases(lifecycleOwner, previewView)
        }, mainExecutor)
    }

    private fun bindUseCases(lifecycleOwner: LifecycleOwner, previewView: PreviewView) {
        val provider = cameraProvider ?: return
        provider.unbindAll()

        preview.surfaceProvider = previewView.surfaceProvider

        val camera = provider.bindToLifecycle(
            lifecycleOwner,
            CameraSelector.DEFAULT_BACK_CAMERA,
            preview,
            imageAnalysis,
            videoCapture
        )

        cameraControl = camera.cameraControl
        cameraInfo = camera.cameraInfo
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
        cameraProvider?.unbindAll()
    }
}
