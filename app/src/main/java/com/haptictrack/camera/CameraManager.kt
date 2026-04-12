package com.haptictrack.camera

import android.content.Context
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

    private val mainHandler = Handler(Looper.getMainLooper())
    private val mainExecutor = java.util.concurrent.Executor { runnable -> mainHandler.post(runnable) }

    private var cameraProvider: ProcessCameraProvider? = null
    private var cameraControl: CameraControl? = null
    private var cameraInfo: CameraInfo? = null

    val preview = Preview.Builder().build()

    val imageAnalysis: ImageAnalysis = ImageAnalysis.Builder()
        .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
        .build()

    private val recorder = Recorder.Builder()
        .setQualitySelector(QualitySelector.from(Quality.HIGHEST))
        .setExecutor(Executors.newSingleThreadExecutor())
        .build()

    val videoCapture = VideoCapture.withOutput(recorder)

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

    fun getMaxZoom(): Float = cameraInfo?.zoomState?.value?.maxZoomRatio ?: 1f

    fun shutdown() {
        cameraProvider?.unbindAll()
    }
}
