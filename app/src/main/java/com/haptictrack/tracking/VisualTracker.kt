package com.haptictrack.tracking

import android.content.Context
import android.graphics.Bitmap
import android.graphics.RectF
import android.util.Log
import org.opencv.android.OpenCVLoader
import org.opencv.android.Utils
import org.opencv.core.Mat
import org.opencv.core.Rect
import org.opencv.imgproc.Imgproc
import org.opencv.video.TrackerVit
import org.opencv.video.TrackerVit_Params
import java.io.File
import java.io.FileOutputStream

/**
 * Frame-to-frame visual tracker using OpenCV VitTracker.
 *
 * Replaced TrackerNano — VitTracker is smaller (698KB vs 1.7MB), 20% faster on ARM,
 * and critically returns meaningful confidence scores (TrackerNano always returned ~0.9).
 * This allows direct confidence-based lost detection instead of the 10-frame
 * detector cross-check heuristic.
 */
class VisualTracker(private val context: Context) {

    companion object {
        private const val TAG = "VisualTracker"
        private const val MIN_CONFIDENCE = 0.4f  // VitTracker returns real scores — lower threshold works
        private const val MODEL_ASSET = "object_tracking_vittrack_2023sep.onnx"

        @Volatile
        private var opencvInitialized = false
    }

    private var tracker: TrackerVit? = null
    private var isTracking = false
    private var lastConfidence = 0f

    private val modelPath: String by lazy { copyAssetToFile(MODEL_ASSET) }

    init {
        if (!opencvInitialized) {
            synchronized(VisualTracker::class.java) {
                if (!opencvInitialized) {
                    opencvInitialized = OpenCVLoader.initLocal()
                    if (opencvInitialized) {
                        Log.i(TAG, "OpenCV initialized successfully")
                    } else {
                        Log.e(TAG, "OpenCV initialization failed")
                    }
                }
            }
        }
    }

    fun init(bitmap: Bitmap, normalizedBox: RectF): Boolean {
        if (!opencvInitialized) return false

        try {
            val mat = bitmapToMat(bitmap)
            val roi = toPixelRect(normalizedBox, bitmap.width, bitmap.height)

            val clampedRoi = Rect(
                roi.x.coerceIn(0, bitmap.width - 2),
                roi.y.coerceIn(0, bitmap.height - 2),
                roi.width.coerceIn(1, bitmap.width - roi.x.coerceIn(0, bitmap.width - 2)),
                roi.height.coerceIn(1, bitmap.height - roi.y.coerceIn(0, bitmap.height - 2))
            )

            val params = TrackerVit_Params().apply {
                set_net(modelPath)
            }
            tracker = TrackerVit.create(params)
            tracker!!.init(mat, clampedRoi)
            isTracking = true
            lastConfidence = 1.0f

            mat.release()
            Log.d(TAG, "INIT box=${fmtBox(normalizedBox)} roi=${clampedRoi}")
            return true
        } catch (e: Exception) {
            Log.e(TAG, "Init failed: ${e.message}")
            isTracking = false
            return false
        }
    }

    fun update(bitmap: Bitmap): TrackerResult? {
        if (!isTracking || tracker == null) return null

        try {
            val mat = bitmapToMat(bitmap)
            val outRect = Rect()
            val found = tracker!!.update(mat, outRect)
            lastConfidence = tracker!!.getTrackingScore()

            mat.release()

            if (!found || lastConfidence < MIN_CONFIDENCE) {
                Log.d(TAG, "LOST confidence=${fmtF(lastConfidence)}")
                return null
            }

            val normBox = RectF(
                outRect.x.toFloat() / bitmap.width,
                outRect.y.toFloat() / bitmap.height,
                (outRect.x + outRect.width).toFloat() / bitmap.width,
                (outRect.y + outRect.height).toFloat() / bitmap.height
            )

            return TrackerResult(normBox, lastConfidence)
        } catch (e: Exception) {
            Log.w(TAG, "Update failed: ${e.message}")
            return null
        }
    }

    fun stop() {
        isTracking = false
        tracker = null
        lastConfidence = 0f
    }

    val isActive: Boolean get() = isTracking

    data class TrackerResult(
        val boundingBox: RectF,
        val confidence: Float
    )

    private fun bitmapToMat(bitmap: Bitmap): Mat {
        val mat = Mat()
        val rgbaBitmap = if (bitmap.config != Bitmap.Config.ARGB_8888) {
            bitmap.copy(Bitmap.Config.ARGB_8888, false)
        } else bitmap
        Utils.bitmapToMat(rgbaBitmap, mat)
        if (rgbaBitmap !== bitmap) rgbaBitmap.recycle()
        Imgproc.cvtColor(mat, mat, Imgproc.COLOR_RGBA2BGR)
        return mat
    }

    private fun toPixelRect(normBox: RectF, width: Int, height: Int): Rect {
        val x = (normBox.left * width).toInt()
        val y = (normBox.top * height).toInt()
        val w = ((normBox.right - normBox.left) * width).toInt()
        val h = ((normBox.bottom - normBox.top) * height).toInt()
        return Rect(x, y, w, h)
    }

    private fun copyAssetToFile(assetName: String): String {
        val file = File(context.filesDir, assetName)
        if (!file.exists()) {
            context.assets.open(assetName).use { input ->
                FileOutputStream(file).use { output ->
                    input.copyTo(output)
                }
            }
        }
        return file.absolutePath
    }

    private fun fmtF(f: Float) = "%.3f".format(f)
    private fun fmtBox(b: RectF) = "[%.2f,%.2f,%.2f,%.2f]".format(b.left, b.top, b.right, b.bottom)
}
