package com.haptictrack.tracking

import android.content.Context
import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import android.graphics.Rect
import android.graphics.RectF
import android.util.Log
import java.nio.ByteBuffer
import java.nio.ByteOrder
import kotlin.math.cos
import kotlin.math.roundToInt
import kotlin.math.sqrt

class VisualTracker(private val context: Context) {

    companion object {
        private const val TAG = "VisualTracker"
        private const val MIN_CONFIDENCE = 0.4f
        private const val MIN_CONFIDENCE_SMALL = 0.25f
        private const val SMALL_ROI_AREA = 15000

        private const val BACKBONE_ASSET = "litetrack_b4_backbone_fp16.tflite"
        private const val TRACK_ASSET = "litetrack_b4_track_fp16.tflite"

        private const val TEMPLATE_SIZE = 127
        private const val SEARCH_SIZE = 256
        private const val SCORE_SIZE = 16
        private const val STRIDE = 16 // SEARCH_SIZE / SCORE_SIZE
        private const val FEAT_H = 8
        private const val FEAT_W = 8
        private const val FEAT_C = 96

        private const val MEAN_R = 0.485f
        private const val MEAN_G = 0.456f
        private const val MEAN_B = 0.406f
        private const val STD_R = 0.229f
        private const val STD_G = 0.224f
        private const val STD_B = 0.225f
    }

    private val backboneGpu: GpuInterpreter
    private val trackGpu: GpuInterpreter

    private val templateInput = ByteBuffer.allocateDirect(4 * TEMPLATE_SIZE * TEMPLATE_SIZE * 3)
        .apply { order(ByteOrder.nativeOrder()) }
    private val featBuffer = ByteBuffer.allocateDirect(4 * FEAT_H * FEAT_W * FEAT_C)
        .apply { order(ByteOrder.nativeOrder()) }
    private val searchInput = ByteBuffer.allocateDirect(4 * SEARCH_SIZE * SEARCH_SIZE * 3)
        .apply { order(ByteOrder.nativeOrder()) }

    private val clsOutput = Array(1) { Array(SCORE_SIZE) { Array(SCORE_SIZE) { FloatArray(1) } } }
    private val regOutput = Array(1) { Array(SCORE_SIZE) { Array(SCORE_SIZE) { FloatArray(4) } } }

    private val hannWindow = FloatArray(SCORE_SIZE * SCORE_SIZE)

    private var isTracking = false
    private var lastConfidence = 0f
    private var roiArea = 0

    private var targetCx = 0f
    private var targetCy = 0f
    private var targetW = 0f
    private var targetH = 0f
    private var baseS = 0f

    init {
        val bbModel = loadTfliteModel(context, BACKBONE_ASSET)
        backboneGpu = createGpuInterpreter(bbModel, modelName = "LiteTrack-Backbone")

        val trModel = loadTfliteModel(context, TRACK_ASSET)
        trackGpu = createGpuInterpreter(trModel, modelName = "LiteTrack-Track")

        val hann1d = FloatArray(SCORE_SIZE) { i ->
            (0.5 * (1.0 - cos(2.0 * Math.PI * i / (SCORE_SIZE - 1)))).toFloat()
        }
        for (y in 0 until SCORE_SIZE) {
            for (x in 0 until SCORE_SIZE) {
                hannWindow[y * SCORE_SIZE + x] = hann1d[y] * hann1d[x]
            }
        }

        Log.i(TAG, "LiteTrack-B4 loaded (template=$TEMPLATE_SIZE, search=$SEARCH_SIZE)")
    }

    fun init(bitmap: Bitmap, normalizedBox: RectF): Boolean {
        try {
            val cx = (normalizedBox.left + normalizedBox.right) / 2f * bitmap.width
            val cy = (normalizedBox.top + normalizedBox.bottom) / 2f * bitmap.height
            val w = normalizedBox.width() * bitmap.width
            val h = normalizedBox.height() * bitmap.height

            val context = (w + h) / 2f
            baseS = sqrt((w + context) * (h + context))

            val crop = cropAndResize(bitmap, cx, cy, baseS, TEMPLATE_SIZE)
            fillBuffer(templateInput, crop, TEMPLATE_SIZE)
            crop.recycle()

            templateInput.rewind()
            featBuffer.rewind()
            backboneGpu.interpreter.run(templateInput, featBuffer)
            featBuffer.rewind()

            val probe = featBuffer.getFloat(0)
            featBuffer.rewind()
            if (probe.isNaN()) {
                Log.e(TAG, "Backbone output NaN — GPU delegate failure")
                return false
            }

            targetCx = cx
            targetCy = cy
            targetW = w
            targetH = h
            isTracking = true
            lastConfidence = 1.0f
            roiArea = (w * h).toInt()

            Log.d(TAG, "INIT box=${fmtBox(normalizedBox)} s=${baseS.toInt()}")
            return true
        } catch (e: Exception) {
            Log.e(TAG, "Init failed: ${e.message}")
            isTracking = false
            return false
        }
    }

    fun update(bitmap: Bitmap): TrackerResult? {
        if (!isTracking) return null

        try {
            val sX = baseS * (SEARCH_SIZE.toFloat() / TEMPLATE_SIZE)

            val crop = cropAndResize(bitmap, targetCx, targetCy, sX, SEARCH_SIZE)
            fillBuffer(searchInput, crop, SEARCH_SIZE)
            crop.recycle()

            featBuffer.rewind()
            searchInput.rewind()
            trackGpu.interpreter.runForMultipleInputsOutputs(
                arrayOf<Any>(featBuffer, searchInput),
                mapOf<Int, Any>(0 to clsOutput, 1 to regOutput)
            )

            var bestScore = -1f
            var bestX = 0
            var bestY = 0
            for (y in 0 until SCORE_SIZE) {
                for (x in 0 until SCORE_SIZE) {
                    val score = sigmoid(clsOutput[0][y][x][0]) * hannWindow[y * SCORE_SIZE + x]
                    if (score > bestScore) {
                        bestScore = score
                        bestX = x
                        bestY = y
                    }
                }
            }

            lastConfidence = bestScore
            val confidenceFloor = if (roiArea < SMALL_ROI_AREA) MIN_CONFIDENCE_SMALL else MIN_CONFIDENCE
            if (bestScore < confidenceFloor) {
                Log.d(TAG, "LOST confidence=${fmtF(bestScore)} (floor=${fmtF(confidenceFloor)}, roi=$roiArea)")
                return null
            }

            // LTRB distances from grid cell center in search-image pixels
            val reg = regOutput[0][bestY][bestX]
            val cellCx = (bestX + 0.5f) * STRIDE
            val cellCy = (bestY + 0.5f) * STRIDE

            val searchLeft = cellCx - reg[0]
            val searchTop = cellCy - reg[1]
            val searchRight = cellCx + reg[2]
            val searchBottom = cellCy + reg[3]

            val searchBboxW = searchRight - searchLeft
            val searchBboxH = searchBottom - searchTop
            val searchBboxCx = (searchLeft + searchRight) / 2f
            val searchBboxCy = (searchTop + searchBottom) / 2f

            val scale = sX / SEARCH_SIZE
            val newCx = targetCx + (searchBboxCx - SEARCH_SIZE / 2f) * scale
            val newCy = targetCy + (searchBboxCy - SEARCH_SIZE / 2f) * scale
            val newW = (searchBboxW * scale).coerceAtLeast(1f)
            val newH = (searchBboxH * scale).coerceAtLeast(1f)

            targetCx = newCx
            targetCy = newCy
            targetW = newW
            targetH = newH

            val ctx = (targetW + targetH) / 2f
            baseS = sqrt((targetW + ctx) * (targetH + ctx))

            roiArea = (targetW * targetH).toInt()

            val normBox = RectF(
                (newCx - newW / 2f) / bitmap.width,
                (newCy - newH / 2f) / bitmap.height,
                (newCx + newW / 2f) / bitmap.width,
                (newCy + newH / 2f) / bitmap.height
            )

            return TrackerResult(normBox, bestScore)
        } catch (e: Exception) {
            Log.w(TAG, "Update failed: ${e.message}")
            return null
        }
    }

    fun stop() {
        isTracking = false
        lastConfidence = 0f
    }

    val isActive: Boolean get() = isTracking

    data class TrackerResult(
        val boundingBox: RectF,
        val confidence: Float
    )

    private fun cropAndResize(bitmap: Bitmap, cx: Float, cy: Float, cropSize: Float, outSize: Int): Bitmap {
        val half = cropSize / 2f
        val srcLeft = (cx - half).roundToInt()
        val srcTop = (cy - half).roundToInt()
        val cropW = cropSize.roundToInt().coerceAtLeast(1)
        val cropH = cropW

        val meanColor = Color.rgb(
            (MEAN_R * 255).toInt(),
            (MEAN_G * 255).toInt(),
            (MEAN_B * 255).toInt()
        )
        val cropBmp = Bitmap.createBitmap(cropW, cropH, Bitmap.Config.ARGB_8888)
        val canvas = Canvas(cropBmp)
        canvas.drawColor(meanColor)

        val visLeft = srcLeft.coerceAtLeast(0)
        val visTop = srcTop.coerceAtLeast(0)
        val visRight = (srcLeft + cropW).coerceAtMost(bitmap.width)
        val visBottom = (srcTop + cropH).coerceAtMost(bitmap.height)

        if (visLeft < visRight && visTop < visBottom) {
            val src = Rect(visLeft, visTop, visRight, visBottom)
            val dst = Rect(visLeft - srcLeft, visTop - srcTop,
                visLeft - srcLeft + src.width(), visTop - srcTop + src.height())
            canvas.drawBitmap(bitmap, src, dst, null)
        }

        val resized = Bitmap.createScaledBitmap(cropBmp, outSize, outSize, true)
        if (resized !== cropBmp) cropBmp.recycle()
        return resized
    }

    private fun fillBuffer(buf: ByteBuffer, bitmap: Bitmap, size: Int) {
        buf.rewind()
        val pixels = IntArray(size * size)
        bitmap.getPixels(pixels, 0, size, 0, 0, size, size)
        for (pixel in pixels) {
            buf.putFloat((Color.red(pixel) / 255f - MEAN_R) / STD_R)
            buf.putFloat((Color.green(pixel) / 255f - MEAN_G) / STD_G)
            buf.putFloat((Color.blue(pixel) / 255f - MEAN_B) / STD_B)
        }
        buf.rewind()
    }

    private fun sigmoid(x: Float): Float = 1f / (1f + kotlin.math.exp(-x))
    private fun fmtF(f: Float) = "%.3f".format(f)
    private fun fmtBox(b: RectF) = "[%.2f,%.2f,%.2f,%.2f]".format(b.left, b.top, b.right, b.bottom)
}
