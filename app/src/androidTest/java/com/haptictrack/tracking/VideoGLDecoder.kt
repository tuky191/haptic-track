package com.haptictrack.tracking

import android.graphics.Bitmap
import android.media.MediaCodec
import android.media.MediaExtractor
import android.media.MediaFormat
import android.util.Log
import com.haptictrack.camera.SurfaceTextureFrameReader
import java.io.File
import java.util.concurrent.LinkedBlockingQueue
import java.util.concurrent.TimeUnit

/**
 * Decodes an MP4 video file through the production [SurfaceTextureFrameReader]
 * GL pipeline. MediaCodec renders into the same Surface that production camera
 * Preview targets, and the existing shader + PBO readback produces ARGB bitmaps —
 * matching what live camera delivers to ObjectTracker.processBitmap.
 *
 * Replaces an earlier pure-Kotlin YUV→ARGB loop that cost 300-500ms per 1080p
 * frame on Snapdragon 870. The hardware GL path costs ~5ms per frame and
 * matches what live camera produces, so test frames now exercise the same
 * conversion code path as production.
 *
 * The PBO ping-pong in SurfaceTextureFrameReader adds a 1-frame latency
 * (the bitmap that emerges corresponds to the previous render). The last
 * frame of the video is therefore lost — acceptable for aggregate-metric
 * assertions (tracking rate, reacq counts).
 *
 * Synchronous pump: each iteration feeds one input, renders one output,
 * waits for the previous-frame bitmap, invokes [onFrame], then loops.
 * The blocking queue parks the GL/processing threads when the test is busy.
 */
class VideoGLDecoder(private val videoFile: File) {

    companion object {
        private const val TAG = "VideoGLDec"
        private const val DEQUEUE_TIMEOUT_US = 10_000L
        private const val BITMAP_TIMEOUT_MS = 2_000L
    }

    private val extractor = MediaExtractor()
    private var videoTrackIndex = -1
    private var inputWidth = 0
    private var inputHeight = 0
    private var mime: String = ""

    private var stf: SurfaceTextureFrameReader? = null
    private var codec: MediaCodec? = null

    init {
        extractor.setDataSource(videoFile.absolutePath)
        for (i in 0 until extractor.trackCount) {
            val format = extractor.getTrackFormat(i)
            val trackMime = format.getString(MediaFormat.KEY_MIME) ?: continue
            if (trackMime.startsWith("video/")) {
                videoTrackIndex = i
                inputWidth = format.getInteger(MediaFormat.KEY_WIDTH)
                inputHeight = format.getInteger(MediaFormat.KEY_HEIGHT)
                mime = trackMime
                break
            }
        }
        require(videoTrackIndex >= 0) { "No video track found in ${videoFile.name}" }
    }

    /**
     * Return a recycler suitable for [ObjectTracker.bitmapRecycler]. Routes
     * the bitmap back to the GL pipeline's pool. Safe to call after [release].
     */
    fun bitmapRecycler(): (Bitmap) -> Unit = { bm -> stf?.releaseAnalysisBitmap(bm) }

    /**
     * Decode the video, calling [onFrame] for each non-skipped frame.
     *
     * @param targetWidth Output bitmaps are downscaled so the longer dimension
     *                    is at most this size, preserving aspect ratio.
     * @param onFrame Receives (frameIndex, bitmap). Returns frames-to-skip;
     *                the next N output buffers are released without rendering
     *                so they don't reach the GL pipeline. Bitmap ownership
     *                stays with the GL pool — release via [bitmapRecycler]
     *                or directly through the recycler.
     */
    fun decodeAll(targetWidth: Int = 640, onFrame: (frameIndex: Int, bitmap: Bitmap) -> Int) {
        extractor.selectTrack(videoTrackIndex)
        val format = extractor.getTrackFormat(videoTrackIndex)
        Log.i(TAG, "Video: ${videoFile.name} ${inputWidth}x${inputHeight} mime=$mime")

        val (outW, outH) = computeOutputSize(inputWidth, inputHeight, targetWidth)
        Log.i(TAG, "Output: ${outW}x${outH}")

        val bitmapQueue = LinkedBlockingQueue<Bitmap>(1)
        val frameReader = SurfaceTextureFrameReader(
            inputWidth = inputWidth,
            inputHeight = inputHeight,
            outputWidth = outW,
            outputHeight = outH,
            onFrame = { bm ->
                // Block here if test thread hasn't taken the previous bitmap yet.
                // That parks the GL/processing threads, which is what we want for
                // strict-sync pumping — no frame loss between renders.
                bitmapQueue.put(bm)
            }
        )
        stf = frameReader
        val surface = frameReader.start()

        val mediaCodec = MediaCodec.createDecoderByType(mime)
        mediaCodec.configure(format, surface, null, 0)
        mediaCodec.start()
        codec = mediaCodec

        val bufferInfo = MediaCodec.BufferInfo()
        var inputDone = false
        var outputDone = false
        // videoFrameIdx is the absolute index into the video stream (counts BOTH
        // skipped and rendered frames) so callers can compute drop rates correctly.
        var videoFrameIdx = 0
        var renderedCount = 0
        var skipCount = 0  // pre-render: drop next N output buffers without rendering
        var pendingPboFrame = -1  // videoFrameIdx of the frame in the PBO queue
        var deliveredCount = 0
        var bitmapsLost = 0

        try {
            while (!outputDone) {
                // Feed input
                if (!inputDone) {
                    val ii = mediaCodec.dequeueInputBuffer(DEQUEUE_TIMEOUT_US)
                    if (ii >= 0) {
                        val ib = mediaCodec.getInputBuffer(ii)!!
                        val ss = extractor.readSampleData(ib, 0)
                        if (ss < 0) {
                            mediaCodec.queueInputBuffer(ii, 0, 0, 0,
                                MediaCodec.BUFFER_FLAG_END_OF_STREAM)
                            inputDone = true
                        } else {
                            mediaCodec.queueInputBuffer(ii, 0, ss,
                                extractor.sampleTime, 0)
                            extractor.advance()
                        }
                    }
                }

                // Drain output
                val oi = mediaCodec.dequeueOutputBuffer(bufferInfo, DEQUEUE_TIMEOUT_US)
                when {
                    oi >= 0 -> {
                        val isEos = (bufferInfo.flags and
                            MediaCodec.BUFFER_FLAG_END_OF_STREAM) != 0
                        if (isEos) {
                            mediaCodec.releaseOutputBuffer(oi, false)
                            outputDone = true
                            if (pendingPboFrame >= 0) bitmapsLost++
                            break
                        }

                        if (skipCount > 0) {
                            mediaCodec.releaseOutputBuffer(oi, false)  // don't render
                            skipCount--
                            videoFrameIdx++
                            continue
                        }

                        // Render to surface — STFrameReader will eventually deliver
                        // a bitmap one render behind (PBO lag).
                        mediaCodec.releaseOutputBuffer(oi, true)
                        val thisFrameIdx = videoFrameIdx
                        videoFrameIdx++
                        renderedCount++

                        if (pendingPboFrame < 0) {
                            // First render — no bitmap available yet (PBO lag).
                            pendingPboFrame = thisFrameIdx
                            continue
                        }

                        val bm = bitmapQueue.poll(BITMAP_TIMEOUT_MS, TimeUnit.MILLISECONDS)
                        if (bm == null) {
                            Log.w(TAG, "Timed out waiting for bitmap from GL pipeline " +
                                "(rendered=$renderedCount, delivered=$deliveredCount)")
                            outputDone = true
                            break
                        }

                        val skips = onFrame(pendingPboFrame, bm)
                        deliveredCount++
                        if (skips > 0) {
                            // Drop the next `skips` codec outputs without rendering.
                            // videoFrameIdx still advances for them so frame numbering
                            // matches the source video.
                            skipCount = skips
                        }
                        pendingPboFrame = thisFrameIdx
                    }
                    oi == MediaCodec.INFO_OUTPUT_FORMAT_CHANGED -> {
                        Log.d(TAG, "Output format changed: ${mediaCodec.outputFormat}")
                    }
                    // INFO_TRY_AGAIN_LATER and INFO_OUTPUT_BUFFERS_CHANGED: keep looping
                }
            }
        } finally {
            Log.i(TAG, "decodeAll done: video frames=$videoFrameIdx rendered=$renderedCount " +
                "delivered=$deliveredCount lostToPboLag=$bitmapsLost from ${videoFile.name}")
        }
    }

    fun release() {
        try {
            codec?.stop()
        } catch (t: Throwable) {
            Log.w(TAG, "codec.stop threw: ${t.message}")
        }
        codec?.release()
        codec = null
        stf?.stop()
        stf = null
        extractor.release()
    }

    /**
     * Scale so that the longer dimension is at most [target], preserving aspect.
     * Even dimensions matter for MediaCodec on some devices; round to even.
     */
    private fun computeOutputSize(inputW: Int, inputH: Int, target: Int): Pair<Int, Int> {
        if (inputW <= 0 || inputH <= 0) return Pair(target, target)
        val outW: Int
        val outH: Int
        if (inputW >= inputH) {
            outW = target
            outH = (target * inputH / inputW).coerceAtLeast(1)
        } else {
            outH = target
            outW = (target * inputW / inputH).coerceAtLeast(1)
        }
        // MediaCodec output Surface texture sizing prefers even dims on some HALs.
        return Pair(outW and 1.inv(), outH and 1.inv())
    }
}
