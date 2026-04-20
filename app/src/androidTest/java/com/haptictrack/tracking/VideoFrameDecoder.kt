package com.haptictrack.tracking

import android.graphics.Bitmap
import android.graphics.ImageFormat
import android.media.Image
import android.media.MediaCodec
import android.media.MediaExtractor
import android.media.MediaFormat
import android.util.Log
import java.io.File

/**
 * Decodes an MP4 video file into a sequence of Bitmaps using MediaExtractor + MediaCodec.
 *
 * Supports frame skipping: the callback returns how many frames to skip before
 * the next decode. Skipped frames are drained from the codec without the expensive
 * YUV→Bitmap conversion, matching live camera behavior where the pipeline processes
 * one frame and drops any that arrived during processing.
 */
class VideoFrameDecoder(private val videoFile: File) {

    companion object {
        private const val TAG = "VideoFrameDec"
        private const val TIMEOUT_US = 10_000L
    }

    private val extractor = MediaExtractor()
    private var decoder: MediaCodec? = null
    private var videoTrackIndex = -1

    init {
        extractor.setDataSource(videoFile.absolutePath)
        for (i in 0 until extractor.trackCount) {
            val format = extractor.getTrackFormat(i)
            val mime = format.getString(MediaFormat.KEY_MIME) ?: continue
            if (mime.startsWith("video/")) {
                videoTrackIndex = i
                break
            }
        }
        require(videoTrackIndex >= 0) { "No video track found in ${videoFile.name}" }
    }

    /**
     * Decode frames from the video, calling [onFrame] for each processed frame.
     *
     * @param targetWidth Decoded frames are downscaled to approximately this width.
     * @param onFrame Called with (frameIndex, bitmap). Returns the number of frames
     *                to skip after this one (0 = process next frame, 3 = skip 3 frames).
     *                The bitmap is NOT recycled by the decoder — caller is responsible.
     */
    fun decodeAll(targetWidth: Int = 640, onFrame: (frameIndex: Int, bitmap: Bitmap) -> Int) {
        extractor.selectTrack(videoTrackIndex)
        val format = extractor.getTrackFormat(videoTrackIndex)
        val mime = format.getString(MediaFormat.KEY_MIME)!!
        val width = format.getInteger(MediaFormat.KEY_WIDTH)
        val height = format.getInteger(MediaFormat.KEY_HEIGHT)
        Log.i(TAG, "Video: ${videoFile.name} ${width}x${height} mime=$mime")

        val codec = MediaCodec.createDecoderByType(mime)
        format.setInteger(MediaFormat.KEY_COLOR_FORMAT,
            android.media.MediaCodecInfo.CodecCapabilities.COLOR_FormatYUV420Flexible)
        codec.configure(format, null, null, 0)
        codec.start()
        decoder = codec

        val bufferInfo = MediaCodec.BufferInfo()
        var inputDone = false
        var frameIndex = 0
        var framesToSkip = 0
        var framesDecoded = 0
        var framesSkipped = 0

        while (true) {
            // Feed input
            if (!inputDone) {
                val inputIndex = codec.dequeueInputBuffer(TIMEOUT_US)
                if (inputIndex >= 0) {
                    val inputBuffer = codec.getInputBuffer(inputIndex)!!
                    val sampleSize = extractor.readSampleData(inputBuffer, 0)
                    if (sampleSize < 0) {
                        codec.queueInputBuffer(inputIndex, 0, 0, 0,
                            MediaCodec.BUFFER_FLAG_END_OF_STREAM)
                        inputDone = true
                    } else {
                        codec.queueInputBuffer(inputIndex, 0, sampleSize,
                            extractor.sampleTime, 0)
                        extractor.advance()
                    }
                }
            }

            // Drain output
            val outputIndex = codec.dequeueOutputBuffer(bufferInfo, TIMEOUT_US)
            if (outputIndex >= 0) {
                if (bufferInfo.flags and MediaCodec.BUFFER_FLAG_END_OF_STREAM != 0) {
                    codec.releaseOutputBuffer(outputIndex, false)
                    break
                }

                if (framesToSkip > 0) {
                    // Skip this frame — release without converting to bitmap
                    codec.releaseOutputBuffer(outputIndex, false)
                    framesToSkip--
                    framesSkipped++
                    frameIndex++
                    continue
                }

                // Process this frame
                val image = codec.getOutputImage(outputIndex)
                if (image != null) {
                    var bitmap = imageToBitmap(image)
                    image.close()

                    if (targetWidth > 0 && bitmap.width > targetWidth) {
                        val scale = targetWidth.toFloat() / bitmap.width
                        val scaled = Bitmap.createScaledBitmap(bitmap,
                            (bitmap.width * scale).toInt(),
                            (bitmap.height * scale).toInt(), true)
                        bitmap.recycle()
                        bitmap = scaled
                    }

                    framesToSkip = onFrame(frameIndex, bitmap)
                    framesDecoded++
                    frameIndex++
                }

                codec.releaseOutputBuffer(outputIndex, false)
            } else if (outputIndex == MediaCodec.INFO_OUTPUT_FORMAT_CHANGED) {
                Log.d(TAG, "Output format changed: ${codec.outputFormat}")
            }
        }

        Log.i(TAG, "Done: $framesDecoded decoded, $framesSkipped skipped, $frameIndex total from ${videoFile.name}")
    }

    fun release() {
        decoder?.stop()
        decoder?.release()
        decoder = null
        extractor.release()
    }

    private fun imageToBitmap(image: Image): Bitmap {
        require(image.format == ImageFormat.YUV_420_888) {
            "Unexpected image format: ${image.format}"
        }

        val width = image.width
        val height = image.height
        val yPlane = image.planes[0]
        val uPlane = image.planes[1]
        val vPlane = image.planes[2]

        val yBuffer = yPlane.buffer
        val uBuffer = uPlane.buffer
        val vBuffer = vPlane.buffer

        val yRowStride = yPlane.rowStride
        val uvRowStride = uPlane.rowStride
        val uvPixelStride = uPlane.pixelStride

        val pixels = IntArray(width * height)

        for (row in 0 until height) {
            for (col in 0 until width) {
                val y = yBuffer.get(row * yRowStride + col).toInt() and 0xFF
                val uvRow = row / 2
                val uvCol = col / 2
                val uvIndex = uvRow * uvRowStride + uvCol * uvPixelStride
                val u = uBuffer.get(uvIndex).toInt() and 0xFF
                val v = vBuffer.get(uvIndex).toInt() and 0xFF

                val yVal = y - 16
                val uVal = u - 128
                val vVal = v - 128
                var r = (1.164f * yVal + 1.596f * vVal).toInt()
                var g = (1.164f * yVal - 0.813f * vVal - 0.391f * uVal).toInt()
                var b = (1.164f * yVal + 2.018f * uVal).toInt()
                r = r.coerceIn(0, 255)
                g = g.coerceIn(0, 255)
                b = b.coerceIn(0, 255)

                pixels[row * width + col] = (0xFF shl 24) or (r shl 16) or (g shl 8) or b
            }
        }

        return Bitmap.createBitmap(pixels, width, height, Bitmap.Config.ARGB_8888)
    }
}
