package com.haptictrack.camera

import android.graphics.Bitmap
import android.graphics.SurfaceTexture
import android.opengl.EGL14
import android.opengl.EGLConfig
import android.opengl.EGLContext
import android.opengl.EGLDisplay
import android.opengl.EGLExt
import android.opengl.EGLSurface
import android.opengl.GLES11Ext
import android.opengl.GLES20
import android.opengl.GLES30
import android.util.Log
import android.util.Size
import android.view.Surface
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.util.concurrent.atomic.AtomicBoolean
import java.util.concurrent.atomic.AtomicReference

/**
 * Reads camera frames from a [SurfaceTexture] via OpenGL on a background thread.
 *
 * During recording mode, CameraX binds Preview + VideoCapture (PRIV + PRIV = 4K).
 * Instead of using PreviewView.getBitmap() (slow, main-thread-bound, 7-10fps),
 * this class provides its own Surface to the Preview use case and reads frames
 * via SurfaceTexture → OpenGL texture → glReadPixels → Bitmap at 20-30fps.
 *
 * Architecture: two threads, decoupled via atomic "latest frame" holder.
 * - GL thread: reads frames from SurfaceTexture at camera rate (~30fps),
 *   stores latest bitmap in [latestFrame]. Old unprocessed frames are recycled.
 * - Processing thread: picks up latest frame when ready, calls [onFrame].
 *   Natural frame dropping when processing is slower than delivery.
 */
class SurfaceTextureFrameReader(
    /** Camera's native buffer size (landscape, e.g. 1600x1200). */
    private val inputWidth: Int = 1600,
    private val inputHeight: Int = 1200,
    /** Output bitmap size (portrait for portrait phones). */
    private val outputWidth: Int = 640,
    private val outputHeight: Int = 854,
    /** Called on processing thread when a frame is ready for ObjectTracker. */
    private val onFrame: (Bitmap) -> Unit,
    /** Called on GL thread at camera rate (~29fps) for viewfinder display. */
    private val onViewfinderFrame: ((Bitmap) -> Unit)? = null
) {

    companion object {
        private const val TAG = "STFrameReader"
        /** Log averaged frame timings every N frames on the GL thread. */
        private const val TIMING_LOG_INTERVAL = 60
        /** Bitmaps in each ring. 3 is enough to absorb one-frame hiccups on either side. */
        private const val POOL_SIZE = 3
    }

    private var glThread: Thread? = null
    private var processingThread: Thread? = null
    private var surfaceTexture: SurfaceTexture? = null
    private var surface: Surface? = null
    private val running = AtomicBoolean(false)
    private val frameAvailable = AtomicBoolean(false)

    /** Latest frame from GL thread. Processing thread swaps it out atomically. */
    private val latestFrame = AtomicReference<Bitmap?>(null)

    /** Bitmaps owned by the processing path; caller releases back via [releaseAnalysisBitmap]. */
    private val processingPool = BitmapRing(POOL_SIZE, outputWidth, outputHeight)
    /** Bitmaps owned by the viewfinder path; cycled without a caller release — Compose drops
     *  references via StateFlow, and we reuse bitmaps that are old enough that Compose has
     *  already swapped them off the RenderThread. */
    private val viewfinderPool = BitmapRing(POOL_SIZE, outputWidth, outputHeight)
    /** In-flight viewfinder bitmaps, oldest first. Once the ring is full, the oldest
     *  bitmap is recycled back into [viewfinderPool] — Compose is ≥2 frames past it. */
    private val inflightViewfinder = ArrayDeque<Bitmap>(POOL_SIZE)

    // EGL state (initialized on GL thread)
    private var eglDisplay: EGLDisplay = EGL14.EGL_NO_DISPLAY
    private var eglContext: EGLContext = EGL14.EGL_NO_CONTEXT
    private var eglSurface: EGLSurface = EGL14.EGL_NO_SURFACE
    private var textureId: Int = 0

    /**
     * Start the GL thread and create the Surface.
     * Returns a [Surface] that should be provided to CameraX Preview via [Preview.SurfaceProvider].
     * Blocks until the Surface is ready.
     */
    fun start(): Surface {
        check(!running.get()) { "Already running" }
        running.set(true)

        val surfaceLatch = java.util.concurrent.CountDownLatch(1)
        var resultSurface: Surface? = null

        glThread = Thread({
            try {
                initEGL()
                textureId = createExternalTexture()
                surfaceTexture = SurfaceTexture(textureId).apply {
                    setDefaultBufferSize(inputWidth, inputHeight)
                    setOnFrameAvailableListener { frameAvailable.set(true) }
                }
                resultSurface = Surface(surfaceTexture)
                surfaceLatch.countDown()

                val bufferBytes = outputWidth * outputHeight * 4

                // Create an FBO for offscreen rendering at output resolution
                val fbo = IntArray(1)
                GLES20.glGenFramebuffers(1, fbo, 0)
                val renderTex = IntArray(1)
                GLES20.glGenTextures(1, renderTex, 0)
                GLES20.glBindTexture(GLES20.GL_TEXTURE_2D, renderTex[0])
                GLES20.glTexImage2D(GLES20.GL_TEXTURE_2D, 0, GLES20.GL_RGBA,
                    outputWidth, outputHeight, 0, GLES20.GL_RGBA, GLES20.GL_UNSIGNED_BYTE, null)
                GLES20.glTexParameteri(GLES20.GL_TEXTURE_2D, GLES20.GL_TEXTURE_MIN_FILTER, GLES20.GL_LINEAR)
                GLES20.glTexParameteri(GLES20.GL_TEXTURE_2D, GLES20.GL_TEXTURE_MAG_FILTER, GLES20.GL_LINEAR)
                GLES20.glBindFramebuffer(GLES20.GL_FRAMEBUFFER, fbo[0])
                GLES20.glFramebufferTexture2D(GLES20.GL_FRAMEBUFFER, GLES20.GL_COLOR_ATTACHMENT0,
                    GLES20.GL_TEXTURE_2D, renderTex[0], 0)

                // Two PBOs for ping-pong async readback. Each frame starts a readback
                // into one PBO and maps the other (which the GPU already finished writing
                // at least one frame ago, so the map is zero-wait on the CPU side).
                // Tradeoff: adds exactly 1 frame of latency to the consumer path.
                val pbos = IntArray(2)
                GLES30.glGenBuffers(2, pbos, 0)
                for (pbo in pbos) {
                    GLES30.glBindBuffer(GLES30.GL_PIXEL_PACK_BUFFER, pbo)
                    GLES30.glBufferData(GLES30.GL_PIXEL_PACK_BUFFER, bufferBytes, null, GLES30.GL_STREAM_READ)
                }
                GLES30.glBindBuffer(GLES30.GL_PIXEL_PACK_BUFFER, 0)
                var pboWrite = 0  // PBO we kick the current-frame readback into
                var pboRead = 1   // PBO whose previous-frame data we map for the CPU
                var haveLastFrame = false  // first iteration has nothing in pboRead yet

                val program = createShaderProgram()

                Log.i(TAG, "GL thread started, output=${outputWidth}x${outputHeight}")
                var glFrameCount = 0
                var readPxNs = 0L
                var copyNs = 0L
                var totalNs = 0L

                // GL render loop: reads frames at camera rate, stores latest
                while (running.get()) {
                    if (frameAvailable.compareAndSet(true, false)) {
                        val st = surfaceTexture ?: continue
                        val frameStart = System.nanoTime()
                        st.updateTexImage()
                        st.getTransformMatrix(texMatrix)

                        // Render external texture to FBO at output resolution
                        GLES20.glBindFramebuffer(GLES20.GL_FRAMEBUFFER, fbo[0])
                        GLES20.glViewport(0, 0, outputWidth, outputHeight)
                        GLES20.glUseProgram(program)

                        // Pass transform matrix (handles rotation, flip from camera HAL)
                        val texMatLoc = GLES20.glGetUniformLocation(program, "uTexMatrix")
                        GLES20.glUniformMatrix4fv(texMatLoc, 1, false, texMatrix, 0)

                        GLES20.glActiveTexture(GLES20.GL_TEXTURE0)
                        GLES20.glBindTexture(GLES11Ext.GL_TEXTURE_EXTERNAL_OES, textureId)
                        GLES20.glUniform1i(GLES20.glGetUniformLocation(program, "sTexture"), 0)

                        drawQuad(program)

                        // Kick off an async readback into the WRITE pbo. Returns immediately;
                        // the actual GPU→CPU transfer overlaps with this thread's next work
                        // (and with the camera producing the next frame).
                        val readStart = System.nanoTime()
                        GLES30.glBindBuffer(GLES30.GL_PIXEL_PACK_BUFFER, pbos[pboWrite])
                        GLES30.glReadPixels(0, 0, outputWidth, outputHeight,
                            GLES30.GL_RGBA, GLES30.GL_UNSIGNED_BYTE, 0)
                        readPxNs += System.nanoTime() - readStart

                        // Map the READ pbo — its readback was issued last frame, so the
                        // driver has had a full frame to finish it. Map blocks only when
                        // the GPU hasn't caught up yet (rare at 30fps).
                        if (haveLastFrame) {
                            val copyStart = System.nanoTime()
                            GLES30.glBindBuffer(GLES30.GL_PIXEL_PACK_BUFFER, pbos[pboRead])
                            val mapped = GLES30.glMapBufferRange(
                                GLES30.GL_PIXEL_PACK_BUFFER, 0, bufferBytes,
                                GLES30.GL_MAP_READ_BIT
                            ) as? ByteBuffer
                            if (mapped != null) {
                                mapped.order(ByteOrder.nativeOrder())

                                // Fill processing bitmap from the mapped PBO.
                                mapped.rewind()
                                val procBitmap = processingPool.acquire()
                                procBitmap.copyPixelsFromBuffer(mapped)

                                // Fill viewfinder bitmap from the same buffer — no bitmap.copy().
                                // Compose may still hold the previous few viewfinder bitmaps via
                                // StateFlow + RenderThread; keep a 3-entry in-flight queue and
                                // recycle the oldest back into the ring once it's ≥2 frames old.
                                if (onViewfinderFrame != null) {
                                    mapped.rewind()
                                    val vfBitmap = viewfinderPool.acquire()
                                    vfBitmap.copyPixelsFromBuffer(mapped)
                                    inflightViewfinder.addLast(vfBitmap)
                                    if (inflightViewfinder.size > POOL_SIZE) {
                                        viewfinderPool.release(inflightViewfinder.removeFirst())
                                    }
                                    onViewfinderFrame.invoke(vfBitmap)
                                }

                                GLES30.glUnmapBuffer(GLES30.GL_PIXEL_PACK_BUFFER)
                                copyNs += System.nanoTime() - copyStart

                                // Hand the processing bitmap off; return any unprocessed
                                // previous frame to the pool so the ring stays in budget.
                                val old = latestFrame.getAndSet(procBitmap)
                                if (old != null) processingPool.release(old)
                            } else {
                                Log.w(TAG, "glMapBufferRange returned null — skipping frame")
                            }
                        }
                        GLES30.glBindBuffer(GLES30.GL_PIXEL_PACK_BUFFER, 0)

                        // Ping-pong for next iteration.
                        val tmp = pboWrite; pboWrite = pboRead; pboRead = tmp
                        haveLastFrame = true

                        totalNs += System.nanoTime() - frameStart
                        glFrameCount++
                        if (glFrameCount % TIMING_LOG_INTERVAL == 0) {
                            val n = TIMING_LOG_INTERVAL
                            Log.i(TAG, "GL frame avg: readPx=${readPxNs / n / 1_000_000.0}ms " +
                                "copy=${copyNs / n / 1_000_000.0}ms " +
                                "total=${totalNs / n / 1_000_000.0}ms " +
                                "(${glFrameCount} frames)")
                            readPxNs = 0L; copyNs = 0L; totalNs = 0L
                        }
                    } else {
                        Thread.sleep(1)
                    }
                }
                Log.i(TAG, "GL thread produced $glFrameCount frames")

                // Cleanup
                GLES30.glDeleteBuffers(2, pbos, 0)
                GLES20.glDeleteFramebuffers(1, fbo, 0)
                GLES20.glDeleteTextures(1, renderTex, 0)
                GLES20.glDeleteTextures(1, intArrayOf(textureId), 0)
                GLES20.glDeleteProgram(program)
                releaseEGL()
                Log.i(TAG, "GL thread stopped")
            } catch (e: Exception) {
                Log.e(TAG, "GL thread error: ${e.message}", e)
                if (eglDisplay != EGL14.EGL_NO_DISPLAY) {
                    releaseEGL()
                }
                surfaceLatch.countDown() // unblock in case of early failure
            }
        }, "STFrameReader-GL")
        glThread?.start()

        surfaceLatch.await(3, java.util.concurrent.TimeUnit.SECONDS)
        surface = resultSurface

        // Start processing thread — picks up latest frame when ready,
        // naturally drops frames when processing is slower than GL delivery
        processingThread = Thread({
            var processedCount = 0
            var processNs = 0L
            var lastReportMs = System.currentTimeMillis()
            Log.i(TAG, "Processing thread started")
            while (running.get()) {
                val frame = latestFrame.getAndSet(null)
                if (frame != null) {
                    val start = System.nanoTime()
                    // Catch any exception from onFrame (tracker.processBitmap, pool
                    // release, etc.) so one bad frame can't silently kill this loop
                    // and freeze tracking. The tracker's own try/finally handles
                    // bitmap ownership on exception; we just need to keep going.
                    try {
                        onFrame(frame)
                    } catch (t: Throwable) {
                        Log.e(TAG, "Processing thread: onFrame threw, continuing", t)
                    }
                    processNs += System.nanoTime() - start
                    processedCount++

                    // Report every 60 processed frames (~5-6s at 10fps processing).
                    if (processedCount % TIMING_LOG_INTERVAL == 0) {
                        val now = System.currentTimeMillis()
                        val elapsedMs = now - lastReportMs
                        val fps = if (elapsedMs > 0)
                            TIMING_LOG_INTERVAL * 1000.0 / elapsedMs else 0.0
                        Log.i(TAG, "Process frame avg: ${processNs / TIMING_LOG_INTERVAL / 1_000_000.0}ms " +
                            "(~${"%.1f".format(fps)}fps, $processedCount total)")
                        processNs = 0L
                        lastReportMs = now
                    }
                } else {
                    Thread.sleep(2) // wait for next frame
                }
            }
            // Return any remaining frame to the pool
            latestFrame.getAndSet(null)?.let { processingPool.release(it) }
            Log.i(TAG, "Processing thread stopped: $processedCount processed")
        }, "STFrameReader-Process")
        processingThread?.start()

        return resultSurface ?: throw IllegalStateException("Failed to create Surface")
    }

    /** Stop both threads and release resources. */
    fun stop() {
        running.set(false)
        processingThread?.join(2000)
        processingThread = null
        glThread?.join(2000)
        glThread = null
        latestFrame.getAndSet(null)?.let { processingPool.release(it) }
        // Drop inflight references but DO NOT recycle them — the most recent
        // inflight bitmap is what Compose is currently painting. Recycling it
        // crashes the Compose draw thread with "Canvas: trying to use a recycled
        // bitmap" (observed on camera switch where stop() runs before Compose
        // can swap to a new bitmap). GC reclaims them once Compose drops its
        // last reference.
        // The pool's free deque is safe — those bitmaps were rotated out of
        // inflight ≥2 frames ago so Compose has long since released them.
        inflightViewfinder.clear()
        processingPool.releaseAll()
        viewfinderPool.releaseAll()
        surface?.release()
        surface = null
        surfaceTexture?.release()
        surfaceTexture = null
    }

    /**
     * Return a processing bitmap to the pool after the consumer finishes with it.
     * Replaces the previous contract of [Bitmap.recycle] inside [onFrame]'s consumer —
     * the pool owns bitmap lifetime now.
     */
    fun releaseAnalysisBitmap(bitmap: Bitmap) {
        processingPool.release(bitmap)
    }

    /** The Surface to provide to CameraX Preview. Only valid after [start]. */
    fun getSurface(): Surface? = surface

    /** Output resolution for SurfaceRequest negotiation. */
    fun getOutputSize(): Size = Size(outputWidth, outputHeight)

    // --- EGL setup ---

    private fun initEGL() {
        eglDisplay = EGL14.eglGetDisplay(EGL14.EGL_DEFAULT_DISPLAY)
        check(eglDisplay != EGL14.EGL_NO_DISPLAY) { "No EGL display" }

        val version = IntArray(2)
        EGL14.eglInitialize(eglDisplay, version, 0, version, 1)

        // GLES 3.0 is required for PBO async readback (glMapBufferRange + GL_PIXEL_PACK_BUFFER).
        // Guaranteed available from Android 4.3 / API 18; we target minSdk=29 so no fallback needed.
        val configAttribs = intArrayOf(
            EGL14.EGL_RENDERABLE_TYPE, EGLExt.EGL_OPENGL_ES3_BIT_KHR,
            EGL14.EGL_SURFACE_TYPE, EGL14.EGL_PBUFFER_BIT,
            EGL14.EGL_RED_SIZE, 8,
            EGL14.EGL_GREEN_SIZE, 8,
            EGL14.EGL_BLUE_SIZE, 8,
            EGL14.EGL_ALPHA_SIZE, 8,
            EGL14.EGL_NONE
        )
        val configs = arrayOfNulls<EGLConfig>(1)
        val numConfigs = IntArray(1)
        EGL14.eglChooseConfig(eglDisplay, configAttribs, 0, configs, 0, 1, numConfigs, 0)
        check(numConfigs[0] > 0) { "No EGL config found" }

        val contextAttribs = intArrayOf(EGL14.EGL_CONTEXT_CLIENT_VERSION, 3, EGL14.EGL_NONE)
        eglContext = EGL14.eglCreateContext(eglDisplay, configs[0]!!, EGL14.EGL_NO_CONTEXT, contextAttribs, 0)

        val pbufferAttribs = intArrayOf(
            EGL14.EGL_WIDTH, outputWidth,
            EGL14.EGL_HEIGHT, outputHeight,
            EGL14.EGL_NONE
        )
        eglSurface = EGL14.eglCreatePbufferSurface(eglDisplay, configs[0]!!, pbufferAttribs, 0)

        EGL14.eglMakeCurrent(eglDisplay, eglSurface, eglSurface, eglContext)
    }

    private fun releaseEGL() {
        EGL14.eglMakeCurrent(eglDisplay, EGL14.EGL_NO_SURFACE, EGL14.EGL_NO_SURFACE, EGL14.EGL_NO_CONTEXT)
        EGL14.eglDestroySurface(eglDisplay, eglSurface)
        EGL14.eglDestroyContext(eglDisplay, eglContext)
        EGL14.eglTerminate(eglDisplay)
    }

    private fun createExternalTexture(): Int {
        val textures = IntArray(1)
        GLES20.glGenTextures(1, textures, 0)
        GLES20.glBindTexture(GLES11Ext.GL_TEXTURE_EXTERNAL_OES, textures[0])
        GLES20.glTexParameteri(GLES11Ext.GL_TEXTURE_EXTERNAL_OES, GLES20.GL_TEXTURE_MIN_FILTER, GLES20.GL_LINEAR)
        GLES20.glTexParameteri(GLES11Ext.GL_TEXTURE_EXTERNAL_OES, GLES20.GL_TEXTURE_MAG_FILTER, GLES20.GL_LINEAR)
        GLES20.glTexParameteri(GLES11Ext.GL_TEXTURE_EXTERNAL_OES, GLES20.GL_TEXTURE_WRAP_S, GLES20.GL_CLAMP_TO_EDGE)
        GLES20.glTexParameteri(GLES11Ext.GL_TEXTURE_EXTERNAL_OES, GLES20.GL_TEXTURE_WRAP_T, GLES20.GL_CLAMP_TO_EDGE)
        return textures[0]
    }

    // --- Shader program for external OES texture ---

    /** SurfaceTexture transform matrix — updated each frame, applied in shader. */
    private val texMatrix = FloatArray(16)

    private val vertexShaderSource = """
        attribute vec4 aPosition;
        attribute vec2 aTexCoord;
        uniform mat4 uTexMatrix;
        varying vec2 vTexCoord;
        void main() {
            gl_Position = aPosition;
            vTexCoord = (uTexMatrix * vec4(aTexCoord, 0.0, 1.0)).xy;
        }
    """.trimIndent()

    private val fragmentShaderSource = """
        #extension GL_OES_EGL_image_external : require
        precision mediump float;
        varying vec2 vTexCoord;
        uniform samplerExternalOES sTexture;
        void main() {
            gl_FragColor = texture2D(sTexture, vTexCoord);
        }
    """.trimIndent()

    private fun createShaderProgram(): Int {
        val vertexShader = loadShader(GLES20.GL_VERTEX_SHADER, vertexShaderSource)
        val fragmentShader = loadShader(GLES20.GL_FRAGMENT_SHADER, fragmentShaderSource)
        val program = GLES20.glCreateProgram()
        GLES20.glAttachShader(program, vertexShader)
        GLES20.glAttachShader(program, fragmentShader)
        GLES20.glLinkProgram(program)
        return program
    }

    private fun loadShader(type: Int, source: String): Int {
        val shader = GLES20.glCreateShader(type)
        GLES20.glShaderSource(shader, source)
        GLES20.glCompileShader(shader)
        val status = IntArray(1)
        GLES20.glGetShaderiv(shader, GLES20.GL_COMPILE_STATUS, status, 0)
        if (status[0] == 0) {
            val log = GLES20.glGetShaderInfoLog(shader)
            GLES20.glDeleteShader(shader)
            throw RuntimeException("Shader compile error: $log")
        }
        return shader
    }

    // --- Quad rendering ---

    // Quad vertices with flipped Y positions to correct for GL→Bitmap coordinate mismatch.
    // GL has Y-up, Bitmap has Y-down. Flipping the quad avoids a separate flipVertically pass.
    private val quadVertices = floatArrayOf(
        -1f,  1f, 0f, 0f,  // top-left (GL) → top-left (Bitmap)
         1f,  1f, 1f, 0f,  // top-right
        -1f, -1f, 0f, 1f,  // bottom-left
         1f, -1f, 1f, 1f   // bottom-right
    )

    private val quadBuffer = ByteBuffer.allocateDirect(quadVertices.size * 4)
        .order(ByteOrder.nativeOrder())
        .asFloatBuffer()
        .apply { put(quadVertices); position(0) }

    private fun drawQuad(program: Int) {
        val posLoc = GLES20.glGetAttribLocation(program, "aPosition")
        val texLoc = GLES20.glGetAttribLocation(program, "aTexCoord")

        quadBuffer.position(0)
        GLES20.glVertexAttribPointer(posLoc, 2, GLES20.GL_FLOAT, false, 16, quadBuffer)
        GLES20.glEnableVertexAttribArray(posLoc)

        quadBuffer.position(2)
        GLES20.glVertexAttribPointer(texLoc, 2, GLES20.GL_FLOAT, false, 16, quadBuffer)
        GLES20.glEnableVertexAttribArray(texLoc)

        GLES20.glDrawArrays(GLES20.GL_TRIANGLE_STRIP, 0, 4)
    }
}
