package com.haptictrack.camera

import android.graphics.Bitmap
import android.graphics.SurfaceTexture
import android.opengl.EGL14
import android.opengl.EGLConfig
import android.opengl.EGLContext
import android.opengl.EGLDisplay
import android.opengl.EGLSurface
import android.opengl.GLES11Ext
import android.opengl.GLES20
import android.util.Log
import android.util.Size
import android.view.Surface
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.util.concurrent.atomic.AtomicBoolean

/**
 * Reads camera frames from a [SurfaceTexture] via OpenGL on a background thread.
 *
 * During recording mode, CameraX binds Preview + VideoCapture (PRIV + PRIV = 4K).
 * Instead of using PreviewView.getBitmap() (slow, main-thread-bound, 7-10fps),
 * this class provides its own Surface to the Preview use case and reads frames
 * via SurfaceTexture → OpenGL texture → glReadPixels → Bitmap at 20-30fps.
 *
 * The output bitmap is downscaled to [outputWidth] × [outputHeight] for analysis.
 */
class SurfaceTextureFrameReader(
    private val outputWidth: Int = 640,
    private val outputHeight: Int = 480,
    private val onFrame: (Bitmap) -> Unit
) {

    companion object {
        private const val TAG = "STFrameReader"
    }

    private var glThread: Thread? = null
    private var surfaceTexture: SurfaceTexture? = null
    private var surface: Surface? = null
    private val running = AtomicBoolean(false)
    private val frameAvailable = AtomicBoolean(false)

    // EGL state (initialized on GL thread)
    private var eglDisplay: EGLDisplay = EGL14.EGL_NO_DISPLAY
    private var eglContext: EGLContext = EGL14.EGL_NO_CONTEXT
    private var eglSurface: EGLSurface = EGL14.EGL_NO_SURFACE
    private var textureId: Int = 0

    // Pre-allocated readback buffer
    private lateinit var readBuffer: ByteBuffer

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
                    setDefaultBufferSize(outputWidth, outputHeight)
                    setOnFrameAvailableListener { frameAvailable.set(true) }
                }
                resultSurface = Surface(surfaceTexture)
                surfaceLatch.countDown()

                readBuffer = ByteBuffer.allocateDirect(outputWidth * outputHeight * 4)
                    .order(ByteOrder.nativeOrder())

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

                val program = createShaderProgram()

                Log.i(TAG, "GL thread started, output=${outputWidth}x${outputHeight}")

                // Render loop
                while (running.get()) {
                    if (frameAvailable.compareAndSet(true, false)) {
                        surfaceTexture?.updateTexImage()

                        // Render external texture to FBO at output resolution
                        GLES20.glBindFramebuffer(GLES20.GL_FRAMEBUFFER, fbo[0])
                        GLES20.glViewport(0, 0, outputWidth, outputHeight)
                        GLES20.glUseProgram(program)

                        GLES20.glActiveTexture(GLES20.GL_TEXTURE0)
                        GLES20.glBindTexture(GLES11Ext.GL_TEXTURE_EXTERNAL_OES, textureId)
                        GLES20.glUniform1i(GLES20.glGetUniformLocation(program, "sTexture"), 0)

                        // Full-screen quad
                        drawQuad(program)

                        // Read pixels
                        readBuffer.rewind()
                        GLES20.glReadPixels(0, 0, outputWidth, outputHeight,
                            GLES20.GL_RGBA, GLES20.GL_UNSIGNED_BYTE, readBuffer)
                        readBuffer.rewind()

                        val bitmap = Bitmap.createBitmap(outputWidth, outputHeight, Bitmap.Config.ARGB_8888)
                        bitmap.copyPixelsFromBuffer(readBuffer)

                        // GL reads bottom-to-top, flip vertically
                        val flipped = flipVertically(bitmap)
                        bitmap.recycle()

                        onFrame(flipped)
                    } else {
                        // No frame ready, sleep briefly to avoid busy-waiting
                        Thread.sleep(1)
                    }
                }

                // Cleanup
                GLES20.glDeleteFramebuffers(1, fbo, 0)
                GLES20.glDeleteTextures(1, renderTex, 0)
                GLES20.glDeleteTextures(1, intArrayOf(textureId), 0)
                GLES20.glDeleteProgram(program)
                releaseEGL()
                Log.i(TAG, "GL thread stopped")
            } catch (e: Exception) {
                Log.e(TAG, "GL thread error: ${e.message}", e)
                surfaceLatch.countDown() // unblock in case of early failure
            }
        }, "STFrameReader-GL")
        glThread?.start()

        surfaceLatch.await(3, java.util.concurrent.TimeUnit.SECONDS)
        surface = resultSurface
        return resultSurface ?: throw IllegalStateException("Failed to create Surface")
    }

    /** Stop the GL thread and release resources. */
    fun stop() {
        running.set(false)
        glThread?.join(2000)
        glThread = null
        surface?.release()
        surface = null
        surfaceTexture?.release()
        surfaceTexture = null
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

        val configAttribs = intArrayOf(
            EGL14.EGL_RENDERABLE_TYPE, EGL14.EGL_OPENGL_ES2_BIT,
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

        val contextAttribs = intArrayOf(EGL14.EGL_CONTEXT_CLIENT_VERSION, 2, EGL14.EGL_NONE)
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

    private val vertexShaderSource = """
        attribute vec4 aPosition;
        attribute vec2 aTexCoord;
        varying vec2 vTexCoord;
        void main() {
            gl_Position = aPosition;
            vTexCoord = aTexCoord;
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

    private val quadVertices = floatArrayOf(
        -1f, -1f, 0f, 0f,  // bottom-left
         1f, -1f, 1f, 0f,  // bottom-right
        -1f,  1f, 0f, 1f,  // top-left
         1f,  1f, 1f, 1f   // top-right
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

    private fun flipVertically(src: Bitmap): Bitmap {
        val matrix = android.graphics.Matrix().apply { postScale(1f, -1f) }
        return Bitmap.createBitmap(src, 0, 0, src.width, src.height, matrix, false)
    }
}
