package com.haptictrack.camera

import android.graphics.SurfaceTexture
import android.opengl.EGL14
import android.opengl.EGLConfig
import android.opengl.EGLContext
import android.opengl.EGLDisplay
import android.opengl.EGLExt
import android.opengl.EGLSurface
import android.opengl.GLES11Ext
import android.opengl.GLES20
import android.os.Handler
import android.os.HandlerThread
import android.util.Log
import android.view.Surface
import androidx.camera.core.CameraEffect
import androidx.camera.core.SurfaceOutput
import androidx.camera.core.SurfaceProcessor
import androidx.camera.core.SurfaceRequest
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.util.concurrent.Executor

/**
 * GPU-based stabilization processor for CameraX VideoCapture.
 *
 * Applies the gyro stabilization homography (mat3) to video frames in a GL shader
 * before they reach the encoder. Runs on its own GL thread.
 *
 * When [videoMatrixProvider] is set, enables lookahead mode: frames are buffered
 * in FBOs for [LOOKAHEAD_FRAMES] frames (~133ms at 30fps), then rendered with a
 * bidirectional-smoothed matrix that uses future gyro data for zero-phase smoothing.
 * Preview stays causal (handled by SurfaceTextureFrameReader).
 */
class StabilizationProcessor(
    private val stabMatrixProvider: () -> FloatArray,
    private val videoMatrixProvider: ((Long) -> FloatArray)? = null,
    private val frameTimestampLogger: ((Long, Long) -> Unit)? = null
) : SurfaceProcessor, SurfaceTexture.OnFrameAvailableListener {

    companion object {
        private const val TAG = "StabProcessor"
        private const val LOOKAHEAD_FRAMES = 4
    }

    private val glThread = HandlerThread("StabProcessor-GL").apply { start() }
    private val glHandler = Handler(glThread.looper)
    val executor: Executor = Executor { glHandler.post(it) }

    // EGL state
    private var eglDisplay: EGLDisplay = EGL14.EGL_NO_DISPLAY
    private var eglContext: EGLContext = EGL14.EGL_NO_CONTEXT
    private var eglConfig: EGLConfig? = null
    private var tempPbuffer: EGLSurface = EGL14.EGL_NO_SURFACE

    // Input (from camera)
    private var inputTextureId: Int = 0
    private var inputSurfaceTexture: SurfaceTexture? = null
    private var inputSurface: Surface? = null
    @Volatile private var released = false

    // Output (to encoder)
    @Volatile private var outputSurface: Surface? = null
    private var outputEglSurface: EGLSurface = EGL14.EGL_NO_SURFACE
    private var outputSurfaceOutput: SurfaceOutput? = null
    private var outputWidth: Int = 0
    private var outputHeight: Int = 0

    // GL programs
    private var causalProgram: Int = 0
    private var copyProgram: Int = 0
    private var renderProgram: Int = 0

    // FBO ring buffer for lookahead
    private val fboTextureIds = IntArray(LOOKAHEAD_FRAMES)
    private val fboIds = IntArray(LOOKAHEAD_FRAMES)
    private var fboInitialized = false
    private var fboWriteIdx = 0
    private data class BufferedFrame(val fboIndex: Int, val timestampNs: Long)
    private val frameRing = ArrayDeque<BufferedFrame>()

    // Matrices
    private val texMatrix = FloatArray(16)

    private var frameCount = 0L

    init {
        glHandler.post { initEGL() }
    }

    override fun onInputSurface(request: SurfaceRequest) {
        val resolution = request.resolution
        Log.i(TAG, "onInputSurface: ${resolution.width}x${resolution.height}")

        glHandler.post {
            cleanupInput()

            inputTextureId = createExternalTexture()
            inputSurfaceTexture = SurfaceTexture(inputTextureId).apply {
                setDefaultBufferSize(resolution.width, resolution.height)
                setOnFrameAvailableListener(this@StabilizationProcessor, glHandler)
            }
            inputSurface = Surface(inputSurfaceTexture)

            if (causalProgram == 0) causalProgram = createCausalProgram()
            if (videoMatrixProvider != null) {
                if (copyProgram == 0) copyProgram = createCopyProgram()
                if (renderProgram == 0) renderProgram = createRenderProgram()
            }

            request.provideSurface(inputSurface!!, executor) {
                Log.d(TAG, "Input surface released by CameraX")
                glHandler.post { cleanupInput() }
            }
        }
    }

    override fun onOutputSurface(surfaceOutput: SurfaceOutput) {
        val size = surfaceOutput.size
        Log.i(TAG, "onOutputSurface: ${size.width}x${size.height} targets=${surfaceOutput.targets}")

        glHandler.post {
            cleanupOutput()

            outputSurfaceOutput = surfaceOutput
            outputWidth = size.width
            outputHeight = size.height

            outputSurface = surfaceOutput.getSurface(executor) { event ->
                if (event.eventCode == SurfaceOutput.Event.EVENT_REQUEST_CLOSE) {
                    Log.d(TAG, "Output surface close requested")
                    glHandler.post {
                        cleanupOutput()
                        surfaceOutput.close()
                    }
                }
            }

            val surfaceAttribs = intArrayOf(EGL14.EGL_NONE)
            outputEglSurface = EGL14.eglCreateWindowSurface(
                eglDisplay, eglConfig!!, outputSurface!!, surfaceAttribs, 0
            )
            if (outputEglSurface == EGL14.EGL_NO_SURFACE) {
                Log.e(TAG, "Failed to create output EGL window surface")
            }
        }
    }

    override fun onFrameAvailable(surfaceTexture: SurfaceTexture) {
        if (released) return
        val eglSurf = outputEglSurface
        if (eglSurf == EGL14.EGL_NO_SURFACE) return
        if (outputSurfaceOutput == null) return

        surfaceTexture.updateTexImage()
        val frameTs = surfaceTexture.timestamp
        frameTimestampLogger?.invoke(frameCount, frameTs)
        surfaceTexture.getTransformMatrix(texMatrix)

        if (videoMatrixProvider != null) {
            renderLookahead(eglSurf, frameTs)
        } else {
            renderCausal(eglSurf, frameTs)
        }

        frameCount++
        if (frameCount % 300 == 0L) {
            Log.d(TAG, "Processed $frameCount video frames (${outputWidth}x${outputHeight})" +
                if (videoMatrixProvider != null) " [lookahead=$LOOKAHEAD_FRAMES]" else "")
        }
    }

    private fun renderCausal(eglSurf: EGLSurface, frameTs: Long) {
        EGL14.eglMakeCurrent(eglDisplay, eglSurf, eglSurf, eglContext)
        GLES20.glViewport(0, 0, outputWidth, outputHeight)
        GLES20.glUseProgram(causalProgram)

        GLES20.glUniformMatrix4fv(
            GLES20.glGetUniformLocation(causalProgram, "uTexMatrix"), 1, false, texMatrix, 0
        )
        GLES20.glUniformMatrix3fv(
            GLES20.glGetUniformLocation(causalProgram, "uStabMatrix"), 1, false, stabMatrixProvider(), 0
        )
        GLES20.glActiveTexture(GLES20.GL_TEXTURE0)
        GLES20.glBindTexture(GLES11Ext.GL_TEXTURE_EXTERNAL_OES, inputTextureId)
        GLES20.glUniform1i(GLES20.glGetUniformLocation(causalProgram, "sTexture"), 0)
        drawQuad(causalProgram)

        EGLExt.eglPresentationTimeANDROID(eglDisplay, eglSurf, frameTs)
        EGL14.eglSwapBuffers(eglDisplay, eglSurf)
    }

    private fun renderLookahead(eglSurf: EGLSurface, frameTs: Long) {
        if (!fboInitialized) initFBOs()
        if (!fboInitialized) { renderCausal(eglSurf, frameTs); return }

        // Step 1: Copy OES texture → FBO (applying texMatrix, no stabilization)
        EGL14.eglMakeCurrent(eglDisplay, eglSurf, eglSurf, eglContext)
        GLES20.glBindFramebuffer(GLES20.GL_FRAMEBUFFER, fboIds[fboWriteIdx])
        GLES20.glViewport(0, 0, outputWidth, outputHeight)
        GLES20.glUseProgram(copyProgram)

        GLES20.glUniformMatrix4fv(
            GLES20.glGetUniformLocation(copyProgram, "uTexMatrix"), 1, false, texMatrix, 0
        )
        GLES20.glActiveTexture(GLES20.GL_TEXTURE0)
        GLES20.glBindTexture(GLES11Ext.GL_TEXTURE_EXTERNAL_OES, inputTextureId)
        GLES20.glUniform1i(GLES20.glGetUniformLocation(copyProgram, "sTexture"), 0)
        drawQuad(copyProgram)

        GLES20.glBindFramebuffer(GLES20.GL_FRAMEBUFFER, 0)

        frameRing.addLast(BufferedFrame(fboWriteIdx, frameTs))
        fboWriteIdx = (fboWriteIdx + 1) % LOOKAHEAD_FRAMES

        // Step 2: If enough frames buffered, render the oldest with lookahead matrix
        if (frameRing.size > LOOKAHEAD_FRAMES) {
            val oldest = frameRing.removeFirst()
            renderFromFBO(eglSurf, oldest)
        }
    }

    private fun renderFromFBO(eglSurf: EGLSurface, frame: BufferedFrame) {
        val stabMatrix = videoMatrixProvider!!(frame.timestampNs)

        EGL14.eglMakeCurrent(eglDisplay, eglSurf, eglSurf, eglContext)
        GLES20.glViewport(0, 0, outputWidth, outputHeight)
        GLES20.glUseProgram(renderProgram)

        GLES20.glUniformMatrix3fv(
            GLES20.glGetUniformLocation(renderProgram, "uStabMatrix"), 1, false, stabMatrix, 0
        )
        GLES20.glActiveTexture(GLES20.GL_TEXTURE0)
        GLES20.glBindTexture(GLES20.GL_TEXTURE_2D, fboTextureIds[frame.fboIndex])
        GLES20.glUniform1i(GLES20.glGetUniformLocation(renderProgram, "sTexture"), 0)
        drawQuad(renderProgram)

        EGLExt.eglPresentationTimeANDROID(eglDisplay, eglSurf, frame.timestampNs)
        EGL14.eglSwapBuffers(eglDisplay, eglSurf)
    }

    private fun flushBufferedFrames(eglSurf: EGLSurface) {
        while (frameRing.isNotEmpty()) {
            val frame = frameRing.removeFirst()
            renderFromFBO(eglSurf, frame)
        }
    }

    fun release() {
        released = true
        glHandler.post {
            cleanupInput()
            cleanupOutput()
            if (causalProgram != 0) { GLES20.glDeleteProgram(causalProgram); causalProgram = 0 }
            if (copyProgram != 0) { GLES20.glDeleteProgram(copyProgram); copyProgram = 0 }
            if (renderProgram != 0) { GLES20.glDeleteProgram(renderProgram); renderProgram = 0 }
            cleanupFBOs()
            releaseEGL()
            glThread.quitSafely()
        }
    }

    // --- EGL setup ---

    private fun initEGL() {
        eglDisplay = EGL14.eglGetDisplay(EGL14.EGL_DEFAULT_DISPLAY)
        check(eglDisplay != EGL14.EGL_NO_DISPLAY) { "No EGL display" }

        val version = IntArray(2)
        EGL14.eglInitialize(eglDisplay, version, 0, version, 1)

        val configAttribs = intArrayOf(
            EGL14.EGL_RENDERABLE_TYPE, EGL14.EGL_OPENGL_ES2_BIT,
            EGL14.EGL_SURFACE_TYPE, EGL14.EGL_WINDOW_BIT or EGL14.EGL_PBUFFER_BIT,
            EGL14.EGL_RED_SIZE, 8,
            EGL14.EGL_GREEN_SIZE, 8,
            EGL14.EGL_BLUE_SIZE, 8,
            EGL14.EGL_ALPHA_SIZE, 8,
            EGLExt.EGL_RECORDABLE_ANDROID, 1,
            EGL14.EGL_NONE
        )
        val configs = arrayOfNulls<EGLConfig>(1)
        val numConfigs = IntArray(1)
        EGL14.eglChooseConfig(eglDisplay, configAttribs, 0, configs, 0, 1, numConfigs, 0)
        check(numConfigs[0] > 0) { "No EGL config found" }
        eglConfig = configs[0]!!

        val contextAttribs = intArrayOf(EGL14.EGL_CONTEXT_CLIENT_VERSION, 2, EGL14.EGL_NONE)
        eglContext = EGL14.eglCreateContext(eglDisplay, eglConfig!!, EGL14.EGL_NO_CONTEXT, contextAttribs, 0)

        val pbufferAttribs = intArrayOf(EGL14.EGL_WIDTH, 1, EGL14.EGL_HEIGHT, 1, EGL14.EGL_NONE)
        tempPbuffer = EGL14.eglCreatePbufferSurface(eglDisplay, eglConfig!!, pbufferAttribs, 0)
        EGL14.eglMakeCurrent(eglDisplay, tempPbuffer, tempPbuffer, eglContext)
    }

    private fun releaseEGL() {
        if (eglDisplay == EGL14.EGL_NO_DISPLAY) return
        EGL14.eglMakeCurrent(eglDisplay, EGL14.EGL_NO_SURFACE, EGL14.EGL_NO_SURFACE, EGL14.EGL_NO_CONTEXT)
        if (tempPbuffer != EGL14.EGL_NO_SURFACE) EGL14.eglDestroySurface(eglDisplay, tempPbuffer)
        EGL14.eglDestroyContext(eglDisplay, eglContext)
        EGL14.eglTerminate(eglDisplay)
        eglDisplay = EGL14.EGL_NO_DISPLAY
    }

    private fun cleanupInput() {
        inputSurfaceTexture?.setOnFrameAvailableListener(null)
        inputSurfaceTexture?.release()
        inputSurface?.release()
        if (inputTextureId != 0) {
            GLES20.glDeleteTextures(1, intArrayOf(inputTextureId), 0)
            inputTextureId = 0
        }
        inputSurfaceTexture = null
        inputSurface = null
    }

    private fun cleanupOutput() {
        if (fboInitialized && outputEglSurface != EGL14.EGL_NO_SURFACE && videoMatrixProvider != null) {
            flushBufferedFrames(outputEglSurface)
        }
        if (outputEglSurface != EGL14.EGL_NO_SURFACE) {
            EGL14.eglDestroySurface(eglDisplay, outputEglSurface)
            outputEglSurface = EGL14.EGL_NO_SURFACE
        }
        outputSurface = null
        outputSurfaceOutput?.close()
        outputSurfaceOutput = null
    }

    // --- FBO management ---

    private fun initFBOs() {
        if (fboInitialized || outputWidth == 0) return

        GLES20.glGenFramebuffers(LOOKAHEAD_FRAMES, fboIds, 0)
        GLES20.glGenTextures(LOOKAHEAD_FRAMES, fboTextureIds, 0)

        for (i in 0 until LOOKAHEAD_FRAMES) {
            GLES20.glBindTexture(GLES20.GL_TEXTURE_2D, fboTextureIds[i])
            GLES20.glTexImage2D(
                GLES20.GL_TEXTURE_2D, 0, GLES20.GL_RGBA,
                outputWidth, outputHeight, 0,
                GLES20.GL_RGBA, GLES20.GL_UNSIGNED_BYTE, null
            )
            GLES20.glTexParameteri(GLES20.GL_TEXTURE_2D, GLES20.GL_TEXTURE_MIN_FILTER, GLES20.GL_LINEAR)
            GLES20.glTexParameteri(GLES20.GL_TEXTURE_2D, GLES20.GL_TEXTURE_MAG_FILTER, GLES20.GL_LINEAR)
            GLES20.glTexParameteri(GLES20.GL_TEXTURE_2D, GLES20.GL_TEXTURE_WRAP_S, GLES20.GL_CLAMP_TO_EDGE)
            GLES20.glTexParameteri(GLES20.GL_TEXTURE_2D, GLES20.GL_TEXTURE_WRAP_T, GLES20.GL_CLAMP_TO_EDGE)

            GLES20.glBindFramebuffer(GLES20.GL_FRAMEBUFFER, fboIds[i])
            GLES20.glFramebufferTexture2D(
                GLES20.GL_FRAMEBUFFER, GLES20.GL_COLOR_ATTACHMENT0,
                GLES20.GL_TEXTURE_2D, fboTextureIds[i], 0
            )

            val status = GLES20.glCheckFramebufferStatus(GLES20.GL_FRAMEBUFFER)
            if (status != GLES20.GL_FRAMEBUFFER_COMPLETE) {
                Log.e(TAG, "FBO $i incomplete: $status")
                cleanupFBOs()
                return
            }
        }

        GLES20.glBindFramebuffer(GLES20.GL_FRAMEBUFFER, 0)
        GLES20.glBindTexture(GLES20.GL_TEXTURE_2D, 0)

        fboInitialized = true
        val mbPerFbo = outputWidth.toLong() * outputHeight * 4 / (1024 * 1024)
        Log.i(TAG, "FBO ring: ${LOOKAHEAD_FRAMES}x ${outputWidth}x${outputHeight} (${mbPerFbo}MB each, ${mbPerFbo * LOOKAHEAD_FRAMES}MB total)")
    }

    private fun cleanupFBOs() {
        if (!fboInitialized) return
        GLES20.glDeleteFramebuffers(LOOKAHEAD_FRAMES, fboIds, 0)
        GLES20.glDeleteTextures(LOOKAHEAD_FRAMES, fboTextureIds, 0)
        fboIds.fill(0)
        fboTextureIds.fill(0)
        fboInitialized = false
        fboWriteIdx = 0
        frameRing.clear()
    }

    // --- GL ---

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

    // Causal shader: OES texture + stabMatrix + texMatrix (original one-pass pipeline)
    private val causalVertexSource = """
        attribute vec4 aPosition;
        attribute vec2 aTexCoord;
        uniform mat4 uTexMatrix;
        uniform mat3 uStabMatrix;
        varying vec2 vTexCoord;
        void main() {
            gl_Position = aPosition;
            vec2 stabUV = (uStabMatrix * vec3(aTexCoord, 1.0)).xy;
            vTexCoord = (uTexMatrix * vec4(stabUV, 0.0, 1.0)).xy;
        }
    """.trimIndent()

    private val oesFragmentSource = """
        #extension GL_OES_EGL_image_external : require
        precision mediump float;
        varying vec2 vTexCoord;
        uniform samplerExternalOES sTexture;
        void main() {
            gl_FragColor = texture2D(sTexture, vTexCoord);
        }
    """.trimIndent()

    // Copy shader: OES texture + texMatrix only (no stabilization) → FBO
    private val copyVertexSource = """
        attribute vec4 aPosition;
        attribute vec2 aTexCoord;
        uniform mat4 uTexMatrix;
        varying vec2 vTexCoord;
        void main() {
            gl_Position = aPosition;
            vTexCoord = (uTexMatrix * vec4(aTexCoord, 0.0, 1.0)).xy;
        }
    """.trimIndent()

    // Render shader: 2D texture + stabMatrix (FBO → output)
    private val renderVertexSource = """
        attribute vec4 aPosition;
        attribute vec2 aTexCoord;
        uniform mat3 uStabMatrix;
        varying vec2 vTexCoord;
        void main() {
            gl_Position = aPosition;
            vTexCoord = (uStabMatrix * vec3(aTexCoord, 1.0)).xy;
        }
    """.trimIndent()

    private val tex2dFragmentSource = """
        precision mediump float;
        varying vec2 vTexCoord;
        uniform sampler2D sTexture;
        void main() {
            gl_FragColor = texture2D(sTexture, vTexCoord);
        }
    """.trimIndent()

    private fun createCausalProgram() = buildProgram(causalVertexSource, oesFragmentSource)
    private fun createCopyProgram() = buildProgram(copyVertexSource, oesFragmentSource)
    private fun createRenderProgram() = buildProgram(renderVertexSource, tex2dFragmentSource)

    private fun buildProgram(vertexSource: String, fragmentSource: String): Int {
        val vs = loadShader(GLES20.GL_VERTEX_SHADER, vertexSource)
        val fs = loadShader(GLES20.GL_FRAGMENT_SHADER, fragmentSource)
        val prog = GLES20.glCreateProgram()
        GLES20.glAttachShader(prog, vs)
        GLES20.glAttachShader(prog, fs)
        GLES20.glLinkProgram(prog)
        return prog
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

    // Standard quad — no Y-flip since we render to a Surface, not a Bitmap.
    private val quadVertices = floatArrayOf(
        -1f, -1f, 0f, 0f,
         1f, -1f, 1f, 0f,
        -1f,  1f, 0f, 1f,
         1f,  1f, 1f, 1f
    )

    private val quadBuffer = ByteBuffer.allocateDirect(quadVertices.size * 4)
        .order(ByteOrder.nativeOrder())
        .asFloatBuffer()
        .apply { put(quadVertices); position(0) }

    private fun drawQuad(prog: Int) {
        val posLoc = GLES20.glGetAttribLocation(prog, "aPosition")
        val texLoc = GLES20.glGetAttribLocation(prog, "aTexCoord")

        quadBuffer.position(0)
        GLES20.glVertexAttribPointer(posLoc, 2, GLES20.GL_FLOAT, false, 16, quadBuffer)
        GLES20.glEnableVertexAttribArray(posLoc)

        quadBuffer.position(2)
        GLES20.glVertexAttribPointer(texLoc, 2, GLES20.GL_FLOAT, false, 16, quadBuffer)
        GLES20.glEnableVertexAttribArray(texLoc)

        GLES20.glDrawArrays(GLES20.GL_TRIANGLE_STRIP, 0, 4)
    }
}

/**
 * CameraEffect that applies gyro stabilization to VideoCapture output.
 * Preview is NOT targeted — SurfaceTextureFrameReader handles preview stabilization separately.
 */
class StabilizationEffect(
    processor: StabilizationProcessor
) : CameraEffect(
    VIDEO_CAPTURE,
    processor.executor,
    processor,
    { error -> Log.e("StabEffect", "Stabilization effect error", error) }
)
