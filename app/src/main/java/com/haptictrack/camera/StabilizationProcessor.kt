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
 */
class StabilizationProcessor(
    private val stabMatrixProvider: () -> FloatArray,
    private val frameTimestampLogger: ((Long, Long) -> Unit)? = null
) : SurfaceProcessor, SurfaceTexture.OnFrameAvailableListener {

    companion object {
        private const val TAG = "StabProcessor"
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

    // GL program
    private var program: Int = 0

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

            if (program == 0) program = createShaderProgram()

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
        frameTimestampLogger?.invoke(frameCount, surfaceTexture.timestamp)
        surfaceTexture.getTransformMatrix(texMatrix)

        EGL14.eglMakeCurrent(eglDisplay, eglSurf, eglSurf, eglContext)
        GLES20.glViewport(0, 0, outputWidth, outputHeight)
        GLES20.glUseProgram(program)

        val texMatLoc = GLES20.glGetUniformLocation(program, "uTexMatrix")
        GLES20.glUniformMatrix4fv(texMatLoc, 1, false, texMatrix, 0)

        val stabMatLoc = GLES20.glGetUniformLocation(program, "uStabMatrix")
        GLES20.glUniformMatrix3fv(stabMatLoc, 1, false, stabMatrixProvider(), 0)

        GLES20.glActiveTexture(GLES20.GL_TEXTURE0)
        GLES20.glBindTexture(GLES11Ext.GL_TEXTURE_EXTERNAL_OES, inputTextureId)
        GLES20.glUniform1i(GLES20.glGetUniformLocation(program, "sTexture"), 0)

        drawQuad()

        EGLExt.eglPresentationTimeANDROID(eglDisplay, eglSurf, surfaceTexture.timestamp)
        EGL14.eglSwapBuffers(eglDisplay, eglSurf)

        frameCount++
        if (frameCount % 300 == 0L) {
            Log.d(TAG, "Processed $frameCount video frames (${outputWidth}x${outputHeight})")
        }
    }

    fun release() {
        released = true
        glHandler.post {
            cleanupInput()
            cleanupOutput()
            if (program != 0) {
                GLES20.glDeleteProgram(program)
                program = 0
            }
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
        if (outputEglSurface != EGL14.EGL_NO_SURFACE) {
            EGL14.eglDestroySurface(eglDisplay, outputEglSurface)
            outputEglSurface = EGL14.EGL_NO_SURFACE
        }
        outputSurface = null
        outputSurfaceOutput?.close()
        outputSurfaceOutput = null
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

    private val vertexShaderSource = """
        attribute vec4 aPosition;
        attribute vec2 aTexCoord;
        uniform mat4 uTexMatrix;
        uniform mat3 uStabMatrix;
        varying vec2 vTexCoord;
        void main() {
            gl_Position = aPosition;
            // .xy discards w — valid because the homography keeps w≈1 for small rotations
            vec2 stabUV = (uStabMatrix * vec3(aTexCoord, 1.0)).xy;
            vTexCoord = (uTexMatrix * vec4(stabUV, 0.0, 1.0)).xy;
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
        val vs = loadShader(GLES20.GL_VERTEX_SHADER, vertexShaderSource)
        val fs = loadShader(GLES20.GL_FRAGMENT_SHADER, fragmentShaderSource)
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

    private fun drawQuad() {
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
