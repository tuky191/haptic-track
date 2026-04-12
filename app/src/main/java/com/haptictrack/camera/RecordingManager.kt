package com.haptictrack.camera

import android.Manifest
import android.content.ContentValues
import android.content.Context
import android.os.Environment
import android.provider.MediaStore
import androidx.annotation.RequiresPermission
import androidx.camera.video.MediaStoreOutputOptions
import androidx.camera.video.Recording
import androidx.camera.video.VideoCapture
import androidx.camera.video.VideoRecordEvent
import java.text.SimpleDateFormat
import java.util.Locale

class RecordingManager(private val context: Context) {

    private var activeRecording: Recording? = null

    val isRecording: Boolean
        get() = activeRecording != null

    @RequiresPermission(Manifest.permission.RECORD_AUDIO)
    fun startRecording(
        videoCapture: VideoCapture<androidx.camera.video.Recorder>,
        onEvent: (VideoRecordEvent) -> Unit
    ) {
        val timestamp = SimpleDateFormat("yyyyMMdd_HHmmss", Locale.US).format(System.currentTimeMillis())
        val contentValues = ContentValues().apply {
            put(MediaStore.MediaColumns.DISPLAY_NAME, "HapticTrack_$timestamp")
            put(MediaStore.MediaColumns.MIME_TYPE, "video/mp4")
            put(MediaStore.MediaColumns.RELATIVE_PATH, Environment.DIRECTORY_MOVIES + "/HapticTrack")
        }

        val outputOptions = MediaStoreOutputOptions.Builder(
            context.contentResolver,
            MediaStore.Video.Media.EXTERNAL_CONTENT_URI
        ).setContentValues(contentValues).build()

        activeRecording = videoCapture.output
            .prepareRecording(context, outputOptions)
            .withAudioEnabled()
            .start(context.mainExecutor, onEvent)
    }

    fun stopRecording() {
        activeRecording?.stop()
        activeRecording = null
    }
}
