package com.haptictrack.audio

import android.content.Context
import android.media.AudioAttributes
import android.speech.tts.TextToSpeech
import android.util.Log
import com.haptictrack.tracking.Cue
import com.haptictrack.tracking.cuePhrase
import java.util.Locale

/**
 * Speaks framing cues to the active media output (the Bluetooth earbud over A2DP).
 *
 * HARD RULE (see plan Global Constraints): media playback only, tagged navigation-guidance.
 * Never start Bluetooth SCO / communication mode — that would seize the mic and break the
 * CAMCORDER recording. A2DP routing happens automatically because we use the media stream.
 */
class VoiceGuide(private val context: Context) {
    private var tts: TextToSpeech? = null
    @Volatile private var ready = false

    fun start() {
        if (tts != null) return
        tts = TextToSpeech(context) { status ->
            if (status == TextToSpeech.SUCCESS) {
                tts?.language = Locale.US
                tts?.setAudioAttributes(
                    AudioAttributes.Builder()
                        .setUsage(AudioAttributes.USAGE_ASSISTANCE_NAVIGATION_GUIDANCE)
                        .setContentType(AudioAttributes.CONTENT_TYPE_SPEECH)
                        .build()
                )
                ready = true
            } else {
                Log.w("VoiceGuide", "TTS init failed: $status")
                tts?.shutdown(); tts = null  // release so a later start() can retry
            }
        }
    }

    /** Speak a cue's phrase, flushing any pending utterance so cues never backlog. */
    fun speak(cue: Cue) {
        if (!ready) return
        val phrase = cuePhrase(cue) ?: return
        tts?.speak(phrase, TextToSpeech.QUEUE_FLUSH, null, "guidance-${cue.name}")
    }

    fun shutdown() {
        tts?.stop(); tts?.shutdown(); tts = null; ready = false
    }
}
