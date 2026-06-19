package com.haptictrack.tracking

import android.content.Context
import android.graphics.Bitmap
import android.graphics.RectF
import android.util.Log
import org.json.JSONObject
import java.io.File
import java.io.PrintWriter
import java.util.concurrent.Executors

/**
 * Per-activation sentry session log for off-device troubleshooting.
 *
 * Each arm() opens a session folder under files/sentry_logs/session_<ts>/ with:
 *  - session.json      header (criteria, start time)
 *  - events.jsonl      one line per event (inspect/classify/match/reject/...)
 *  - <seq>_<tag>.png   frames saved at decision points (candidate box in the event)
 *
 * Pull with: adb pull /sdcard/Android/data/com.haptictrack/files/sentry_logs/
 * All IO runs on a single background thread so the processing thread never blocks.
 * Auto-prunes to the most recent [MAX_SESSIONS] sessions.
 */
class SentryLogger(context: Context) {

    companion object {
        private const val TAG = "SentryLog"
        private const val MAX_SESSIONS = 20
    }

    private val baseDir = File(context.getExternalFilesDir(null), "sentry_logs")
    private val io = Executors.newSingleThreadExecutor()

    @Volatile private var sessionDir: File? = null
    @Volatile private var events: PrintWriter? = null
    private var seq = 0
    private var startNs = 0L

    val active: Boolean get() = sessionDir != null

    fun arm(criteria: SentryCriteria) {
        if (active) return
        startNs = System.currentTimeMillis()
        io.execute {
            try {
                baseDir.mkdirs()
                prune()
                val dir = File(baseDir, "session_$startNs").apply { mkdirs() }
                File(dir, "session.json").writeText(JSONObject().apply {
                    put("startMs", startNs)
                    put("gender", criteria.gender.name)
                    put("ageMin", criteria.ageMin)
                    put("ageMax", criteria.ageMax)
                }.toString())
                events = PrintWriter(File(dir, "events.jsonl").bufferedWriter(), false)
                sessionDir = dir
                Log.i(TAG, "Sentry session started: ${dir.absolutePath}")
            } catch (e: Exception) {
                Log.w(TAG, "arm failed: ${e.message}")
            }
        }
    }

    /** Log one event. [box] is the candidate's screen-normalized bbox; [attr] its classification. */
    fun event(type: String, box: RectF? = null, attr: FaceAttributes? = null, note: String? = null) {
        val tMs = System.currentTimeMillis()
        io.execute {
            val w = events ?: return@execute
            val o = JSONObject()
            o.put("t", tMs - startNs)
            o.put("type", type)
            if (box != null) o.put("box", "${f(box.left)},${f(box.top)},${f(box.right)},${f(box.bottom)}")
            if (attr != null) {
                o.put("gender", attr.genderLabel)
                o.put("age", attr.age)
                o.put("ageBucket", attr.ageBucket)
                o.put("genderConf", attr.genderConfidence)
            }
            if (note != null) o.put("note", note)
            w.println(o.toString())
            w.flush()
        }
    }

    /** Save a frame for the current session. Takes ownership of [bmp] (recycled after write). */
    fun saveFrame(tag: String, bmp: Bitmap) {
        val dir = sessionDir
        if (dir == null) { bmp.recycle(); return }
        val n = seq++
        io.execute {
            try {
                File(dir, "%04d_%s.png".format(n, tag)).outputStream().use {
                    bmp.compress(Bitmap.CompressFormat.PNG, 100, it)
                }
            } catch (e: Exception) {
                Log.w(TAG, "saveFrame failed: ${e.message}")
            } finally {
                bmp.recycle()
            }
        }
    }

    fun disarm(matched: Int, inspected: Int) {
        val dir = sessionDir ?: return
        val durMs = System.currentTimeMillis() - startNs
        io.execute {
            try {
                events?.flush(); events?.close()
                File(dir, "summary.json").writeText(JSONObject().apply {
                    put("durationMs", durMs)
                    put("inspected", inspected)
                    put("matched", matched)
                }.toString())
                Log.i(TAG, "Sentry session ended (${durMs}ms, inspected=$inspected matched=$matched)")
            } catch (e: Exception) {
                Log.w(TAG, "disarm failed: ${e.message}")
            }
        }
        events = null
        sessionDir = null
        seq = 0
    }

    private fun prune() {
        val sessions = baseDir.listFiles { f -> f.isDirectory && f.name.startsWith("session_") } ?: return
        if (sessions.size < MAX_SESSIONS) return
        sessions.sortedBy { it.name }.dropLast(MAX_SESSIONS - 1).forEach { it.deleteRecursively() }
    }

    private fun f(v: Float) = "%.4f".format(v)
}
