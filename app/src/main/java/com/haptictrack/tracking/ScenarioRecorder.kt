package com.haptictrack.tracking

import android.graphics.RectF
import android.util.Base64
import android.util.Log
import org.json.JSONArray
import org.json.JSONObject
import java.io.File
import java.nio.ByteBuffer
import java.nio.ByteOrder

/**
 * Records ReacquisitionEngine inputs to a JSON file for deterministic replay testing.
 *
 * Captures:
 * - Lock state (label, COCO label, bounding box, gallery embeddings, color histogram, person attrs)
 * - Per-frame detection list with all signals (embedding, color histogram, person attrs)
 * - Engine events (LOST, REACQUIRE, TIMEOUT, CLEAR) with frame numbers
 *
 * Output: scenario.json in the session's debug directory.
 *
 * Note: only frames processed via the detector path (ObjectTracker.processImage) are recorded.
 * Visual tracker frames are not captured — replayed scenarios test ReacquisitionEngine only.
 *
 * Thread safety: all methods are synchronized. Called from UI thread (lock/clear)
 * and camera thread (recordFrame/recordEvent).
 *
 * Serialization is deferred to [stop] to keep the camera thread fast.
 */
class ScenarioRecorder {

    companion object {
        private const val TAG = "ScenarioRec"
    }

    private var recording = false
    private var lockState: JSONObject? = null
    private var rawFrames = mutableListOf<FrameData>()
    private var rawEvents = mutableListOf<EventData>()
    private var frameIndex = 0
    private var outputFile: File? = null

    val isRecording: Boolean get() = recording

    /** Per-frame snapshot — stored raw, serialized lazily in [stop]. */
    private data class FrameData(
        val index: Int,
        val detections: List<TrackedObject>
    )

    private data class EventData(
        val frame: Int,
        val type: String,
        val details: JSONObject?
    )

    /**
     * Start recording. Call after ReacquisitionEngine.lock().
     * [sessionDir] is the debug session directory where scenario.json will be written.
     */
    @Synchronized
    fun start(
        sessionDir: File,
        trackingId: Int,
        label: String?,
        cocoLabel: String?,
        boundingBox: RectF,
        embeddings: List<FloatArray>,
        colorHistogram: FloatArray?
    ) {
        lockState = JSONObject().apply {
            put("trackingId", trackingId)
            put("label", label ?: JSONObject.NULL)
            put("cocoLabel", cocoLabel ?: JSONObject.NULL)
            put("boundingBox", boxToJson(boundingBox))
            put("embeddings", JSONArray().apply {
                embeddings.forEach { put(floatArrayToBase64(it)) }
            })
            put("colorHistogram", colorHistogram?.let { floatArrayToBase64(it) } ?: JSONObject.NULL)
        }
        rawFrames = mutableListOf()
        rawEvents = mutableListOf()
        frameIndex = 0
        outputFile = File(sessionDir, "scenario.json")
        recording = true
        Log.d(TAG, "Recording started: ${outputFile?.absolutePath}")
    }

    /**
     * Record one frame of detections. Call with the same list passed to processFrame().
     * Stores references only — serialization is deferred to [stop].
     */
    @Synchronized
    fun recordFrame(detections: List<TrackedObject>) {
        if (!recording) return
        rawFrames.add(FrameData(frameIndex, detections.toList()))
        frameIndex++
    }

    /**
     * Record an event that happened on the current frame.
     */
    @Synchronized
    fun recordEvent(type: String, details: JSONObject? = null) {
        if (!recording) return
        rawEvents.add(EventData(frameIndex - 1, type, details))
    }

    /**
     * Stop recording and flush to disk. Serialization happens here, off the camera thread.
     */
    @Synchronized
    fun stop() {
        if (!recording) return
        recording = false
        flush()
        Log.d(TAG, "Recording stopped: $frameIndex frames, ${rawEvents.size} events → ${outputFile?.name}")
    }

    private fun flush() {
        val file = outputFile ?: return
        val lock = lockState ?: return

        val framesJson = JSONArray()
        for (frame in rawFrames) {
            framesJson.put(JSONObject().apply {
                put("index", frame.index)
                put("detections", JSONArray().apply {
                    frame.detections.forEach { put(trackedObjectToJson(it)) }
                })
            })
        }

        val eventsJson = JSONArray()
        for (event in rawEvents) {
            eventsJson.put(JSONObject().apply {
                put("frame", event.frame)
                put("type", event.type)
                if (event.details != null) put("details", event.details)
            })
        }

        val scenario = JSONObject().apply {
            put("version", 1)
            put("lock", lock)
            put("frames", framesJson)
            put("events", eventsJson)
        }
        try {
            file.writeText(scenario.toString())
        } catch (e: Exception) {
            Log.e(TAG, "Failed to write scenario: ${e.message}")
        }
    }

    // --- Serialization helpers ---

    private fun trackedObjectToJson(obj: TrackedObject): JSONObject {
        return JSONObject().apply {
            put("id", obj.id)
            put("label", obj.label ?: JSONObject.NULL)
            put("confidence", obj.confidence.toDouble())
            put("boundingBox", boxToJson(obj.boundingBox))
            put("embedding", obj.embedding?.let { floatArrayToBase64(it) } ?: JSONObject.NULL)
            put("colorHistogram", obj.colorHistogram?.let { floatArrayToBase64(it) } ?: JSONObject.NULL)
            put("reIdEmbedding", obj.reIdEmbedding?.let { floatArrayToBase64(it) } ?: JSONObject.NULL)
            put("faceEmbedding", obj.faceEmbedding?.let { floatArrayToBase64(it) } ?: JSONObject.NULL)
        }
    }

    private fun boxToJson(box: RectF): JSONArray {
        return JSONArray().apply {
            put(box.left.toDouble())
            put(box.top.toDouble())
            put(box.right.toDouble())
            put(box.bottom.toDouble())
        }
    }
}

// --- Shared serialization for recorder and replay ---

/** Encode FloatArray as base64 (little-endian IEEE 754). Compact for embeddings (~1KB for 256-dim). */
fun floatArrayToBase64(arr: FloatArray): String {
    val buf = ByteBuffer.allocate(arr.size * 4).order(ByteOrder.LITTLE_ENDIAN)
    arr.forEach { buf.putFloat(it) }
    return Base64.encodeToString(buf.array(), Base64.NO_WRAP)
}

/** Decode base64 back to FloatArray. */
fun base64ToFloatArray(b64: String): FloatArray {
    val bytes = Base64.decode(b64, Base64.NO_WRAP)
    val buf = ByteBuffer.wrap(bytes).order(ByteOrder.LITTLE_ENDIAN)
    return FloatArray(bytes.size / 4) { buf.getFloat() }
}

/** Deserialize a JSONArray [left, top, right, bottom] to RectF. */
fun jsonToBox(arr: JSONArray): RectF {
    return RectF(
        arr.getDouble(0).toFloat(),
        arr.getDouble(1).toFloat(),
        arr.getDouble(2).toFloat(),
        arr.getDouble(3).toFloat()
    )
}

/** Deserialize a JSONObject to TrackedObject. */
fun jsonToTrackedObject(obj: JSONObject): TrackedObject {
    return TrackedObject(
        id = obj.getInt("id"),
        label = if (!obj.isNull("label")) obj.getString("label") else null,
        confidence = obj.getDouble("confidence").toFloat(),
        boundingBox = jsonToBox(obj.getJSONArray("boundingBox")),
        embedding = if (!obj.isNull("embedding")) base64ToFloatArray(obj.getString("embedding")) else null,
        colorHistogram = if (!obj.isNull("colorHistogram")) base64ToFloatArray(obj.getString("colorHistogram")) else null,
        reIdEmbedding = if (obj.has("reIdEmbedding") && !obj.isNull("reIdEmbedding")) base64ToFloatArray(obj.getString("reIdEmbedding")) else null,
        faceEmbedding = if (obj.has("faceEmbedding") && !obj.isNull("faceEmbedding")) base64ToFloatArray(obj.getString("faceEmbedding")) else null
    )
}
