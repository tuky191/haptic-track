package com.haptictrack.tracking

import android.util.Log
import org.json.JSONArray
import org.json.JSONObject
import java.io.File

/**
 * Audit instrumentation (#92): measures how stable consecutive embeddings
 * of the locked object are across frames. The noise floor — how much an
 * embedding of the same object varies frame-to-frame just from input jitter
 * (aspect distortion, segmenter flicker, lighting, micro-movement) — bounds
 * how cleanly any model can separate same-object from different-object.
 *
 * For each embedder (MNV3, OSNet, MobileFaceNet), we keep a ring of the most
 * recent audit-sample embeddings of the locked object. Each new sample is
 * compared against samples 1, 5, and 30 entries back. At session end we dump
 * percentile stats and the raw history to JSON for offline plotting.
 *
 * "k=1, 5, 30" are in audit samples, not frames. With the default sampling
 * cadence in [ObjectTracker] (every 5 confirmed frames during VT-confirmed
 * tracking), k=1 ≈ 5 frames ≈ 167 ms; k=30 ≈ 150 frames ≈ 5 s. The exact
 * sampling interval is recorded in the JSON so the conversion is unambiguous.
 *
 * Cost is dominated by the embedder calls themselves — the ring + cosine
 * bookkeeping is microseconds. Compile-time gated by the audit flag in
 * [CropDebugCapture].
 */
class EmbeddingStabilityLogger {

    companion object {
        private const val TAG = "EmbedAudit"
        const val RING_CAPACITY = 64           // must be ≥ max(K_VALUES) + 1
        val K_VALUES = intArrayOf(1, 5, 30)    // samples back to compare against
    }

    private class Ring(capacity: Int) {
        val embeddings = arrayOfNulls<FloatArray>(capacity)
        val frames = IntArray(capacity)
        var size = 0
        var head = 0   // next write index

        fun add(frame: Int, emb: FloatArray, capacity: Int) {
            embeddings[head] = emb
            frames[head] = frame
            head = (head + 1) % capacity
            if (size < capacity) size++
        }

        /** Returns the embedding [k] samples back from the most recent, or null if not enough samples yet. */
        fun nthBack(k: Int, capacity: Int): Pair<Int, FloatArray>? {
            if (size <= k) return null
            val idx = (head - 1 - k + capacity) % capacity
            val emb = embeddings[idx] ?: return null
            return frames[idx] to emb
        }
    }

    private data class EmbedderStats(
        val name: String,
        val ring: Ring = Ring(RING_CAPACITY),
        // Per-k history of measured cosine similarities. Indexed by entry order.
        val deltas: HashMap<Int, MutableList<Float>> = HashMap<Int, MutableList<Float>>().apply {
            for (k in K_VALUES) put(k, ArrayList())
        },
        // Per-sample frame indices, for offline correlation.
        val sampledFrames: MutableList<Int> = ArrayList(),
    )

    private val embedders = LinkedHashMap<String, EmbedderStats>()

    /** Sampling cadence (in confirmed VT frames) — recorded in the output JSON for clarity. */
    @Volatile var samplingIntervalFrames: Int = 0

    /**
     * Record a fresh embedding of the locked object. Computes cosine similarity
     * against entries 1, 5, 30 samples back (when available) and accumulates them.
     *
     * Synchronized so that audit work running on a background thread can safely
     * call [record] while the foreground thread calls [clear] / [flush] in
     * `clearLock`.
     */
    @Synchronized
    fun record(embedderName: String, frameIndex: Int, embedding: FloatArray?) {
        if (!CropDebugCapture.AUDIT_ENABLED) return
        val emb = embedding ?: return
        val stats = embedders.getOrPut(embedderName) { EmbedderStats(embedderName) }
        // Compute deltas first (against the existing ring), then add.
        for (k in K_VALUES) {
            val past = stats.ring.nthBack(k - 1, RING_CAPACITY) ?: continue
            val sim = cosineSimilarity(past.second, emb)
            stats.deltas[k]?.add(sim)
        }
        // Defensive copy — the caller may reuse the array.
        stats.ring.add(frameIndex, emb.copyOf(), RING_CAPACITY)
        stats.sampledFrames.add(frameIndex)
    }

    /** Reset all state — call on lock/clear. */
    @Synchronized
    fun clear() {
        embedders.clear()
    }

    /**
     * Write `embedding_stability.json` under [parentSessionDir]. Returns the
     * resulting JSON object so callers (e.g. VideoReplayTest) can aggregate
     * across sessions without re-reading from disk.
     */
    @Synchronized
    fun flush(parentSessionDir: File?): JSONObject? {
        if (!CropDebugCapture.AUDIT_ENABLED) return null
        if (embedders.isEmpty()) return null
        val out = JSONObject()
        out.put("samplingIntervalFrames", samplingIntervalFrames)
        out.put("k_values", JSONArray(K_VALUES.toList()))
        val embObj = JSONObject()
        for ((name, stats) in embedders) {
            val e = JSONObject()
            e.put("samples", stats.sampledFrames.size)
            e.put("frames", JSONArray(stats.sampledFrames))
            for (k in K_VALUES) {
                val list = stats.deltas[k] ?: continue
                if (list.isEmpty()) continue
                val sorted = list.sorted()
                val k1 = JSONObject()
                k1.put("n", list.size)
                k1.put("p10", percentile(sorted, 0.10))
                k1.put("p50", percentile(sorted, 0.50))
                k1.put("p90", percentile(sorted, 0.90))
                k1.put("mean", list.average())
                k1.put("history", JSONArray(list))
                e.put("k$k", k1)
            }
            embObj.put(name, e)
        }
        out.put("embedders", embObj)

        if (parentSessionDir != null) {
            try {
                File(parentSessionDir, "embedding_stability.json")
                    .writeText(out.toString(2))
            } catch (e: Exception) {
                Log.w(TAG, "Failed to write embedding_stability.json: ${e.message}")
            }
        }
        return out
    }

    private fun percentile(sorted: List<Float>, p: Double): Double {
        if (sorted.isEmpty()) return 0.0
        val idx = ((sorted.size - 1) * p).toInt().coerceIn(0, sorted.size - 1)
        return sorted[idx].toDouble()
    }
}
