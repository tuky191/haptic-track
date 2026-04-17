package com.haptictrack.tracking

import org.json.JSONArray
import org.json.JSONObject
import org.junit.Assert.*
import org.junit.Test
import org.junit.runner.RunWith
import org.robolectric.RobolectricTestRunner

/**
 * Replays captured scenarios through ReacquisitionEngine and asserts expected events.
 *
 * Scenario JSON files in src/test/resources/scenarios/ are captured on-device
 * by [ScenarioRecorder] during real tracking sessions. Each file contains:
 * - lock state (label, embeddings, color histogram, person attributes)
 * - per-frame detection lists with all signals
 * - recorded events (LOST, REACQUIRE, TIMEOUT) with frame indices
 *
 * Tests replay frames through a fresh engine and verify the same events occur.
 * This makes real-world failures into reproducible regression tests.
 */
@RunWith(RobolectricTestRunner::class)
class ScenarioReplayTest {

    // No-arg convenience: uses maxFramesLost from scenario JSON if present
    private fun replay(scenario: JSONObject): ReplayResult {
        val maxLost = scenario.optInt("maxFramesLost", 450)
        val engine = ReacquisitionEngine(maxFramesLost = maxLost)
        return replayWithEngine(scenario, engine)
    }

    data class ReplayEvent(
        val frame: Int,
        val type: String,
        val objectId: Int?,
        val label: String?
    )

    data class ReplayResult(
        val engine: ReacquisitionEngine,
        val events: List<ReplayEvent>
    )

    // --- Scenario loading ---

    private fun loadScenario(name: String): JSONObject {
        val stream = javaClass.classLoader!!.getResourceAsStream("scenarios/$name")
            ?: throw IllegalArgumentException("Scenario not found: scenarios/$name")
        return JSONObject(stream.bufferedReader().readText())
    }

    // --- Assertion helpers ---

    /** Assert that the scenario produces a REACQUIRE event for the given label. */
    private fun assertReacquiresLabel(result: ReplayResult, expectedLabel: String, message: String? = null) {
        val reacquire = result.events.firstOrNull { it.type == "REACQUIRE" }
        assertNotNull("${message ?: "Expected REACQUIRE"} but got events: ${result.events}", reacquire)
        assertEquals(message ?: "Expected reacquire label", expectedLabel, reacquire!!.label)
    }

    /** Assert that the scenario never reacquires the given label. */
    private fun assertNeverReacquiresLabel(result: ReplayResult, rejectedLabel: String, message: String? = null) {
        val bad = result.events.filter { it.type == "REACQUIRE" && it.label == rejectedLabel }
        assertTrue("${message ?: "Should not reacquire '$rejectedLabel'"} but did at frames: ${bad.map { it.frame }}", bad.isEmpty())
    }

    /** Assert that the scenario times out (no successful reacquisition). */
    private fun assertTimesOut(result: ReplayResult, message: String? = null) {
        val timeout = result.events.any { it.type == "TIMEOUT" }
        assertTrue("${message ?: "Expected TIMEOUT"} but got events: ${result.events}", timeout)
    }

    // --- Synthetic scenario tests (validate the harness itself) ---

    @Test
    fun `replay harness handles empty frame sequence`() {
        val scenario = buildSyntheticScenario(
            label = "cup",
            frames = listOf(emptyList())
        )
        val result = replay(scenario)
        assertEquals(1, result.events.size)
        assertEquals("LOST", result.events[0].type)
    }

    @Test
    fun `replay harness detects reacquisition`() {
        val emb = floatArrayOf(0.9f, 0.3f, 0.1f)
        val scenario = buildSyntheticScenario(
            label = "cup",
            embedding = emb,
            frames = listOf(
                emptyList(), // frame 0: lost
                listOf(SyntheticDetection(id = 55, label = "cup", box = floatArrayOf(0.42f, 0.42f, 0.62f, 0.62f),
                    embedding = floatArrayOf(0.85f, 0.35f, 0.1f))) // frame 1: reacquire
            )
        )
        val result = replay(scenario)
        val lost = result.events.filter { it.type == "LOST" }
        val reacq = result.events.filter { it.type == "REACQUIRE" }
        assertEquals(1, lost.size)
        assertEquals(1, reacq.size)
        assertEquals(55, reacq[0].objectId)
    }

    @Test
    fun `replay harness detects timeout`() {
        val scenario = buildSyntheticScenario(
            label = "cup",
            maxFramesLost = 5,
            frames = List(10) { emptyList() } // 10 empty frames, timeout at 5
        )
        val result = replay(scenario)
        assertTrue(result.events.any { it.type == "TIMEOUT" })
    }

    // --- Real captured scenarios ---

    @Test
    fun `cup reacquisition - replays recorded events`() {
        val scenario = loadScenario("cup_reacquisition.json")
        val result = replay(scenario)

        // Session had: LOST at 0, REACQUIRE cup at 8, LOST at 10, REACQUIRE cup at 14, TIMEOUT at 15
        val lostEvents = result.events.filter { it.type == "LOST" }
        val reacqEvents = result.events.filter { it.type == "REACQUIRE" }

        assertTrue("Should have at least one LOST event", lostEvents.isNotEmpty())
        assertTrue("Should reacquire at least once", reacqEvents.isNotEmpty())
        // Every reacquisition should be a cup, not a keyboard/bed/person
        reacqEvents.forEach { event ->
            assertEquals("Reacquired object should be cup", "cup", event.label)
        }
    }

    @Test
    fun `cup reacquisition - never locks on wrong category`() {
        val scenario = loadScenario("cup_reacquisition.json")
        val result = replay(scenario)

        assertNeverReacquiresLabel(result, "keyboard")
        assertNeverReacquiresLabel(result, "bed")
        assertNeverReacquiresLabel(result, "person")
        assertNeverReacquiresLabel(result, "laptop")
    }

    @Test
    fun `mouse cascade - always reacquires mouse never wrong category`() {
        val scenario = loadScenario("mouse_cascade_reacquisition.json")
        val result = replay(scenario)

        val reacqEvents = result.events.filter { it.type == "REACQUIRE" }
        assertTrue("Should reacquire multiple times", reacqEvents.size >= 4)
        reacqEvents.forEach { event ->
            assertEquals("Every reacquisition should be mouse", "mouse", event.label)
        }
    }

    @Test
    fun `mouse cascade - never locks on keyboard tv or person`() {
        val scenario = loadScenario("mouse_cascade_reacquisition.json")
        val result = replay(scenario)

        assertNeverReacquiresLabel(result, "keyboard")
        assertNeverReacquiresLabel(result, "tv")
        assertNeverReacquiresLabel(result, "person")
        assertNeverReacquiresLabel(result, "scissors")
    }

    // --- Helpers for building synthetic scenarios ---

    data class SyntheticDetection(
        val id: Int,
        val label: String?,
        val box: FloatArray = floatArrayOf(0.4f, 0.4f, 0.6f, 0.6f),
        val confidence: Float = 0.8f,
        val embedding: FloatArray? = null,
        val colorHistogram: FloatArray? = null
    )

    private fun buildSyntheticScenario(
        label: String?,
        cocoLabel: String? = null,
        box: FloatArray = floatArrayOf(0.4f, 0.4f, 0.6f, 0.6f),
        embedding: FloatArray? = null,
        maxFramesLost: Int? = null,
        frames: List<List<SyntheticDetection>>
    ): JSONObject {
        val lockJson = JSONObject().apply {
            put("trackingId", 1)
            put("label", label ?: JSONObject.NULL)
            put("cocoLabel", cocoLabel ?: JSONObject.NULL)
            put("boundingBox", JSONArray().apply {
                box.forEach { v -> put(v.toDouble()) }
            })
            put("embeddings", JSONArray().also { arr ->
                if (embedding != null) arr.put(floatArrayToBase64(embedding))
            })
            put("colorHistogram", JSONObject.NULL)
            put("personAttributes", JSONObject.NULL)
        }

        val framesJson = JSONArray()
        frames.forEachIndexed { idx, dets ->
            val detsJson = JSONArray()
            for (d in dets) {
                val boxArr = JSONArray()
                d.box.forEach { v -> boxArr.put(v.toDouble()) }
                detsJson.put(JSONObject().apply {
                    put("id", d.id)
                    put("label", d.label ?: JSONObject.NULL)
                    put("confidence", d.confidence.toDouble())
                    put("boundingBox", boxArr)
                    put("embedding", d.embedding?.let { floatArrayToBase64(it) } ?: JSONObject.NULL)
                    put("colorHistogram", d.colorHistogram?.let { floatArrayToBase64(it) } ?: JSONObject.NULL)
                    put("personAttributes", JSONObject.NULL)
                })
            }
            framesJson.put(JSONObject().apply {
                put("index", idx)
                put("detections", detsJson)
            })
        }

        return JSONObject().apply {
            put("version", 1)
            put("lock", lockJson)
            put("frames", framesJson)
            put("events", JSONArray())
            if (maxFramesLost != null) put("maxFramesLost", maxFramesLost)
        }
    }

    private fun replayWithEngine(scenario: JSONObject, engine: ReacquisitionEngine): ReplayResult {
        val lockJson = scenario.getJSONObject("lock")

        val embeddings = mutableListOf<FloatArray>()
        val embJson = lockJson.getJSONArray("embeddings")
        for (i in 0 until embJson.length()) {
            embeddings.add(base64ToFloatArray(embJson.getString(i)))
        }
        val colorHist = if (!lockJson.isNull("colorHistogram"))
            base64ToFloatArray(lockJson.getString("colorHistogram")) else null
        val personAttrs = if (!lockJson.isNull("personAttributes"))
            jsonToPersonAttributes(lockJson.getJSONObject("personAttributes")) else null
        val cocoLabel = if (!lockJson.isNull("cocoLabel"))
            lockJson.getString("cocoLabel") else null

        engine.lock(
            trackingId = lockJson.getInt("trackingId"),
            boundingBox = jsonToBox(lockJson.getJSONArray("boundingBox")),
            label = if (!lockJson.isNull("label")) lockJson.getString("label") else null,
            embeddings = embeddings,
            colorHist = colorHist,
            personAttrs = personAttrs,
            cocoLabel = cocoLabel
        )

        val events = mutableListOf<ReplayEvent>()
        val framesJson = scenario.getJSONArray("frames")

        for (i in 0 until framesJson.length()) {
            val frame = framesJson.getJSONObject(i)
            val detections = mutableListOf<TrackedObject>()
            val detsJson = frame.getJSONArray("detections")
            for (j in 0 until detsJson.length()) {
                detections.add(jsonToTrackedObject(detsJson.getJSONObject(j)))
            }

            val wasSearching = engine.isSearching
            val prevLost = engine.framesLost
            val result = engine.processFrame(detections)

            val nowLost = engine.framesLost
            when {
                wasSearching && result != null && nowLost == 0 ->
                    events.add(ReplayEvent(i, "REACQUIRE", result.id, result.label))
                nowLost == 1 && prevLost == 0 ->
                    events.add(ReplayEvent(i, "LOST", null, null))
                engine.hasTimedOut && prevLost <= engine.maxFramesLost ->
                    events.add(ReplayEvent(i, "TIMEOUT", null, null))
            }
        }

        return ReplayResult(engine, events)
    }
}
