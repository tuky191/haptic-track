package com.haptictrack.tracking

import org.json.JSONArray
import org.json.JSONObject
import org.junit.Assert.*
import org.junit.Test
import org.junit.runner.RunWith
import org.robolectric.RobolectricTestRunner

/**
 * Phase 0 ablation tests (#123) for model stack modernization (#129).
 *
 * Ablation A: replay all scenarios with person-attribute scoring disabled
 *             (disableAttributes = true). If all pass → Crossroad-0230 is safe to drop.
 *
 * Ablation D: classify re-acquisition failures as detection-side vs scoring-side.
 *             Determines whether Phase 2 (better detector) is high-value.
 *
 * Ablation B (segmentation) cannot be tested via scenario replay — embeddings
 * are pre-computed at capture time. Requires on-device testing.
 */
@RunWith(RobolectricTestRunner::class)
class AblationTest {

    // ── Replay infrastructure (mirrors ScenarioReplayTest) ──

    data class ReplayEvent(
        val frame: Int,
        val type: String,
        val objectId: Int?,
        val label: String?
    )

    data class ReplayResult(
        val engine: ReacquisitionEngine,
        val events: List<ReplayEvent>,
        val framesTracked: Int = 0,
        val totalFrames: Int = 0
    ) {
        val trackingRate: Int get() = if (totalFrames > 0) framesTracked * 100 / totalFrames else 0
        val reacquisitions: Int get() = events.count { it.type == "REACQUIRE" }
        val losses: Int get() = events.count { it.type == "LOST" }
        val timedOut: Boolean get() = events.any { it.type == "TIMEOUT" }
        fun wrongCategoryReacqs(validLabels: Set<String>): List<ReplayEvent> =
            events.filter { it.type == "REACQUIRE" && it.label !in validLabels }
    }

    private fun loadScenario(name: String): JSONObject {
        val stream = javaClass.classLoader!!.getResourceAsStream("scenarios/$name")
            ?: throw IllegalArgumentException("Scenario not found: scenarios/$name")
        return JSONObject(stream.bufferedReader().readText())
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
        var framesTracked = 0

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
            if (nowLost == 0) framesTracked++
            when {
                wasSearching && result != null && nowLost == 0 ->
                    events.add(ReplayEvent(i, "REACQUIRE", result.id, result.label))
                nowLost == 1 && prevLost == 0 ->
                    events.add(ReplayEvent(i, "LOST", null, null))
                engine.hasTimedOut && prevLost <= engine.maxFramesLost ->
                    events.add(ReplayEvent(i, "TIMEOUT", null, null))
            }
        }

        return ReplayResult(engine, events, framesTracked, framesJson.length())
    }

    private fun replayNoAttrs(scenario: JSONObject): ReplayResult {
        val maxLost = scenario.optInt("maxFramesLost", 450)
        val engine = ReacquisitionEngine(maxFramesLost = maxLost, disableAttributes = true)
        return replayWithEngine(scenario, engine)
    }

    private fun replayBaseline(scenario: JSONObject): ReplayResult {
        val maxLost = scenario.optInt("maxFramesLost", 450)
        val engine = ReacquisitionEngine(maxFramesLost = maxLost)
        return replayWithEngine(scenario, engine)
    }

    companion object {
        val PERSON_LABELS = setOf("person", "boy", "girl", "man", "woman", "human face")

        val ALL_SCENARIOS = listOf(
            "boy_indoor_wife_swap.json",
            "boy_label_flicker.json",
            "chair_living_room_wrong_reacq.json",
            "chair_lost_no_recovery.json",
            "cup_no_hop_limit.json",
            "cup_reacquisition.json",
            "man_desk_camera_swing.json",
            "man_multi_person_lock_holds.json",
            "mouse_cascade_reacquisition.json",
            "person_boy_flicker.json",
            "person_playground_tracking.json",
            "person_tracking_recovery.json",
            "two_people_indoor.json"
        )
    }

    // ══════════════════════════════════════════════════════════════
    // ABLATION A: Disable person-attribute scoring
    // ══════════════════════════════════════════════════════════════

    @Test
    fun `ablation A - cup no hop limit`() {
        val result = replayNoAttrs(loadScenario("cup_no_hop_limit.json"))
        assertTrue("Should reacquire >= 6 times, got ${result.reacquisitions}", result.reacquisitions >= 6)
        result.events.filter { it.type == "REACQUIRE" }.forEach {
            assertNotEquals("Should never reacquire as person", "person", it.label)
        }
        assertFalse("Should not timeout", result.timedOut)
    }

    @Test
    fun `ablation A - cup reacquisition - no wrong category`() {
        val result = replayNoAttrs(loadScenario("cup_reacquisition.json"))
        result.events.filter { it.type == "REACQUIRE" }.forEach {
            assertEquals("Reacquired object should be cup", "cup", it.label)
        }
    }

    @Test
    fun `ablation A - mouse cascade`() {
        val result = replayNoAttrs(loadScenario("mouse_cascade_reacquisition.json"))
        assertTrue("Should reacquire >= 4 times, got ${result.reacquisitions}", result.reacquisitions >= 4)
        result.events.filter { it.type == "REACQUIRE" }.forEach {
            assertEquals("Every reacquisition should be mouse", "mouse", it.label)
        }
    }

    @Test
    fun `ablation A - boy label flicker`() {
        val result = replayNoAttrs(loadScenario("boy_label_flicker.json"))
        assertTrue("Should reacquire >= 4 times, got ${result.reacquisitions}", result.reacquisitions >= 4)
        val wrong = result.wrongCategoryReacqs(PERSON_LABELS)
        assertTrue("No wrong-category reacqs (got: ${wrong.map { "${it.label}@F${it.frame}" }})", wrong.isEmpty())
        assertTrue("Tracking rate >= 50%, got ${result.trackingRate}%", result.trackingRate >= 50)
    }

    @Test
    fun `ablation A - person recovery`() {
        val result = replayNoAttrs(loadScenario("person_tracking_recovery.json"))
        assertTrue("Should reacquire >= 2 times, got ${result.reacquisitions}", result.reacquisitions >= 2)
        assertTrue("Tracking rate >= 75%, got ${result.trackingRate}%", result.trackingRate >= 75)
        val wrong = result.wrongCategoryReacqs(PERSON_LABELS)
        assertTrue("No wrong-category reacqs", wrong.isEmpty())
    }

    @Test
    fun `ablation A - person boy flicker`() {
        val result = replayNoAttrs(loadScenario("person_boy_flicker.json"))
        assertTrue("Should reacquire >= 1, got ${result.reacquisitions}", result.reacquisitions >= 1)
        assertTrue("Tracking rate >= 50%, got ${result.trackingRate}%", result.trackingRate >= 50)
        val wrong = result.wrongCategoryReacqs(PERSON_LABELS)
        assertTrue("No wrong-category reacqs", wrong.isEmpty())
    }

    @Test
    fun `ablation A - chair lost no recovery`() {
        val result = replayNoAttrs(loadScenario("chair_lost_no_recovery.json"))
        assertEquals("Should have 0 reacquisitions", 0, result.reacquisitions)
    }

    @Test
    fun `ablation A - man desk swing`() {
        val result = replayNoAttrs(loadScenario("man_desk_camera_swing.json"))
        assertTrue("Should reacquire >= 3, got ${result.reacquisitions}", result.reacquisitions >= 3)
        assertTrue("Tracking rate >= 50%, got ${result.trackingRate}%", result.trackingRate >= 50)
        val wrong = result.wrongCategoryReacqs(PERSON_LABELS)
        assertTrue("No wrong-category reacqs", wrong.isEmpty())
    }

    @Test
    fun `ablation A - playground no wrong category`() {
        val result = replayNoAttrs(loadScenario("person_playground_tracking.json"))
        val wrong = result.wrongCategoryReacqs(PERSON_LABELS)
        assertTrue("No wrong-category reacqs", wrong.isEmpty())
        assertFalse("Should not timeout", result.timedOut)
    }

    @Test
    fun `ablation A - chair living room`() {
        val result = replayNoAttrs(loadScenario("chair_living_room_wrong_reacq.json"))
        assertTrue("Should reacquire >= 1, got ${result.reacquisitions}", result.reacquisitions >= 1)
        assertFalse("Should not timeout", result.timedOut)
        assertFalse("Should not reacquire person",
            result.events.any { it.type == "REACQUIRE" && it.label == "person" })
    }

    @Test
    fun `ablation A - two people indoor`() {
        val result = replayNoAttrs(loadScenario("two_people_indoor.json"))
        val wrong = result.wrongCategoryReacqs(PERSON_LABELS)
        assertTrue("No wrong-category reacqs", wrong.isEmpty())
        assertTrue("Should reacquire >= 1, got ${result.reacquisitions}", result.reacquisitions >= 1)
        assertFalse("Should not timeout", result.timedOut)
    }

    @Test
    fun `ablation A - man multi-person lock holds`() {
        val result = replayNoAttrs(loadScenario("man_multi_person_lock_holds.json"))
        val wrong = result.wrongCategoryReacqs(PERSON_LABELS)
        assertTrue("No wrong-category reacqs", wrong.isEmpty())
        assertTrue("Should reacquire >= 1, got ${result.reacquisitions}", result.reacquisitions >= 1)
        assertFalse("Should not timeout", result.timedOut)
    }

    @Test
    fun `ablation A - boy indoor wife swap`() {
        val result = replayNoAttrs(loadScenario("boy_indoor_wife_swap.json"))
        val wrong = result.wrongCategoryReacqs(PERSON_LABELS)
        assertTrue("No wrong-category reacqs", wrong.isEmpty())
    }

    // ══════════════════════════════════════════════════════════════
    // ABLATION A COMPARISON: measure score delta (not pass/fail)
    // ══════════════════════════════════════════════════════════════

    @Test
    fun `ablation A comparison - all scenarios produce same or better results`() {
        val regressions = mutableListOf<String>()

        for (name in ALL_SCENARIOS) {
            val scenario = loadScenario(name)
            val baseline = replayBaseline(scenario)
            val noAttrs = replayNoAttrs(scenario)

            val baseLabel = name.removeSuffix(".json")

            // Check: no new timeouts
            if (noAttrs.timedOut && !baseline.timedOut) {
                regressions.add("$baseLabel: NEW TIMEOUT (baseline had none)")
            }

            // Check: reacquisition count didn't drop significantly
            if (baseline.reacquisitions > 0 && noAttrs.reacquisitions < baseline.reacquisitions - 1) {
                regressions.add("$baseLabel: reacqs dropped ${baseline.reacquisitions} → ${noAttrs.reacquisitions}")
            }

            // Check: tracking rate didn't drop by more than 5%
            if (noAttrs.trackingRate < baseline.trackingRate - 5) {
                regressions.add("$baseLabel: tracking rate ${baseline.trackingRate}% → ${noAttrs.trackingRate}%")
            }

            // Check: no new wrong-category reacquisitions
            val baselineWrong = baseline.wrongCategoryReacqs(PERSON_LABELS).size
            val noAttrsWrong = noAttrs.wrongCategoryReacqs(PERSON_LABELS).size
            if (noAttrsWrong > baselineWrong) {
                regressions.add("$baseLabel: wrong-category reacqs $baselineWrong → $noAttrsWrong")
            }

            println("  $baseLabel: reacqs=${baseline.reacquisitions}→${noAttrs.reacquisitions} " +
                    "tracking=${baseline.trackingRate}%→${noAttrs.trackingRate}% " +
                    "timeout=${baseline.timedOut}→${noAttrs.timedOut}")
        }

        assertTrue(
            "Ablation A regressions:\n${regressions.joinToString("\n")}",
            regressions.isEmpty()
        )
    }

    // ══════════════════════════════════════════════════════════════
    // ABLATION D: Failure-mode classification
    // ══════════════════════════════════════════════════════════════

    data class FailureClassification(
        val scenario: String,
        val totalLosses: Int,
        val totalReacqs: Int,
        val totalTimeouts: Int,
        val framesSearchingWithCorrectCandidate: Int,
        val framesSearchingWithoutCorrectCandidate: Int
    )

    @Test
    fun `ablation D - classify failures as detection-side vs scoring-side`() {
        val classifications = mutableListOf<FailureClassification>()
        var totalDetectionSide = 0
        var totalScoringSide = 0

        for (name in ALL_SCENARIOS) {
            val scenario = loadScenario(name)
            val lockJson = scenario.getJSONObject("lock")
            val lockLabel = if (!lockJson.isNull("label")) lockJson.getString("label") else null
            val lockCocoLabel = if (!lockJson.isNull("cocoLabel")) lockJson.getString("cocoLabel") else null
            val lockIsPerson = (lockCocoLabel ?: lockLabel) == "person"
            val lockBox = jsonToBox(lockJson.getJSONArray("boundingBox"))

            val framesJson = scenario.getJSONArray("frames")
            val engine = ReacquisitionEngine(maxFramesLost = scenario.optInt("maxFramesLost", 450))

            val embeddings = mutableListOf<FloatArray>()
            val embJson = lockJson.getJSONArray("embeddings")
            for (i in 0 until embJson.length()) embeddings.add(base64ToFloatArray(embJson.getString(i)))
            val colorHist = if (!lockJson.isNull("colorHistogram"))
                base64ToFloatArray(lockJson.getString("colorHistogram")) else null
            val personAttrs = if (!lockJson.isNull("personAttributes"))
                jsonToPersonAttributes(lockJson.getJSONObject("personAttributes")) else null

            engine.lock(
                trackingId = lockJson.getInt("trackingId"),
                boundingBox = lockBox,
                label = lockLabel,
                embeddings = embeddings,
                colorHist = colorHist,
                personAttrs = personAttrs,
                cocoLabel = lockCocoLabel
            )

            var losses = 0
            var reacqs = 0
            var timeouts = 0
            var framesWithCorrect = 0
            var framesWithoutCorrect = 0

            for (i in 0 until framesJson.length()) {
                val frame = framesJson.getJSONObject(i)
                val detections = mutableListOf<TrackedObject>()
                val detsJson = frame.getJSONArray("detections")
                for (j in 0 until detsJson.length()) {
                    detections.add(jsonToTrackedObject(detsJson.getJSONObject(j)))
                }

                val wasSearching = engine.isSearching
                val prevLost = engine.framesLost
                engine.processFrame(detections)
                val nowLost = engine.framesLost

                if (nowLost == 1 && prevLost == 0) losses++
                if (wasSearching && nowLost == 0) reacqs++
                if (engine.hasTimedOut && prevLost <= engine.maxFramesLost) timeouts++

                if (engine.isSearching) {
                    val matchingLabel = if (lockIsPerson) "person" else (lockLabel ?: "")
                    val hasSameCategoryCandidate = detections.any { det ->
                        val detIsPerson = det.label == "person"
                        if (lockIsPerson) detIsPerson else !detIsPerson && det.label == matchingLabel
                    }
                    if (hasSameCategoryCandidate) framesWithCorrect++ else framesWithoutCorrect++
                }
            }

            totalDetectionSide += framesWithoutCorrect
            totalScoringSide += framesWithCorrect

            classifications.add(FailureClassification(
                scenario = name.removeSuffix(".json"),
                totalLosses = losses,
                totalReacqs = reacqs,
                totalTimeouts = timeouts,
                framesSearchingWithCorrectCandidate = framesWithCorrect,
                framesSearchingWithoutCorrectCandidate = framesWithoutCorrect
            ))
        }

        println("\n╔══════════════════════════════════════════════════════════════════════╗")
        println("║  ABLATION D: Failure-Mode Classification                           ║")
        println("╠══════════════════════════════════════════════════════════════════════╣")
        println("║  During search frames, is the correct candidate in the detector    ║")
        println("║  output (scoring-side) or missing entirely (detection-side)?       ║")
        println("╚══════════════════════════════════════════════════════════════════════╝\n")

        for (c in classifications) {
            val total = c.framesSearchingWithCorrectCandidate + c.framesSearchingWithoutCorrectCandidate
            if (total == 0) {
                println("  ${c.scenario}: no search frames (losses=${c.totalLosses}, reacqs=${c.totalReacqs})")
                continue
            }
            val detPct = c.framesSearchingWithoutCorrectCandidate * 100 / total
            val scorePct = c.framesSearchingWithCorrectCandidate * 100 / total
            println("  ${c.scenario}: " +
                    "search_frames=$total " +
                    "det_miss=${c.framesSearchingWithoutCorrectCandidate} (${detPct}%) " +
                    "score_miss=${c.framesSearchingWithCorrectCandidate} (${scorePct}%) " +
                    "losses=${c.totalLosses} reacqs=${c.totalReacqs} timeouts=${c.totalTimeouts}")
        }

        val grandTotal = totalDetectionSide + totalScoringSide
        if (grandTotal > 0) {
            println("\n  ── TOTALS ──")
            println("  Detection-side (candidate missing): $totalDetectionSide / $grandTotal (${totalDetectionSide * 100 / grandTotal}%)")
            println("  Scoring-side   (candidate present): $totalScoringSide / $grandTotal (${totalScoringSide * 100 / grandTotal}%)")
            println("\n  If detection-side > 50%, Phase 2 (YOLO11n) is high-value.")
            println("  If scoring-side > 50%, Phase 2 is low-priority — invest in identity instead.")
        }
    }
}
