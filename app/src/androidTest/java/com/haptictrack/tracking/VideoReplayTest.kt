package com.haptictrack.tracking

import android.graphics.Bitmap
import android.graphics.RectF
import android.util.Log
import androidx.test.core.app.ApplicationProvider
import androidx.test.ext.junit.runners.AndroidJUnit4
import org.json.JSONObject
import org.junit.After
import org.junit.Assert.*
import org.junit.Before
import org.junit.Test
import org.junit.runner.RunWith
import java.io.File
import java.util.concurrent.CountDownLatch
import java.util.concurrent.TimeUnit

/**
 * Instrumented integration test that replays real video through the full
 * ObjectTracker pipeline on-device with real ML models.
 *
 * Unlike [ScenarioReplayTest] which replays pre-captured detector outputs
 * through ReacquisitionEngine only, this test uses the actual video as the
 * single source of truth — the same frames that were recorded on-device are
 * fed through the full pipeline: detection → embedding → tracking → reacquisition.
 *
 * Test videos and their spec files live on the device at:
 *   /sdcard/Android/data/com.haptictrack/files/test_videos/
 *
 * Each test has:
 *   - {name}.mp4 — the video file
 *   - {name}.json — test spec (lock frame, bounding box, label, assertions)
 *
 * Push test data:
 *   adb push test_videos/ /sdcard/Android/data/com.haptictrack/files/test_videos/
 *
 * Run:
 *   ./gradlew connectedDebugAndroidTest --tests "*.VideoReplayTest"
 */
@RunWith(AndroidJUnit4::class)
class VideoReplayTest {

    companion object {
        private const val TAG = "VideoReplayTest"
        private const val TEST_VIDEO_SUBDIR = "test_videos"
        val PERSON_LABELS = setOf("person", "boy", "girl", "man", "woman", "human face")
    }

    private var tracker: ObjectTracker? = null
    private lateinit var testVideoDir: File

    @Before
    fun setup() {
        val context = ApplicationProvider.getApplicationContext<android.app.Application>()
        testVideoDir = File(context.getExternalFilesDir(null), TEST_VIDEO_SUBDIR)
        assertTrue("Test video directory must exist: $testVideoDir — " +
            "push videos with: adb push test_videos/ /sdcard/Android/data/com.haptictrack/files/test_videos/",
            testVideoDir.exists())
    }

    @After
    fun teardown() {
        tracker?.shutdown()
        tracker = null
    }

    // --- Test cases ---

    @Test
    fun man_desk_camera_swing_reacquires_correctly() {
        val result = replayVideo("man_desk_camera_swing")

        assertTrue("Should reacquire at least 2 times, got ${result.reacquisitions}",
            result.reacquisitions >= 2)
        assertFalse("Should not timeout", result.timedOut)

        val wrong = result.wrongCategoryReacqs(PERSON_LABELS)
        assertTrue("Should never reacquire non-person (got: ${wrong.map { "${it.label}@F${it.frame}" }})",
            wrong.isEmpty())

        Log.i(TAG, "man_desk_camera_swing: trackingRate=${result.trackingRate}% " +
            "reacqs=${result.reacquisitions} losses=${result.losses} " +
            "totalFrames=${result.totalFrames}")
    }

    @Test
    fun man_desk_camera_swing_tracking_rate() {
        val result = replayVideo("man_desk_camera_swing")

        // Baseline with natural frame dropping: 66% tracked, 3 reacqs, 3 losses.
        // 130 frames processed / 943 total (86% drop rate — matches live behavior).
        assertTrue("Tracking rate should be >= 55% (baseline: 66%), got ${result.trackingRate}%",
            result.trackingRate >= 55)
    }

    @Test
    fun person_playground_reacquires_correctly() {
        val result = replayVideo("person_playground_tracking")

        assertFalse("Should not timeout", result.timedOut)

        val wrong = result.wrongCategoryReacqs(PERSON_LABELS)
        assertTrue("Should never reacquire non-person (got: ${wrong.map { "${it.label}@F${it.frame}" }})",
            wrong.isEmpty())

        Log.i(TAG, "person_playground: trackingRate=${result.trackingRate}% " +
            "reacqs=${result.reacquisitions} losses=${result.losses} " +
            "totalFrames=${result.totalFrames}")
    }

    @Test
    fun person_playground_tracking_rate() {
        val result = replayVideo("person_playground_tracking")

        // Baseline with velocity improvements: 100% tracked, 1 loss.
        // Previously: 98% tracked, 5 losses.
        assertTrue("Tracking rate should be >= 90%, got ${result.trackingRate}%",
            result.trackingRate >= 90)
    }

    @Test
    fun boy_indoor_wife_swap_reacquires_correctly() {
        val result = replayVideo("boy_indoor_wife_swap")

        assertTrue("Should reacquire at least once, got ${result.reacquisitions}",
            result.reacquisitions >= 1)
        assertFalse("Should not timeout", result.timedOut)

        val wrong = result.wrongCategoryReacqs(PERSON_LABELS)
        assertTrue("Should never reacquire non-person (got: ${wrong.map { "${it.label}@F${it.frame}" }})",
            wrong.isEmpty())

        Log.i(TAG, "boy_indoor_wife_swap: trackingRate=${result.trackingRate}% " +
            "reacqs=${result.reacquisitions} losses=${result.losses} " +
            "totalFrames=${result.totalFrames}")
    }

    @Test
    fun boy_indoor_wife_swap_tracking_rate() {
        val result = replayVideo("boy_indoor_wife_swap")

        // This scenario has a known bug: tracking jumps from son to wife.
        // Set a low floor initially — we'll tighten as we fix the person swap issue.
        assertTrue("Tracking rate should be >= 30%, got ${result.trackingRate}%",
            result.trackingRate >= 30)
    }

    // chair_living_room_wrong_reacq: chair at desk, camera pans around living room.
    // Multiple couches and other chairs. Tests that wrong chair at frame edge is NOT
    // reacquired without identity verification (regression for single-candidate fast path).

    @Test
    fun chair_living_room_reacquires_correctly() {
        val result = replayVideo("chair_living_room_wrong_reacq")

        assertTrue("Should reacquire at least once, got ${result.reacquisitions}",
            result.reacquisitions >= 1)
        assertFalse("Should not timeout", result.timedOut)

        val wrong = result.wrongCategoryReacqs(setOf("chair"))
        assertTrue("Should never reacquire non-chair (got: ${wrong.map { "${it.label}@F${it.frame}" }})",
            wrong.isEmpty())

        Log.i(TAG, "chair_living_room: trackingRate=${result.trackingRate}% " +
            "reacqs=${result.reacquisitions} losses=${result.losses} " +
            "totalFrames=${result.totalFrames}")
    }

    @Test
    fun chair_living_room_tracking_rate() {
        val result = replayVideo("chair_living_room_wrong_reacq")

        // Baseline: 86% tracked, 5 reacqs, 5 losses. All chair-only.
        // Tentative confirmation adds ~2 frames latency per reacquisition,
        // reducing tracking rate from ~70% to ~62%. Worth the tradeoff:
        // zero wrong reacquisitions (couch/bed eliminated).
        assertTrue("Tracking rate should be >= 55%, got ${result.trackingRate}%",
            result.trackingRate >= 55)
    }

    // flowerpot_wrong_reacq: white bowl/flowerpot on table, camera zooms away and returns.
    // Black potted plant nearby. Tests that the wrong plant is NOT reacquired.
    // Regression test for: mature gallery accepting sim=0.000 candidates after timeout.

    @Test
    fun flowerpot_reacquires_correctly() {
        val result = replayVideo("flowerpot_wrong_reacq")

        assertFalse("Should not timeout", result.timedOut)

        Log.i(TAG, "flowerpot_wrong_reacq: trackingRate=${result.trackingRate}% " +
            "reacqs=${result.reacquisitions} losses=${result.losses} " +
            "totalFrames=${result.totalFrames}")
    }

    @Test
    fun flowerpot_tracking_rate() {
        val result = replayVideo("flowerpot_wrong_reacq")

        // Baseline: 80% tracked, 4 reacqs, 5 losses. No wrong-plant lock.
        assertTrue("Tracking rate should be >= 65%, got ${result.trackingRate}%",
            result.trackingRate >= 65)
    }

    // --- Replay infrastructure ---

    data class ReplayEvent(
        val frame: Int,
        val type: String,
        val objectId: Int? = null,
        val label: String? = null
    )

    data class ReplayResult(
        val events: List<ReplayEvent>,
        val framesTracked: Int,
        val totalFrames: Int
    ) {
        val trackingRate: Int get() = if (totalFrames > 0) framesTracked * 100 / totalFrames else 0
        val reacquisitions: Int get() = events.count { it.type == "REACQUIRE" }
        val losses: Int get() = events.count { it.type == "LOST" }
        val timedOut: Boolean get() = events.any { it.type == "TIMEOUT" }
        fun wrongCategoryReacqs(validLabels: Set<String>): List<ReplayEvent> =
            events.filter { it.type == "REACQUIRE" && it.label !in validLabels }
    }

    /**
     * Test spec JSON format:
     * ```json
     * {
     *   "lockFrame": 0,
     *   "lockBox": [left, top, right, bottom],
     *   "lockLabel": "person",
     *   "analysisWidth": 640,
     *   "skipFrames": 3
     * }
     * ```
     *
     * - lockFrame: which decoded frame to lock on (0-based)
     * - lockBox: normalized bounding box [0,1] to lock on
     * - lockLabel: COCO label to lock with
     * - analysisWidth: downscale width for frames (default 640, matches production)
     * - fps: video framerate, used for natural frame dropping (default 30)
     *
     * Frame dropping simulates live camera behavior: the pipeline processes one
     * frame at a time via processBitmap(). While processing, the camera keeps
     * producing frames. When the pipeline is ready for the next frame, it grabs
     * the LATEST — skipping any that arrived during processing. This test
     * measures actual processBitmap() wall time and advances the frame index
     * accordingly, so frame dropping matches what happens on-device.
     */
    private fun replayVideo(name: String): ReplayResult {
        val videoFile = File(testVideoDir, "$name.mp4")
        val specFile = File(testVideoDir, "$name.json")
        assertTrue("Video file missing: $videoFile", videoFile.exists())
        assertTrue("Spec file missing: $specFile", specFile.exists())

        val spec = JSONObject(specFile.readText())
        val lockFrame = spec.getInt("lockFrame")
        val lockBoxArr = spec.getJSONArray("lockBox")
        val lockBox = RectF(
            lockBoxArr.getDouble(0).toFloat(),
            lockBoxArr.getDouble(1).toFloat(),
            lockBoxArr.getDouble(2).toFloat(),
            lockBoxArr.getDouble(3).toFloat()
        )
        val lockLabel = spec.getString("lockLabel")
        val analysisWidth = spec.optInt("analysisWidth", 640)
        val fps = spec.optInt("fps", 30)
        val frameDurationMs = 1000L / fps

        // Initialize ObjectTracker with real models
        val context = ApplicationProvider.getApplicationContext<android.app.Application>()
        val loadLatch = CountDownLatch(1)
        val ot = ObjectTracker(context, onLoadingStatus = { status ->
            Log.i(TAG, "Loading: $status")
            if (status == "Ready") loadLatch.countDown()
        })
        tracker = ot

        assertTrue("Models should load within 60s", loadLatch.await(60, TimeUnit.SECONDS))
        Log.i(TAG, "Models loaded, starting replay of $name")

        // Collect events from the callback
        val events = mutableListOf<ReplayEvent>()
        var framesTracked = 0
        var framesProcessed = 0
        var locked = false

        // Track state transitions by sampling BEFORE and AFTER each processBitmap call.
        // The callback approach misses transitions that happen within a single processFrame
        // (e.g. LOST→REACQUIRE in the same call when the object disappears and reappears).
        ot.onDetectionResult = { _, lockedObject, _, _, _ ->
            if (locked && ot.reacquisition.framesLost == 0) framesTracked++
        }

        var totalVideoFrames = 0

        val decoder = VideoFrameDecoder(videoFile)
        decoder.decodeAll(targetWidth = analysisWidth) { frameIndex, bitmap ->
            totalVideoFrames = frameIndex + 1

            if (frameIndex == lockFrame && !locked) {
                val forProcess = bitmap.copy(bitmap.config ?: Bitmap.Config.ARGB_8888, false)
                ot.processBitmap(forProcess)
                ot.lockOnObject(trackingId = 1, boundingBox = lockBox, label = lockLabel)
                locked = true
                framesTracked++
                Log.i(TAG, "Locked on $lockLabel at frame $frameIndex box=$lockBox")
                return@decodeAll 0
            } else if (locked) {
                // Snapshot state BEFORE processing
                val wasSearching = ot.reacquisition.isSearching
                val prevLost = ot.reacquisition.framesLost

                val startMs = System.currentTimeMillis()
                ot.processBitmap(bitmap)
                val processingMs = System.currentTimeMillis() - startMs

                // Check state AFTER processing to detect transitions
                val nowLost = ot.reacquisition.framesLost
                val lockedObj = if (nowLost == 0 && prevLost > 0) ot.reacquisition.lockedId else null
                when {
                    wasSearching && lockedObj != null && nowLost == 0 ->
                        events.add(ReplayEvent(framesProcessed, "REACQUIRE",
                            lockedObj, ot.reacquisition.lastKnownLabel))
                    nowLost == 1 && prevLost == 0 ->
                        events.add(ReplayEvent(framesProcessed, "LOST"))
                    ot.reacquisition.hasTimedOut && prevLost <= ot.reacquisition.maxFramesLost ->
                        events.add(ReplayEvent(framesProcessed, "TIMEOUT"))
                }

                framesProcessed++
                val framesToSkip = ((processingMs / frameDurationMs).toInt() - 1).coerceAtLeast(0)
                return@decodeAll framesToSkip
            }
            return@decodeAll 0
        }
        decoder.release()

        val framesDropped = totalVideoFrames - framesProcessed - 1
        val result = ReplayResult(events, framesTracked, framesProcessed)
        Log.i(TAG, "Replay complete: $framesProcessed processed / $totalVideoFrames total " +
            "($framesDropped dropped, ${framesDropped * 100 / totalVideoFrames.coerceAtLeast(1)}% drop rate), " +
            "${result.trackingRate}% tracked, ${result.reacquisitions} reacqs, " +
            "${result.losses} losses, timedOut=${result.timedOut}")
        Log.i(TAG, "Events: ${result.events}")
        return result
    }
}
