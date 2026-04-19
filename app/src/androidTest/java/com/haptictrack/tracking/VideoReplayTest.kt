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

        assertTrue("Should reacquire at least 2 times, got ${result.reacquisitions}",
            result.reacquisitions >= 2)
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

        assertTrue("Tracking rate should be >= 50%, got ${result.trackingRate}%",
            result.trackingRate >= 50)
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
        var wasSearching = false
        var prevFramesLost = 0

        ot.onDetectionResult = { _, lockedObject, _, _, _ ->
            val nowLost = ot.reacquisition.framesLost
            val isSearching = ot.reacquisition.isSearching

            if (locked) {
                when {
                    wasSearching && lockedObject != null && nowLost == 0 ->
                        events.add(ReplayEvent(framesProcessed, "REACQUIRE",
                            lockedObject.id, lockedObject.label))
                    nowLost == 1 && prevFramesLost == 0 ->
                        events.add(ReplayEvent(framesProcessed, "LOST"))
                    ot.reacquisition.hasTimedOut && prevFramesLost <= ot.reacquisition.maxFramesLost ->
                        events.add(ReplayEvent(framesProcessed, "TIMEOUT"))
                }
                if (nowLost == 0) framesTracked++
                wasSearching = isSearching
                prevFramesLost = nowLost
            }
        }

        // Decode and process frames with natural frame dropping.
        // The decoder callback returns how many frames to skip after each processed
        // frame. Skipped frames are drained from the codec without YUV→Bitmap
        // conversion, so they're fast. This matches live behavior where the pipeline
        // processes one frame and drops any that arrived during processing.
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
                return@decodeAll 0 // process next frame immediately
            } else if (locked) {
                val startMs = System.currentTimeMillis()
                ot.processBitmap(bitmap)
                val processingMs = System.currentTimeMillis() - startMs

                framesProcessed++
                // Skip frames that would have arrived during processing
                val framesToSkip = ((processingMs / frameDurationMs).toInt() - 1).coerceAtLeast(0)
                return@decodeAll framesToSkip
            }
            return@decodeAll 0
        }
        decoder.release()

        val framesDropped = totalVideoFrames - framesProcessed - 1 // -1 for lock frame
        val result = ReplayResult(events, framesTracked, framesProcessed)
        Log.i(TAG, "Replay complete: $framesProcessed processed / $totalVideoFrames total " +
            "($framesDropped dropped, ${framesDropped * 100 / totalVideoFrames.coerceAtLeast(1)}% drop rate), " +
            "${result.trackingRate}% tracked, ${result.reacquisitions} reacqs, " +
            "${result.losses} losses, timedOut=${result.timedOut}")
        Log.i(TAG, "Events: ${result.events}")
        return result
    }
}
