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

        // Baseline: 86% from full pipeline (vs 58% from scenario replay which lacks VitTracker).
        // 942 frames, 5 losses, 5 reacqs, all person labels.
        assertTrue("Tracking rate should be >= 75% (baseline: 86%), got ${result.trackingRate}%",
            result.trackingRate >= 75)
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
     * - skipFrames: process every Nth frame to speed up test (default 1 = every frame)
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
        val skipFrames = spec.optInt("skipFrames", 1)

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
        var frameCount = 0
        var locked = false
        var wasSearching = false
        var prevFramesLost = 0

        ot.onDetectionResult = { allObjects, lockedObject, _, _, _ ->
            val nowLost = ot.reacquisition.framesLost
            val isSearching = ot.reacquisition.isSearching

            if (locked) {
                // Track events
                when {
                    wasSearching && lockedObject != null && nowLost == 0 ->
                        events.add(ReplayEvent(frameCount, "REACQUIRE",
                            lockedObject.id, lockedObject.label))
                    nowLost == 1 && prevFramesLost == 0 ->
                        events.add(ReplayEvent(frameCount, "LOST"))
                    ot.reacquisition.hasTimedOut && prevFramesLost <= ot.reacquisition.maxFramesLost ->
                        events.add(ReplayEvent(frameCount, "TIMEOUT"))
                }
                if (nowLost == 0) framesTracked++
                wasSearching = isSearching
                prevFramesLost = nowLost
            }
        }

        // Decode and process frames
        val decoder = VideoFrameDecoder(videoFile)
        decoder.decodeAll(targetWidth = analysisWidth) { frameIndex, bitmap ->
            if (frameIndex % skipFrames != 0) return@decodeAll

            if (frameIndex == lockFrame && !locked) {
                // Feed one frame first so ObjectTracker has lastFrameBitmap.
                // processBitmapInternal recycles the bitmap in its finally block,
                // so we need a separate copy for the lock frame.
                val forProcess = bitmap.copy(bitmap.config ?: Bitmap.Config.ARGB_8888, false)
                ot.processBitmap(forProcess)
                // Now lock — lockOnObject reads lastFrameBitmap (set by processBitmap)
                ot.lockOnObject(trackingId = 1, boundingBox = lockBox, label = lockLabel)
                locked = true
                framesTracked++ // lock frame counts as tracked
                Log.i(TAG, "Locked on $lockLabel at frame $frameIndex box=$lockBox")
            } else if (locked) {
                // processBitmapInternal recycles bitmap in its finally block
                ot.processBitmap(bitmap)
                frameCount++
            }
        }
        decoder.release()

        val result = ReplayResult(events, framesTracked, frameCount)
        Log.i(TAG, "Replay complete: ${result.totalFrames} frames, " +
            "${result.trackingRate}% tracked, ${result.reacquisitions} reacqs, " +
            "${result.losses} losses, timedOut=${result.timedOut}")
        Log.i(TAG, "Events: ${result.events}")
        return result
    }
}
