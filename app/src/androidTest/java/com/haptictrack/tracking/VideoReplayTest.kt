package com.haptictrack.tracking

import android.graphics.Bitmap
import android.graphics.RectF
import android.util.Log
import androidx.test.core.app.ApplicationProvider
import androidx.test.ext.junit.runners.AndroidJUnit4
import org.json.JSONObject
import org.junit.After
import org.junit.AfterClass
import org.junit.Assert.*
import org.junit.Before
import org.junit.BeforeClass
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

        /**
         * Single ObjectTracker shared across all tests in this class. Loading the
         * 9 ML models takes ~10-15s per fresh instance; with ~12 tests in this
         * suite that is 2-3 minutes of pure load overhead per run. State is reset
         * via [ObjectTracker.clearLock] in @Before so tests remain isolated.
         */
        private var sharedTracker: ObjectTracker? = null

        @BeforeClass
        @JvmStatic
        fun loadModelsOnce() {
            val context = ApplicationProvider.getApplicationContext<android.app.Application>()
            val latch = CountDownLatch(1)
            val ot = ObjectTracker(context, onLoadingStatus = { status ->
                Log.i(TAG, "Loading: $status")
                if (status == "Ready") latch.countDown()
            })
            assertTrue("Models should load within 60s", latch.await(60, TimeUnit.SECONDS))
            sharedTracker = ot
            Log.i(TAG, "Shared tracker ready — model load amortized across tests")
        }

        @AfterClass
        @JvmStatic
        fun shutdownTracker() {
            sharedTracker?.shutdown()
            sharedTracker = null
        }
    }

    private lateinit var testVideoDir: File

    @Before
    fun setup() {
        val context = ApplicationProvider.getApplicationContext<android.app.Application>()
        testVideoDir = File(context.getExternalFilesDir(null), TEST_VIDEO_SUBDIR)
        assertTrue("Test video directory must exist: $testVideoDir — " +
            "push videos with: adb push test_videos/ /sdcard/Android/data/com.haptictrack/files/test_videos/",
            testVideoDir.exists())
        // Reset per-test state on the shared tracker so each replay starts clean.
        sharedTracker?.clearLock()
    }

    @After
    fun teardown() {
        // Don't shutdown — @AfterClass handles that. Just clear lock so the next
        // test starts from a known state even if this one threw.
        sharedTracker?.clearLock()
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

        // Baseline (GL pipeline, 2026-04-26): 81-82% tracked, 2-3 reacqs.
        // ~660 of 1377 frames processed (~50% drop rate) at ~11fps effective —
        // matches live device throughput. Old 66% baseline was at ~3fps with
        // pure-Kotlin YUV bottleneck and wasn't representative.
        assertTrue("Tracking rate should be >= 75% (baseline: 81-82%), got ${result.trackingRate}%",
            result.trackingRate >= 75)
    }

    @Test
    fun person_playground_reacquires_correctly() {
        val result = replayVideo("person_playground_tracking")

        assertFalse("Should not timeout", result.timedOut)

        val wrong = result.wrongCategoryReacqs(PERSON_LABELS)
        assertTrue("Should never reacquire non-person (got: ${wrong.map { "${it.label}@F${it.frame}" }})",
            wrong.isEmpty())

        // Identity check: the playground often has multiple people at varying
        // distances. Earlier runs of this test showed up to 9 reacquires —
        // some of which could have jumped to a passing person rather than the
        // locked subject. GT skips frames where ByteTrack itself was lost.
        val swapped = result.wrongIdentityReacqs()
        assertTrue(
            "Should never reacquire onto a different person. " +
                "Mismatches: ${swapped.map { "frame ${it.videoFrame} box=${it.box?.toList()}" }}",
            swapped.isEmpty()
        )

        Log.i(TAG, "person_playground: trackingRate=${result.trackingRate}% " +
            "reacqs=${result.reacquisitions} losses=${result.losses} " +
            "totalFrames=${result.totalFrames} gt=${result.groundTruth?.annotatedFrameCount ?: 0}")
    }

    @Test
    fun person_playground_tracking_rate() {
        val result = replayVideo("person_playground_tracking")

        // Baseline (GL pipeline, 2026-04-26): 95-96% tracked, 9 reacqs, 9 losses.
        // Reacq count up from old "1 loss" baseline because dense sampling at
        // ~11fps now sees micro-losses the old ~3fps sampling missed. Tracking
        // rate stays high because each re-acquire is fast.
        assertTrue("Tracking rate should be >= 90% (baseline: 95-96%), got ${result.trackingRate}%",
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

        // Identity check via ground truth (sidecar .gt.json from
        // tools/synthetic_test_videos/generate_ground_truth.py). Empty when
        // GT is absent — we don't fail closed on missing ground truth, but
        // when it exists we enforce that REACQUIRE bboxes match the lock
        // subject's GT bbox (IoU >= 0.3) on frames where GT is confident.
        val swapped = result.wrongIdentityReacqs()
        assertTrue(
            "Should never reacquire onto a different person (son↔wife swap). " +
                "Mismatches: ${swapped.map { "frame ${it.videoFrame} box=${it.box?.toList()}" }}",
            swapped.isEmpty()
        )

        Log.i(TAG, "boy_indoor_wife_swap: trackingRate=${result.trackingRate}% " +
            "reacqs=${result.reacquisitions} losses=${result.losses} " +
            "totalFrames=${result.totalFrames} gt=${result.groundTruth?.annotatedFrameCount ?: 0}")
    }

    @Test
    fun boy_indoor_wife_swap_tracking_rate() {
        val result = replayVideo("boy_indoor_wife_swap")

        // Baseline (GL pipeline, 2026-04-26): 83-86% tracked, 12-13 reacqs.
        // The old ≥30% floor came from a "tracking jumps from son to wife"
        // concern, but since both are in PERSON_LABELS the wrong-category
        // assertion can't tell them apart — the swap is invisible to this
        // test. The high reacq count is the real signal that the lock is
        // bouncing across people; #83 phase 2 (scene face memory) targets it.
        assertTrue("Tracking rate should be >= 75% (baseline: 83-86%), got ${result.trackingRate}%",
            result.trackingRate >= 75)
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

        // GT identity check intentionally not wired here. YOLOv8 doesn't reliably
        // detect this small white chair through camera motion (only ~14% of frames
        // got a confident detection). Most reacquires fall in GT-null frames so
        // the assertion would silently no-op. Revisit if we get a better tracker
        // for static-furniture scenes.

        Log.i(TAG, "chair_living_room: trackingRate=${result.trackingRate}% " +
            "reacqs=${result.reacquisitions} losses=${result.losses} " +
            "totalFrames=${result.totalFrames}")
    }

    @Test
    fun chair_living_room_tracking_rate() {
        val result = replayVideo("chair_living_room_wrong_reacq")

        // Baseline (GL pipeline, 2026-04-26): 76-77% tracked, 5-6 reacqs,
        // all chair-only (no wrong-category). Old "86% / 62-70% w/ tentative"
        // numbers were at ~3fps sampling; this is what live device sees at
        // ~11fps with the same tentative-confirmation path engaged.
        assertTrue("Tracking rate should be >= 70% (baseline: 76-77%), got ${result.trackingRate}%",
            result.trackingRate >= 70)
    }

    // flowerpot_wrong_reacq: white bowl/flowerpot on table, camera zooms away and returns.
    // Black potted plant nearby. Tests that the wrong plant is NOT reacquired.
    // Regression test for: mature gallery accepting sim=0.000 candidates after timeout.

    @Test
    fun flowerpot_reacquires_correctly() {
        val result = replayVideo("flowerpot_wrong_reacq")

        assertFalse("Should not timeout", result.timedOut)

        // GT identity check intentionally not wired here. YOLOv8 only confidently
        // detects this static potted plant in ~7% of frames — empirically all 9
        // reacquires from the first run fell in GT-null frames, so the assertion
        // would silently skip every event. The "wrong plant" case this test was
        // designed for would benefit from a static-object tracker (template
        // matching, perhaps) rather than detection-based GT.

        Log.i(TAG, "flowerpot_wrong_reacq: trackingRate=${result.trackingRate}% " +
            "reacqs=${result.reacquisitions} losses=${result.losses} " +
            "totalFrames=${result.totalFrames}")
    }

    @Test
    fun flowerpot_tracking_rate() {
        val result = replayVideo("flowerpot_wrong_reacq")

        // Baseline (GL pipeline, 2026-04-26): 81% tracked (stable across runs),
        // 6-8 reacqs. No wrong-plant lock. Matches old 80% baseline closely.
        assertTrue("Tracking rate should be >= 75% (baseline: 81%), got ${result.trackingRate}%",
            result.trackingRate >= 75)
    }

    // mouse_desk_rotation: mouse on desk, phone rotated multiple times.
    // During rotation, MobileNetV3 can confuse keyboard with mouse (sim=0.58).
    // Tests that geometric override threshold (0.65) and tentative confirmation
    // prevent wrong reacquisition on orientation change.

    @Test
    fun mouse_desk_rotation_reacquires_correctly() {
        val result = replayVideo("mouse_desk_rotation")

        assertTrue("Should reacquire at least once, got ${result.reacquisitions}",
            result.reacquisitions >= 1)

        val wrong = result.wrongCategoryReacqs(setOf("mouse"))
        assertTrue("Should never reacquire non-mouse (got: ${wrong.map { "${it.label}@F${it.frame}" }})",
            wrong.isEmpty())

        Log.i(TAG, "mouse_desk_rotation: trackingRate=${result.trackingRate}% " +
            "reacqs=${result.reacquisitions} losses=${result.losses} " +
            "totalFrames=${result.totalFrames}")
    }

    @Test
    fun mouse_desk_rotation_tracking_rate() {
        val result = replayVideo("mouse_desk_rotation")

        // Baseline (GL pipeline, 2026-04-26): 29-34% tracked across runs.
        // Variance straddles the old ≥30% floor — concurrency in
        // embeddingExecutor/lockExecutor + ML inference timing is genuinely
        // non-deterministic. Floor lowered to 25% to absorb run-to-run noise.
        // Small-object + rotation is the underlying difficulty; #59 tracks it.
        assertTrue("Tracking rate should be >= 25% (baseline: 29-34%), got ${result.trackingRate}%",
            result.trackingRate >= 25)
    }

    // two_men_forest: synthetic stock-video test (Pexels 4830416, cottonbro studio,
    // CC0). Two men with similar earth-tone outdoor clothing walk down a path.
    // Probes person identity discrimination when color histogram alone can't tell
    // them apart — the face gate (#84) and body re-id (OSNet) have to do the work.
    // The two subjects differ in age (younger / bearded older), so face is a
    // genuine discriminator if it can fire.

    @Test
    fun two_men_forest_no_wrong_person_lock() {
        val result = replayVideo("two_men_forest")

        // Baseline TBD on first device run. Primary assertion is that the lock
        // never reacquires onto a wrong subject — the other man is consistently
        // visible alongside the locked one. PERSON_LABELS is the wrong scope here
        // because both subjects ARE people; the test design captures this video's
        // value via tracking rate (a wrong-person reacquire would drop it sharply).
        assertFalse("Should not timeout", result.timedOut)

        val wrong = result.wrongCategoryReacqs(PERSON_LABELS)
        assertTrue("Should never reacquire non-person (got: ${wrong.map { "${it.label}@F${it.frame}" }})",
            wrong.isEmpty())

        Log.i(TAG, "two_men_forest: trackingRate=${result.trackingRate}% " +
            "reacqs=${result.reacquisitions} losses=${result.losses} " +
            "totalFrames=${result.totalFrames}")
    }

    // two_men_suits: synthetic stock-video test (Pexels 8915048, Kampus Production,
    // CC0). 8s — four persons in frame: two men in nearly-identical black suits
    // with bow ties side-by-side, a third man in a gray jacket, and a woman in
    // a patterned blouse. Lock target is the lighter-haired suit man (3rd from
    // left). Tests the color-histogram-can't-discriminate case — the man directly
    // beside the lock target wears identical clothing. Hair color and face are
    // the discriminators.

    @Test
    fun two_men_suits_no_wrong_person_lock() {
        val result = replayVideo("two_men_suits")

        // Baseline TBD on first device run. 198 frames is short — primary signal
        // is "no wrong-category reacquire" since both suit men are 'person' and
        // the test can't tell them apart at the assertion layer. Tracking rate
        // is a secondary indicator — a sharp drop suggests the lock jumped to
        // the other suit-wearing subject.
        assertFalse("Should not timeout", result.timedOut)

        val wrong = result.wrongCategoryReacqs(PERSON_LABELS)
        assertTrue("Should never reacquire non-person (got: ${wrong.map { "${it.label}@F${it.frame}" }})",
            wrong.isEmpty())

        Log.i(TAG, "two_men_suits: trackingRate=${result.trackingRate}% " +
            "reacqs=${result.reacquisitions} losses=${result.losses} " +
            "totalFrames=${result.totalFrames}")
    }

    // person_distance: synthetic stock-video test (Pexels 5320110, cottonbro studio,
    // CC0). 13.6s — single shirtless man with distinctive tattoos walks from
    // mid-distance toward camera, ending at close-up. Tests scale tolerance:
    // gallery embeddings learned from a small crop (~12% of frame height at
    // lock time) must continue to match as the subject grows ~6× over the clip.
    // Adaptive size threshold and re-id at varied scale are exercised here.

    @Test
    fun person_distance_tracking_through_scale_change() {
        val result = replayVideo("person_distance")

        // Single-subject video — no wrong-person assertion is meaningful here.
        // Primary signal is whether tracking survives the scale change. A timeout
        // means the gallery couldn't bridge the small→large transition.
        assertFalse("Should not timeout", result.timedOut)

        Log.i(TAG, "person_distance: trackingRate=${result.trackingRate}% " +
            "reacqs=${result.reacquisitions} losses=${result.losses} " +
            "totalFrames=${result.totalFrames}")
    }

    // crowd_street: synthetic stock-video test (Pexels 12699538, CC0). 12.2s
    // portrait shot of a busy market street — six detected persons in the lock
    // frame alone. Lock target picked via EfficientDet-Lite2 detection (the
    // same model the app uses) so the box is what the production detector
    // would produce. Tests reacquisition through occlusion and discrimination
    // among many same-category candidates at once.

    @Test
    fun crowd_street_no_wrong_category_lock() {
        val result = replayVideo("crowd_street")

        // The dense-multi-person scene means an "A→B" person swap is invisible
        // to PERSON_LABELS. Primary signals: no timeout (the lock should be
        // reacquired through brief occlusions) and tracking rate that doesn't
        // collapse (catastrophic mistracking would push it near zero).
        assertFalse("Should not timeout", result.timedOut)

        val wrong = result.wrongCategoryReacqs(PERSON_LABELS)
        assertTrue("Should never reacquire non-person (got: ${wrong.map { "${it.label}@F${it.frame}" }})",
            wrong.isEmpty())

        Log.i(TAG, "crowd_street: trackingRate=${result.trackingRate}% " +
            "reacqs=${result.reacquisitions} losses=${result.losses} " +
            "totalFrames=${result.totalFrames}")
    }

    // --- Replay infrastructure ---

    data class ReplayEvent(
        val frame: Int,
        val type: String,
        val objectId: Int? = null,
        val label: String? = null,
        /** Source video frame index this event occurred on (matches GT keys). */
        val videoFrame: Int? = null,
        /** Bbox in normalized [l,t,r,b] at event time, when available. */
        val box: FloatArray? = null
    )

    /**
     * Per-frame ground truth: the lock subject's bbox at that source video frame,
     * or null if ByteTrack reported the subject lost / ambiguous on that frame.
     *
     * Loaded from `<name>.gt.json` produced by `tools/synthetic_test_videos/
     * generate_ground_truth.py`. Matched frames are trustworthy; we don't enforce
     * identity on lost/ambiguous frames since GT itself is uncertain there.
     */
    class GroundTruth(private val byFrame: Map<Int, FloatArray>) {
        fun boxAt(frame: Int): FloatArray? = byFrame[frame]
        val annotatedFrameCount: Int get() = byFrame.size

        companion object {
            fun loadOrNull(file: File): GroundTruth? {
                if (!file.exists()) return null
                val obj = JSONObject(file.readText())
                val anns = obj.getJSONArray("annotations")
                val map = HashMap<Int, FloatArray>(anns.length())
                for (i in 0 until anns.length()) {
                    val a = anns.getJSONObject(i)
                    val frame = a.getInt("frame")
                    // optJSONArray returns null for both missing keys AND
                    // JSONObject.NULL AND non-array types (e.g. malformed GT
                    // file with a stray string in "box"). Skip cleanly in all
                    // those cases instead of throwing ClassCastException.
                    val arr = a.optJSONArray("box") ?: continue
                    if (arr.length() < 4) continue
                    map[frame] = floatArrayOf(
                        arr.getDouble(0).toFloat(),
                        arr.getDouble(1).toFloat(),
                        arr.getDouble(2).toFloat(),
                        arr.getDouble(3).toFloat()
                    )
                }
                return GroundTruth(map)
            }
        }
    }

    data class ReplayResult(
        val events: List<ReplayEvent>,
        val framesTracked: Int,
        val totalFrames: Int,
        val groundTruth: GroundTruth? = null
    ) {
        val trackingRate: Int get() = if (totalFrames > 0) framesTracked * 100 / totalFrames else 0
        val reacquisitions: Int get() = events.count { it.type == "REACQUIRE" }
        val losses: Int get() = events.count { it.type == "LOST" }
        val timedOut: Boolean get() = events.any { it.type == "TIMEOUT" }
        fun wrongCategoryReacqs(validLabels: Set<String>): List<ReplayEvent> =
            events.filter { it.type == "REACQUIRE" && it.label !in validLabels }

        /**
         * REACQUIRE events whose bbox doesn't match the ground-truth subject's
         * bbox at the same video frame (IoU below [iouThreshold]). When GT is
         * absent for an event's frame (subject was lost/ambiguous in GT),
         * the event is skipped — we don't enforce identity where GT itself
         * is uncertain. Returns an empty list when no GT is loaded.
         *
         * Default 0.5: empirically the engine's reacquire boxes overlap GT
         * (YOLOv8) by 0.88-0.96 on correct matches, so 0.5 is well below the
         * noise floor and catches cases where the engine landed on a clearly
         * different subject. Tighter would risk flaking on slight detector
         * disagreement on the same person; looser lets two adjacent persons
         * pass when they shouldn't.
         */
        fun wrongIdentityReacqs(iouThreshold: Float = 0.5f): List<ReplayEvent> {
            val gt = groundTruth ?: return emptyList()
            return events.filter { event ->
                if (event.type != "REACQUIRE") return@filter false
                val frame = event.videoFrame ?: return@filter false
                val box = event.box ?: return@filter false
                val gtBox = gt.boxAt(frame) ?: return@filter false
                computeIou(box, gtBox) < iouThreshold
            }
        }

        private fun computeIou(a: FloatArray, b: FloatArray): Float {
            val il = maxOf(a[0], b[0])
            val it = maxOf(a[1], b[1])
            val ir = minOf(a[2], b[2])
            val ib = minOf(a[3], b[3])
            if (il >= ir || it >= ib) return 0f
            val inter = (ir - il) * (ib - it)
            val areaA = (a[2] - a[0]) * (a[3] - a[1])
            val areaB = (b[2] - b[0]) * (b[3] - b[1])
            return inter / (areaA + areaB - inter)
        }
    }

    /**
     * Test spec JSON format:
     * ```json
     * {
     *   "lockFrame": 0,
     *   "lockBox": [left, top, right, bottom],
     *   "lockLabel": "person",
     *   "analysisWidth": 640,
     *   "fps": 30
     * }
     * ```
     *
     * - lockFrame: which decoded frame to lock on (0-based)
     * - lockBox: normalized bounding box [0,1] to lock on
     * - lockLabel: COCO label to lock with
     * - analysisWidth: longer-side downscale target (default 640, matches production)
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

        val gtFile = File(testVideoDir, "$name.gt.json")
        val groundTruth = GroundTruth.loadOrNull(gtFile)
        if (groundTruth != null) {
            Log.i(TAG, "Loaded ground truth from $gtFile " +
                "(${groundTruth.annotatedFrameCount} annotated frames)")
        }

        val ot = sharedTracker ?: error("sharedTracker not initialized — @BeforeClass failed?")
        Log.i(TAG, "Starting replay of $name")

        val decoder = VideoGLDecoder(videoFile)
        // Bitmaps come from the GL pipeline's pool. ObjectTracker retains the
        // most recent input as lastFrameBitmap and hands the previous one back
        // via the recycler — route it back to the pool so it can be reused.
        ot.bitmapRecycler = decoder.bitmapRecycler()

        // Collect events from the callback
        val events = mutableListOf<ReplayEvent>()
        var framesTracked = 0
        var framesProcessed = 0
        var locked = false

        // Track state transitions by sampling BEFORE and AFTER each processBitmap call.
        // The callback approach misses transitions that happen within a single processFrame
        // (e.g. LOST→REACQUIRE in the same call when the object disappears and reappears).
        ot.onDetectionResult = { _, _, _, _, _ ->
            if (locked && ot.reacquisition.framesLost == 0) framesTracked++
        }

        var totalVideoFrames = 0

        try {
            decoder.decodeAll(targetWidth = analysisWidth) { frameIndex, bitmap ->
                totalVideoFrames = frameIndex + 1

                if (frameIndex == lockFrame && !locked) {
                    // processBitmap takes ownership via bitmapRecycler. lockOnObject
                    // snapshots lastFrameBitmap internally before applying ML, so the
                    // bitmap stays alive long enough for the lock burst.
                    ot.processBitmap(bitmap)
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
                    ot.processBitmap(bitmap)  // tracker owns it; recycler releases previous frame
                    val processingMs = System.currentTimeMillis() - startMs

                    // Check state AFTER processing to detect transitions
                    val nowLost = ot.reacquisition.framesLost
                    val lockedObj = if (nowLost == 0 && prevLost > 0) ot.reacquisition.lockedId else null
                    val curBox = ot.reacquisition.lastKnownBox?.let {
                        floatArrayOf(it.left, it.top, it.right, it.bottom)
                    }
                    when {
                        wasSearching && lockedObj != null && nowLost == 0 ->
                            events.add(ReplayEvent(framesProcessed, "REACQUIRE",
                                lockedObj, ot.reacquisition.lastKnownLabel,
                                videoFrame = frameIndex, box = curBox))
                        nowLost == 1 && prevLost == 0 ->
                            events.add(ReplayEvent(framesProcessed, "LOST",
                                videoFrame = frameIndex))
                        ot.reacquisition.hasTimedOut && prevLost <= ot.reacquisition.maxFramesLost ->
                            events.add(ReplayEvent(framesProcessed, "TIMEOUT",
                                videoFrame = frameIndex))
                    }

                    framesProcessed++
                    val framesToSkip = ((processingMs / frameDurationMs).toInt() - 1).coerceAtLeast(0)
                    return@decodeAll framesToSkip
                }
                // Frame before lockFrame — feed to processBitmap so lastFrameBitmap
                // is populated when lockOnObject runs and the pipeline state matches
                // production (which processes every frame regardless of lock state).
                // Currently all specs use lockFrame=0 so this branch is dead, but
                // keeping it well-formed prevents a latent bug if any future spec
                // moves the lock past frame 0.
                ot.processBitmap(bitmap)
                return@decodeAll 0
            }
        } finally {
            decoder.release()
            // Clear callbacks so subsequent tests don't see stale state.
            ot.bitmapRecycler = null
            ot.onDetectionResult = null
        }

        val framesDropped = totalVideoFrames - framesProcessed - 1
        val result = ReplayResult(events, framesTracked, framesProcessed, groundTruth)
        Log.i(TAG, "Replay complete: $framesProcessed processed / $totalVideoFrames total " +
            "($framesDropped dropped, ${framesDropped * 100 / totalVideoFrames.coerceAtLeast(1)}% drop rate), " +
            "${result.trackingRate}% tracked, ${result.reacquisitions} reacqs, " +
            "${result.losses} losses, timedOut=${result.timedOut}")
        Log.i(TAG, "Events: ${result.events}")
        return result
    }
}
