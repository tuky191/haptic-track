package com.haptictrack.tracking

import android.graphics.RectF
import kotlin.math.abs
import kotlin.math.hypot

/** Haptic cue the sentry emits on state transitions. */
enum class SentryCue { SCANNING, INSPECTING, MATCH, REJECT }

/**
 * Active auto-lock ("sentry") state machine. Driven once per detection frame
 * from IDLE. Hunts for a person matching [criteria]:
 *
 *  SCANNING  — wide zoom; pick the most CENTERED person (zoom is center-based,
 *              no pan on a phone, so only a centered candidate survives zoom-in).
 *  INSPECTING— zoom in on that candidate to enlarge the face, classify, compare.
 *              match → lock+record; no-match / no face within timeout → cooldown
 *              the candidate, zoom out, resume scanning.
 *  MATCHED   — a match was locked; normal tracking takes over until cleared.
 *
 * Model-agnostic: classification is injected via [classify] returning
 * [FaceAttributes] (null = no usable face yet), so whichever age/gender model
 * ships slots in unchanged.
 *
 * All side effects go through callbacks so the controller is unit-testable.
 */
class SentryController(
    private val criteria: () -> SentryCriteria,
    private val setZoomTarget: (Float) -> Unit,
    /** Current applied camera zoom — inspect zoom is computed relative to this so it converges. */
    private val currentZoom: () -> Float,
    private val minZoom: () -> Float,
    private val maxZoom: () -> Float,
    /** Classify a candidate against the current frame. Null = no usable face this frame. */
    private val classify: (TrackedObject) -> FaceAttributes?,
    /** Lock + start recording on a matched candidate. */
    private val lock: (TrackedObject) -> Unit,
    private val haptic: (SentryCue) -> Unit,
    /** Structured event sink for session logging (type, candidate box, attrs, note). No-op by default. */
    private val onEvent: (type: String, box: RectF?, attr: FaceAttributes?, note: String?) -> Unit = { _, _, _, _ -> },
) {
    companion object {
        const val SCAN_ZOOM = 1.0f
        /** Candidate must be within this normalized distance of frame center to inspect. */
        const val CENTER_TOLERANCE = 0.22f
        /** Target person-box height as a fraction of frame when inspecting (big enough face). */
        const val INSPECT_OCCUPANCY = 0.75f
        /** Classify every Nth inspecting frame (BlazeFace+genderage is not free). */
        const val CLASSIFY_INTERVAL = 3
        /** Give up on a candidate after this many inspecting frames with no decision. */
        const val INSPECT_TIMEOUT = 45
        /** Frames a rejected candidate is skipped before it can be inspected again. */
        const val REJECT_COOLDOWN = 90
        /** IoU above which a candidate is considered "the same" across frames when id is unstable. */
        const val SAME_CANDIDATE_IOU = 0.3f
        /** Candidate must stay centered this many consecutive scan frames before we commit to inspect
         *  (avoids zooming in on a 1-frame flicker, esp. at distance). */
        const val CONFIRM_FRAMES = 3
        /** Tolerate this many consecutive missing-detection frames during inspect before declaring lost
         *  (distant subjects flicker; a single miss shouldn't abort + zoom out). */
        const val INSPECT_MISS_TOLERANCE = 8
        /** Occupancy deadzone: hold zoom when the box is within ±this fraction of INSPECT_OCCUPANCY. */
        const val INSPECT_DEADZONE = 0.15f
        /** A candidate overlapping a recently-rejected (wide-zoom) box by ≥ this IoU is skipped. */
        const val COOLDOWN_IOU = 0.3f
    }

    enum class State { OFF, SCANNING, INSPECTING, MATCHED }

    var state: State = State.OFF
        private set

    val phase: SentryPhase get() = when (state) {
        State.OFF -> SentryPhase.OFF
        State.SCANNING -> SentryPhase.SCANNING
        State.INSPECTING -> SentryPhase.INSPECTING
        State.MATCHED -> SentryPhase.MATCHED
    }

    private var inspectKey: String? = null
    private var inspectBox: RectF? = null         // latest candidate box (zoomed frame)
    private var inspectStartBox: RectF? = null    // box at inspect start (wide zoom) — used for cooldown
    private var inspectFrames = 0
    private var inspectMissed = 0                  // consecutive frames the candidate wasn't found
    private var centeringKey: String? = null      // candidate currently building a centering streak
    private var centeringStreak = 0
    /** Recently-rejected wide-zoom boxes (+ frames remaining) — skip candidates overlapping these. */
    private val cooled = ArrayList<Pair<RectF, Int>>()

    fun setEnabled(enabled: Boolean) {
        if (enabled) {
            if (state == State.OFF) {
                state = State.SCANNING
                resetScanState()
                haptic(SentryCue.SCANNING)
            }
        } else {
            state = State.OFF
            resetInspect()
            resetScanState()
        }
    }

    /** Call when the user/system clears the lock — re-arm scanning if still on. */
    fun onLockCleared() {
        if (state == State.MATCHED) {
            state = State.SCANNING
            resetInspect()
            resetScanState()
            haptic(SentryCue.SCANNING)
        }
    }

    /** Drive one detection frame. [persons] are person detections in normalized screen coords. */
    fun onFrame(persons: List<TrackedObject>) {
        // Age out the cooldown list every frame.
        if (cooled.isNotEmpty()) {
            val it = cooled.listIterator()
            while (it.hasNext()) {
                val (box, n) = it.next()
                if (n - 1 <= 0) it.remove() else it.set(box to (n - 1))
            }
        }
        when (state) {
            State.OFF, State.MATCHED -> return
            State.SCANNING -> scan(persons)
            State.INSPECTING -> inspect(persons)
        }
    }

    private fun scan(persons: List<TrackedObject>) {
        setZoomTarget(SCAN_ZOOM)
        val candidate = persons
            .filter { !isCooled(it.boundingBox) }
            .minByOrNull { centerDistance(it.boundingBox) }
        if (candidate == null || centerDistance(candidate.boundingBox) > CENTER_TOLERANCE) {
            centeringKey = null; centeringStreak = 0
            return
        }
        // Build a centering streak — only commit once the same candidate has held center,
        // so we don't zoom in on a transient/flickery detection (the distance failure mode).
        val k = keyOf(candidate)
        if (k == centeringKey) centeringStreak++ else { centeringKey = k; centeringStreak = 1 }
        if (centeringStreak < CONFIRM_FRAMES) return

        inspectKey = k
        inspectBox = RectF(candidate.boundingBox)
        inspectStartBox = RectF(candidate.boundingBox)
        inspectFrames = 0
        inspectMissed = 0
        centeringKey = null; centeringStreak = 0
        state = State.INSPECTING
        setZoomTarget(inspectZoomFor(candidate.boundingBox))  // start zooming in
        haptic(SentryCue.INSPECTING)
        onEvent("INSPECT_START", candidate.boundingBox, null, null)
    }

    private fun inspect(persons: List<TrackedObject>) {
        val candidate = matchCandidate(persons)
        if (candidate == null) {
            // Distant subjects flicker — tolerate a few missing frames before giving up,
            // and hold the zoom rather than yo-yoing back out on a single dropped detection.
            inspectMissed++
            if (inspectMissed > INSPECT_MISS_TOLERANCE) rejectCurrent(null, "lost")
            return
        }
        inspectMissed = 0
        inspectBox = RectF(candidate.boundingBox)
        setZoomTarget(inspectZoomFor(candidate.boundingBox))
        inspectFrames++

        // Classify on the first inspect frame, then every CLASSIFY_INTERVAL.
        if ((inspectFrames - 1) % CLASSIFY_INTERVAL == 0) {
            val attr = classify(candidate)
            if (attr != null) {
                val matched = criteria().matches(attr)
                onEvent("CLASSIFY", candidate.boundingBox, attr, if (matched) "match" else "no-match")
                if (matched) {
                    state = State.MATCHED
                    haptic(SentryCue.MATCH)
                    onEvent("MATCH", candidate.boundingBox, attr, null)
                    lock(candidate)
                    resetInspect()
                    return
                } else {
                    rejectCurrent(attr, "no-match"); return
                }
            }
        }
        if (inspectFrames >= INSPECT_TIMEOUT) rejectCurrent(null, "timeout-no-face")
    }

    private fun rejectCurrent(attr: FaceAttributes?, reason: String) {
        onEvent("REJECT", inspectBox, attr, reason)
        // Cool the candidate by its WIDE-ZOOM box so the cooldown survives zooming back out
        // (a zoomed box wouldn't match the wide-zoom box next scan, and the same person would
        // be re-inspected instantly — the thrash we saw in the logs).
        inspectStartBox?.let { cooled.add(RectF(it) to REJECT_COOLDOWN) }
        resetInspect()
        state = State.SCANNING
        setZoomTarget(SCAN_ZOOM)
        haptic(SentryCue.REJECT)
    }

    private fun resetInspect() {
        inspectKey = null; inspectBox = null; inspectStartBox = null
        inspectFrames = 0; inspectMissed = 0
    }

    private fun resetScanState() {
        centeringKey = null; centeringStreak = 0
        cooled.clear()
    }

    private fun isCooled(box: RectF): Boolean =
        cooled.any { FrameToFrameTracker.computeIou(it.first, box) >= COOLDOWN_IOU }

    /** Re-locate the inspected candidate in this frame by id, else by IoU to the last box. */
    private fun matchCandidate(persons: List<TrackedObject>): TrackedObject? {
        val key = inspectKey ?: return null
        persons.firstOrNull { keyOf(it) == key }?.let { return it }
        val box = inspectBox ?: return null
        return persons
            .map { it to FrameToFrameTracker.computeIou(it.boundingBox, box) }
            .filter { it.second >= SAME_CANDIDATE_IOU }
            .maxByOrNull { it.second }?.first
    }

    /**
     * Zoom that brings [box] to INSPECT_OCCUPANCY of frame height, relative to the CURRENT
     * applied zoom (so it converges instead of oscillating). Holds when already in the deadzone.
     */
    private fun inspectZoomFor(box: RectF): Float {
        val z = currentZoom().coerceAtLeast(0.1f)
        val h = box.height().coerceAtLeast(0.02f)
        if (h in INSPECT_OCCUPANCY * (1f - INSPECT_DEADZONE)..INSPECT_OCCUPANCY * (1f + INSPECT_DEADZONE)) {
            return z  // already framed — hold
        }
        return (z * (INSPECT_OCCUPANCY / h)).coerceIn(minZoom(), maxZoom())
    }

    private fun centerDistance(box: RectF): Float =
        hypot((box.centerX() - 0.5f), (box.centerY() - 0.5f))

    /** Stable-ish candidate key: detector id when present, else a quantized center. */
    private fun keyOf(obj: TrackedObject): String =
        if (obj.id >= 0) "id:${obj.id}"
        else "pos:${(obj.boundingBox.centerX() * 20).toInt()},${(obj.boundingBox.centerY() * 20).toInt()}"
}
