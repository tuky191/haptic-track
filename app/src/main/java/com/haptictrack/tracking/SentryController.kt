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
    private var inspectBox: RectF? = null
    private var inspectFrames = 0
    private val cooldowns = HashMap<String, Int>()

    fun setEnabled(enabled: Boolean) {
        if (enabled) {
            if (state == State.OFF) {
                state = State.SCANNING
                haptic(SentryCue.SCANNING)
            }
        } else {
            state = State.OFF
            resetInspect()
            cooldowns.clear()
        }
    }

    /** Call when the user/system clears the lock — re-arm scanning if still on. */
    fun onLockCleared() {
        if (state == State.MATCHED) {
            state = State.SCANNING
            resetInspect()
            haptic(SentryCue.SCANNING)
        }
    }

    /** Drive one detection frame. [persons] are person detections in normalized screen coords. */
    fun onFrame(persons: List<TrackedObject>) {
        // Age out cooldowns every frame.
        if (cooldowns.isNotEmpty()) {
            val iter = cooldowns.entries.iterator()
            while (iter.hasNext()) {
                val e = iter.next()
                val v = e.value - 1
                if (v <= 0) iter.remove() else e.setValue(v)
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
            .filter { keyOf(it) !in cooldowns }
            .minByOrNull { centerDistance(it.boundingBox) }
            ?: return
        if (centerDistance(candidate.boundingBox) <= CENTER_TOLERANCE) {
            inspectKey = keyOf(candidate)
            inspectBox = RectF(candidate.boundingBox)
            inspectFrames = 0
            state = State.INSPECTING
            setZoomTarget(inspectZoomFor(candidate.boundingBox))  // start zooming in immediately
            haptic(SentryCue.INSPECTING)
            onEvent("INSPECT_START", candidate.boundingBox, null, null)
        }
        // else: not centered enough — stay scanning (handheld: user pans toward them).
    }

    private fun inspect(persons: List<TrackedObject>) {
        val candidate = matchCandidate(persons)
        if (candidate == null) {
            // Lost the candidate (walked off / occluded) — back to scanning.
            rejectCurrent(null, "lost"); return
        }
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
        inspectKey?.let { cooldowns[it] = REJECT_COOLDOWN }
        resetInspect()
        state = State.SCANNING
        setZoomTarget(SCAN_ZOOM)
        haptic(SentryCue.REJECT)
    }

    private fun resetInspect() {
        inspectKey = null; inspectBox = null; inspectFrames = 0
    }

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

    /** Zoom that brings [box] to INSPECT_OCCUPANCY of frame height, relative to current scan zoom. */
    private fun inspectZoomFor(box: RectF): Float {
        val h = box.height().coerceAtLeast(0.01f)
        val target = SCAN_ZOOM * (INSPECT_OCCUPANCY / h)
        return target.coerceIn(minZoom(), maxZoom())
    }

    private fun centerDistance(box: RectF): Float =
        hypot((box.centerX() - 0.5f), (box.centerY() - 0.5f))

    /** Stable-ish candidate key: detector id when present, else a quantized center. */
    private fun keyOf(obj: TrackedObject): String =
        if (obj.id >= 0) "id:${obj.id}"
        else "pos:${(obj.boundingBox.centerX() * 20).toInt()},${(obj.boundingBox.centerY() * 20).toInt()}"
}
