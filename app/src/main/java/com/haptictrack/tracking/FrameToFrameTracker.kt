package com.haptictrack.tracking

import android.graphics.RectF
import android.util.Log

/**
 * Assigns stable tracking IDs to per-frame detections using IoU matching.
 * MediaPipe doesn't provide tracking IDs, so we do it ourselves.
 *
 * Each frame, we match new detections to previous detections by bounding box
 * overlap (Intersection over Union). Matched detections keep their ID;
 * unmatched ones get a new ID.
 */
class FrameToFrameTracker(
    private val minIou: Float = 0.2f,
    /** Lower IoU threshold when velocity prediction shifts boxes before matching. */
    private val minIouWithPrediction: Float = 0.1f
) {

    private var nextId: Int = 1
    private var previousFrame: List<TrackedObject> = emptyList()

    /**
     * Velocity-aware ID matching: shifts previous frame's boxes by the velocity
     * vector before computing IoU. This handles fast-moving objects that would
     * otherwise lose their ID because the box displacement exceeds the IoU threshold.
     *
     * Uses a lower IoU threshold ([minIouWithPrediction]) since the prediction
     * accounts for most of the displacement.
     */
    fun assignIdsWithVelocity(
        detections: List<TrackedObject>,
        velocityX: Float,
        velocityY: Float
    ): List<TrackedObject> {
        if (previousFrame.isEmpty()) {
            val result = detections.map { it.copy(id = nextId++) }
            previousFrame = result
            return result
        }

        // Shift previous boxes by velocity to predict where they should be now
        val shiftedPrev = previousFrame.map { prev ->
            val shifted = RectF(
                (prev.boundingBox.left + velocityX).coerceIn(0f, 1f),
                (prev.boundingBox.top + velocityY).coerceIn(0f, 1f),
                (prev.boundingBox.right + velocityX).coerceIn(0f, 1f),
                (prev.boundingBox.bottom + velocityY).coerceIn(0f, 1f)
            )
            prev.copy(boundingBox = shifted)
        }

        val used = mutableSetOf<Int>()
        val result = mutableListOf<TrackedObject>()

        for (detection in detections) {
            var bestIdx = -1
            var bestIou = minIouWithPrediction

            for ((idx, prev) in shiftedPrev.withIndex()) {
                if (idx in used) continue
                val iou = computeIou(detection.boundingBox, prev.boundingBox)
                if (iou > bestIou) {
                    bestIou = iou
                    bestIdx = idx
                }
            }

            if (bestIdx >= 0) {
                val prevId = previousFrame[bestIdx].id
                used.add(bestIdx)
                result.add(detection.copy(id = prevId))
                Log.v("FTFTracker", "MATCH(vel) label=\"${detection.label}\" iou=${String.format("%.2f", bestIou)} → id=$prevId")
            } else {
                val newId = nextId++
                result.add(detection.copy(id = newId))
                Log.d("FTFTracker", "NEW label=\"${detection.label}\" → id=$newId box=[${String.format("%.2f,%.2f,%.2f,%.2f", detection.boundingBox.left, detection.boundingBox.top, detection.boundingBox.right, detection.boundingBox.bottom)}]")
            }
        }

        previousFrame = result
        return result
    }

    fun assignIds(detections: List<TrackedObject>): List<TrackedObject> {
        if (previousFrame.isEmpty()) {
            val result = detections.map { it.copy(id = nextId++) }
            previousFrame = result
            return result
        }

        val used = mutableSetOf<Int>() // indices of previous detections already matched
        val result = mutableListOf<TrackedObject>()

        // For each new detection, find best matching previous detection by IoU
        for (detection in detections) {
            var bestIdx = -1
            var bestIou = minIou

            for ((idx, prev) in previousFrame.withIndex()) {
                if (idx in used) continue
                val iou = computeIou(detection.boundingBox, prev.boundingBox)
                if (iou > bestIou) {
                    bestIou = iou
                    bestIdx = idx
                }
            }

            if (bestIdx >= 0) {
                val prevId = previousFrame[bestIdx].id
                used.add(bestIdx)
                result.add(detection.copy(id = prevId))
                Log.v("FTFTracker", "MATCH label=\"${detection.label}\" iou=${String.format("%.2f", bestIou)} → id=$prevId")
            } else {
                val newId = nextId++
                result.add(detection.copy(id = newId))
                Log.d("FTFTracker", "NEW label=\"${detection.label}\" → id=$newId box=[${String.format("%.2f,%.2f,%.2f,%.2f", detection.boundingBox.left, detection.boundingBox.top, detection.boundingBox.right, detection.boundingBox.bottom)}]")
            }
        }

        previousFrame = result
        return result
    }

    fun reset() {
        previousFrame = emptyList()
        nextId = 1
    }

    companion object {
        /** Delegates to the shared top-level [computeIou] in EmbeddingUtils. */
        fun computeIou(a: RectF, b: RectF): Float = com.haptictrack.tracking.computeIou(a, b)
    }
}
