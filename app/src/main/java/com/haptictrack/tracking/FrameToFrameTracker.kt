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
    private val minIou: Float = 0.2f
) {

    private var nextId: Int = 1
    private var previousFrame: List<TrackedObject> = emptyList()

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
        fun computeIou(a: RectF, b: RectF): Float {
            val interLeft = maxOf(a.left, b.left)
            val interTop = maxOf(a.top, b.top)
            val interRight = minOf(a.right, b.right)
            val interBottom = minOf(a.bottom, b.bottom)

            if (interLeft >= interRight || interTop >= interBottom) return 0f

            val interArea = (interRight - interLeft) * (interBottom - interTop)
            val aArea = a.width() * a.height()
            val bArea = b.width() * b.height()
            val unionArea = aArea + bArea - interArea

            return if (unionArea > 0f) interArea / unionArea else 0f
        }
    }
}
