package com.haptictrack.tracking

import android.content.Context
import android.graphics.Bitmap
import android.graphics.Color
import android.graphics.RectF
import android.util.Log
import org.tensorflow.lite.Interpreter
import java.nio.ByteBuffer
import java.nio.ByteOrder

/**
 * YOLOv8n-oiv7 label enricher: 601 Open Images classes via TFLite.
 *
 * NOT used as the primary detector (EfficientDet-Lite2 handles that).
 * Instead, runs on-demand to upgrade coarse COCO labels to finer OIV7 labels:
 * - At lock time: enriches the locked object's label
 * - During re-acquisition: enriches candidate labels for better discrimination
 *
 * Input: [1, 640, 640, 3] float32 (NHWC, 0-1 normalized)
 * Output: [1, 605, 8400] float32 (4 box coords + 601 class scores × 8400 anchors)
 */
class Yolov8Detector(context: Context) {

    companion object {
        private const val TAG = "Yolov8Det"
        private const val MODEL_ASSET = "yolov8n_oiv7.tflite"
        private const val LABELS_ASSET = "oiv7_labels.txt"
        private const val INPUT_SIZE = 640
        private const val OUTPUT_FEATURES = 605  // 4 box coords + 601 class scores
        private const val NUM_ANCHORS = 8400
        private const val NUM_CLASSES = 601
        private const val CONFIDENCE_THRESHOLD = 0.25f
        private const val NMS_IOU_THRESHOLD = 0.45f
        private const val MAX_RESULTS = 15

        /** Minimum IoU to consider a YOLOv8 detection as matching an EfficientDet box. */
        private const val ENRICH_IOU_THRESHOLD = 0.3f

        /** Body-part labels filtered from enrichment results. */
        private val BODY_PART_LABELS = setOf(
            "Human arm", "Human beard", "Human ear", "Human eye",
            "Human foot", "Human hair", "Human hand",
            "Human head", "Human leg", "Human mouth", "Human nose",
            "Clothing"
        )

        /** OIV7 labels that are valid enrichments for COCO "person". */
        private val PERSON_ENRICHMENTS = setOf(
            "Person", "Man", "Woman", "Boy", "Girl",
            "Human face", "Glasses"
        )
    }

    private val interpreter: Interpreter
    private val labels: List<String>

    // Pre-allocated buffers
    private val inputBuffer: ByteBuffer = ByteBuffer.allocateDirect(4 * INPUT_SIZE * INPUT_SIZE * 3).apply {
        order(ByteOrder.nativeOrder())
    }
    private val outputArray = Array(1) { Array(OUTPUT_FEATURES) { FloatArray(NUM_ANCHORS) } }

    init {
        val model = loadTfliteModel(context, MODEL_ASSET)
        interpreter = createGpuInterpreter(model, cpuThreads = 4)

        labels = context.assets.open(LABELS_ASSET).bufferedReader().use { it.readLines() }
        Log.i(TAG, "Loaded YOLOv8n-oiv7: ${labels.size} classes, input ${INPUT_SIZE}x${INPUT_SIZE}")
    }

    /**
     * Enrich a single detection's label by running YOLOv8 on the full frame
     * and finding the best-matching OIV7 detection by IoU.
     *
     * Returns the finer OIV7 label (lowercased), or the original label if no match.
     */
    fun enrichLabel(bitmap: Bitmap, box: RectF, coarseLabel: String?): String? {
        val dummy = TrackedObject(id = -1, boundingBox = box, label = coarseLabel)
        val result = enrichLabels(bitmap, listOf(dummy))
        val enriched = result[-1]
        if (enriched != null) {
            Log.d(TAG, "Enriched '$coarseLabel' → '$enriched'")
            return enriched
        }
        Log.d(TAG, "No enrichment for '$coarseLabel'")
        return coarseLabel
    }

    /**
     * Enrich labels for multiple detections in one pass.
     * Runs YOLOv8 once, then matches each input box to the best OIV7 detection.
     */
    fun enrichLabels(bitmap: Bitmap, objects: List<TrackedObject>): Map<Int, String> {
        val detections = detect(bitmap)
        val enriched = mutableMapOf<Int, String>()

        // Pre-filter body parts once
        val usable = detections.filter { it.label !in BODY_PART_LABELS }

        for (obj in objects) {
            // Compute IoU once per candidate, sort by it
            val best = usable
                .map { det -> det to computeIou(det.box, obj.boundingBox) }
                .filter { (_, iou) -> iou > ENRICH_IOU_THRESHOLD }
                .sortedByDescending { (_, iou) -> iou }
                .let { ranked ->
                    if (obj.label == "person") {
                        ranked.firstOrNull { (det, _) -> det.label in PERSON_ENRICHMENTS }
                    } else {
                        ranked.firstOrNull()
                    }
                }

            if (best != null) {
                enriched[obj.id] = best.first.label.lowercase()
            }
        }

        if (enriched.isNotEmpty()) {
            Log.d(TAG, "Enriched ${enriched.size}/${objects.size} labels: $enriched")
        }
        return enriched
    }

    fun close() {
        interpreter.close()
    }

    data class Detection(
        val box: RectF,
        val label: String,
        val confidence: Float
    )

    private fun detect(bitmap: Bitmap): List<Detection> {
        val resized = Bitmap.createScaledBitmap(bitmap, INPUT_SIZE, INPUT_SIZE, true)
        fillInputBuffer(resized)
        if (resized !== bitmap) resized.recycle()

        interpreter.run(inputBuffer, outputArray)

        val raw = mutableListOf<Detection>()
        val output = outputArray[0]

        for (anchor in 0 until NUM_ANCHORS) {
            var bestClass = 0
            var bestScore = 0f
            for (cls in 0 until NUM_CLASSES) {
                val score = output[4 + cls][anchor]
                if (score > bestScore) {
                    bestScore = score
                    bestClass = cls
                }
            }

            if (bestScore < CONFIDENCE_THRESHOLD) continue

            val cx = output[0][anchor]
            val cy = output[1][anchor]
            val w = output[2][anchor]
            val h = output[3][anchor]

            val left = (cx - w / 2f).coerceIn(0f, 1f)
            val top = (cy - h / 2f).coerceIn(0f, 1f)
            val right = (cx + w / 2f).coerceIn(0f, 1f)
            val bottom = (cy + h / 2f).coerceIn(0f, 1f)

            if (right - left < 0.01f || bottom - top < 0.01f) continue

            raw.add(Detection(
                box = RectF(left, top, right, bottom),
                label = labels.getOrElse(bestClass) { "unknown" },
                confidence = bestScore
            ))
        }

        return nms(raw).take(MAX_RESULTS)
    }

    private fun nms(detections: List<Detection>): List<Detection> {
        val sorted = detections.sortedByDescending { it.confidence }
        val kept = mutableListOf<Detection>()

        val suppressed = BooleanArray(sorted.size)
        for (i in sorted.indices) {
            if (suppressed[i]) continue
            kept.add(sorted[i])
            for (j in i + 1 until sorted.size) {
                if (suppressed[j]) continue
                if (sorted[i].label == sorted[j].label &&
                    computeIou(sorted[i].box, sorted[j].box) > NMS_IOU_THRESHOLD) {
                    suppressed[j] = true
                }
            }
        }
        return kept
    }

    private fun fillInputBuffer(bitmap: Bitmap) {
        inputBuffer.rewind()
        val pixels = IntArray(INPUT_SIZE * INPUT_SIZE)
        bitmap.getPixels(pixels, 0, INPUT_SIZE, 0, 0, INPUT_SIZE, INPUT_SIZE)
        for (pixel in pixels) {
            inputBuffer.putFloat(Color.red(pixel) / 255f)
            inputBuffer.putFloat(Color.green(pixel) / 255f)
            inputBuffer.putFloat(Color.blue(pixel) / 255f)
        }
        inputBuffer.rewind()
    }
}
