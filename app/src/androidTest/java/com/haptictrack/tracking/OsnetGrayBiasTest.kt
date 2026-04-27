package com.haptictrack.tracking

import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import android.graphics.Rect
import android.graphics.RectF
import android.util.Log
import androidx.test.core.app.ApplicationProvider
import androidx.test.ext.junit.runners.AndroidJUnit4
import org.junit.After
import org.junit.Before
import org.junit.Test
import org.junit.runner.RunWith
import kotlin.random.Random

/**
 * On-device diagnostic for the hypothesis that OSNet's response to canonical
 * inputs is dominated by the letterbox gray fill when bbox aspect mismatches
 * its 1:2 native target. Looking for evidence that two different subjects
 * with similar gray-padding ratios produce embeddings whose cosine similarity
 * is biased upward by a shared "gray-canvas response" feature.
 *
 * Runs OSNet on a fixed grid of synthetic + extracted-crop inputs and prints
 * a similarity matrix to logcat. No assertions — pull the logcat output to
 * read the result.
 */
@RunWith(AndroidJUnit4::class)
class OsnetGrayBiasTest {

    companion object {
        private const val TAG = "OsnetGrayBias"
    }

    private lateinit var pri: PersonReIdEmbedder

    @Before
    fun setup() {
        val context = ApplicationProvider.getApplicationContext<android.app.Application>()
        pri = PersonReIdEmbedder(context)
    }

    @After
    fun teardown() {
        pri.close()
    }

    @Test
    fun gray_canvas_baseline_and_padding_effect() {
        // Build inputs at OSNet's native 128×256.
        val gray = solid(Color.rgb(127, 127, 127))
        val black = solid(Color.BLACK)
        val white = solid(Color.WHITE)
        val noiseA = randomNoise(seed = 42)
        val noiseB = randomNoise(seed = 1337)

        // Two synthetic 128×256 "people" — these fill the canvas natively at OSNet's 1:2.
        val personA = personPatch(seed = 11)
        val personB = personPatch(seed = 22)
        val aFull = personA
        val bFull = personB

        // Two near-square 128×128 "people" — these mimic the real-world case
        // where a bbox aspect doesn't match OSNet's 1:2. We then test two
        // treatments of the same square source:
        //   1. Stretch (legacy pre-#100): resize 128×128 → 128×256, distorting vertically.
        //   2. Canonical letterbox (#100): center at 128×128 in 128×256 with gray fill.
        val squareA = personPatchSquare(seed = 11)
        val squareB = personPatchSquare(seed = 22)
        val aStretch = stretchToOsnet(squareA)
        val bStretch = stretchToOsnet(squareB)
        val aPad = paddedNearSquare(squareA)
        val bPad = paddedNearSquare(squareB)

        val embs = mapOf(
            "gray"      to embed(gray),
            "black"     to embed(black),
            "white"     to embed(white),
            "noiseA"    to embed(noiseA),
            "noiseB"    to embed(noiseB),
            "A_full"    to embed(aFull),
            "B_full"    to embed(bFull),
            "A_stretch" to embed(aStretch),
            "B_stretch" to embed(bStretch),
            "A_pad"     to embed(aPad),
            "B_pad"     to embed(bPad),
        ).filterValues { it != null }.mapValues { it.value!! }

        Log.i(TAG, "==== OSNet response baselines ====")
        for ((name, emb) in embs) {
            val mean = emb.average()
            val std = kotlin.math.sqrt(emb.sumOf { ((it - mean) * (it - mean)).toDouble() } / emb.size)
            Log.i(TAG, "  %-8s norm=%.3f mean=%.4f std=%.4f first8=%s".format(
                name, l2Norm(emb), mean, std,
                emb.take(8).joinToString(",") { "%.2f".format(it) }
            ))
        }

        Log.i(TAG, "")
        Log.i(TAG, "==== Cosine similarity matrix ====")
        val names = embs.keys.toList()
        val header = "          " + names.joinToString("") { "%-8s".format(it) }
        Log.i(TAG, header)
        for (a in names) {
            val row = StringBuilder("%-10s".format(a))
            for (b in names) {
                val sim = cosineSimilarity(embs[a]!!, embs[b]!!)
                row.append("%-8s".format("%.3f".format(sim)))
            }
            Log.i(TAG, row.toString())
        }

        // Specific comparisons — what we care about for the hypothesis.
        Log.i(TAG, "")
        Log.i(TAG, "==== Hypothesis-directed metrics ====")
        Log.i(TAG, "  A_full vs B_full   (different, native-aspect):    %.3f"
            .format(cosineSimilarity(embs["A_full"]!!, embs["B_full"]!!)))
        Log.i(TAG, "  A_stretch vs B_stretch (different, square→stretch): %.3f"
            .format(cosineSimilarity(embs["A_stretch"]!!, embs["B_stretch"]!!)))
        Log.i(TAG, "  A_pad  vs B_pad    (different, square→letterbox): %.3f"
            .format(cosineSimilarity(embs["A_pad"]!!, embs["B_pad"]!!)))
        Log.i(TAG, "  A_stretch vs A_pad (same source, stretch vs letterbox): %.3f"
            .format(cosineSimilarity(embs["A_stretch"]!!, embs["A_pad"]!!)))
        Log.i(TAG, "  gray   vs A_pad    (gray vs letterboxed person):  %.3f"
            .format(cosineSimilarity(embs["gray"]!!, embs["A_pad"]!!)))
        Log.i(TAG, "  gray   vs A_stretch (gray vs stretched person):   %.3f"
            .format(cosineSimilarity(embs["gray"]!!, embs["A_stretch"]!!)))

        // Separation = same-person sim - different-person sim. Bigger is better.
        Log.i(TAG, "")
        Log.i(TAG, "==== Treatment effect on separation ====")
        Log.i(TAG, "  Stretch:   sameAvsB = -, diffAB = %.3f"
            .format(cosineSimilarity(embs["A_stretch"]!!, embs["B_stretch"]!!)))
        Log.i(TAG, "  Letterbox: sameAvsB = -, diffAB = %.3f  (smaller diffAB → cleaner)"
            .format(cosineSimilarity(embs["A_pad"]!!, embs["B_pad"]!!)))
    }

    private fun embed(bmp: Bitmap): FloatArray? {
        val canonical = CanonicalCrop(
            bitmap = bmp,
            sourceBoxNormalized = RectF(0f, 0f, 1f, 1f),
            sourceCropPx = Rect(0, 0, bmp.width, bmp.height),
            padding = Padding(0, 0, 0, 0),
            targetWidth = 128,
            targetHeight = 256,
        )
        return pri.embed(canonical)
    }

    private fun solid(color: Int): Bitmap = Bitmap.createBitmap(128, 256, Bitmap.Config.ARGB_8888).apply {
        eraseColor(color)
    }

    private fun randomNoise(seed: Int): Bitmap {
        val rng = Random(seed)
        val pixels = IntArray(128 * 256) { Color.rgb(rng.nextInt(256), rng.nextInt(256), rng.nextInt(256)) }
        return Bitmap.createBitmap(pixels, 128, 256, Bitmap.Config.ARGB_8888)
    }

    /** Synthetic full-aspect (128×256) "person-ish" patch — a colored region with structure. */
    private fun personPatch(seed: Int): Bitmap {
        val rng = Random(seed)
        val bmp = Bitmap.createBitmap(128, 256, Bitmap.Config.ARGB_8888)
        val canvas = Canvas(bmp)
        canvas.drawColor(Color.rgb(rng.nextInt(60, 120), rng.nextInt(60, 120), rng.nextInt(60, 120)))
        val headColor = Color.rgb(rng.nextInt(150, 230), rng.nextInt(120, 200), rng.nextInt(100, 180))
        val p = Paint().apply { color = headColor; isAntiAlias = true }
        canvas.drawCircle(64f, 50f, 28f, p)
        p.color = Color.rgb(rng.nextInt(50, 200), rng.nextInt(50, 200), rng.nextInt(50, 200))
        canvas.drawRect(32f, 80f, 96f, 180f, p)
        p.color = Color.rgb(rng.nextInt(40, 100), rng.nextInt(40, 100), rng.nextInt(40, 100))
        canvas.drawRect(40f, 180f, 60f, 250f, p)
        canvas.drawRect(68f, 180f, 88f, 250f, p)
        return bmp
    }

    /** Synthetic near-square (128×128) "person-ish" patch — head + torso, no legs. */
    private fun personPatchSquare(seed: Int): Bitmap {
        val rng = Random(seed)
        val bmp = Bitmap.createBitmap(128, 128, Bitmap.Config.ARGB_8888)
        val canvas = Canvas(bmp)
        canvas.drawColor(Color.rgb(rng.nextInt(60, 120), rng.nextInt(60, 120), rng.nextInt(60, 120)))
        val headColor = Color.rgb(rng.nextInt(150, 230), rng.nextInt(120, 200), rng.nextInt(100, 180))
        val p = Paint().apply { color = headColor; isAntiAlias = true }
        canvas.drawCircle(64f, 32f, 22f, p)
        p.color = Color.rgb(rng.nextInt(50, 200), rng.nextInt(50, 200), rng.nextInt(50, 200))
        canvas.drawRect(36f, 60f, 92f, 120f, p)
        return bmp
    }

    /**
     * Stretch a 128×128 source to 128×256 (legacy pre-#100 OSNet treatment of
     * near-square bboxes — vertical 2× stretch, no aspect preservation).
     */
    private fun stretchToOsnet(src: Bitmap): Bitmap {
        return Bitmap.createScaledBitmap(src, 128, 256, true)
    }

    /**
     * Center a 128×128 source inside a 128×256 canvas with neutral-gray fill.
     * Mimics canonical letterbox of a 1:1 bbox into OSNet's 1:2 target
     * (≈ 50% of canvas becomes gray padding).
     */
    private fun paddedNearSquare(src: Bitmap): Bitmap {
        val out = Bitmap.createBitmap(128, 256, Bitmap.Config.ARGB_8888)
        val c = Canvas(out)
        c.drawColor(Color.rgb(127, 127, 127))
        val dstRect = Rect(0, 64, 128, 192)
        val srcRect = Rect(0, 0, src.width, src.height)
        c.drawBitmap(src, srcRect, dstRect, null)
        return out
    }
}
