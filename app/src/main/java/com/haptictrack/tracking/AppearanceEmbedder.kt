package com.haptictrack.tracking

import android.content.Context
import android.graphics.Bitmap
import android.graphics.Matrix
import android.graphics.RectF
import android.util.Log
import com.google.mediapipe.framework.image.BitmapImageBuilder
import com.google.mediapipe.tasks.core.BaseOptions
import com.google.mediapipe.tasks.core.Delegate
import com.google.mediapipe.tasks.vision.imageembedder.ImageEmbedder

/**
 * Wraps MediaPipe Image Embedder to extract visual feature vectors from object crops.
 *
 * Used to distinguish "which cup" vs "a cup" — at lock time we store the locked
 * object's embedding, then during re-acquisition we compare candidates by cosine
 * similarity against the stored reference.
 *
 * Model: MobileNetV3 Large — ~10MB, ~8ms per crop.
 *
 * #100 input flow: every embed call consumes a [CanonicalCrop] at exactly
 * [MNV3_INPUT_SIZE]² so MediaPipe's internal `keep_aspect_ratio=false`
 * stretch is bypassed — the model sees uniformly aspect-preserved input
 * across every detection. When segmentation succeeds, the masked canonical
 * is fed instead of the raw canonical so the model embeds foreground only.
 */
class AppearanceEmbedder(
    context: Context,
    private val cropper: CanonicalCropper = CanonicalCropper(),
) {

    companion object {
        private const val TAG = "AppearEmbed"
        private const val MODEL_PATH = "mobilenet_v3_large_embedder.tflite"
        /**
         * MobileNetV3 Large native input. We pre-letterbox to this so MediaPipe
         * skips its internal resize and can't silently re-stretch our inputs.
         */
        const val MNV3_INPUT_SIZE = 224
    }

    private val embedder: ImageEmbedder
    private val segmenter: ObjectSegmenter = ObjectSegmenter(context, cropper)

    init {
        embedder = try {
            val gpuOptions = BaseOptions.builder()
                .setModelAssetPath(MODEL_PATH)
                .setDelegate(Delegate.GPU)
                .build()
            ImageEmbedder.createFromOptions(context, ImageEmbedder.ImageEmbedderOptions.builder()
                .setBaseOptions(gpuOptions).setL2Normalize(true).setQuantize(false).build())
        } catch (e: Exception) {
            Log.w(TAG, "GPU delegate failed, falling back to CPU: ${e.message}")
            val cpuOptions = BaseOptions.builder().setModelAssetPath(MODEL_PATH).build()
            ImageEmbedder.createFromOptions(context, ImageEmbedder.ImageEmbedderOptions.builder()
                .setBaseOptions(cpuOptions).setL2Normalize(true).setQuantize(false).build())
        }
    }

    /**
     * Result of a single segmentation pass that provides both embedding and masked crop.
     * Avoids running the segmenter twice when both signals are needed.
     */
    data class EmbedResult(
        val embedding: FloatArray?,
        /** Masked crop bitmap — caller must recycle when done. Null if segmentation failed. */
        val maskedCrop: Bitmap?
    )

    /**
     * Compute an MNV3 embedding from a prepared [CanonicalCrop]. Caller owns
     * [canonical] and is responsible for recycling. Required dims: 224×224.
     */
    fun embed(canonical: CanonicalCrop): FloatArray? {
        if (canonical.targetWidth != MNV3_INPUT_SIZE || canonical.targetHeight != MNV3_INPUT_SIZE) {
            Log.w(TAG, "Canonical dims ${canonical.targetWidth}×${canonical.targetHeight} != expected ${MNV3_INPUT_SIZE}²")
            return null
        }
        return embedBitmap(canonical.bitmap)
    }

    /**
     * Extract an embedding using segmentation masking. Returns null if segmentation
     * fails — caller should NOT match against unmasked embeddings since they include
     * too much background noise for reliable identity discrimination.
     */
    fun embed(bitmap: Bitmap, normalizedBox: RectF): FloatArray? {
        return embedAndCrop(bitmap, normalizedBox, fallback = false).also { it.maskedCrop?.recycle() }.embedding
    }

    /**
     * Extract an embedding with raw crop fallback. Used at lock time or during
     * gallery accumulation when we need at least some embedding even if segmentation fails.
     */
    fun embedWithFallback(bitmap: Bitmap, normalizedBox: RectF): FloatArray? {
        return embedAndCrop(bitmap, normalizedBox, fallback = true).also { it.maskedCrop?.recycle() }.embedding
    }

    /**
     * Single segmentation pass that returns both the embedding and the masked
     * crop. When [fallback] is true and segmentation fails, the embedding is
     * computed from a raw MNV3 canonical (no mask), and [EmbedResult.maskedCrop]
     * is null. When [fallback] is false and segmentation fails, both fields are null.
     *
     * The returned [EmbedResult.maskedCrop], when present, is the segmenter
     * canonical at [ObjectSegmenter.MODEL_SIZE]² with non-foreground pixels
     * blacked out — useful for color-histogram computation. Caller recycles it.
     */
    fun embedAndCrop(bitmap: Bitmap, normalizedBox: RectF, fallback: Boolean = false): EmbedResult {
        // Try the segmenter canonical path first.
        val segCanonical = cropper.prepare(
            bitmap, normalizedBox,
            targetWidth = ObjectSegmenter.MODEL_SIZE, targetHeight = ObjectSegmenter.MODEL_SIZE,
        )
        val masked: Bitmap? = if (segCanonical != null) {
            try { segmenter.segmentCanonical(segCanonical) }
            finally { segCanonical.bitmap.recycle() }
        } else null

        if (masked != null) {
            // Downscale masked 512² → 224² for MNV3. Both letterbox-fit the same
            // source aspect, so this is a clean shrink — no aspect distortion.
            val mnv3Input = Bitmap.createScaledBitmap(masked, MNV3_INPUT_SIZE, MNV3_INPUT_SIZE, true)
            val embedding = try {
                embedBitmap(mnv3Input)
            } catch (e: Exception) {
                Log.w(TAG, "Embed failed: ${e.message}")
                null
            }
            if (mnv3Input !== masked) mnv3Input.recycle()
            return EmbedResult(embedding, masked)
        }

        // Segmentation skipped or failed. Caller may want a raw fallback.
        if (!fallback) return EmbedResult(null, null)
        val mnv3Canonical = cropper.prepare(
            bitmap, normalizedBox,
            targetWidth = MNV3_INPUT_SIZE, targetHeight = MNV3_INPUT_SIZE,
        ) ?: return EmbedResult(null, null)
        return try {
            EmbedResult(embedBitmap(mnv3Canonical.bitmap), null)
        } finally {
            mnv3Canonical.bitmap.recycle()
        }
    }

    /**
     * Result of augmented embedding: the embeddings plus the masked crop for histogram use.
     */
    data class AugmentedResult(
        val embeddings: List<FloatArray>,
        /** Masked crop bitmap — caller must recycle when done. Null if segmentation failed. */
        val maskedCrop: Bitmap?
    )

    /**
     * Embed the canonical crop plus augmented variants (rotated 90/180/270, flipped).
     * Uses the segmenter to mask out background pixels before embedding when possible.
     * Returns embeddings covering multiple orientations plus the masked segmenter
     * canonical for histogram use (caller recycles). Single segmentation pass.
     */
    fun embedWithAugmentations(bitmap: Bitmap, normalizedBox: RectF): AugmentedResult {
        // Obtain the same pair (mnv3-input, masked-crop) as embedAndCrop, but
        // we need to keep the mnv3-input around for rotations.
        val segCanonical = cropper.prepare(
            bitmap, normalizedBox,
            targetWidth = ObjectSegmenter.MODEL_SIZE, targetHeight = ObjectSegmenter.MODEL_SIZE,
        )
        val masked: Bitmap? = if (segCanonical != null) {
            try { segmenter.segmentCanonical(segCanonical) }
            finally { segCanonical.bitmap.recycle() }
        } else null

        // Build the mnv3-input source: from masked canonical when available,
        // otherwise from a raw mnv3 canonical (lock time wants gallery diversity
        // even when segmentation fails).
        // Wrap createScaledBitmap so a downscale OOM doesn't leak `masked` —
        // the caller can't reach it via AugmentedResult if we throw before
        // returning.
        val (mnv3Source, ownsSource) = if (masked != null) {
            try {
                Pair(Bitmap.createScaledBitmap(masked, MNV3_INPUT_SIZE, MNV3_INPUT_SIZE, true), true)
            } catch (t: Throwable) {
                Log.w(TAG, "Augmented downscale failed: ${t.message}")
                masked.recycle()
                return AugmentedResult(emptyList(), null)
            }
        } else {
            val raw = cropper.prepare(
                bitmap, normalizedBox,
                targetWidth = MNV3_INPUT_SIZE, targetHeight = MNV3_INPUT_SIZE,
            )
            if (raw == null) return AugmentedResult(emptyList(), masked)
            Pair(raw.bitmap, true)  // canonical owner — we recycle below
        }

        val results = mutableListOf<FloatArray>()
        try {
            embedBitmap(mnv3Source)?.let { results.add(it) }

            for (degrees in listOf(90f, 180f, 270f)) {
                val rotated = rotateBitmap(mnv3Source, degrees)
                embedBitmap(rotated)?.let { results.add(it) }
                if (rotated !== mnv3Source) rotated.recycle()
            }

            val flipped = flipBitmap(mnv3Source)
            embedBitmap(flipped)?.let { results.add(it) }
            flipped.recycle()
        } catch (e: Exception) {
            Log.w(TAG, "Augmented embedding failed: ${e.message}")
        } finally {
            if (ownsSource) mnv3Source.recycle()
        }

        Log.d(TAG, "Generated ${results.size} augmented embeddings")
        return AugmentedResult(results, masked)
    }

    private fun embedBitmap(bitmap: Bitmap): FloatArray? {
        return try {
            val mpImage = BitmapImageBuilder(bitmap).build()
            val result = embedder.embed(mpImage)
            val embedding = result.embeddingResult().embeddings().firstOrNull()
            val raw = embedding?.floatEmbedding()
            if (raw != null && raw.isNotEmpty()) raw else null
        } catch (e: Exception) {
            Log.w(TAG, "Embed failed: ${e.message}")
            null
        }
    }

    private fun rotateBitmap(bitmap: Bitmap, degrees: Float): Bitmap {
        val matrix = Matrix().apply { postRotate(degrees) }
        return Bitmap.createBitmap(bitmap, 0, 0, bitmap.width, bitmap.height, matrix, true)
    }

    private fun flipBitmap(bitmap: Bitmap): Bitmap {
        val matrix = Matrix().apply { preScale(-1f, 1f) }
        return Bitmap.createBitmap(bitmap, 0, 0, bitmap.width, bitmap.height, matrix, true)
    }

    /**
     * Extract the object contour as normalized [0,1] points for UI overlay.
     */
    fun extractContour(bitmap: Bitmap, normalizedBox: RectF): List<android.graphics.PointF> {
        return segmenter.extractContour(bitmap, normalizedBox)
    }

    /**
     * Run segmentation and return the tight foreground bounding box in
     * normalized source-frame coordinates. Cheaper than [extractContour].
     */
    fun extractTightBbox(bitmap: Bitmap, normalizedBox: RectF): RectF? {
        return segmenter.extractTightBbox(bitmap, normalizedBox)
    }

    /**
     * Audit/debug only — returns the segmenter's masked crop without computing
     * an embedding. Caller must recycle. Null if segmentation failed (large
     * crop, empty mask, etc.). Used by [CropDebugCapture] for visual review.
     */
    fun debugMaskedCrop(bitmap: Bitmap, normalizedBox: RectF): Bitmap? =
        segmenter.segmentAndCrop(bitmap, normalizedBox)

    fun shutdown() {
        embedder.close()
        segmenter.shutdown()
    }
}
