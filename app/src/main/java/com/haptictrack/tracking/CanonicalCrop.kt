package com.haptictrack.tracking

import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Rect
import android.graphics.RectF
import android.util.Log
import kotlin.math.max

/**
 * Letterbox padding in target-pixel space. Sums to (targetW - drawW, targetH - drawH).
 */
data class Padding(val left: Int, val top: Int, val right: Int, val bottom: Int) {
    val isZero: Boolean get() = left == 0 && top == 0 && right == 0 && bottom == 0
}

/**
 * Aspect-preserving canonical crop ready for an embedder. The bitmap is
 * exactly [targetWidth] × [targetHeight] with the source content centered
 * and neutral-gray letterboxed; [padding] records the gray border on each
 * side so a downstream consumer can ignore it (e.g. when threading a mask).
 *
 * Caller of [CanonicalCropper.prepare] owns [bitmap] and is responsible
 * for recycling once all consumers (MNV3, OSNet, segmenter, face) are done.
 */
data class CanonicalCrop(
    val bitmap: Bitmap,
    /** Original normalized bbox in the source frame — for back-mapping. */
    val sourceBoxNormalized: RectF,
    /** The pixel rect in source-bitmap coords actually cropped (post-pad, post-clamp). */
    val sourceCropPx: Rect,
    val padding: Padding,
    val targetWidth: Int,
    val targetHeight: Int,
) {
    /** Width of the rendered source region inside the canonical bitmap (target pixels). */
    val drawWidth: Int get() = targetWidth - padding.left - padding.right
    /** Height of the rendered source region inside the canonical bitmap (target pixels). */
    val drawHeight: Int get() = targetHeight - padding.top - padding.bottom
}

/**
 * Builds canonical crops — one source-of-truth crop preparation feeding all
 * embedders. Replaces the previous per-embedder `cropNormalized` +
 * stretch-resize idiom that produced differently-distorted views of the
 * same subject across MNV3/OSNet/EdgeFace-XS/segmenter (#91 audit).
 *
 * Defaults agreed in #100:
 *  - Neutral-gray letterbox fill (0x7F7F7F) — channel-balanced, less mean
 *    shift than black for ImageNet-trained models. Empirical validation
 *    deferred to the audit re-baseline.
 *  - 5% padding around the bbox (matches the segmenter's existing pre-crop
 *    pad, applied uniformly across targets). Tunable per call.
 *  - 28×28 minimum source pixels — below that no embedder produces useful
 *    output. Folds in the spirit of #98.
 */
class CanonicalCropper {

    companion object {
        private const val TAG = "CanonicalCropper"
        /** Neutral-fill color for letterbox padding (gray, channel-balanced). */
        const val FILL_COLOR: Int = 0xFF7F7F7F.toInt()
        /** Default 5% bbox padding to match the segmenter's existing prep. */
        const val DEFAULT_PADDING_FRACTION: Float = 0.05f
        /** Below this many source pixels in either dim, refuse the crop. */
        const val DEFAULT_MIN_SOURCE_PIXELS: Int = 28
    }

    /**
     * Prepare a canonical crop from a normalized bbox in [source].
     *
     * Returns null if the source bbox is below [minSourcePixels] in either
     * dimension, or if any pixel-level crop step fails.
     */
    fun prepare(
        source: Bitmap,
        normalizedBox: RectF,
        targetWidth: Int,
        targetHeight: Int,
        paddingFraction: Float = DEFAULT_PADDING_FRACTION,
        minSourcePixels: Int = DEFAULT_MIN_SOURCE_PIXELS,
    ): CanonicalCrop? {
        val imgW = source.width
        val imgH = source.height
        if (imgW <= 0 || imgH <= 0 || targetWidth <= 0 || targetHeight <= 0) return null

        // Min-pixel guard on the raw (pre-pad) bbox so callers don't waste
        // embedder budget on inputs no model produces useful output for.
        val rawW = ((normalizedBox.right - normalizedBox.left) * imgW).toInt()
        val rawH = ((normalizedBox.bottom - normalizedBox.top) * imgH).toInt()
        if (rawW < minSourcePixels || rawH < minSourcePixels) return null

        // Pad and clamp into source pixel space.
        val padX = paddingFraction * (normalizedBox.right - normalizedBox.left)
        val padY = paddingFraction * (normalizedBox.bottom - normalizedBox.top)
        val cl = ((normalizedBox.left - padX) * imgW).toInt().coerceIn(0, imgW - 1)
        val ct = ((normalizedBox.top - padY) * imgH).toInt().coerceIn(0, imgH - 1)
        val cr = ((normalizedBox.right + padX) * imgW).toInt().coerceIn(cl + 1, imgW)
        val cb = ((normalizedBox.bottom + padY) * imgH).toInt().coerceIn(ct + 1, imgH)
        val cw = cr - cl
        val ch = cb - ct
        if (cw <= 0 || ch <= 0) return null

        // Aspect-preserving fit. Larger source aspect → fit width, pad top/bottom.
        // Smaller-or-equal source aspect → fit height, pad left/right.
        val srcAspect = cw.toFloat() / ch
        val dstAspect = targetWidth.toFloat() / targetHeight
        val drawW: Int
        val drawH: Int
        if (srcAspect > dstAspect) {
            drawW = targetWidth
            drawH = max(1, (targetWidth / srcAspect).toInt())
        } else {
            drawH = targetHeight
            drawW = max(1, (targetHeight * srcAspect).toInt())
        }
        val padLeft = (targetWidth - drawW) / 2
        val padTop = (targetHeight - drawH) / 2
        val padRight = targetWidth - drawW - padLeft
        val padBottom = targetHeight - drawH - padTop

        val canonical = try {
            Bitmap.createBitmap(targetWidth, targetHeight, Bitmap.Config.ARGB_8888)
        } catch (e: Throwable) {
            Log.w(TAG, "Bitmap alloc failed: ${e.message}")
            return null
        }
        try {
            val canvas = Canvas(canonical)
            canvas.drawColor(FILL_COLOR)
            val srcRect = Rect(cl, ct, cr, cb)
            val dstRect = Rect(padLeft, padTop, padLeft + drawW, padTop + drawH)
            canvas.drawBitmap(source, srcRect, dstRect, null)
        } catch (e: Throwable) {
            canonical.recycle()
            Log.w(TAG, "Canvas draw failed: ${e.message}")
            return null
        }

        return CanonicalCrop(
            bitmap = canonical,
            sourceBoxNormalized = RectF(normalizedBox),
            sourceCropPx = Rect(cl, ct, cr, cb),
            padding = Padding(padLeft, padTop, padRight, padBottom),
            targetWidth = targetWidth,
            targetHeight = targetHeight,
        )
    }
}
