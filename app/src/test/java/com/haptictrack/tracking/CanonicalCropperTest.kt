package com.haptictrack.tracking

import android.graphics.Bitmap
import android.graphics.RectF
import org.junit.Assert.assertEquals
import org.junit.Assert.assertNotNull
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Before
import org.junit.Test
import org.junit.runner.RunWith
import org.robolectric.RobolectricTestRunner

@RunWith(RobolectricTestRunner::class)
class CanonicalCropperTest {

    private lateinit var cropper: CanonicalCropper
    private lateinit var source: Bitmap

    @Before
    fun setup() {
        cropper = CanonicalCropper()
        // 1000×1000 source so percent boxes map to integer pixel sizes cleanly.
        source = Bitmap.createBitmap(1000, 1000, Bitmap.Config.ARGB_8888)
    }

    // --- Sum invariant: padding + drawDims = target dims ---

    @Test
    fun `padding plus draw dims sums to target dims`() {
        // Source 200×200 (aspect 1.0) into OSNet target 128×256 (aspect 0.5).
        // Source is wider than target → fit width, pad top/bottom.
        val box = RectF(0.4f, 0.4f, 0.6f, 0.6f)
        val out = cropper.prepare(source, box, targetWidth = 128, targetHeight = 256)!!
        assertEquals(128, out.targetWidth)
        assertEquals(256, out.targetHeight)
        assertEquals(out.targetWidth, out.bitmap.width)
        assertEquals(out.targetHeight, out.bitmap.height)
        assertEquals(0, out.padding.left)
        assertEquals(0, out.padding.right)
        assertTrue("top+bottom pad should fill the target height", out.padding.top + out.padding.bottom > 0)
    }

    // --- Aspect preservation cases ---

    @Test
    fun `square source into square target has no padding`() {
        // 200×200 → 224×224 (no aspect mismatch).
        val box = RectF(0.4f, 0.4f, 0.6f, 0.6f)
        val out = cropper.prepare(source, box, targetWidth = 224, targetHeight = 224)!!
        assertTrue("no padding expected for matching aspect", out.padding.isZero)
    }

    @Test
    fun `near-square source into 1 to 2 target letterboxes top and bottom`() {
        // 146×133 person bbox (aspect ≈ 1.10), OSNet target 128×256 (aspect 0.5).
        // srcAspect (≈1.10) > dstAspect (0.5) → fit width to 128, drawH ≈ 116.
        // Padding goes top+bottom to fill the remaining 256 - drawH rows.
        val box = RectF(0.4f, 0.4f, 0.4f + 0.146f, 0.4f + 0.133f)
        val out = cropper.prepare(source, box, targetWidth = 128, targetHeight = 256)!!
        assertEquals(0, out.padding.left)
        assertEquals(0, out.padding.right)
        assertTrue("expected top+bottom letterbox padding", out.padding.top + out.padding.bottom > 0)
    }

    @Test
    fun `tall source into 2 to 1 target letterboxes top and bottom`() {
        // 73×258 person bbox (aspect ≈ 0.283) into MNV3 target 224×224 (aspect 1.0).
        // srcAspect (0.283) < dstAspect (1.0) → fit height, pad left/right.
        val box = RectF(0.4f, 0.2f, 0.4f + 0.073f, 0.2f + 0.258f)
        val out = cropper.prepare(source, box, targetWidth = 224, targetHeight = 224)!!
        assertEquals(0, out.padding.top)
        assertEquals(0, out.padding.bottom)
        assertTrue("expected left+right letterbox padding for tall source", out.padding.left + out.padding.right > 0)
    }

    @Test
    fun `wide source into square target letterboxes top and bottom`() {
        // 300×100 source (aspect 3.0) into 224×224 target (aspect 1.0).
        val box = RectF(0.2f, 0.5f, 0.5f, 0.6f)
        val out = cropper.prepare(source, box, targetWidth = 224, targetHeight = 224)!!
        assertEquals(0, out.padding.left)
        assertEquals(0, out.padding.right)
        assertTrue("wide source should pad top+bottom", out.padding.top + out.padding.bottom > 0)
    }

    // --- Min-pixel guard (#98) ---

    @Test
    fun `bbox smaller than min source pixels returns null`() {
        // 20×20 bbox in a 1000×1000 source — below default 28×28 minimum.
        val box = RectF(0.5f, 0.5f, 0.52f, 0.52f)
        assertNull(cropper.prepare(source, box, 224, 224))
    }

    @Test
    fun `bbox just above min source pixels returns crop`() {
        // 30×30 bbox — clear of the 28-pixel threshold and float precision.
        val box = RectF(0.5f, 0.5f, 0.53f, 0.53f)
        assertNotNull(cropper.prepare(source, box, 224, 224))
    }

    @Test
    fun `min source pixels override is respected`() {
        // 12×12 bbox — below default 28, but custom override to 10 lets it pass.
        val box = RectF(0.5f, 0.5f, 0.512f, 0.512f)
        assertNotNull(cropper.prepare(source, box, 224, 224, minSourcePixels = 10))
    }

    // --- Edge clamping ---

    @Test
    fun `bbox at frame edge does not crash and returns valid crop`() {
        // Bbox flush against the right and bottom edges of the source.
        val box = RectF(0.85f, 0.85f, 1.0f, 1.0f)
        val out = cropper.prepare(source, box, 224, 224)
        assertNotNull(out)
    }

    @Test
    fun `bbox extending past frame edge is clamped`() {
        // Caller provides a box that, with 5% pad, would go past the right edge.
        // The cropper should clamp without error.
        val box = RectF(0.9f, 0.4f, 0.99f, 0.5f)
        val out = cropper.prepare(source, box, 224, 224)
        assertNotNull(out)
    }

    // --- Null cases ---

    @Test
    fun `inverted bbox returns null`() {
        val box = RectF(0.6f, 0.6f, 0.4f, 0.4f)
        assertNull(cropper.prepare(source, box, 224, 224))
    }

    @Test
    fun `zero-size bbox returns null`() {
        val box = RectF(0.5f, 0.5f, 0.5f, 0.5f)
        assertNull(cropper.prepare(source, box, 224, 224))
    }

    @Test
    fun `zero target dims returns null`() {
        val box = RectF(0.4f, 0.4f, 0.6f, 0.6f)
        assertNull(cropper.prepare(source, box, 0, 224))
        assertNull(cropper.prepare(source, box, 224, 0))
    }

    // --- Source bitmap is preserved (we own the canonical, source remains caller's) ---

    @Test
    fun `source bitmap is not recycled`() {
        val box = RectF(0.4f, 0.4f, 0.6f, 0.6f)
        val out = cropper.prepare(source, box, 224, 224)!!
        out.bitmap.recycle()
        assertTrue("source must not be recycled by the cropper", !source.isRecycled)
    }

    // --- Source-box round-trip ---

    @Test
    fun `sourceBoxNormalized matches the input box`() {
        val box = RectF(0.123f, 0.456f, 0.789f, 0.987f)
        val out = cropper.prepare(source, box, 224, 224)!!
        assertEquals(box.left, out.sourceBoxNormalized.left, 1e-6f)
        assertEquals(box.top, out.sourceBoxNormalized.top, 1e-6f)
        assertEquals(box.right, out.sourceBoxNormalized.right, 1e-6f)
        assertEquals(box.bottom, out.sourceBoxNormalized.bottom, 1e-6f)
    }

    @Test
    fun `sourceCropPx covers the padded bbox in source pixels`() {
        // 1000×1000 source, bbox covers 40-60 in both dims (200 px).
        // Default 5% padding adds 0.01 each side → bbox 0.39..0.61 → 390..610 source px.
        val box = RectF(0.4f, 0.4f, 0.6f, 0.6f)
        val out = cropper.prepare(source, box, 224, 224)!!
        // Width and height should match (square bbox + 5% pad).
        assertEquals(out.sourceCropPx.width(), out.sourceCropPx.height())
        // Should be ~220 px (200 raw + 5% × 200 each side = 220).
        assertTrue("expected ~220 px crop, got ${out.sourceCropPx.width()}", out.sourceCropPx.width() in 215..225)
    }

    @Test
    fun `drawWidth and drawHeight reflect aspect-preserved fit`() {
        // Square source into 1:2 portrait target → fit width, drawHeight < targetHeight.
        val box = RectF(0.4f, 0.4f, 0.6f, 0.6f)
        val out = cropper.prepare(source, box, 128, 256)!!
        assertEquals(128, out.drawWidth)
        assertTrue("drawHeight should be < targetHeight for square→portrait", out.drawHeight < 256)
        // drawHeight + paddingTop + paddingBottom == targetHeight
        assertEquals(256, out.drawHeight + out.padding.top + out.padding.bottom)
    }

    // Note: actual pixel-fill color is verified visually during the audit
    // re-baseline (composite JPEGs). Robolectric's default legacy graphics
    // mode no-ops Canvas operations, so getPixel-based tests are unreliable
    // here without GraphicsMode.NATIVE. Math/dimension tests above cover
    // the refactoring contract; pixel content is an empirical concern.
}
