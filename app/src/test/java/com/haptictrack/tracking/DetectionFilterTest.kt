package com.haptictrack.tracking

import android.graphics.RectF
import org.junit.Assert.*
import org.junit.Before
import org.junit.Test
import org.junit.runner.RunWith
import org.robolectric.RobolectricTestRunner

@RunWith(RobolectricTestRunner::class)
class DetectionFilterTest {

    private lateinit var filter: DetectionFilter

    @Before
    fun setup() {
        filter = DetectionFilter()
    }

    // --- Valid objects pass through ---

    @Test
    fun `classified object with good confidence passes`() {
        val obj = obj(label = "Food", confidence = 0.8f,
            left = 0.3f, top = 0.3f, right = 0.6f, bottom = 0.6f)
        assertTrue(filter.isValid(obj))
    }

    @Test
    fun `filter returns only valid objects`() {
        val good = obj(label = "Food", confidence = 0.8f,
            left = 0.3f, top = 0.3f, right = 0.6f, bottom = 0.6f)
        val noise = obj(label = null, confidence = 0f,
            left = 0.0f, top = 0.0f, right = 0.1f, bottom = 0.8f)

        val result = filter.filter(listOf(good, noise))
        assertEquals(1, result.size)
        assertEquals("Food", result[0].label)
    }

    // --- Unclassified objects rejected ---

    @Test
    fun `rejects object with no label`() {
        val obj = obj(label = null, confidence = 0f,
            left = 0.3f, top = 0.3f, right = 0.6f, bottom = 0.6f)
        assertFalse(filter.isValid(obj))
    }

    // --- Confidence threshold ---

    @Test
    fun `rejects object below confidence threshold`() {
        val obj = obj(label = "Food", confidence = 0.1f,
            left = 0.3f, top = 0.3f, right = 0.6f, bottom = 0.6f)
        assertFalse(filter.isValid(obj))
    }

    @Test
    fun `accepts object at confidence threshold`() {
        val obj = obj(label = "Food", confidence = 0.5f,
            left = 0.3f, top = 0.3f, right = 0.6f, bottom = 0.6f)
        assertTrue(filter.isValid(obj))
    }

    // --- Size filtering ---

    @Test
    fun `rejects tiny box`() {
        // 0.02 * 0.02 = 0.0004 area, below minBoxArea of 0.005
        val obj = obj(label = "Food", confidence = 0.8f,
            left = 0.5f, top = 0.5f, right = 0.52f, bottom = 0.52f)
        assertFalse(filter.isValid(obj))
    }

    @Test
    fun `rejects box covering most of frame`() {
        // 0.9 * 0.9 = 0.81 area, above maxBoxArea of 0.7
        val obj = obj(label = "Food", confidence = 0.8f,
            left = 0.05f, top = 0.05f, right = 0.95f, bottom = 0.95f)
        assertFalse(filter.isValid(obj))
    }

    @Test
    fun `accepts medium-sized box`() {
        // 0.3 * 0.3 = 0.09 area
        val obj = obj(label = "Food", confidence = 0.8f,
            left = 0.3f, top = 0.3f, right = 0.6f, bottom = 0.6f)
        assertTrue(filter.isValid(obj))
    }

    // --- Aspect ratio filtering ---

    @Test
    fun `rejects extremely thin vertical box`() {
        // width=0.02, height=0.5 → aspect ratio 0.04
        val obj = obj(label = "Home good", confidence = 0.5f,
            left = 0.5f, top = 0.2f, right = 0.52f, bottom = 0.7f)
        assertFalse(filter.isValid(obj))
    }

    @Test
    fun `rejects extremely wide horizontal box`() {
        // width=0.8, height=0.02 → aspect ratio 40.0
        val obj = obj(label = "Home good", confidence = 0.5f,
            left = 0.1f, top = 0.5f, right = 0.9f, bottom = 0.52f)
        assertFalse(filter.isValid(obj))
    }

    @Test
    fun `accepts reasonable aspect ratio`() {
        // width=0.3, height=0.4 → aspect ratio 0.75
        val obj = obj(label = "Food", confidence = 0.8f,
            left = 0.3f, top = 0.2f, right = 0.6f, bottom = 0.6f)
        assertTrue(filter.isValid(obj))
    }

    // --- Invalid tracking ID ---

    @Test
    fun `rejects negative tracking ID`() {
        val obj = obj(id = -1, label = "Food", confidence = 0.8f,
            left = 0.3f, top = 0.3f, right = 0.6f, bottom = 0.6f)
        assertFalse(filter.isValid(obj))
    }

    // --- Edge cases ---

    @Test
    fun `rejects zero-size box`() {
        val obj = obj(label = "Food", confidence = 0.8f,
            left = 0.5f, top = 0.5f, right = 0.5f, bottom = 0.5f)
        assertFalse(filter.isValid(obj))
    }

    @Test
    fun `empty input returns empty output`() {
        assertEquals(0, filter.filter(emptyList()).size)
    }

    @Test
    fun `custom thresholds are respected`() {
        val strict = DetectionFilter(minConfidence = 0.9f)
        val obj = obj(label = "Food", confidence = 0.85f,
            left = 0.3f, top = 0.3f, right = 0.6f, bottom = 0.6f)
        assertFalse(strict.isValid(obj))
    }

    // --- Helpers ---

    private fun obj(
        id: Int = 1,
        left: Float = 0f,
        top: Float = 0f,
        right: Float = 0.1f,
        bottom: Float = 0.1f,
        label: String? = null,
        confidence: Float = 0.8f
    ) = TrackedObject(
        id = id,
        boundingBox = RectF(left, top, right, bottom),
        label = label,
        confidence = confidence
    )
}
