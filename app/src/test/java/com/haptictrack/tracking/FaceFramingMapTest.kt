package com.haptictrack.tracking

import android.graphics.RectF
import org.junit.Assert.assertEquals
import org.junit.Test
import org.junit.runner.RunWith
import org.robolectric.RobolectricTestRunner

@RunWith(RobolectricTestRunner::class)
class FaceFramingMapTest {
    @Test fun `face within person maps into screen coords`() {
        val person = RectF(0.40f, 0.20f, 0.60f, 0.80f)        // 0.2 wide, 0.6 tall
        val faceInPerson = RectF(0.25f, 0.0f, 0.75f, 0.20f)   // top-center band of the person crop
        val screen = mapFaceToScreen(faceInPerson, person)
        assertEquals(0.40f + 0.25f * 0.20f, screen.left, 1e-4f)
        assertEquals(0.20f + 0.0f * 0.60f, screen.top, 1e-4f)
        assertEquals(0.40f + 0.75f * 0.20f, screen.right, 1e-4f)
        assertEquals(0.20f + 0.20f * 0.60f, screen.bottom, 1e-4f)
    }
}
