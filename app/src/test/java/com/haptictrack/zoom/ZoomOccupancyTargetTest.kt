// app/src/test/java/com/haptictrack/zoom/ZoomOccupancyTargetTest.kt
package com.haptictrack.zoom

import android.graphics.RectF
import org.junit.Assert.assertTrue
import org.junit.Test
import org.junit.runner.RunWith
import org.robolectric.RobolectricTestRunner

@RunWith(RobolectricTestRunner::class)
class ZoomOccupancyTargetTest {
    @Test fun `larger occupancy target zooms in more`() {
        val box = RectF(0.45f, 0.45f, 0.55f, 0.55f)  // small subject
        val small = ZoomController().apply { occupancyTarget = 0.10f }.calculateZoom(box, 1f, 10f)
        val large = ZoomController().apply { occupancyTarget = 0.40f }.calculateZoom(box, 1f, 10f)
        assertTrue("bigger target -> more zoom", large > small)
    }
}
