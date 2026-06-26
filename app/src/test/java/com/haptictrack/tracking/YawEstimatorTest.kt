package com.haptictrack.tracking

import android.graphics.PointF
import org.junit.Assert.assertTrue
import org.junit.Test
import org.junit.runner.RunWith
import org.robolectric.RobolectricTestRunner

@RunWith(RobolectricTestRunner::class)
class YawEstimatorTest {
    // Eyes span x in [0.4,0.6] at y=0.5; nose y=0.55.
    private fun yaw(noseX: Float) =
        estimateYawDeg(PointF(0.4f, 0.5f), PointF(0.6f, 0.5f), PointF(noseX, 0.55f))!!

    @Test fun `nose centered between eyes is frontal`() {
        assertTrue(kotlin.math.abs(yaw(0.5f)) < 8f)
    }
    @Test fun `nose toward image-right eye means facing right (positive)`() {
        assertTrue(yaw(0.58f) > 15f)
    }
    @Test fun `nose toward image-left eye means facing left (negative)`() {
        assertTrue(yaw(0.42f) < -15f)
    }
    @Test fun `coincident eyes return null`() {
        assertTrue(estimateYawDeg(PointF(0.5f, 0.5f), PointF(0.5f, 0.5f), PointF(0.5f, 0.55f)) == null)
    }
}
