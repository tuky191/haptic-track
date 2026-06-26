// app/src/test/java/com/haptictrack/camera/GyroRollExposureTest.kt
package com.haptictrack.camera

import org.junit.Assert.assertEquals
import org.junit.Test
import org.junit.runner.RunWith
import org.robolectric.RobolectricTestRunner
import org.robolectric.RuntimeEnvironment

@RunWith(RobolectricTestRunner::class)
class GyroRollExposureTest {
    @Test fun `roll is zero before any sensor sample`() {
        val stab = GyroStabilizer(RuntimeEnvironment.getApplication())
        assertEquals(0f, stab.currentRollDeg(), 0.001f)
    }
}
