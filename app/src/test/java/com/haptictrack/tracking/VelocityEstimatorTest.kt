package com.haptictrack.tracking

import android.graphics.RectF
import org.junit.Assert.*
import org.junit.Before
import org.junit.Test
import org.junit.runner.RunWith
import org.robolectric.RobolectricTestRunner

@RunWith(RobolectricTestRunner::class)
class VelocityEstimatorTest {

    private lateinit var estimator: VelocityEstimator

    @Before
    fun setup() {
        estimator = VelocityEstimator()
    }

    @Test
    fun `first update produces zero velocity`() {
        estimator.update(0.5f, 0.5f)
        assertEquals(0f, estimator.velocityX, 0.001f)
        assertEquals(0f, estimator.velocityY, 0.001f)
        assertEquals(0f, estimator.speed, 0.001f)
    }

    @Test
    fun `constant rightward motion produces positive velocityX`() {
        // Move 0.05 per frame to the right
        for (i in 0..10) {
            estimator.update(0.3f + i * 0.05f, 0.5f)
        }
        assertTrue("velocityX should be positive", estimator.velocityX > 0.04f)
        assertEquals(0f, estimator.velocityY, 0.005f)
    }

    @Test
    fun `constant downward motion produces positive velocityY`() {
        for (i in 0..10) {
            estimator.update(0.5f, 0.3f + i * 0.03f)
        }
        assertEquals(0f, estimator.velocityX, 0.005f)
        assertTrue("velocityY should be positive", estimator.velocityY > 0.025f)
    }

    @Test
    fun `diagonal motion produces velocity in both axes`() {
        for (i in 0..10) {
            estimator.update(0.3f + i * 0.04f, 0.3f + i * 0.03f)
        }
        assertTrue("velocityX should be positive", estimator.velocityX > 0.03f)
        assertTrue("velocityY should be positive", estimator.velocityY > 0.02f)
        assertTrue("speed should reflect both", estimator.speed > 0.04f)
    }

    @Test
    fun `stationary object has near-zero velocity`() {
        for (i in 0..20) {
            estimator.update(0.5f, 0.5f)
        }
        assertEquals(0f, estimator.velocityX, 0.001f)
        assertEquals(0f, estimator.velocityY, 0.001f)
        assertFalse(estimator.isHighVelocity())
    }

    @Test
    fun `high velocity detected for fast motion`() {
        // 0.05 per frame > HIGH_VELOCITY_THRESHOLD (0.03)
        for (i in 0..10) {
            estimator.update(0.2f + i * 0.05f, 0.5f)
        }
        assertTrue(estimator.isHighVelocity())
    }

    @Test
    fun `very high velocity detected for very fast motion`() {
        // 0.08 per frame > VERY_HIGH_VELOCITY_THRESHOLD (0.06)
        for (i in 0..10) {
            estimator.update(0.1f + i * 0.08f, 0.5f)
        }
        assertTrue(estimator.isVeryHighVelocity())
    }

    @Test
    fun `slow motion does not trigger high velocity`() {
        // 0.01 per frame < HIGH_VELOCITY_THRESHOLD (0.03)
        for (i in 0..10) {
            estimator.update(0.4f + i * 0.01f, 0.5f)
        }
        assertFalse(estimator.isHighVelocity())
    }

    @Test
    fun `predictPosition extrapolates linearly`() {
        // Establish rightward velocity
        for (i in 0..10) {
            estimator.update(0.3f + i * 0.05f, 0.5f)
        }
        val predicted = estimator.predictPosition(3)
        // Last position: 0.3 + 10*0.05 = 0.8
        // Velocity ~0.05, predict 3 ahead: ~0.8 + 0.15 = ~0.95
        assertTrue("Predicted X should be ahead of last position", predicted.x > 0.85f)
        assertEquals(0.5f, predicted.y, 0.05f)
    }

    @Test
    fun `predictPosition clamps to 0-1 range`() {
        // Move toward right edge
        for (i in 0..10) {
            estimator.update(0.7f + i * 0.05f, 0.5f)
        }
        val predicted = estimator.predictPosition(10)
        assertEquals(1f, predicted.x, 0.001f) // clamped
    }

    @Test
    fun `predictBox shifts box by velocity`() {
        for (i in 0..10) {
            estimator.update(0.3f + i * 0.04f, 0.5f)
        }
        val box = RectF(0.6f, 0.4f, 0.8f, 0.6f)
        val predicted = estimator.predictBox(box, 2)
        // ~0.04 velocity * 2 frames = ~0.08 shift
        assertTrue("Predicted box left should shift right", predicted.left > box.left)
        assertTrue("Width should be preserved",
            kotlin.math.abs(predicted.width() - box.width()) < 0.02f)
    }

    @Test
    fun `reset clears all state`() {
        for (i in 0..5) {
            estimator.update(0.3f + i * 0.05f, 0.5f)
        }
        assertTrue(estimator.speed > 0f)

        estimator.reset()
        assertEquals(0f, estimator.velocityX, 0.001f)
        assertEquals(0f, estimator.velocityY, 0.001f)
        assertEquals(0f, estimator.speed, 0.001f)
        assertFalse(estimator.isHighVelocity())
    }

    @Test
    fun `smoothing dampens sudden direction changes`() {
        // Move right for several frames
        for (i in 0..5) {
            estimator.update(0.3f + i * 0.05f, 0.5f)
        }
        val rightVelocity = estimator.velocityX
        assertTrue(rightVelocity > 0)

        // Sudden stop
        val lastX = 0.3f + 5 * 0.05f
        estimator.update(lastX, 0.5f)

        // Velocity should decrease but not instantly go to zero (smoothing)
        assertTrue("Velocity should decrease after stop", estimator.velocityX < rightVelocity)
        assertTrue("Velocity should not instantly zero out", estimator.velocityX > 0f)
    }

    @Test
    fun `velocity reversal tracked correctly`() {
        // Move right
        for (i in 0..5) {
            estimator.update(0.3f + i * 0.05f, 0.5f)
        }
        assertTrue(estimator.velocityX > 0)

        // Move left
        val lastX = 0.3f + 5 * 0.05f
        for (i in 1..10) {
            estimator.update(lastX - i * 0.05f, 0.5f)
        }
        assertTrue("Velocity should reverse to negative", estimator.velocityX < 0)
    }
}
