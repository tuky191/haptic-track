package com.haptictrack.tracking

import org.junit.Assert.*
import org.junit.Test
import org.junit.runner.RunWith
import org.robolectric.RobolectricTestRunner

@RunWith(RobolectricTestRunner::class)
class PersonAttributesTest {

    // --- PersonAttributes.similarity ---

    @Test
    fun `identical attributes have high similarity`() {
        val a = attrs(isMale = true, hasBackpack = true, hasLongPants = true,
            upperColor = "red", lowerColor = "blue")
        val sim = a.similarity(a)
        assertEquals(1f, sim, 0.01f)
    }

    @Test
    fun `identical attributes without color have neutral color component`() {
        // attrSim=1.0 * 0.6 + colorSim=0.5(neutral) * 0.4 = 0.8
        val a = attrs(isMale = true, hasBackpack = true, hasLongPants = true)
        val sim = a.similarity(a)
        assertEquals(0.8f, sim, 0.01f)
    }

    @Test
    fun `completely opposite attributes have low similarity`() {
        val a = attrs(isMale = true, hasBackpack = true, hasLongPants = true,
            hasLongSleeves = true, hasHat = true, hasLongHair = false, hasBag = false, hasCoatJacket = true)
        val b = attrs(isMale = false, hasBackpack = false, hasLongPants = false,
            hasLongSleeves = false, hasHat = false, hasLongHair = true, hasBag = true, hasCoatJacket = false)
        val sim = a.similarity(b)
        assertTrue("Opposite attributes should have low similarity: $sim", sim < 0.4f)
    }

    @Test
    fun `same gender different accessories has moderate similarity`() {
        val a = attrs(isMale = true, hasBackpack = true, hasHat = true)
        val b = attrs(isMale = true, hasBackpack = false, hasHat = false)
        val sim = a.similarity(b)
        assertTrue("Same gender, different accessories: $sim", sim > 0.4f && sim < 0.9f)
    }

    @Test
    fun `color match boosts similarity`() {
        val a = attrs(isMale = true, upperColor = "red", lowerColor = "blue")
        val b = attrs(isMale = true, upperColor = "red", lowerColor = "blue")
        val c = attrs(isMale = true, upperColor = "green", lowerColor = "black")
        val simMatch = a.similarity(b)
        val simMismatch = a.similarity(c)
        assertTrue("Color match ($simMatch) should beat mismatch ($simMismatch)",
            simMatch > simMismatch)
    }

    @Test
    fun `similarity works without raw probabilities`() {
        val a = attrs(isMale = true, hasBackpack = true, hasLongPants = true, rawProbs = null)
        val b = attrs(isMale = true, hasBackpack = true, hasLongPants = true, rawProbs = null)
        val sim = a.similarity(b)
        assertEquals("Hard comparison: all match", 1f, sim, 0.01f)
    }

    @Test
    fun `hard comparison counts mismatches`() {
        val a = attrs(isMale = true, hasBackpack = true, rawProbs = null)
        val b = attrs(isMale = false, hasBackpack = false, rawProbs = null)
        val sim = a.similarity(b)
        // 6 matches out of 8 (only isMale and hasBackpack differ)
        assertEquals(6f / 8f, sim, 0.01f)
    }

    // --- PersonAttributes.summary ---

    @Test
    fun `summary includes gender and clothing`() {
        val a = attrs(isMale = true, hasBackpack = true, hasLongSleeves = true,
            hasLongPants = true, upperColor = "red", lowerColor = "blue")
        val summary = a.summary()
        assertTrue("Should contain 'man'", summary.contains("man"))
        assertTrue("Should contain 'backpack'", summary.contains("backpack"))
        assertTrue("Should contain 'long sleeves'", summary.contains("long sleeves"))
        assertTrue("Should contain 'red'", summary.contains("red"))
    }

    @Test
    fun `summary for woman with hat`() {
        val a = attrs(isMale = false, hasHat = true)
        val summary = a.summary()
        assertTrue("Should contain 'woman'", summary.contains("woman"))
        assertTrue("Should contain 'hat'", summary.contains("hat"))
    }

    // --- quantizeColor ---

    @Test
    fun `quantize black`() {
        assertEquals("black", quantizeColor(0f, 0f, 0.05f))
    }

    @Test
    fun `quantize white`() {
        assertEquals("white", quantizeColor(0f, 0.05f, 0.95f))
    }

    @Test
    fun `quantize gray`() {
        assertEquals("gray", quantizeColor(0f, 0.05f, 0.5f))
    }

    @Test
    fun `quantize red`() {
        assertEquals("red", quantizeColor(0f, 0.8f, 0.8f))
        assertEquals("red", quantizeColor(355f, 0.8f, 0.8f))
    }

    @Test
    fun `quantize blue`() {
        assertEquals("blue", quantizeColor(220f, 0.8f, 0.8f))
    }

    @Test
    fun `quantize green`() {
        assertEquals("green", quantizeColor(120f, 0.8f, 0.8f))
    }

    @Test
    fun `quantize yellow`() {
        assertEquals("yellow", quantizeColor(55f, 0.8f, 0.8f))
    }

    @Test
    fun `quantize orange`() {
        assertEquals("orange", quantizeColor(25f, 0.8f, 0.8f))
    }

    @Test
    fun `quantize purple`() {
        assertEquals("purple", quantizeColor(275f, 0.8f, 0.8f))
    }

    @Test
    fun `quantize pink`() {
        assertEquals("pink", quantizeColor(330f, 0.8f, 0.8f))
    }

    // --- Helpers ---

    private fun attrs(
        isMale: Boolean = false,
        hasBag: Boolean = false,
        hasBackpack: Boolean = false,
        hasHat: Boolean = false,
        hasLongSleeves: Boolean = false,
        hasLongPants: Boolean = false,
        hasLongHair: Boolean = false,
        hasCoatJacket: Boolean = false,
        upperColor: String? = null,
        lowerColor: String? = null,
        rawProbs: FloatArray? = floatArrayOf(
            if (isMale) 0.9f else 0.1f,
            if (hasBag) 0.9f else 0.1f,
            if (hasBackpack) 0.9f else 0.1f,
            if (hasHat) 0.9f else 0.1f,
            if (hasLongSleeves) 0.9f else 0.1f,
            if (hasLongPants) 0.9f else 0.1f,
            if (hasLongHair) 0.9f else 0.1f,
            if (hasCoatJacket) 0.9f else 0.1f
        )
    ) = PersonAttributes(
        isMale = isMale,
        hasBag = hasBag,
        hasBackpack = hasBackpack,
        hasHat = hasHat,
        hasLongSleeves = hasLongSleeves,
        hasLongPants = hasLongPants,
        hasLongHair = hasLongHair,
        hasCoatJacket = hasCoatJacket,
        upperColor = upperColor,
        lowerColor = lowerColor,
        rawProbabilities = rawProbs
    )
}
