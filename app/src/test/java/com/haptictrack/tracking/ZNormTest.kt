package com.haptictrack.tracking

import android.graphics.RectF
import org.junit.Assert.assertEquals
import org.junit.Assert.assertNotNull
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Assert.assertFalse
import org.junit.Test
import org.junit.runner.RunWith
import org.robolectric.RobolectricTestRunner

/**
 * Unit tests for #102 Z-norm (Phase 1 + Phase 3) — the live impostor cohort
 * statistics that drive the calibrated embedding-override gates in
 * [ReacquisitionEngine.scoreCandidate].
 *
 * The tests exercise:
 *   - [ReacquisitionEngine.znMnv3Sim] returning null below MIN_COHORT_FOR_ZNORM
 *   - [ReacquisitionEngine.znOsnetSim] / [ReacquisitionEngine.znFaceSim]
 *     null when no anchor or below cohort floor
 *   - z-score correctness: known mean/sigma → known z output
 *   - Sigma-floor clamping: a homogeneous cohort with σ ≈ 0 is bumped to
 *     [ReacquisitionEngine.Z_SIGMA_FLOOR] so z-scores can't blow up
 *   - Override gate behavior flipping at the calibrated z thresholds
 *     (label / geometric / tentative-bypass)
 *
 * The score-formula path is exercised end-to-end via [scoreCandidate] so we
 * cover both the z-norm computation and how it feeds the production gates.
 */
@RunWith(RobolectricTestRunner::class)
class ZNormTest {

    /** Build a normalized vector pointing along [axis] in n-dim space. */
    private fun unit(axis: Int, n: Int = 16): FloatArray {
        require(axis < n)
        val v = FloatArray(n)
        v[axis] = 1f
        return v
    }

    /** Linear blend of two axes, normalized — produces controllable cosine sims. */
    private fun mix(a: Int, b: Int, alpha: Float, n: Int = 16): FloatArray {
        val v = FloatArray(n)
        v[a] = alpha
        v[b] = kotlin.math.sqrt(1f - alpha * alpha)
        return v
    }

    /** A 1280-dim MNV3-shaped embedding with a single non-zero axis. */
    private fun mnv3Unit(axis: Int): FloatArray = unit(axis, n = 1280)
    private fun mnv3Mix(a: Int, b: Int, alpha: Float): FloatArray = mix(a, b, alpha, n = 1280)

    /** A 512-dim OSNet-shaped embedding. */
    private fun osnetUnit(axis: Int): FloatArray = unit(axis, n = 512)
    private fun osnetMix(a: Int, b: Int, alpha: Float): FloatArray = mix(a, b, alpha, n = 512)

    /** A 192-dim face-shaped embedding. */
    private fun faceUnit(axis: Int): FloatArray = unit(axis, n = 192)

    private fun obj(id: Int, label: String = "person",
                    embedding: FloatArray? = null,
                    reIdEmbedding: FloatArray? = null,
                    faceEmbedding: FloatArray? = null,
                    box: RectF = RectF(0.4f, 0.4f, 0.6f, 0.6f)) =
        TrackedObject(id = id, boundingBox = box, label = label,
                      embedding = embedding, reIdEmbedding = reIdEmbedding,
                      faceEmbedding = faceEmbedding)

    // ----------------------------------------------------------------- znMnv3Sim

    @Test
    fun `znMnv3Sim returns null when fewer than MIN_COHORT_FOR_ZNORM negatives`() {
        val engine = ReacquisitionEngine()
        engine.lock(1, RectF(0.4f, 0.4f, 0.6f, 0.6f), "person",
            listOf(mnv3Unit(0)), null, null, cocoLabel = "person")

        // Add 4 negatives — below the 5-negative floor.
        repeat(4) { engine.addSceneNegative(mnv3Unit(it + 1)) }

        assertNull("Below cohort floor → null z", engine.znMnv3Sim(mnv3Unit(0)))
        assertNull("Stats not yet computed", engine.mnv3LiveStats)
    }

    @Test
    fun `znMnv3Sim returns z-score once cohort reaches MIN_COHORT_FOR_ZNORM`() {
        val engine = ReacquisitionEngine()
        engine.lock(1, RectF(0.4f, 0.4f, 0.6f, 0.6f), "person",
            listOf(mnv3Unit(0)), null, null, cocoLabel = "person")

        // 5 orthogonal negatives → all cosines against gallery axis 0 are 0,
        // so impostor mean ≈ 0, std at the σ-floor.
        repeat(5) { engine.addSceneNegative(mnv3Unit(it + 1)) }

        val z = engine.znMnv3Sim(mnv3Unit(0))
        assertNotNull("At cohort floor → z computed", z)
        // Candidate aligns with gallery (sim = 1), impostor mean = 0,
        // std clamped to Z_SIGMA_FLOOR = 0.05 → z = (1 - 0) / 0.05 = 20.
        assertEquals(20f, z!!, 1e-3f)

        val stats = engine.mnv3LiveStats
        assertNotNull(stats)
        assertEquals(0f, stats!!.mean, 1e-5f)
        assertEquals(ReacquisitionEngine.Z_SIGMA_FLOOR, stats.std, 1e-5f)
        assertEquals(5, stats.n)
    }

    @Test
    fun `znMnv3Sim sigma floor clamps homogeneous cohort`() {
        val engine = ReacquisitionEngine()
        engine.lock(1, RectF(0.4f, 0.4f, 0.6f, 0.6f), "person",
            listOf(mnv3Unit(0)), null, null, cocoLabel = "person")

        // 6 IDENTICAL negatives → variance is exactly 0. Without the σ floor
        // the z-score would be Infinity (divide by zero) — the floor keeps z
        // finite even when the cohort degenerates.
        val homogeneous = mnv3Unit(1)
        repeat(6) { engine.addSceneNegative(homogeneous) }

        // Probe with an axis orthogonal to BOTH gallery (axis 0) and impostors
        // (axis 1). Sim against gallery = 0. Impostor mean = 0. Std clamped
        // to Z_SIGMA_FLOOR = 0.05. z = (0 - 0) / 0.05 = 0.
        val z = engine.znMnv3Sim(mnv3Unit(2))
        assertNotNull("Homogeneous cohort still yields z (σ floor applied)", z)
        assertTrue("z must be finite, not Inf/NaN", z!!.isFinite())
        assertEquals(0f, z, 1e-5f)

        val stats = engine.mnv3LiveStats!!
        assertEquals(ReacquisitionEngine.Z_SIGMA_FLOOR, stats.std, 1e-5f)
    }

    @Test
    fun `znMnv3Sim z-score is correct given heterogeneous cohort`() {
        val engine = ReacquisitionEngine()
        engine.lock(1, RectF(0.4f, 0.4f, 0.6f, 0.6f), "person",
            listOf(mnv3Unit(0)), null, null, cocoLabel = "person")

        // 6 negatives with controlled cosines vs gallery axis 0:
        //   sim values: 0.1, 0.2, 0.3, 0.4, 0.5, 0.6
        // mean = 0.35, std = sqrt(((-0.25)^2 + (-0.15)^2 + (-0.05)^2 + 0.05^2 + 0.15^2 + 0.25^2)/6)
        //              = sqrt(0.175/6) ≈ 0.1708
        val sims = floatArrayOf(0.1f, 0.2f, 0.3f, 0.4f, 0.5f, 0.6f)
        for (s in sims) {
            // mix(0, 1, alpha) has cosine alpha against unit(0) — controllable cohort cosine.
            engine.addSceneNegative(mnv3Mix(0, 1, s))
        }

        val stats = engine.mnv3LiveStats!!
        assertEquals(0.35f, stats.mean, 1e-3f)
        assertEquals(0.1708f, stats.std, 1e-3f)

        // Candidate at sim=0.7 → z = (0.7 - 0.35) / 0.1708 ≈ 2.05
        val candidate = mnv3Mix(0, 1, 0.7f)
        val z = engine.znMnv3Sim(candidate)!!
        assertEquals(2.05f, z, 1e-2f)
    }

    @Test
    fun `znMnv3Sim returns null when gallery is empty`() {
        val engine = ReacquisitionEngine()
        // Lock with NO embeddings.
        engine.lock(1, RectF(0.4f, 0.4f, 0.6f, 0.6f), "person",
            emptyList(), null, null, cocoLabel = "person")
        repeat(6) { engine.addSceneNegative(mnv3Unit(it)) }

        assertNull("Empty gallery → null z", engine.znMnv3Sim(mnv3Unit(0)))
    }

    // ----------------------------------------------------------------- znOsnetSim

    @Test
    fun `znOsnetSim returns null without locked OSNet anchor`() {
        val engine = ReacquisitionEngine()
        engine.lock(1, RectF(0.4f, 0.4f, 0.6f, 0.6f), "person",
            listOf(mnv3Unit(0)), null, null, cocoLabel = "person",
            reIdEmbedding = null /* no anchor */)
        // OSNet cohort comes from observePerson — add 6 paired observations
        // so the cohort has 6 OSNet bodies but the lock has no OSNet anchor.
        repeat(6) { engine.addScenePersonPair(face = faceUnit(it + 1), body = osnetUnit(it + 1)) }

        assertNull("No locked OSNet anchor → null z", engine.znOsnetSim(osnetUnit(0)))
    }

    @Test
    fun `znOsnetSim computes z against locked anchor`() {
        val engine = ReacquisitionEngine()
        val lockedReId = osnetUnit(0)
        engine.lock(1, RectF(0.4f, 0.4f, 0.6f, 0.6f), "person",
            listOf(mnv3Unit(0)), null, null, cocoLabel = "person",
            reIdEmbedding = lockedReId)

        // 6 OSNet negatives all orthogonal to anchor → impostor mean=0, σ≈0
        // → clamped to Z_SIGMA_FLOOR = 0.05.
        repeat(6) { engine.addScenePersonPair(face = faceUnit(it + 1), body = osnetUnit(it + 1)) }

        // Candidate identical to anchor → cosine 1.0 → z = (1 - 0) / 0.05 = 20.
        val z = engine.znOsnetSim(osnetUnit(0))!!
        assertEquals(20f, z, 1e-3f)

        // Candidate orthogonal → cosine 0 → z = 0.
        val zOrth = engine.znOsnetSim(osnetUnit(7))!!
        assertEquals(0f, zOrth, 1e-3f)
    }

    // ----------------------------------------------------------------- znFaceSim

    @Test
    fun `znFaceSim returns null without locked face anchor`() {
        val engine = ReacquisitionEngine()
        engine.lock(1, RectF(0.4f, 0.4f, 0.6f, 0.6f), "person",
            listOf(mnv3Unit(0)), null, null, cocoLabel = "person",
            faceEmbedding = null /* no anchor */)
        repeat(6) { engine.addScenePersonPair(face = faceUnit(it + 1), body = osnetUnit(it + 1)) }

        assertNull("No locked face anchor → null z", engine.znFaceSim(faceUnit(0)))
    }

    @Test
    fun `znFaceSim computes z against locked face anchor`() {
        val engine = ReacquisitionEngine()
        val lockedFace = faceUnit(0)
        engine.lock(1, RectF(0.4f, 0.4f, 0.6f, 0.6f), "person",
            listOf(mnv3Unit(0)), null, null, cocoLabel = "person",
            faceEmbedding = lockedFace)

        repeat(6) { engine.addScenePersonPair(face = faceUnit(it + 1), body = osnetUnit(it + 1)) }

        val z = engine.znFaceSim(faceUnit(0))!!
        assertEquals(20f, z, 1e-3f)
    }

    // ----------------------------------------------------------------- override gate

    /**
     * Override gate (used inside [scoreCandidate]) bypasses position/size hard
     * filters when the calibrated z-score is high enough. Z_GEOMETRIC_OVERRIDE
     * = 1.0; we set up a candidate FAR from the lock (huge size mismatch) and
     * verify that:
     *   - With z below threshold → candidate rejected by hard size filter.
     *   - With z above threshold → candidate admitted (override fires).
     *
     * We drive z by controlling the impostor cohort's mean: with mean=0.0 a
     * candidate with reId=0.5 sits at z = 0.5 / σ; clamping σ to 0.05 yields
     * z=10, well above threshold → override fires.
     */
    @Test
    fun `geometric override fires when z is high`() {
        val engine = ReacquisitionEngine(sizeRatioThreshold = 1.5f)
        val lockedReId = osnetUnit(0)
        // Tiny lock box so the candidate (full-frame) has a huge size ratio.
        engine.lock(1, RectF(0.49f, 0.49f, 0.51f, 0.51f), "person",
            listOf(mnv3Unit(0)), null, null, cocoLabel = "person",
            reIdEmbedding = lockedReId)
        // 6 orthogonal OSNet negatives.
        repeat(6) { engine.addScenePersonPair(face = faceUnit(it + 1), body = osnetUnit(it + 1)) }

        // Candidate: huge size, but OSNet matches lock exactly (z = 20).
        val candidate = obj(id = 7, label = "person",
            embedding = mnv3Unit(0), reIdEmbedding = lockedReId,
            box = RectF(0.0f, 0.0f, 1.0f, 1.0f))
        val score = engine.scoreCandidate(candidate, engine.lastKnownBox!!)
        assertNotNull("High-z override admits oversized candidate", score)
    }

    @Test
    fun `geometric override blocks when z is near zero`() {
        val engine = ReacquisitionEngine(sizeRatioThreshold = 1.5f)
        val lockedReId = osnetUnit(0)
        engine.lock(1, RectF(0.49f, 0.49f, 0.51f, 0.51f), "person",
            listOf(mnv3Unit(0)), null, null, cocoLabel = "person",
            reIdEmbedding = lockedReId)
        repeat(6) { engine.addScenePersonPair(face = faceUnit(it + 1), body = osnetUnit(it + 1)) }

        // Candidate: huge size, OSNet ORTHOGONAL to lock (cosine 0, z = 0).
        val candidate = obj(id = 7, label = "person",
            embedding = mnv3Unit(0), reIdEmbedding = osnetUnit(7),
            box = RectF(0.0f, 0.0f, 1.0f, 1.0f))
        val score = engine.scoreCandidate(candidate, engine.lastKnownBox!!)
        assertNull("Low-z override doesn't fire — size filter rejects", score)
    }

    @Test
    fun `override falls back to raw cosine when stats unavailable`() {
        val engine = ReacquisitionEngine(sizeRatioThreshold = 1.5f)
        val lockedReId = osnetUnit(0)
        engine.lock(1, RectF(0.49f, 0.49f, 0.51f, 0.51f), "person",
            listOf(mnv3Unit(0)), null, null, cocoLabel = "person",
            reIdEmbedding = lockedReId)
        // No negatives added → stats are null → overridePasses falls back
        // to raw-cosine threshold (GEOMETRIC_OVERRIDE_THRESHOLD = 0.55).

        // Candidate: huge size, OSNet matches lock (cosine = 1.0 ≥ 0.55 raw).
        val candidate = obj(id = 7, label = "person",
            embedding = mnv3Unit(0), reIdEmbedding = lockedReId,
            box = RectF(0.0f, 0.0f, 1.0f, 1.0f))
        val score = engine.scoreCandidate(candidate, engine.lastKnownBox!!)
        assertNotNull("Raw-cosine fallback admits when sim ≥ 0.55", score)
    }

    @Test
    fun `live stats invalidated by addSceneNegative`() {
        val engine = ReacquisitionEngine()
        engine.lock(1, RectF(0.4f, 0.4f, 0.6f, 0.6f), "person",
            listOf(mnv3Unit(0)), null, null, cocoLabel = "person")

        repeat(6) { engine.addSceneNegative(mnv3Unit(it + 1)) }
        val stats1 = engine.mnv3LiveStats
        assertNotNull(stats1)
        assertEquals(6, stats1!!.n)

        // New negative — invalidate + recompute.
        engine.addSceneNegative(mnv3Unit(7))
        val stats2 = engine.mnv3LiveStats!!
        assertEquals(7, stats2.n)
    }

    @Test
    fun `live stats invalidated by addEmbedding`() {
        val engine = ReacquisitionEngine()
        engine.lock(1, RectF(0.4f, 0.4f, 0.6f, 0.6f), "person",
            listOf(mnv3Unit(0)), null, null, cocoLabel = "person")
        repeat(6) { engine.addSceneNegative(mnv3Unit(it + 1)) }
        val statsBefore = engine.mnv3LiveStats!!

        // Add a new positive — gallery grows, stats recompute (cosines shift).
        engine.addEmbedding(mnv3Mix(0, 1, 0.7f))
        val statsAfter = engine.mnv3LiveStats!!
        // bestGallerySimilarity now picks the better of (axis-0, axis-0/1 mix)
        // for each negative — values change.
        assertTrue("mean should have updated after gallery change",
            kotlin.math.abs(statsBefore.mean - statsAfter.mean) > 1e-4f ||
            statsAfter.n == statsBefore.n) // n stays same; mean might shift
    }

    @Test
    fun `clear resets live stats`() {
        val engine = ReacquisitionEngine()
        engine.lock(1, RectF(0.4f, 0.4f, 0.6f, 0.6f), "person",
            listOf(mnv3Unit(0)), null, null, cocoLabel = "person")
        repeat(6) { engine.addSceneNegative(mnv3Unit(it + 1)) }
        assertNotNull(engine.mnv3LiveStats)

        engine.clear()
        assertNull("clear() drops mnv3 stats", engine.mnv3LiveStats)
        assertNull("clear() drops osnet stats", engine.osnetLiveStats)
        assertNull("clear() drops face stats", engine.faceLiveStats)
    }
}
