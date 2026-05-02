package com.haptictrack.camera

import org.junit.Assert.*
import org.junit.Test

class GyroStabilizerTest {

    private fun Q(w: Double, x: Double, y: Double, z: Double) = GyroStabilizer.Quat(w, x, y, z)

    @Test
    fun `identity quaternion has unit norm`() {
        val q = Q(1.0, 0.0, 0.0, 0.0)
        assertEquals(1.0, q.norm(), 1e-10)
    }

    @Test
    fun `quaternion multiplication identity`() {
        val a = Q(0.7071, 0.7071, 0.0, 0.0).normalized()
        val id = Q(1.0, 0.0, 0.0, 0.0)
        val result = a * id
        assertEquals(a.w, result.w, 1e-6)
        assertEquals(a.x, result.x, 1e-6)
    }

    @Test
    fun `q times conjugate gives identity`() {
        val q = Q(0.5, 0.5, 0.5, 0.5).normalized()
        val result = q * q.conjugate()
        assertEquals(1.0, result.w, 1e-6)
        assertEquals(0.0, result.x, 1e-6)
        assertEquals(0.0, result.y, 1e-6)
        assertEquals(0.0, result.z, 1e-6)
    }

    @Test
    fun `slerp at t=0 returns first quaternion`() {
        val a = Q(1.0, 0.0, 0.0, 0.0)
        val b = Q(0.7071, 0.7071, 0.0, 0.0).normalized()
        val result = slerp(a, b, 0.0)
        assertEquals(a.w, result.w, 1e-6)
        assertEquals(a.x, result.x, 1e-6)
    }

    @Test
    fun `slerp at t=1 returns second quaternion`() {
        val a = Q(1.0, 0.0, 0.0, 0.0)
        val b = Q(0.7071, 0.7071, 0.0, 0.0).normalized()
        val result = slerp(a, b, 1.0)
        assertEquals(b.w, result.w, 1e-4)
        assertEquals(b.x, result.x, 1e-4)
    }

    @Test
    fun `identity quaternion produces identity rotation matrix`() {
        val q = Q(1.0, 0.0, 0.0, 0.0)
        val m = q.toRotationMatrix()
        assertEquals(1.0, m[0], 1e-10) // m[0][0]
        assertEquals(0.0, m[1], 1e-10) // m[0][1]
        assertEquals(0.0, m[3], 1e-10) // m[1][0]
        assertEquals(1.0, m[4], 1e-10) // m[1][1]
        assertEquals(1.0, m[8], 1e-10) // m[2][2]
    }

    @Test
    fun `90-degree rotation around Z axis`() {
        // q = cos(45°) + sin(45°)*k = (0.7071, 0, 0, 0.7071)
        val q = Q(0.7071067811865476, 0.0, 0.0, 0.7071067811865476)
        val m = q.toRotationMatrix()
        // Should be [[0,-1,0],[1,0,0],[0,0,1]]
        assertEquals(0.0, m[0], 1e-6)
        assertEquals(-1.0, m[1], 1e-6)
        assertEquals(1.0, m[3], 1e-6)
        assertEquals(0.0, m[4], 1e-6)
        assertEquals(1.0, m[8], 1e-6)
    }

    @Test
    fun `correction of identity raw and smoothed gives identity matrix`() {
        val raw = Q(0.5, 0.5, 0.5, 0.5).normalized()
        val smoothed = raw // same orientation — no correction needed
        val correction = raw * smoothed.conjugate()
        val m = correction.toRotationMatrix()
        assertEquals(1.0, m[0], 1e-6)
        assertEquals(0.0, m[1], 1e-6)
        assertEquals(1.0, m[4], 1e-6)
    }

    private fun slerp(a: GyroStabilizer.Quat, b: GyroStabilizer.Quat, t: Double): GyroStabilizer.Quat {
        var dot = a.dot(b)
        val b2 = if (dot < 0) { dot = -dot; GyroStabilizer.Quat(-b.w, -b.x, -b.y, -b.z) } else b
        return if (dot > 0.9995) {
            GyroStabilizer.Quat(a.w + t * (b2.w - a.w), a.x + t * (b2.x - a.x),
              a.y + t * (b2.y - a.y), a.z + t * (b2.z - a.z)).normalized()
        } else {
            val theta = kotlin.math.acos(dot.coerceIn(-1.0, 1.0))
            val sinTheta = kotlin.math.sin(theta)
            val wa = kotlin.math.sin((1 - t) * theta) / sinTheta
            val wb = kotlin.math.sin(t * theta) / sinTheta
            GyroStabilizer.Quat(wa * a.w + wb * b2.w, wa * a.x + wb * b2.x,
              wa * a.y + wb * b2.y, wa * a.z + wb * b2.z).normalized()
        }
    }
}
