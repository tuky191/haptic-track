package com.haptictrack.tracking

import org.junit.Assert.assertEquals
import org.junit.Test

class GuidanceTypesTest {
    @Test fun `framing target cycles and wraps`() {
        assertEquals(FramingTarget.UPPER_BODY, FramingTarget.FULL_BODY.next())
        assertEquals(FramingTarget.FACE_HEAD, FramingTarget.UPPER_BODY.next())
        assertEquals(FramingTarget.FULL_BODY, FramingTarget.FACE_HEAD.next())
    }

    @Test fun `guidance mode cycles through all four and wraps`() {
        assertEquals(GuidanceMode.HAPTIC, GuidanceMode.OFF.next())
        assertEquals(GuidanceMode.VOICE, GuidanceMode.HAPTIC.next())
        assertEquals(GuidanceMode.BOTH, GuidanceMode.VOICE.next())
        assertEquals(GuidanceMode.OFF, GuidanceMode.BOTH.next())
    }
}
