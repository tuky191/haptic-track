package com.haptictrack.tracking

import org.junit.Assert.assertEquals
import org.junit.Assert.assertNotNull
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Test

class CuePhraseTest {
    @Test fun `none has no phrase`() { assertNull(cuePhrase(Cue.NONE)) }
    @Test fun `every other cue has a non-blank phrase`() {
        for (c in Cue.entries) if (c != Cue.NONE) {
            val p = cuePhrase(c)
            assertNotNull("missing phrase for $c", p)
            assertTrue("blank phrase for $c", p!!.isNotBlank())
        }
    }
    @Test fun `move left says left`() { assertEquals("move left", cuePhrase(Cue.MOVE_LEFT)) }
}
