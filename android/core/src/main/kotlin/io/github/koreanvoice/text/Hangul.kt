package io.github.koreanvoice.text

/**
 * Hangul syllable <-> conjoining jamo helpers (equivalent to the `jamo` Python package
 * functions `h2j`, `j2h` and `hangul_to_jamo` used by g2pkk and MeloTTS).
 */
object Hangul {
    const val SBASE = 0xAC00
    const val LBASE = 0x1100
    const val VBASE = 0x1161
    const val TBASE = 0x11A7
    const val LCOUNT = 19
    const val VCOUNT = 21
    const val TCOUNT = 28
    const val NCOUNT = VCOUNT * TCOUNT
    const val SCOUNT = LCOUNT * NCOUNT

    fun isSyllable(c: Char): Boolean = c.code in SBASE until SBASE + SCOUNT
    fun isLead(c: Char): Boolean = c.code in 0x1100..0x1112
    fun isVowel(c: Char): Boolean = c.code in 0x1161..0x1175
    fun isTail(c: Char): Boolean = c.code in 0x11A8..0x11C2

    /** Decompose every Hangul syllable into conjoining jamo; everything else is kept as is. */
    fun h2j(s: CharSequence): String {
        val sb = StringBuilder(s.length * 3)
        for (c in s) appendJamo(sb, c)
        return sb.toString()
    }

    fun appendJamo(sb: StringBuilder, c: Char) {
        if (!isSyllable(c)) {
            sb.append(c); return
        }
        val idx = c.code - SBASE
        sb.append((LBASE + idx / NCOUNT).toChar())
        sb.append((VBASE + (idx % NCOUNT) / TCOUNT).toChar())
        val t = idx % TCOUNT
        if (t != 0) sb.append((TBASE + t).toChar())
    }

    fun compose(lead: Char, vowel: Char, tail: Char? = null): Char {
        val l = lead.code - LBASE
        val v = vowel.code - VBASE
        val t = if (tail == null) 0 else tail.code - TBASE
        return (SBASE + (l * VCOUNT + v) * TCOUNT + t).toChar()
    }

    /** Final consonant (conjoining tail jamo) of a syllable, or null. */
    fun tailOf(c: Char): Char? {
        if (!isSyllable(c)) return null
        val t = (c.code - SBASE) % TCOUNT
        return if (t == 0) null else (TBASE + t).toChar()
    }

    /** Initial consonant (conjoining lead jamo) of a syllable, or null. */
    fun leadOf(c: Char): Char? = if (!isSyllable(c)) null else (LBASE + (c.code - SBASE) / NCOUNT).toChar()

    /** Vowel index 0..20 of a syllable, or -1. */
    fun vowelIndex(c: Char): Int = if (!isSyllable(c)) -1 else ((c.code - SBASE) % NCOUNT) / TCOUNT

    /** Replace the tail of [c] with [tail] (null removes it). */
    fun withTail(c: Char, tail: Char?): Char {
        val base = c.code - (c.code - SBASE) % TCOUNT
        return (base + if (tail == null) 0 else tail.code - TBASE).toChar()
    }

    fun withLead(c: Char, lead: Char): Char {
        val idx = c.code - SBASE
        val rest = idx % NCOUNT
        return (SBASE + (lead.code - LBASE) * NCOUNT + rest).toChar()
    }

    fun hasTail(c: Char): Boolean = tailOf(c) != null

    private val COMPOSE_CVC = Regex("[ᄀ-ᄒ][ᅡ-ᅵ][ᆨ-ᇂ]")
    private val COMPOSE_CV = Regex("[ᄀ-ᄒ][ᅡ-ᅵ]")
    private val PLACEHOLDER = Regex("(^|[^ᄀ-ᄒ])([ᅡ-ᅵ])")

    /** g2pkk `utils.compose`: insert ᄋ before bare vowels, then assemble syllables. */
    fun composeLetters(letters: String): String {
        var s = PLACEHOLDER.replace(letters, "$1ᄋ$2")
        s = COMPOSE_CVC.replace(s) { m -> compose(m.value[0], m.value[1], m.value[2]).toString() }
        s = COMPOSE_CV.replace(s) { m -> compose(m.value[0], m.value[1]).toString() }
        return s
    }

    /** True if the whole string consists of complete Hangul syllables. */
    fun isAllSyllables(s: String): Boolean = s.isNotEmpty() && s.all { isSyllable(it) }
}
