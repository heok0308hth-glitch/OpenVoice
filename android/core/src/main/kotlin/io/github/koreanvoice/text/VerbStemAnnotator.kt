package io.github.koreanvoice.text

/**
 * Replacement for g2pkk's mecab-based `annotate()`.
 *
 * Inside MeloTTS, g2pkk is called on one word at a time, and of all mecab tags only "/P"
 * (a verb/adjective stem ending in ㄴ ㄵ ㅁ ㄻ ㄺ ㄼ ㄾ, e.g. 신고 → [신꼬], 읽고 → [일꼬], 넓게 → [널께])
 * changes the pronunciation. This class predicts that tag with a 3-level back-off table distilled
 * from mecab-ko's decisions on ~1.9M distinct Korean Wikipedia words (tools/distill_p.py):
 *   A: previous syllable + rest of word, B: stem syllable + rest of word, C: stem + next syllable,
 * then a fallback rule (double-coda stems are verbal unless the syllable is a known noun).
 * Agreement with mecab: 99.997% of weighted positions on the training words and 99.85% on
 * held-out words.
 */
class VerbStemAnnotator(private val levels: List<Map<String, Boolean>>) {

    fun annotate(s: String): String {
        if (s.none { Hangul.isSyllable(it) }) return s
        val sb = StringBuilder(s.length + 8)
        var i = 0
        while (i < s.length) {
            if (!Hangul.isSyllable(s[i])) {
                sb.append(s[i]); i++; continue
            }
            var j = i
            while (j < s.length && Hangul.isSyllable(s[j])) j++
            val word = s.substring(i, j)
            for (k in word.indices) {
                sb.append(word[k])
                if (k < word.length - 1 && isCandidate(word, k) && predict(word, k)) sb.append("/P")
            }
            i = j
        }
        return sb.toString()
    }

    fun predict(w: String, i: Int): Boolean {
        val keys = arrayOf(w.substring(maxOf(0, i - 1)), w.substring(i), w.substring(i, i + 2))
        for (lvl in 0 until 3) levels[lvl][keys[lvl]]?.let { return it }
        val t = Hangul.tailOf(w[i])
        return t != null && t in DOUBLE_CODAS && w[i] !in NOUN_DOUBLE
    }

    companion object {
        private const val FINALS = "ᆫᆬᆷᆱᆰᆲᆴ"
        private const val DOUBLE_CODAS = "ᆬᆱᆰᆲᆴ"
        private const val NOUN_DOUBLE = "닭흙칡삶앎"

        /** True when a /P tag on syllable i would trigger one of g2pkk's tensification rules. */
        fun isCandidate(w: String, i: Int): Boolean {
            val t = Hangul.tailOf(w[i]) ?: return false
            if (t !in FINALS) return false
            val o = Hangul.leadOf(w[i + 1]) ?: return false
            return if (t == 'ᆰ') o == 'ᄀ' || o == 'ᄁ' else o in "ᄀᄃᄉᄌ"
        }

        fun load(): VerbStemAnnotator {
            val maps = List(3) { HashMap<String, Boolean>() }
            for (line in G2pKo.readResource("p_table.tsv").lines()) {
                if (line.isEmpty() || line.startsWith("#")) continue
                val parts = line.split('\t')
                if (parts.size != 3) continue
                val lvl = "ABC".indexOf(parts[0])
                if (lvl >= 0) maps[lvl][parts[1]] = parts[2] == "1"
            }
            return VerbStemAnnotator(maps)
        }
    }
}
