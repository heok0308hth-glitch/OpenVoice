package io.github.koreanvoice.text

/**
 * Splits normalized Korean text into synthesis chunks.
 *
 * - Sentences end at . ! ? … ; paragraphs at line breaks.
 * - Sentences longer than [maxChars] are split at commas first, then after clause-final endings
 *   (-고, -며, -지만, -는데, -면서, -어서 …), then at the space nearest the middle, so each piece is
 *   still a natural prosodic phrase.
 * - Very short sentences in the same paragraph are merged (better prosody, fewer model calls).
 * Every chunk carries the pause that should follow it.
 */
class TextChunker(
    private val maxChars: Int = 90,
    private val mergeUpTo: Int = 50,
    private val minChars: Int = 18,
) {
    data class Chunk(val text: String, val pauseMs: Int, val paragraphEnd: Boolean)

    fun split(normalized: String): List<Chunk> {
        val out = ArrayList<Chunk>()
        val paragraphs = normalized.split('\n').map { it.trim() }.filter { it.isNotEmpty() }
        for (para in paragraphs) {
            val pieces = ArrayList<String>()
            for (s in sentences(para)) pieces.addAll(splitLong(s))
            val merged = merge(pieces)
            merged.forEachIndexed { i, t ->
                val last = i == merged.size - 1
                out.add(Chunk(t, if (last) PAUSE_PARAGRAPH else pauseFor(t), last))
            }
        }
        return out
    }

    fun sentences(p: String): List<String> {
        val res = ArrayList<String>()
        val m = SENT_END.findAll(p)
        var start = 0
        for (x in m) {
            val end = x.range.last + 1
            val s = p.substring(start, end).trim()
            if (s.isNotEmpty()) res.add(s)
            start = end
        }
        val rest = p.substring(start).trim()
        if (rest.isNotEmpty()) res.add(rest)
        return res
    }

    private fun splitLong(s: String): List<String> {
        if (s.length <= maxChars) return listOf(ensureEnding(s))
        val cut = bestCut(s)
        val left = s.substring(0, cut).trim().trimEnd(',')
        val right = s.substring(cut).trim()
        return splitLong("$left,") + splitLong(right)
    }

    /** Index at which to cut a too-long sentence. */
    private fun bestCut(s: String): Int {
        val mid = s.length / 2
        val lo = (s.length * 0.25).toInt(); val hi = (s.length * 0.75).toInt()
        fun nearest(cands: List<Int>) = cands.filter { it in lo..hi }.minByOrNull { kotlin.math.abs(it - mid) }
        val commas = s.indices.filter { s[it] == ',' }.map { it + 1 }
        nearest(commas)?.let { return it }
        val spaces = s.indices.filter { s[it] == ' ' }
        val clause = spaces.filter { i -> CLAUSE_ENDINGS.any { e -> s.regionMatches(i - e.length, e, 0, e.length) } }
        nearest(clause)?.let { return it }
        nearest(spaces)?.let { return it }
        spaces.minByOrNull { kotlin.math.abs(it - mid) }?.let { return it }
        return mid
    }

    private fun merge(pieces: List<String>): List<String> {
        val out = ArrayList<String>()
        for (p in pieces) {
            val last = out.lastOrNull()
            if (last != null && (last.length < minChars || p.length < minChars) && last.length + p.length + 1 <= mergeUpTo) {
                out[out.size - 1] = "$last $p"
            } else out.add(p)
        }
        return out
    }

    private fun ensureEnding(s: String): String = if (s.last() in ".!?…,") s else "$s."

    private fun pauseFor(t: String): Int = when (t.last()) {
        ',' -> PAUSE_COMMA
        '?', '!' -> PAUSE_QUESTION
        '…' -> PAUSE_ELLIPSIS
        else -> PAUSE_SENTENCE
    }

    companion object {
        const val PAUSE_COMMA = 180
        const val PAUSE_SENTENCE = 380
        const val PAUSE_QUESTION = 450
        const val PAUSE_ELLIPSIS = 550
        const val PAUSE_PARAGRAPH = 750
        private val SENT_END = Regex("[.!?…]+(?=\\s|$)")
        private val CLAUSE_ENDINGS = listOf("지만", "는데", "은데", "면서", "어서", "아서", "니까", "므로", "도록", "는지", "고", "며", "서", "면", "나")
    }
}
