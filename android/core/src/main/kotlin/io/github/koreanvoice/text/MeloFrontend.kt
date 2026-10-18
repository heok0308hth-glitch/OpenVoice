package io.github.koreanvoice.text

/**
 * Kotlin replica of MeloTTS-Korean's text frontend (melo/text/korean.py + cleaner +
 * cleaned_text_to_sequence + intersperse), producing exactly the model inputs:
 * phoneme ids, tone ids, language ids, the word→phone map used to expand BERT features, and the
 * BERT input ids.
 */
class MeloFrontend(
    private val g2p: G2pKo = G2pKo(),
    val tokenizer: WordPieceTokenizer = WordPieceTokenizer.load(),
    symbols: List<String> = loadSymbols(),
) {
    private val symbolToId: Map<String, Int> = symbols.withIndex().associate { it.value to it.index }
    private val wordCache = object : LinkedHashMap<String, String>(4096, 0.75f, true) {
        override fun removeEldestEntry(eldest: MutableMap.MutableEntry<String, String>?) = size > 20000
    }

    class Result(
        val normText: String,
        val phones: List<String>,
        /** Interspersed ids (blank = 0 between every symbol), length N = 2 * phones + 1. */
        val ids: LongArray,
        val tones: LongArray,
        val langs: LongArray,
        /** word2ph after intersperse adjustment; one entry per BERT token incl. [CLS]/[SEP]. */
        val word2ph: IntArray,
        val bertIds: LongArray,
    )

    /** MeloTTS `korean.normalize` (CJK removal, etc_dictionary, english_dictionary, lowercase). */
    fun meloNormalize(text: String): String {
        var t = text.trim()
        t = CJK.replace(t, "")
        for ((k, v) in ETC_DICT) t = t.replace(k, v)
        t = Regex("[A-Za-z]+").replace(t) { m -> MELO_EN[m.value] ?: m.value }
        return t.lowercase()
    }

    /** korean_text_to_phonemes(): normalize → g2pkk → hangul_to_jamo. */
    fun wordToPhonemes(word: String): String {
        synchronized(wordCache) { wordCache[word]?.let { return it } }
        val g = g2p.convert(meloNormalize(word))
        val res = Hangul.h2j(g)
        synchronized(wordCache) { wordCache[word] = res }
        return res
    }

    fun process(text: String): Result {
        val norm = meloNormalize(text)
        val tokens = tokenizer.tokenize(norm)
        val groups = ArrayList<MutableList<String>>()
        for (t in tokens) {
            if (!t.startsWith("#") || groups.isEmpty()) groups.add(mutableListOf(t))
            else groups.last().add(t.replace("#", ""))
        }
        val phs = ArrayList<String>()
        val word2ph = ArrayList<Int>()
        word2ph.add(1)
        for (group in groups) {
            val text0 = group.joinToString("")
            when {
                text0 == "[UNK]" -> { phs.add("_"); word2ph.add(1) }
                text0 in PUNCTUATION -> { phs.add(text0); word2ph.add(1) }
                else -> {
                    // Symbols unknown to the model would raise KeyError in MeloTTS; drop them instead.
                    val phonemes = wordToPhonemes(text0).map { it.toString() }.filter { it in symbolToId }
                    phs.addAll(phonemes)
                    word2ph.addAll(distributePhone(phonemes.size, group.size).toList())
                }
            }
        }
        word2ph.add(1)
        val phones = ArrayList<String>(phs.size + 2)
        phones.add("_"); phones.addAll(phs); phones.add("_")
        val n = phones.size * 2 + 1
        val ids = LongArray(n); val tones = LongArray(n); val langs = LongArray(n)
        for ((i, p) in phones.withIndex()) {
            ids[2 * i + 1] = symbolToId.getValue(p).toLong()
            tones[2 * i + 1] = KR_TONE_START.toLong()
            langs[2 * i + 1] = KR_LANG_ID.toLong()
        }
        val w2p = IntArray(word2ph.size) { word2ph[it] * 2 }
        w2p[0] += 1
        return Result(norm, phones, ids, tones, langs, w2p, tokenizer.encode(tokens))
    }

    companion object {
        const val KR_TONE_START = 11 // num_zh_tones(6) + num_ja_tones(1) + num_en_tones(4)
        const val KR_LANG_ID = 4
        val PUNCTUATION = setOf("!", "?", "…", ",", ".", "'", "-", "¿", "¡")
        private val CJK = Regex("[⺀-⺙⺛-⻳⼀-⿕々〇〡-〩〸-〺〻㐀-䶵一-鿃豈-鶴侮-頻並-龎]")
        private val ETC_DICT = linkedMapOf("1+1" to "원플러스원", "2+1" to "투플러스원")
        private val MELO_EN = mapOf(
            "KOREA" to "코리아", "IDOL" to "아이돌", "IT" to "아이티", "IQ" to "아이큐", "UP" to "업", "DOWN" to "다운",
            "PC" to "피씨", "CCTV" to "씨씨티비", "SNS" to "에스엔에스", "AI" to "에이아이", "CEO" to "씨이오",
            "A" to "에이", "B" to "비", "C" to "씨", "D" to "디", "E" to "이", "F" to "에프", "G" to "지", "H" to "에이치",
            "I" to "아이", "J" to "제이", "K" to "케이", "L" to "엘", "M" to "엠", "N" to "엔", "O" to "오", "P" to "피",
            "Q" to "큐", "R" to "알", "S" to "에스", "T" to "티", "U" to "유", "V" to "브이", "W" to "더블유",
            "X" to "엑스", "Y" to "와이", "Z" to "제트",
        )

        fun distributePhone(nPhone: Int, nWord: Int): IntArray {
            val out = IntArray(nWord)
            if (nWord == 0) return out
            val q = nPhone / nWord; val r = nPhone % nWord
            for (i in 0 until nWord) out[i] = q + if (i < r) 1 else 0
            return out
        }

        fun loadSymbols(): List<String> = G2pKo.readResource("symbols.txt").split("\n").let {
            if (it.last().isEmpty()) it.dropLast(1) else it
        }
    }
}
