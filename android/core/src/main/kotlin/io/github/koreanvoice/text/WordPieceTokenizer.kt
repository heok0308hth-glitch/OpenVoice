package io.github.koreanvoice.text

/**
 * BERT uncased tokenizer (BasicTokenizer + WordPiece) matching HuggingFace `BertTokenizer` with
 * the kykim/bert-kor-base settings: do_lower_case=true, strip_accents=false,
 * tokenize_chinese_chars=true, max_input_chars_per_word=100.
 */
class WordPieceTokenizer(vocabLines: List<String>) {
    private val vocab: Map<String, Int>
    val unkId: Int
    val clsId: Int
    val sepId: Int

    init {
        val m = HashMap<String, Int>(vocabLines.size * 2)
        vocabLines.forEachIndexed { i, t -> if (t.isNotEmpty() || i == 0) m.putIfAbsent(t, i) }
        vocab = m
        unkId = m.getValue("[UNK]"); clsId = m.getValue("[CLS]"); sepId = m.getValue("[SEP]")
    }

    fun tokenize(text: String): List<String> {
        val out = ArrayList<String>()
        for (word in basicTokenize(text)) wordPiece(word, out)
        return out
    }

    fun id(token: String): Int = vocab[token] ?: unkId

    /** [CLS] tokens [SEP] ids, like `tokenizer(text)["input_ids"]`. */
    fun encode(tokens: List<String>): LongArray {
        val ids = LongArray(tokens.size + 2)
        ids[0] = clsId.toLong()
        tokens.forEachIndexed { i, t -> ids[i + 1] = id(t).toLong() }
        ids[ids.size - 1] = sepId.toLong()
        return ids
    }

    fun basicTokenize(text: String): List<String> {
        val cleaned = StringBuilder(text.length)
        var i = 0
        while (i < text.length) {
            val cp = text.codePointAt(i)
            i += Character.charCount(cp)
            if (cp == 0 || cp == 0xFFFD || isControl(cp)) continue
            if (isWhitespace(cp)) cleaned.append(' ')
            else if (isChinese(cp)) cleaned.append(' ').appendCodePoint(cp).append(' ')
            else cleaned.appendCodePoint(cp)
        }
        val out = ArrayList<String>()
        for (tok in cleaned.split(' ')) {
            if (tok.isEmpty()) continue
            val lower = tok.lowercase()
            var cur = StringBuilder()
            var k = 0
            while (k < lower.length) {
                val cp = lower.codePointAt(k)
                k += Character.charCount(cp)
                if (isPunctuation(cp)) {
                    if (cur.isNotEmpty()) { out.add(cur.toString()); cur = StringBuilder() }
                    out.add(String(Character.toChars(cp)))
                } else cur.appendCodePoint(cp)
            }
            if (cur.isNotEmpty()) out.add(cur.toString())
        }
        return out
    }

    private fun wordPiece(word: String, out: MutableList<String>) {
        if (word.codePointCount(0, word.length) > 100) { out.add("[UNK]"); return }
        val pieces = ArrayList<String>()
        var start = 0
        while (start < word.length) {
            var end = word.length
            var cur: String? = null
            while (start < end) {
                var sub = word.substring(start, end)
                if (start > 0) sub = "##$sub"
                if (vocab.containsKey(sub)) { cur = sub; break }
                end = word.offsetByCodePoints(end, -1)
            }
            if (cur == null) { out.add("[UNK]"); return }
            pieces.add(cur)
            start = end
        }
        out.addAll(pieces)
    }

    companion object {
        fun load(): WordPieceTokenizer = WordPieceTokenizer(G2pKo.readResource("bert_vocab.txt").split("\n").let {
            if (it.isNotEmpty() && it.last().isEmpty()) it.dropLast(1) else it
        })

        fun isWhitespace(cp: Int): Boolean =
            cp == ' '.code || cp == '\t'.code || cp == '\n'.code || cp == '\r'.code ||
                Character.getType(cp) == Character.SPACE_SEPARATOR.toInt()

        fun isControl(cp: Int): Boolean {
            if (cp == '\t'.code || cp == '\n'.code || cp == '\r'.code) return false
            val t = Character.getType(cp)
            return t == Character.CONTROL.toInt() || t == Character.FORMAT.toInt()
        }

        fun isPunctuation(cp: Int): Boolean {
            if (cp in 33..47 || cp in 58..64 || cp in 91..96 || cp in 123..126) return true
            return when (Character.getType(cp).toByte()) {
                Character.CONNECTOR_PUNCTUATION, Character.DASH_PUNCTUATION, Character.START_PUNCTUATION,
                Character.END_PUNCTUATION, Character.INITIAL_QUOTE_PUNCTUATION, Character.FINAL_QUOTE_PUNCTUATION,
                Character.OTHER_PUNCTUATION -> true
                else -> false
            }
        }

        fun isChinese(cp: Int): Boolean =
            cp in 0x4E00..0x9FFF || cp in 0x3400..0x4DBF || cp in 0x20000..0x2A6DF || cp in 0x2A700..0x2B73F ||
                cp in 0x2B740..0x2B81F || cp in 0x2B820..0x2CEAF || cp in 0xF900..0xFAFF || cp in 0x2F800..0x2FA1F
    }
}
