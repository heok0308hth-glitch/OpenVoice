package io.github.koreanvoice.expressive

import java.text.Normalizer

/**
 * Chatterbox multilingual text tokenizer (grapheme_mtl_merged_expanded_v1.json, HuggingFace `tokenizers`
 * BPE with a Whitespace pre-tokenizer and 118 added tokens), plus the MTLTokenizer preprocessing:
 * lowercase → NFKD (Hangul syllables become conjoining jamo) → "[ko]" prefix → spaces as [SPACE].
 */
class CbTokenizer(json: String) {
    private val vocab: Map<String, Int>
    private val mergeRank: Map<Pair<String, String>, Int>
    private val added: List<Pair<String, Int>>      // content -> id, longest first
    val unkId: Int

    init {
        val root = MiniJson.parse(json) as Map<*, *>
        val model = root["model"] as Map<*, *>
        @Suppress("UNCHECKED_CAST")
        vocab = (model["vocab"] as Map<String, Any>).mapValues { (it.value as Number).toInt() }
        val merges = model["merges"] as List<*>
        val mr = HashMap<Pair<String, String>, Int>()
        merges.forEachIndexed { i, m ->
            val parts = when (m) {
                is String -> m.split(' ', limit = 2)
                is List<*> -> m.map { it as String }
                else -> error("bad merge")
            }
            mr[parts[0] to parts[1]] = i
        }
        mergeRank = mr
        added = (root["added_tokens"] as List<*>).map { a ->
            a as Map<*, *>; (a["content"] as String) to (a["id"] as Number).toInt()
        }.sortedByDescending { it.first.length }
        unkId = vocab["[UNK]"] ?: 1
    }

    /** MTLTokenizer.encode(text, language_id="ko") (without [START]/[STOP]). */
    fun encodeKorean(text: String): IntArray {
        var t = Normalizer.normalize(text.lowercase(), Normalizer.Form.NFKD)
        t = "[ko]$t"
        t = t.replace(" ", "[SPACE]")
        return encode(t)
    }

    fun encode(text: String): IntArray {
        val out = ArrayList<Int>()
        var i = 0
        val buf = StringBuilder()
        fun flush() { if (buf.isNotEmpty()) { encodePlain(buf.toString(), out); buf.clear() } }
        while (i < text.length) {
            var hit: Pair<String, Int>? = null
            if (text[i] == '[') for (a in added) if (text.startsWith(a.first, i)) { hit = a; break }
            if (hit != null) { flush(); out.add(hit.second); i += hit.first.length }
            else { buf.append(text[i]); i++ }
        }
        flush()
        return out.toIntArray()
    }

    /** Whitespace pre-tokenizer (\w+|[^\w\s]+, Unicode) followed by BPE. */
    private fun encodePlain(s: String, out: MutableList<Int>) {
        var i = 0
        while (i < s.length) {
            val cp = s.codePointAt(i)
            if (Character.isWhitespace(cp)) { i += Character.charCount(cp); continue }
            val word = isWordChar(cp)
            val start = i
            while (i < s.length) {
                val c = s.codePointAt(i)
                if (Character.isWhitespace(c) || isWordChar(c) != word) break
                i += Character.charCount(c)
            }
            bpe(s.substring(start, i), out)
        }
    }

    private fun bpe(word: String, out: MutableList<Int>) {
        val syms = ArrayList<String>()
        var k = 0
        while (k < word.length) { val n = Character.charCount(word.codePointAt(k)); syms.add(word.substring(k, k + n)); k += n }
        while (syms.size > 1) {
            var best = -1; var bestRank = Int.MAX_VALUE
            for (j in 0 until syms.size - 1) {
                val r = mergeRank[syms[j] to syms[j + 1]] ?: continue
                if (r < bestRank) { bestRank = r; best = j }
            }
            if (best < 0) break
            syms[best] = syms[best] + syms[best + 1]
            syms.removeAt(best + 1)
        }
        for (sym in syms) out.add(vocab[sym] ?: unkId)
    }

    companion object {
        /** Unicode \w as in the Rust regex crate: Alphabetic, marks, decimal digits, connector punctuation, join controls. */
        fun isWordChar(cp: Int): Boolean {
            if (Character.isAlphabetic(cp) || Character.isDigit(cp)) return true
            return when (Character.getType(cp).toByte()) {
                Character.NON_SPACING_MARK, Character.ENCLOSING_MARK, Character.COMBINING_SPACING_MARK,
                Character.CONNECTOR_PUNCTUATION -> true
                else -> cp == 0x200C || cp == 0x200D
            }
        }

        /** Port of chatterbox `punc_norm`. */
        fun puncNorm(input: String): String {
            if (input.isEmpty()) return "You need to add some text for me to talk."
            var text = input
            if (text[0].isLowerCase()) text = text[0].uppercaseChar() + text.substring(1)
            text = text.split(Regex("\\s+")).filter { it.isNotEmpty() }.joinToString(" ")
            for ((a, b) in listOf("..." to ", ", "…" to ", ", ":" to ",", " - " to ", ", ";" to ", ", "—" to "-", "–" to "-",
                " ," to ",", "“" to "\"", "”" to "\"", "‘" to "'", "’" to "'")) text = text.replace(a, b)
            text = text.trimEnd(' ')
            val enders = listOf(".", "!", "?", "-", ",", "、", "，", "。", "？", "！")
            if (enders.none { text.endsWith(it) }) text += "."
            return text
        }
    }
}

/** Tiny JSON reader (objects, arrays, strings, numbers, booleans, null) for the tokenizer and manifests. */
object MiniJson {
    fun parse(s: String): Any? = Parser(s).run { val v = value(); v }

    private class Parser(val s: String) {
        var i = 0
        fun ws() { while (i < s.length && s[i].isWhitespace()) i++ }
        fun value(): Any? {
            ws()
            return when (s[i]) {
                '{' -> obj()
                '[' -> arr()
                '"' -> str()
                't' -> { i += 4; true }
                'f' -> { i += 5; false }
                'n' -> { i += 4; null }
                else -> num()
            }
        }
        fun obj(): Map<String, Any?> {
            val m = LinkedHashMap<String, Any?>()
            i++; ws()
            if (s[i] == '}') { i++; return m }
            while (true) {
                ws(); val k = str(); ws(); i++ /* : */
                m[k] = value(); ws()
                if (s[i] == ',') { i++; continue }
                i++; return m
            }
        }
        fun arr(): List<Any?> {
            val a = ArrayList<Any?>()
            i++; ws()
            if (s[i] == ']') { i++; return a }
            while (true) {
                a.add(value()); ws()
                if (s[i] == ',') { i++; continue }
                i++; return a
            }
        }
        fun str(): String {
            val sb = StringBuilder(); i++
            while (true) {
                val c = s[i++]
                when (c) {
                    '"' -> return sb.toString()
                    '\\' -> {
                        when (val e = s[i++]) {
                            'n' -> sb.append('\n'); 't' -> sb.append('\t'); 'r' -> sb.append('\r'); 'b' -> sb.append('\b')
                            'f' -> sb.append('\u000C'); 'u' -> { sb.append(s.substring(i, i + 4).toInt(16).toChar()); i += 4 }
                            else -> sb.append(e)
                        }
                    }
                    else -> sb.append(c)
                }
            }
        }
        fun num(): Number {
            val st = i
            while (i < s.length && (s[i].isDigit() || s[i] in "+-.eE")) i++
            val t = s.substring(st, i)
            return if (t.any { it in ".eE" }) t.toDouble() else t.toLong()
        }
    }
}
