package io.github.koreanvoice.text

import java.io.BufferedReader
import java.io.InputStreamReader
import java.util.regex.Matcher
import java.util.regex.Pattern
import java.util.zip.GZIPInputStream

/**
 * Kotlin port of g2pkk 0.1.2 (Apache-2.0, https://github.com/harmlessman/g2pkk, a fork of
 * Kyubyong Park's g2pK), which is the Korean grapheme-to-phoneme converter MeloTTS-Korean was
 * trained with. Called with the defaults MeloTTS uses (descriptive=False, group_vowels=False,
 * to_syl=True).
 *
 * The only part of g2pkk that needs a morphological analyzer is `annotate` (mecab-ko). Here it is
 * replaced by [VerbStemAnnotator], a small decision table distilled from mecab's output on Korean
 * Wikipedia (see tools/distill_p.py), because the /J, /E and /B tags have no effect in the way
 * MeloTTS calls g2pkk (one word at a time, numbers already spelled out by [KoreanNormalizer]).
 */
class G2pKo(private val res: Resources = Resources.shared) {

    fun convert(input: String): String {
        var s = idioms(input)
        s = convertEng(s)
        s = res.annotator.annotate(s)
        s = convertNum(s)
        var inp = Hangul.h2j(s)
        inp = special(inp)
        inp = PJEB.matcher(inp).replaceAll("")
        for (rule in res.table) inp = rule.apply(inp)
        inp = links(inp)
        return Hangul.composeLetters(inp)
    }

    private fun idioms(s: String): String {
        var out = s
        for ((p, rep) in res.idioms) out = p.matcher(out).replaceAll(rep)
        return out
    }

    // ---------------------------------------------------------------- English (g2pkk english.py)
    fun convertEng(string: String): String {
        val words = LinkedHashSet<String>()
        val m = ENG_WORD.matcher(string)
        while (m.find()) words.add(m.group())
        var out = string
        for (engWord in words) {
            val ret = englishToHangul(engWord) ?: continue
            out = out.replace(engWord, ret)
        }
        return out
    }

    /** Hangul spelling of one English word via CMUdict, or null if the word is unknown. */
    fun englishToHangul(engWord: String): String? {
        val arpabets = res.cmu(engWord.lowercase()) ?: return null
        val phonemes = adjust(arpabets)
        val ret = StringBuilder()
        val n = phonemes.size
        for (i in 0 until n) {
            val p = phonemes[i]
            val pPrev = if (i > 0) phonemes[i - 1] else "^"
            val pNext = if (i < n - 1) phonemes[i + 1] else "$"
            val pNext2 = if (i < n - 2) phonemes[i + 1] else "$" // sic: g2pkk uses i+1 here
            val p0 = p[0]
            val next0 = pNext[0]
            val prev2 = if (pPrev.length >= 2) pPrev.substring(0, 2) else pPrev
            when {
                p.length == 1 && p0 in "PTK" -> when {
                    prev2 in SHORT_VOWELS && pNext == "$" -> ret.append(toJong(p))
                    prev2 in SHORT_VOWELS && next0 !in "AEIOULRMN" -> ret.append(toJong(p))
                    next0 in "\$BCDFGHJKLMNPQRSTVWXYZ" -> ret.append(toCho(p)).append('ᅳ')
                    else -> ret.append(toCho(p))
                }
                p.length == 1 && p0 in "BDG" -> {
                    ret.append(toCho(p))
                    if (next0 in SYL_FINAL_OR_CONS) ret.append('ᅳ')
                }
                p in setOf("S", "Z", "F", "V", "TH", "DH", "SH", "ZH") -> {
                    ret.append(toCho(p))
                    when (p) {
                        "S", "Z", "F", "V", "TH", "DH" -> if (next0 in SYL_FINAL_OR_CONS) ret.append('ᅳ')
                        "SH" -> when {
                            next0 == '$' -> ret.append('ᅵ')
                            next0 in CONSONANTS -> ret.append('ᅲ')
                            else -> ret.append('Y')
                        }
                        "ZH" -> if (next0 in SYL_FINAL_OR_CONS) ret.append('ᅵ')
                    }
                }
                p in setOf("TS", "DZ", "CH", "JH") -> {
                    ret.append(toCho(p))
                    if (next0 in SYL_FINAL_OR_CONS) ret.append(if (p == "TS" || p == "DZ") 'ᅳ' else 'ᅵ')
                }
                p == "M" || p == "N" || p == "NG" -> {
                    if ((p == "M" || p == "N") && next0 in VOWELS) ret.append(toCho(p)) else ret.append(toJong(p))
                }
                p == "L" -> when {
                    pPrev == "^" -> ret.append(toCho(p))
                    next0 in "\$BCDFGHJKLPQRSTVWXZ" -> ret.append(toJong(p))
                    pPrev == "M" || pPrev == "N" -> ret.append(toCho(p))
                    next0 in VOWELS -> ret.append("\u11af\u1105")
                    (pNext == "M" || pNext == "N") && pNext2[0] !in VOWELS -> ret.append("\u11af\u1105\u1173")
                }
                p == "ER" -> {
                    if (pPrev[0] in VOWELS) ret.append('ᄋ')
                    ret.append(toJung(p))
                    if (next0 in VOWELS) ret.append('ᄅ')
                }
                p == "R" -> if (next0 in VOWELS) ret.append(toCho(p))
                p0 in "AEIOU" -> ret.append(toJung(p))
                else -> ret.append(toCho(p))
            }
        }
        var r = reconstruct(ret.toString())
        r = Hangul.composeLetters(r)
        r = JAMO_ANY.matcher(r).replaceAll("")
        return r
    }

    private fun adjust(arpabets: List<String>): List<String> {
        var s = " " + arpabets.joinToString(" ") + " $"
        s = s.replace(Regex("\\d"), "")
        s = s.replace(" T S ", " TS ").replace(" D Z ", " DZ ").replace(" AW ER ", " AWER ")
            .replace(" IH R $", " IH ER ").replace(" EH R $", " EH ER ").replace(" $", "")
        return s.trim('$', ' ').split(Regex("\\s+")).filter { it.isNotEmpty() }
    }

    private fun reconstruct(string: String): String {
        var s = string
        for ((a, b) in RECONSTRUCT) s = s.replace(a, b)
        return s
    }

    // ---------------------------------------------------------------- numerals (g2pkk numerals.py)
    private fun convertNum(string: String): String {
        var s = string
        val m = BOUND_NUM.matcher(s)
        val tokens = LinkedHashSet<Pair<String, String>>()
        while (m.find()) tokens.add(m.group(1) to m.group(2))
        for ((num, bn) in tokens) {
            val spelled = KoreanNumbers.g2pkProcessNum(num, sino = bn !in BOUND_NOUNS)
            s = s.replace("$num$bn/B", "$spelled$bn/B")
        }
        val sb = StringBuilder(s.length)
        for (c in s) sb.append(if (c in '0'..'9') "영일이삼사오육칠팔구"[c - '0'] else c)
        return sb.toString()
    }

    // ---------------------------------------------------------------- special.py (descriptive=False)
    private fun special(input: String): String {
        var out = input
        out = rx("([ᄌᄍᄎ])ᅧ", out, "$1ᅥ")                       // jyeo
        out = rx("([ᄀᄁᄂᄃᄄᄅᄆᄇᄈᄉᄊᄌᄍᄎᄏᄐᄑᄒ])ᅴ", out, "$1ᅵ")    // consonant_ui
        out = out.replace("/J", "")                               // josa_ui
        // jamo (rule 16)
        out = rx("([\u1100\u1173])\u11ae\u110b", out, "$1\u1109")
        out = rx("([\u110b\u1173])[\u11bd\u11be\u11c0\u11c2]\u110b", out, "$1\u1109")
        out = rx("([\u110b\u1173])[\u11bf]\u110b", out, "$1\u1100")
        out = rx("([\u110b\u1173])[\u11c1]\u110b", out, "$1\u1107")
        // rieulgiyeok
        out = rx("ᆰ/P([ᄀᄁ])", out, "ᆯᄁ")
        // rieulbieub
        out = rx("([ᆲᆴ])/Pᄀ", out, "$1ᄁ")
        out = rx("([ᆲᆴ])/Pᄃ", out, "$1ᄄ")
        out = rx("([ᆲᆴ])/Pᄉ", out, "$1ᄊ")
        out = rx("([ᆲᆴ])/Pᄌ", out, "$1ᄍ")
        // verb_nieun
        for ((a, b) in VERB_NIEUN) out = rx(a, out, b)
        // balb
        out = rx("(\u1107\u1161)\u11b2((\$|[^\u110b\u1112]))", out, "$1\u11b8$2")
        out = rx("(\u1102\u1165)\u11b2([\u110c\u110d]\u116e|[\u1103\u1104]\u116e)", out, "$1\u11b8$2")
        // palatalize
        out = rx("ᆮᄋ([ᅵᅧ])", out, "ᄌ$1")
        out = rx("ᇀᄋ([ᅵᅧ])", out, "ᄎ$1")
        out = rx("ᆴᄋ([ᅵᅧ])", out, "ᆯᄎ$1")
        out = rx("ᆮᄒ([ᅵ])", out, "ᄎ$1")
        // modifying_rieul
        for ((a, b) in MODIFYING_RIEUL) out = rx(a, out, b)
        return out
    }

    private fun links(input: String): String {
        var out = input
        for (table in LINKS) for ((a, b) in table) out = out.replace(a, b)
        return out
    }

    /** Raw resources shared by all G2pKo instances (loaded lazily from the classpath). */
    class Resources private constructor() {
        val table: List<TableRule> by lazy { parseTable() }
        val idioms: List<Pair<Pattern, String>> by lazy { parseIdioms() }
        val annotator: VerbStemAnnotator by lazy { VerbStemAnnotator.load() }
        private val cmuMap: Map<String, List<String>> by lazy { loadCmu() }
        fun cmu(word: String): List<String>? = cmuMap[word]

        private fun parseTable(): List<TableRule> {
            val lines = readResource("table.csv").removePrefix("\uFEFF").lines().filter { it.isNotEmpty() }
            val onsets = lines[0].split(",")
            val out = ArrayList<TableRule>()
            for (line in lines.drop(1)) {
                val cols = line.split(",")
                val coda = cols[0]
                for (i in 1 until onsets.size) {
                    val cell = cols.getOrElse(i) { "" }
                    if (cell.isEmpty()) continue
                    val str2 = if ("(" in cell) cell.substringBefore("(") else cell
                    out.add(TableRule(coda + onsets[i], str2))
                }
            }
            return out
        }

        private fun parseIdioms(): List<Pair<Pattern, String>> =
            readResource("idioms.txt").lines().mapNotNull { raw ->
                val line = raw.split("#")[0].trim()
                if ("===" !in line) null else {
                    val (a, b) = line.split("===")
                    Pattern.compile(a) to Matcher.quoteReplacement(b)
                }
            }

        private fun loadCmu(): Map<String, List<String>> {
            val map = HashMap<String, List<String>>(140_000)
            val stream = G2pKo::class.java.getResourceAsStream("/koreanvoice/cmudict_first.tsv.gz")
                ?: error("cmudict resource missing")
            BufferedReader(InputStreamReader(GZIPInputStream(stream), Charsets.UTF_8)).useLines { seq ->
                seq.forEach { l ->
                    val tab = l.indexOf('\t')
                    if (tab > 0) map[l.substring(0, tab)] = l.substring(tab + 1).split(' ')
                }
            }
            return map
        }

        companion object {
            val shared: Resources by lazy { Resources() }
        }
    }

    /** One row/column cell of g2pkk's table.csv, compiled from Python regex syntax. */
    class TableRule(pattern: String, replacement: String) {
        // Python's \W is Unicode-aware; spell it out so desktop Java and Android (ICU) agree.
        private val p: Pattern = Pattern.compile(pattern.replace("\\W", PY_NON_WORD))
        private val rep: String = replacement.replace(Regex("\\\\(\\d)"), "\\$$1")
        fun apply(s: String): String = p.matcher(s).replaceAll(rep)
    }

    companion object {
        /**
         * Python 3 `\W` (complement of str.isalnum() or '_'). Pattern.UNICODE_CHARACTER_CLASS would do
         * the same on desktop Java but throws on Android, whose regex engine is ICU.
         */
        const val PY_NON_WORD = "[^\\p{L}\\p{N}_]"
        private val PJEB: Pattern = Pattern.compile("/[PJEB]")
        private val ENG_WORD: Pattern = Pattern.compile("[A-Za-z']+")
        private val JAMO_ANY: Pattern = Pattern.compile("[ᄀ-ᇿ]")
        private val BOUND_NUM: Pattern = Pattern.compile("([0-9][0-9,]*)([ㄱ-힣]+)/B")
        private val BOUND_NOUNS = "군데 권 개 그루 닢 대 두 마리 모 모금 뭇 발 발짝 방 번 벌 보루 살 수 술 시 쌈 움큼 정 짝 채 척 첩 축 켤레 톨 통"
            .split(" ").toSet()
        private val SHORT_VOWELS = setOf("AE", "AH", "AX", "EH", "IH", "IX", "UH")
        private const val VOWELS = "AEIOUY"
        private const val CONSONANTS = "BCDFGHJKLMNPQRSTVWXZ"
        private const val SYL_FINAL_OR_CONS = "\$BCDFGHJKLMNPQRSTVWXZ"
        private val patternCache = HashMap<String, Pattern>()

        private fun rx(pattern: String, input: String, replacement: String): String {
            val p = synchronized(patternCache) { patternCache.getOrPut(pattern) { Pattern.compile(pattern) } }
            return p.matcher(input).replaceAll(replacement)
        }

        private fun toCho(p: String): String = CHO[p] ?: p
        private fun toJung(p: String): String = JUNG[p] ?: p
        private fun toJong(p: String): String = JONG[p] ?: p

        private val CHO = mapOf(
            "B" to "ᄇ", "CH" to "ᄎ", "D" to "ᄃ", "DH" to "ᄃ", "DZ" to "ᄌ", "F" to "ᄑ", "G" to "ᄀ", "HH" to "ᄒ",
            "JH" to "ᄌ", "K" to "ᄏ", "L" to "ᄅ", "M" to "ᄆ", "N" to "ᄂ", "NG" to "ᄋ", "P" to "ᄑ", "R" to "ᄅ",
            "S" to "ᄉ", "SH" to "ᄉ", "T" to "ᄐ", "TH" to "ᄉ", "TS" to "ᄎ", "V" to "ᄇ", "W" to "W", "Y" to "Y",
            "Z" to "ᄌ", "ZH" to "ᄌ",
        )
        private val JUNG = mapOf(
            "AA" to "ᅡ", "AE" to "ᅢ", "AH" to "ᅥ", "AO" to "ᅩ", "AW" to "\u1161\u110b\u116e", "AWER" to "\u1161\u110b\u116f", "AY" to "\u1161\u110b\u1175",
            "EH" to "ᅦ", "ER" to "ᅥ", "EY" to "\u1166\u110b\u1175", "IH" to "ᅵ", "IY" to "ᅵ", "OW" to "ᅩ", "OY" to "\u1169\u110b\u1175",
            "UH" to "ᅮ", "UW" to "ᅮ",
        )
        private val JONG = mapOf(
            "B" to "ᆸ", "CH" to "ᆾ", "D" to "ᆮ", "DH" to "ᆮ", "F" to "ᇁ", "G" to "ᆨ", "HH" to "ᇂ", "JH" to "ᆽ",
            "K" to "ᆨ", "L" to "ᆯ", "M" to "ᆷ", "N" to "ᆫ", "NG" to "ᆼ", "P" to "ᆸ", "R" to "ᆯ", "S" to "ᆺ",
            "SH" to "ᆺ", "T" to "ᆺ", "TH" to "ᆺ", "V" to "ᆸ", "W" to "ᆼ", "Y" to "ᆼ", "Z" to "ᆽ", "ZH" to "ᆽ",
        )
        private val RECONSTRUCT = listOf(
            "\u1100\u1173W" to "\u1100W", "\u1112\u1173W" to "\u1112W", "\u110f\u1173W" to "\u110fW",
            "\u1102Y\u1165" to "\u1102\u1175\u110b\u1165", "\u1103Y\u1165" to "\u1103\u1175\u110b\u1165", "\u1105Y\u1165" to "\u1105\u1175\u110b\u1165",
            "Yᅵ" to "ᅵ", "Yᅡ" to "ᅣ", "Yᅢ" to "ᅤ", "Yᅥ" to "ᅧ", "Yᅦ" to "ᅨ", "Yᅩ" to "ᅭ", "Yᅮ" to "ᅲ",
            "Wᅡ" to "ᅪ", "Wᅢ" to "ᅫ", "Wᅥ" to "ᅯ", "Wᅩ" to "ᅯ", "Wᅮ" to "ᅮ", "Wᅦ" to "ᅰ", "Wᅵ" to "ᅱ",
            "ᅳᅵ" to "ᅴ", "Y" to "ᅵ", "W" to "ᅮ",
        )
        private val VERB_NIEUN = listOf(
            "([ᆫᆷ])/Pᄀ" to "$1ᄁ", "([ᆫᆷ])/Pᄃ" to "$1ᄄ", "([ᆫᆷ])/Pᄉ" to "$1ᄊ", "([ᆫᆷ])/Pᄌ" to "$1ᄍ",
            "ᆬ/Pᄀ" to "ᆫᄁ", "ᆬ/Pᄃ" to "ᆫᄄ", "ᆬ/Pᄉ" to "ᆫᄊ", "ᆬ/Pᄌ" to "ᆫᄍ",
            "ᆱ/Pᄀ" to "ᆷᄁ", "ᆱ/Pᄃ" to "ᆷᄄ", "ᆱ/Pᄉ" to "ᆷᄊ", "ᆱ/Pᄌ" to "ᆷᄍ",
        )
        private val MODIFYING_RIEUL = listOf(
            "ᆯ/E ᄀ" to "ᆯ ᄁ", "ᆯ/E ᄃ" to "ᆯ ᄄ", "ᆯ/E ᄇ" to "ᆯ ᄈ", "ᆯ/E ᄉ" to "ᆯ ᄊ", "ᆯ/E ᄌ" to "ᆯ ᄍ",
            "\u11af\u1100\u1165\u11af" to "\u11af\u1101\u1165\u11af",
            "\u11af\u1107\u1161\u11a9\u110b\u1166" to "\u11af\u1108\u1161\u1101\u1166",
            "\u11af\u1109\u1166\u1105\u1161" to "\u11af\u110a\u1166\u1105\u1161",
            "\u11af\u1109\u116e\u1105\u1169\u11a8" to "\u11af\u110a\u116e\u1105\u1169\u11a8",
            "\u11af\u110c\u1175\u1105\u1161\u1103\u1169" to "\u11af\u110d\u1175\u1105\u1161\u1103\u1169",
            "\u11af\u110c\u1175\u110b\u1165\u11ab\u110c\u1165\u11bc" to "\u11af\u110d\u1175\u110b\u1165\u11ab\u110c\u1165\u11bc",
            "\u11af\u110c\u1175\u11ab\u1103\u1162" to "\u11af\u110d\u1175\u11ab\u1103\u1162",
        )
        private val LINKS: List<List<Pair<String, String>>> = listOf(
            listOf("ᆨᄋ" to "ᄀ", "ᆩᄋ" to "ᄁ", "ᆫᄋ" to "ᄂ", "ᆮᄋ" to "ᄃ", "ᆯᄋ" to "ᄅ", "ᆷᄋ" to "ᄆ", "ᆸᄋ" to "ᄇ",
                "ᆺᄋ" to "ᄉ", "ᆻᄋ" to "ᄊ", "ᆽᄋ" to "ᄌ", "ᆾᄋ" to "ᄎ", "ᆿᄋ" to "ᄏ", "ᇀᄋ" to "ᄐ", "ᇁᄋ" to "ᄑ"),
            listOf("ᆪᄋ" to "ᆨᄊ", "ᆬᄋ" to "ᆫᄌ", "ᆰᄋ" to "ᆯᄀ", "ᆱᄋ" to "ᆯᄆ", "ᆲᄋ" to "ᆯᄇ", "ᆳᄋ" to "ᆯᄊ",
                "ᆴᄋ" to "ᆯᄐ", "ᆵᄋ" to "ᆯᄑ", "ᆹᄋ" to "ᆸᄊ"),
            listOf("ᆨ ᄋ" to " ᄀ", "ᆩ ᄋ" to " ᄁ", "ᆫ ᄋ" to " ᄂ", "ᆮ ᄋ" to " ᄃ", "ᆯ ᄋ" to " ᄅ", "ᆷ ᄋ" to " ᄆ",
                "ᆸ ᄋ" to " ᄇ", "ᆺ ᄋ" to " ᄉ", "ᆻ ᄋ" to " ᄊ", "ᆽ ᄋ" to " ᄌ", "ᆾ ᄋ" to " ᄎ", "ᆿ ᄋ" to " ᄏ",
                "ᇀ ᄋ" to " ᄐ", "ᇁ ᄋ" to " ᄑ", "ᆪ ᄋ" to "ᆨ ᄊ", "ᆬ ᄋ" to "ᆫ ᄌ", "ᆰ ᄋ" to "ᆯ ᄀ", "ᆱ ᄋ" to "ᆯ ᄆ",
                "ᆲ ᄋ" to "ᆯ ᄇ", "ᆳ ᄋ" to "ᆯ ᄊ", "ᆴ ᄋ" to "ᆯ ᄐ", "ᆵ ᄋ" to "ᆯ ᄑ", "ᆹ ᄋ" to "ᆸ ᄊ"),
            listOf("ᇂᄋ" to "ᄋ", "ᆭᄋ" to "ᄂ", "ᆶᄋ" to "ᄅ"),
        )

        internal fun readResource(name: String): String =
            (G2pKo::class.java.getResourceAsStream("/koreanvoice/$name") ?: error("missing resource $name"))
                .use { String(it.readBytes(), Charsets.UTF_8) }
    }
}
