package io.github.koreanvoice.text

import java.text.Normalizer

/**
 * Korean text normalization done before the MeloTTS frontend.
 *
 * MeloTTS-Korean itself only removes Hanja and maps a few uppercase acronyms; g2pkk then reads
 * digits one by one ("2024" → 이영이사) unless a mecab bound-noun tag is present, and crashes on
 * symbols missing from the model's symbol table. This normalizer turns arbitrary Korean prose into
 * text made only of Hangul syllables, spaces and the punctuation the model was trained with:
 *  - numbers: Sino-Korean vs native readings chosen by the counter (3개 → 세 개, 3개월 → 삼 개월,
 *    13시 → 십삼 시, 3시 → 세 시), dates, times, phone numbers, decimals, fractions, ranges,
 *    ordinals (2번째 → 두 번째), currency, units and percentages;
 *  - Hanja → Hangul (Unihan readings, with the word-initial sound rule 두음법칙);
 *  - English: loanword dictionary → CMUdict + g2pkk transliteration → letter-by-letter spelling;
 *  - symbols, emoji, URLs, quotes and brackets; ㅋㅋ/ㅎㅎ style jamo.
 * Line breaks are preserved so paragraph structure survives for the chunker.
 */
class KoreanNormalizer(private val g2p: G2pKo = G2pKo()) {

    fun normalize(input: String): String =
        input.replace("\r\n", "\n").replace('\r', '\n').split('\n').joinToString("\n") { normalizeLine(it) }

    fun normalizeLine(line0: String): String {
        var s = compatJamo(line0)
        s = nfkcNonHangul(s)
        s = URL.replace(s) { " 링크 " }
        s = EMAIL.replace(s) { m -> " " + readAddress(m.groupValues[1]) + " 골뱅이 " + readAddress(m.groupValues[2]) + " " }
        s = hanja(s)
        s = numbers(s)
        s = english(s)
        s = symbols(s)
        return cleanup(s)
    }

    // ------------------------------------------------------------------ jamo / unicode
    private fun compatJamo(s: String): String {
        if (s.none { it.code in 0x3131..0x318E }) return s
        var t = Regex("[ㅠㅜ]{2,}").replace(s, " ")
        val sb = StringBuilder()
        for (c in t) {
            val code = c.code
            when {
                code in 0x3131..0x314E -> {
                    val lead = COMPAT_TO_LEAD[c]
                    if (lead != null) sb.append(Hangul.compose(lead, 'ᅳ')) // ㅋ → 크
                }
                code in 0x314F..0x3163 -> sb.append(Hangul.compose('ᄋ', (0x1161 + (code - 0x314F)).toChar()))
                code in 0x3164..0x318E -> {}
                else -> sb.append(c)
            }
        }
        t = sb.toString()
        return Regex("(.)\\1{4,}").replace(t) { m -> m.groupValues[1].repeat(4) }
    }

    private fun nfkcNonHangul(s: String): String {
        val sb = StringBuilder(s.length)
        for (c in s) {
            if (Hangul.isSyllable(c) || c.code < 0x80) sb.append(c)
            else when (c) {
                '℃' -> sb.append("°C")
                '℉' -> sb.append("°F")
                '㎡' -> sb.append("m²")
                '㎥' -> sb.append("m³")
                '·', 'ㆍ', '･', '‧', '•' -> sb.append('·')
                '…' -> sb.append("...")
                '～', '〜' -> sb.append('~')
                else -> sb.append(Normalizer.normalize(c.toString(), Normalizer.Form.NFKC))
            }
        }
        return sb.toString()
    }

    // ------------------------------------------------------------------ Hanja
    private fun hanja(s: String): String {
        if (s.none { WordPieceTokenizer.isChinese(it.code) }) return s
        val map = hanjaMap
        val sb = StringBuilder(s.length)
        for (i in s.indices) {
            val c = s[i]
            val r = map[c]
            if (r == null) {
                if (!WordPieceTokenizer.isChinese(c.code)) sb.append(c)
                continue
            }
            val wordInitial = i == 0 || !(map.containsKey(s[i - 1]) || s[i - 1].isLetterOrDigit())
            sb.append(if (wordInitial) initialSoundRule(r) else r)
        }
        return sb.toString()
    }

    /** 두음법칙: 녀→여, 력→역, 로→노, 리→이 at the start of a Sino-Korean word. */
    private fun initialSoundRule(r: Char): Char {
        val lead = Hangul.leadOf(r) ?: return r
        val v = Hangul.vowelIndex(r)
        val iy = v in IOTIZED
        return when (lead) {
            'ᄅ' -> Hangul.withLead(r, if (iy) 'ᄋ' else 'ᄂ')
            'ᄂ' -> if (iy) Hangul.withLead(r, 'ᄋ') else r
            else -> r
        }
    }

    // ------------------------------------------------------------------ numbers
    private fun numbers(input: String): String {
        if (input.none { it in '0'..'9' }) return input
        var s = ALNUM.replace(input) { m -> " " + ALNUM_WORDS.getValue(m.value.lowercase()) + " " }
        s = Regex("(\\d)·(\\d{1,2})(?!\\d)").replace(s) { m -> KoreanNumbers.digits(m.groupValues[1]) + KoreanNumbers.digits(m.groupValues[2]) }
        // phone numbers
        s = PHONE.replace(s) { m -> m.groupValues.drop(1).filter { it.isNotEmpty() }.joinToString(", ") { KoreanNumbers.digits(it, true) } + " " }
        // dates
        s = DATE.replace(s) { m ->
            val y = m.groupValues[1]; val mo = m.groupValues[2].toInt(); val d = m.groupValues[3].toInt()
            if (mo in 1..12 && d in 1..31) " ${KoreanNumbers.sino(y)} 년 ${KoreanNumbers.month(mo)} ${KoreanNumbers.sino(d.toLong())} 일 " else m.value
        }
        // times hh:mm(:ss)
        s = TIME.replace(s) { m ->
            val h = m.groupValues[1].toInt(); val mi = m.groupValues[2].toInt(); val sec = m.groupValues[3]
            if (h > 24 || mi > 59) return@replace m.value
            val sb = StringBuilder(" ").append(hour(h)).append(" 시")
            if (mi > 0) sb.append(" ").append(KoreanNumbers.sino(mi.toLong())).append(" 분")
            if (sec.isNotEmpty() && sec.toInt() > 0) sb.append(" ").append(KoreanNumbers.sino(sec)).append(" 초")
            sb.append(" ").toString()
        }
        // negative numbers
        s = Regex("(^|[\\s(\\[=])-(\\d)").replace(s) { m -> m.groupValues[1] + "마이너스 " + m.groupValues[2] }
        // scores / ratios 3:2
        s = Regex("(\\d+)\\s*:\\s*(\\d+)").replace(s) { m -> "${KoreanNumbers.sino(m.groupValues[1])} 대 ${KoreanNumbers.sino(m.groupValues[2])}" }
        // currency prefixes
        s = Regex("([₩$€¥£])\\s?(\\d[\\d,]*(?:\\.\\d+)?)\\s?(만|억|조)?").replace(s) { m ->
            val unit = when (m.groupValues[1]) { "₩" -> "원"; "$" -> "달러"; "€" -> "유로"; "¥" -> "엔"; else -> "파운드" }
            " " + readNumber(m.groupValues[2]) + m.groupValues[3] + " " + unit + " "
        }
        // fractions a/b (not dates, handled above)
        s = Regex("(?<![\\d/])(\\d{1,4})/(\\d{1,4})(?![\\d/])").replace(s) { m ->
            "${KoreanNumbers.sino(m.groupValues[2])}분의 ${KoreanNumbers.sino(m.groupValues[1])}"
        }
        // percent / units
        s = Regex("(\\d[\\d,]*(?:\\.\\d+)?)\\s?(%p|%|퍼센트)").replace(s) { m ->
            readNumber(m.groupValues[1]) + if (m.groupValues[2] == "%p") " 퍼센트포인트" else " 퍼센트"
        }
        s = UNIT_RE.replace(s) { m -> readNumber(m.groupValues[1]) + " " + UNITS.getValue(m.groupValues[2]) + m.groupValues[3] }
        // ranges with counter: 3~5개, 10-20명
        s = Regex("(\\d[\\d,]*)\\s?[~\\-]\\s?(\\d[\\d,]*)\\s?($COUNTER_ALT)").replace(s) { m ->
            val c = m.groupValues[3]
            readWithCounter(m.groupValues[1], c, withCounter = false) + "에서 " + readWithCounter(m.groupValues[2], c, true)
        }
        // arithmetic (before units so "3 x 4" is not read as 3배)
        s = Regex("(\\d)\\s?\\+\\s?(?=\\d)").replace(s) { m -> m.groupValues[1] + " 더하기 " }
        s = Regex("(\\d)\\s?[×*xX]\\s?(?=\\d)").replace(s) { m -> m.groupValues[1] + " 곱하기 " }
        s = Regex("(\\d)\\s?÷\\s?(?=\\d)").replace(s) { m -> m.groupValues[1] + " 나누기 " }
        s = Regex("(\\d)\\s?=\\s?(?=\\d)").replace(s) { m -> m.groupValues[1] + " 는 " }
        // 번 used as a label (3번 출구 → 삼 번 출구) and decades/age groups (20대 → 이십 대)
        s = Regex("(\\d+)\\s?번(\\s?)(?=$LABEL_NOUNS)").replace(s) { m -> KoreanNumbers.sino(m.groupValues[1]) + " 번" + m.groupValues[2] }
        s = Regex("(\\d*0)\\s?대(?![가-힣]*[을를이가]\\s?(?:타|몰|운전|주차|샀|구입))").replace(s) { m -> KoreanNumbers.sino(m.groupValues[1]) + " 대" }
        // ordinals
        s = Regex("(\\d+)\\s?번째").replace(s) { m -> (KoreanNumbers.parseIntSafe(m.groupValues[1])?.let { KoreanNumbers.ordinalPrefix(it) } ?: KoreanNumbers.sino(m.groupValues[1])) + " 번째" }
        s = Regex("(\\d+)\\s?째").replace(s) { m ->
            val n = KoreanNumbers.parseIntSafe(m.groupValues[1])
            val w = when (n) { 1 -> "첫"; 2 -> "둘"; 3 -> "셋"; 4 -> "넷"; else -> n?.let { KoreanNumbers.native(it, false) } ?: KoreanNumbers.sino(m.groupValues[1]) }
            w + "째"
        }
        s = Regex("제(\\d+)").replace(s) { m -> "제" + KoreanNumbers.sino(m.groupValues[1]) }
        // number + counter
        s = Regex("(\\d[\\d,]*(?:\\.\\d+)?)\\s?($COUNTER_ALT)").replace(s) { m -> readWithCounter(m.groupValues[1], m.groupValues[2], true) }
        s = Regex("(\\d)\\s?~\\s?(?=\\d)").replace(s) { m -> m.groupValues[1] + "에서 " }
        // remaining plain numbers
        s = Regex("\\d[\\d,]*(?:\\.\\d+)?").replace(s) { m -> readNumber(m.value) }
        return s
    }

    private fun hour(h: Int): String = if (h in 1..12) KoreanNumbers.native(h, true)!! else KoreanNumbers.sino(h.toLong())

    /** Plain number: commas, decimals, leading zeros. */
    private fun readNumber(raw: String): String {
        val n = raw.replace(",", "")
        if ('.' in n) {
            val (a, b) = n.split('.', limit = 2)
            return KoreanNumbers.decimal(a, b.filter { it.isDigit() })
        }
        if (n.length > 1 && n.startsWith("0")) return KoreanNumbers.digits(n)
        return KoreanNumbers.sino(n)
    }

    private fun readWithCounter(raw: String, counter: String, withCounter: Boolean): String {
        val n = raw.replace(",", "")
        val value = if ('.' in n || n.length > 1 && n.startsWith("0")) null else KoreanNumbers.parseIntSafe(n)
        val word = when {
            counter == "월" && value != null && value in 1..12 -> return if (withCounter) KoreanNumbers.month(value) else KoreanNumbers.sino(n)
            counter == "시" && value != null -> hour(value)
            counter in NATIVE_COUNTERS && value != null && value in 1..99 -> KoreanNumbers.native(value, true)!!
            else -> readNumber(raw)
        }
        return if (withCounter) "$word $counter" else word
    }

    // ------------------------------------------------------------------ English
    private fun english(s: String): String {
        if (s.none { it in 'A'..'Z' || it in 'a'..'z' }) return s
        return Regex("[A-Za-z]+(?:'[A-Za-z]+)?").replace(s) { m ->
            val w = m.value
            val known = if (w.length == 1) null else LOANWORDS[w.lowercase()] ?: ACRONYM_WORDS[w.lowercase()]?.takeIf { w.any { it.isUpperCase() } }
            if (known != null) return@replace " $known "
            // split camelCase (iPhone → i Phone, PDFViewer → PDF Viewer)
            var t = Regex("([a-z])([A-Z])").replace(w, "$1 $2")
            t = Regex("([A-Z]+)([A-Z][a-z])").replace(t, "$1 $2")
            " " + t.split(' ').filter { it.isNotEmpty() }.joinToString(" ") { englishWord(it) } + " "
        }
    }

    fun englishWord(w: String): String {
        val lower = w.lowercase()
        if (w.length == 1) return if (w[0].isUpperCase()) LETTERS.getValue(w[0]) else LOANWORDS[lower] ?: LETTERS.getValue(w[0].uppercaseChar())
        if (w.all { it in 'A'..'Z' }) {
            return MELO_EN_DICT[w] ?: ACRONYM_WORDS[lower] ?: LOANWORDS[lower]?.takeIf { w.length <= 3 } ?: spellAscii(w)
        }
        LOANWORDS[lower]?.let { return it }
        g2p.englishToHangul(lower.replace("'", ""))?.takeIf { it.isNotEmpty() && Hangul.isAllSyllables(it) }?.let { return it }
        g2p.englishToHangul(lower)?.takeIf { it.isNotEmpty() && Hangul.isAllSyllables(it) }?.let { return it }
        return spellAscii(w)
    }

    /** e-mail / domain parts: words read as words, separators as 점. */
    private fun readAddress(a: String): String =
        Regex("[A-Za-z]+|\\d+|[._-]").findAll(a).joinToString(" ") { t ->
            when {
                t.value == "." -> "점"
                t.value == "_" || t.value == "-" -> ""
                t.value[0].isDigit() -> KoreanNumbers.digits(t.value)
                t.value.length <= 3 -> spellAscii(t.value)
                else -> englishWord(t.value)
            }
        }

    private fun spellAscii(w: String): String {
        val sb = StringBuilder()
        for (c in w) {
            when {
                c.isLetter() && c.code < 128 -> sb.append(LETTERS[c.uppercaseChar()])
                c in '0'..'9' -> sb.append(KoreanNumbers.digits(c.toString()))
                c == '.' -> sb.append(" 점 ")
                c == '-' || c == '_' -> sb.append(" ")
            }
        }
        return sb.toString().trim()
    }

    // ------------------------------------------------------------------ symbols / punctuation
    private fun symbols(input: String): String {
        var s = input
        s = s.replace(Regex("\\.{2,}"), "…").replace(Regex("…+"), "…")
        s = s.replace("&", " 앤드 ").replace("@", " 앳 ")
        s = Regex("\\s\\+\\s").replace(s, " 플러스 ")
        s = Regex("[(\\[{「『〈《【〔]").replace(s, ", ")
        s = Regex("[)\\]}」』〉》】〕]").replace(s, ", ")
        s = Regex("[\"“”„‟«»`´]").replace(s, " ")
        s = Regex("[‘’']").replace(s, "")
        s = Regex("[;:]").replace(s, ",")
        s = Regex("[—–―]").replace(s, ", ")
        s = Regex("·").replace(s, ", ")
        s = Regex("(?<=[가-힣])\\s*~\\s*(?=[가-힣])").replace(s, "에서 ")
        s = Regex("[/\\\\|~^_*#=<>+]").replace(s, " ")
        s = Regex("(?<=[가-힣])-(?=[가-힣])").replace(s, " ")
        s = s.replace("-", " ")
        s = Regex("[!！]+").replace(s, "!")
        s = Regex("[?？]+").replace(s, "?")
        s = Regex("[?!]{2,}").replace(s) { m -> if ('?' in m.value) "?" else "!" }
        return s
    }

    private fun cleanup(input: String): String {
        val sb = StringBuilder(input.length)
        for (c in input) {
            when {
                Hangul.isSyllable(c) -> sb.append(c)
                c == ' ' || c == '\t' -> sb.append(' ')
                c in ",.!?…" -> sb.append(c)
                else -> sb.append(' ')
            }
        }
        var s = sb.toString()
        s = Regex("\\s+").replace(s, " ")
        s = Regex("\\s+([,.!?…])").replace(s, "$1")
        s = Regex(",+").replace(s, ",")
        s = Regex("[,]([.!?…])").replace(s, "$1")
        s = Regex("([.!?…])[,]+").replace(s, "$1")
        s = Regex("^[\\s,.!?…]+").replace(s, "")
        s = Regex("([,.!?…])(?=[가-힣])").replace(s, "$1 ")
        return s.trim()
    }

    companion object {
        private val URL = Regex("(https?://|www\\.)[^\\s가-힣]+")
        private val EMAIL = Regex("([A-Za-z0-9._%+-]+)@([A-Za-z0-9.-]+\\.[A-Za-z]{2,})")
        private val PHONE = Regex("(?<!\\d)(0\\d{1,2})[-. ](\\d{3,4})[-. ](\\d{4})(?!\\d)")
        private val DATE = Regex("(?<!\\d)(\\d{4})[-./](\\d{1,2})[-./](\\d{1,2})(?!\\d)\\.?")
        private val TIME = Regex("(?<!\\d)(\\d{1,2}):(\\d{2})(?::(\\d{2}))?(?!\\d)")
        private val ALNUM_WORDS = mapOf(
            "mp3" to "엠피스리", "mp4" to "엠피포", "5g" to "파이브지", "4g" to "포지", "3g" to "쓰리지", "3d" to "쓰리디",
            "2d" to "투디", "4k" to "포케이", "8k" to "에잇케이", "ps4" to "피에스포", "ps5" to "피에스파이브",
            "covid-19" to "코비드 일구", "k-pop" to "케이팝", "b2b" to "비투비", "b2c" to "비투씨", "1+1" to "원 플러스 원",
            "2+1" to "투 플러스 원", "24/7" to "이십사 칠", "iot" to "아이오티", "gpt-4" to "지피티 포",
        )
        private val ALNUM = Regex("(?i)(?<![A-Za-z0-9])(" + ALNUM_WORDS.keys.sortedByDescending { it.length }
            .joinToString("|") { Regex.escape(it) } + ")(?![A-Za-z0-9])")
        private const val LABEL_NOUNS = "출구|버스|선|타자|트랙|국도|게이트|좌석|문제|항목|방|채널|홀|레인|라인|타석|국|선수|후보|마을버스|승강장|플랫폼"
        private val IOTIZED = setOf(2, 3, 6, 7, 12, 17, 20) // ㅑ ㅒ ㅕ ㅖ ㅛ ㅠ ㅣ

        private val COMPAT_TO_LEAD: Map<Char, Char> = mapOf(
            'ㄱ' to 'ᄀ', 'ㄲ' to 'ᄁ', 'ㄴ' to 'ᄂ', 'ㄷ' to 'ᄃ', 'ㄸ' to 'ᄄ', 'ㄹ' to 'ᄅ', 'ㅁ' to 'ᄆ', 'ㅂ' to 'ᄇ',
            'ㅃ' to 'ᄈ', 'ㅅ' to 'ᄉ', 'ㅆ' to 'ᄊ', 'ㅇ' to 'ᄋ', 'ㅈ' to 'ᄌ', 'ㅉ' to 'ᄍ', 'ㅊ' to 'ᄎ', 'ㅋ' to 'ᄏ',
            'ㅌ' to 'ᄐ', 'ㅍ' to 'ᄑ', 'ㅎ' to 'ᄒ',
        )

        /** Counters read with native numerals (1–99). Everything else is Sino-Korean. */
        val NATIVE_COUNTERS = setOf(
            "개", "명", "마리", "살", "시", "시간", "달", "사람", "번", "배", "권", "대", "벌", "켤레", "그루", "송이",
            "잔", "병", "장", "채", "척", "통", "곳", "군데", "가지", "줄", "칸", "판", "바퀴", "방울", "쌍", "상자",
            "봉지", "그릇", "접시", "조각", "알", "끼", "차례", "걸음", "모금", "마디", "자루", "개비", "켤레", "포기",
            "다발", "묶음", "꼬치", "덩어리", "숟가락", "젓가락", "사발", "주먹",
        )

        private val SINO_COUNTERS = listOf(
            "개월", "개국", "개소", "개년", "개조", "개사", "주일", "주년", "년", "월", "일", "원", "층", "호", "회", "차", "등", "위", "세", "학년", "교시",
            "분", "초", "주", "점", "도", "명분", "인", "인분", "박", "km", "kg", "달러", "유로", "엔", "파운드",
            "만", "억", "조", "킬로", "미터", "센티", "그램", "리터", "퍼센트", "쪽", "페이지", "장면", "부", "편",
            "화", "기", "대대", "년대", "세기", "강", "과", "항", "조항", "번지", "동", "단계", "급", "배속", "타",
            "승", "패", "골", "점차", "표", "석", "칼로리", "와트", "볼트",
        )
        private val COUNTER_ALT: String = (NATIVE_COUNTERS + SINO_COUNTERS).distinct().sortedByDescending { it.length }
            .joinToString("|") { Regex.escape(it) }

        private val UNITS: Map<String, String> = linkedMapOf(
            "km/h" to "킬로미터 퍼 아워", "m/s" to "미터 퍼 세컨드", "kWh" to "킬로와트시", "mAh" to "밀리암페어시",
            "km²" to "제곱킬로미터", "m²" to "제곱미터", "m³" to "세제곱미터", "cm²" to "제곱센티미터",
            "km" to "킬로미터", "cm" to "센티미터", "mm" to "밀리미터", "nm" to "나노미터", "kg" to "킬로그램",
            "mg" to "밀리그램", "ml" to "밀리리터", "mL" to "밀리리터", "°C" to "도", "°F" to "화씨 도", "°" to "도",
            "GHz" to "기가헤르츠", "MHz" to "메가헤르츠", "kHz" to "킬로헤르츠", "Hz" to "헤르츠",
            "TB" to "테라바이트", "GB" to "기가바이트", "MB" to "메가바이트", "KB" to "킬로바이트", "kB" to "킬로바이트",
            "kW" to "킬로와트", "kcal" to "킬로칼로리", "cal" to "칼로리", "dB" to "데시벨", "ppm" to "피피엠",
            "m" to "미터", "g" to "그램", "L" to "리터", "l" to "리터", "W" to "와트", "V" to "볼트", "t" to "톤",
        )
        private val UNIT_RE = Regex("(\\d[\\d,]*(?:\\.\\d+)?)\\s?(" + UNITS.keys.sortedByDescending { it.length }
            .joinToString("|") { Regex.escape(it) } + ")(?![A-Za-z])(\\s?)")

        val LETTERS: Map<Char, String> = mapOf(
            'A' to "에이", 'B' to "비", 'C' to "씨", 'D' to "디", 'E' to "이", 'F' to "에프", 'G' to "지", 'H' to "에이치",
            'I' to "아이", 'J' to "제이", 'K' to "케이", 'L' to "엘", 'M' to "엠", 'N' to "엔", 'O' to "오", 'P' to "피",
            'Q' to "큐", 'R' to "알", 'S' to "에스", 'T' to "티", 'U' to "유", 'V' to "브이", 'W' to "더블유",
            'X' to "엑스", 'Y' to "와이", 'Z' to "제트",
        )

        /** MeloTTS ko_dictionary.english_dictionary (uppercase keys). */
        private val MELO_EN_DICT = mapOf(
            "KOREA" to "코리아", "IDOL" to "아이돌", "IT" to "아이티", "IQ" to "아이큐", "UP" to "업", "DOWN" to "다운",
            "PC" to "피씨", "CCTV" to "씨씨티비", "SNS" to "에스엔에스", "AI" to "에이아이", "CEO" to "씨이오",
        )

        /** Acronyms pronounced as words. */
        private val ACRONYM_WORDS = mapOf(
            "nasa" to "나사", "nato" to "나토", "unesco" to "유네스코", "unicef" to "유니세프", "opec" to "오펙",
            "kospi" to "코스피", "kosdaq" to "코스닥", "asean" to "아세안", "fifa" to "피파", "covid" to "코비드",
            "korea" to "코리아", "seoul" to "서울", "ok" to "오케이", "kakao" to "카카오", "naver" to "네이버",
            "lol" to "롤", "wifi" to "와이파이", "led" to "엘이디", "laser" to "레이저", "radar" to "레이더",
        )

        /** Common loanwords with their conventional Korean spelling (외래어 표기). */
        private val LOANWORDS: Map<String, String> = mapOf(
            "google" to "구글", "youtube" to "유튜브", "facebook" to "페이스북", "instagram" to "인스타그램",
            "twitter" to "트위터", "iphone" to "아이폰", "ipad" to "아이패드", "android" to "안드로이드",
            "samsung" to "삼성", "galaxy" to "갤럭시", "apple" to "애플", "microsoft" to "마이크로소프트",
            "windows" to "윈도우", "window" to "윈도우", "computer" to "컴퓨터", "internet" to "인터넷",
            "email" to "이메일", "mail" to "메일", "app" to "앱", "apps" to "앱", "smartphone" to "스마트폰",
            "phone" to "폰", "coffee" to "커피", "cafe" to "카페", "ok" to "오케이", "okay" to "오케이",
            "hello" to "헬로", "hi" to "하이", "bye" to "바이", "thank" to "땡큐", "thanks" to "땡스",
            "you" to "유", "i" to "아이", "wifi" to "와이파이", "bluetooth" to "블루투스", "online" to "온라인",
            "offline" to "오프라인", "program" to "프로그램", "service" to "서비스", "system" to "시스템",
            "data" to "데이터", "file" to "파일", "video" to "비디오", "music" to "뮤직", "game" to "게임",
            "news" to "뉴스", "team" to "팀", "project" to "프로젝트", "open" to "오픈", "voice" to "보이스",
            "openvoice" to "오픈보이스", "melo" to "멜로", "chatgpt" to "챗지피티", "gpt" to "지피티",
            "claude" to "클로드", "netflix" to "넷플릭스", "amazon" to "아마존", "tesla" to "테슬라",
            "kakaotalk" to "카카오톡", "naver" to "네이버", "kakao" to "카카오", "coupang" to "쿠팡",
            "hyundai" to "현대", "lg" to "엘지", "sk" to "에스케이", "kpop" to "케이팝", "bts" to "비티에스",
            "web" to "웹", "site" to "사이트", "website" to "웹사이트", "blog" to "블로그", "click" to "클릭",
            "download" to "다운로드", "upload" to "업로드", "update" to "업데이트", "login" to "로그인",
            "logout" to "로그아웃", "password" to "패스워드", "user" to "유저", "server" to "서버",
            "cloud" to "클라우드", "software" to "소프트웨어", "hardware" to "하드웨어", "chip" to "칩",
            "battery" to "배터리", "camera" to "카메라", "speaker" to "스피커", "mike" to "마이크",
            "microphone" to "마이크", "audio" to "오디오", "mp3" to "엠피스리", "pdf" to "피디에프",
            "office" to "오피스", "word" to "워드", "excel" to "엑셀", "powerpoint" to "파워포인트",
            "bus" to "버스", "taxi" to "택시", "hotel" to "호텔", "pizza" to "피자", "chicken" to "치킨",
            "beer" to "비어", "juice" to "주스", "cake" to "케이크", "bread" to "브레드", "ice" to "아이스",
            "cream" to "크림", "shopping" to "쇼핑", "mall" to "몰", "mart" to "마트", "center" to "센터",
            "studio" to "스튜디오", "concert" to "콘서트", "drama" to "드라마", "movie" to "무비",
            "show" to "쇼", "style" to "스타일", "fashion" to "패션", "design" to "디자인", "brand" to "브랜드",
            "marketing" to "마케팅", "business" to "비즈니스", "startup" to "스타트업", "meeting" to "미팅",
            "schedule" to "스케줄", "test" to "테스트", "level" to "레벨", "version" to "버전", "beta" to "베타",
            "alpha" to "알파", "plus" to "플러스", "pro" to "프로", "max" to "맥스", "mini" to "미니",
            "ultra" to "울트라", "smart" to "스마트", "tv" to "티브이", "radio" to "라디오", "robot" to "로봇",
            "model" to "모델", "deep" to "딥", "learning" to "러닝", "machine" to "머신", "chat" to "챗",
            "bot" to "봇", "free" to "프리", "premium" to "프리미엄", "event" to "이벤트", "sale" to "세일",
            "coupon" to "쿠폰", "point" to "포인트", "card" to "카드", "bank" to "뱅크", "pay" to "페이",
            "korean" to "코리안", "english" to "잉글리시", "america" to "아메리카", "london" to "런던",
            "paris" to "파리", "new" to "뉴", "york" to "욕", "tokyo" to "도쿄", "japan" to "재팬",
            "china" to "차이나", "seoul" to "서울", "busan" to "부산", "the" to "더", "and" to "앤드",
            "of" to "오브", "in" to "인", "to" to "투", "a" to "어", "is" to "이즈", "for" to "포", "with" to "위드",
        )

        @Volatile private var hanjaCache: Map<Char, Char>? = null
        private val hanjaMap: Map<Char, Char>
            get() = hanjaCache ?: synchronized(this) {
                hanjaCache ?: G2pKo.readResource("hanja.tsv").lines().mapNotNull { l ->
                    if (l.startsWith("#") || l.length < 3 || l[1] != '\t') null else l[0] to l[2]
                }.toMap().also { hanjaCache = it }
            }
    }
}
