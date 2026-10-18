package io.github.koreanvoice.text

import java.math.BigInteger

/** Korean numeral readers (Sino-Korean and native Korean). */
object KoreanNumbers {
    private const val DIGITS = "영일이삼사오육칠팔구"
    private val SMALL_UNITS = arrayOf("", "십", "백", "천")
    private val BIG_UNITS = arrayOf("", "만", "억", "조", "경", "해")

    /** Sino-Korean cardinal: 10000 → 만, 12345 → 만이천삼백사십오, 100000000 → 일억. */
    fun sino(numStr: String): String {
        val s = numStr.trimStart('0')
        if (s.isEmpty()) return "영"
        if (s.length > 24) return digits(numStr)
        val groups = ArrayList<String>()
        var end = s.length
        while (end > 0) { groups.add(s.substring(maxOf(0, end - 4), end)); end -= 4 }
        val sb = StringBuilder()
        for (g in groups.indices.reversed()) {
            val chunk = groups[g].padStart(4, '0')
            if (chunk == "0000") continue
            val part = StringBuilder()
            for (k in 0 until 4) {
                val d = chunk[k] - '0'
                if (d == 0) continue
                val unit = SMALL_UNITS[3 - k]
                if (d == 1 && unit.isNotEmpty()) part.append(unit) else part.append(DIGITS[d]).append(unit)
            }
            // 만 alone (10000) is read 만, not 일만; 억/조 keep 일 (일억, 일조).
            if (g == 1 && part.toString() == "일") part.clear()
            sb.append(part).append(BIG_UNITS[g])
        }
        return sb.toString()
    }

    fun sino(n: Long): String = sino(n.toString())

    /** Digit-by-digit reading, 0 → 공 when [zeroAsGong] (phone numbers), else 영. */
    fun digits(s: String, zeroAsGong: Boolean = false): String {
        val sb = StringBuilder()
        for (c in s) if (c in '0'..'9') sb.append(if (c == '0' && zeroAsGong) '공' else DIGITS[c - '0'])
        return sb.toString()
    }

    private val NATIVE_ONES = arrayOf("", "하나", "둘", "셋", "넷", "다섯", "여섯", "일곱", "여덟", "아홉")
    private val NATIVE_ONES_MOD = arrayOf("", "한", "두", "세", "네", "다섯", "여섯", "일곱", "여덟", "아홉")
    private val NATIVE_TENS = arrayOf("", "열", "스물", "서른", "마흔", "쉰", "예순", "일흔", "여든", "아흔")

    /**
     * Native Korean numeral for 1..99 (null otherwise). [modifier] gives the form used before a
     * counter: 한 개, 두 명, 스무 살.
     */
    fun native(n: Int, modifier: Boolean): String? {
        if (n !in 1..99) return null
        val t = n / 10; val o = n % 10
        if (modifier && n == 20) return "스무"
        val tens = NATIVE_TENS[t]
        val ones = if (modifier) NATIVE_ONES_MOD[o] else NATIVE_ONES[o]
        return tens + ones
    }

    /** Ordinal with 번째/째: 1 → 첫, 2 → 두, 20 → 스무, 21 → 스물한. */
    fun ordinalPrefix(n: Int): String? = if (n == 1) "첫" else native(n, true)

    /** Month names with the customary irregular forms 유월/시월. */
    fun month(n: Int): String = when (n) { 6 -> "유월"; 10 -> "시월"; else -> sino(n.toLong()) + "월" }

    /** Decimal number "3.14" → 삼 점 일사. */
    fun decimal(intPart: String, frac: String): String = sino(intPart.ifEmpty { "0" }) + "점" + digits(frac)

    fun parseIntSafe(s: String): Int? = s.toBigIntegerOrNull()?.takeIf { it < BigInteger.valueOf(Int.MAX_VALUE.toLong()) }?.toInt()

    /** Port of g2pkk numerals.process_num, kept for exactness inside [G2pKo]. */
    fun g2pkProcessNum(numIn: String, sino: Boolean): String {
        val num = numIn.replace(",", "")
        if (num == "0") return "영"
        if (!sino && num == "20") return "스무"
        val names = "일이삼사오육칠팔구"
        val mods = arrayOf("한", "두", "세", "네", "다섯", "여섯", "일곱", "여덟", "아홉")
        val decs = arrayOf("열", "스물", "서른", "마흔", "쉰", "예순", "일흔", "여든", "아흔")
        fun dn(d: Char) = if (d in '1'..'9') names[d - '1'].toString() else ""
        val spelled = ArrayList<String>()
        for ((idx, digit) in num.withIndex()) {
            val i = num.length - idx - 1
            var name = ""
            if (sino) {
                if (i == 0) name = dn(digit)
                else if (i == 1) name = (dn(digit) + "십").replace("일십", "십")
            } else {
                if (i == 0) name = if (digit in '1'..'9') mods[digit - '1'] else ""
                else if (i == 1) name = if (digit in '1'..'9') decs[digit - '1'] else ""
            }
            if (digit == '0') {
                if (i % 4 == 0) {
                    val last = spelled.subList(maxOf(0, spelled.size - minOf(3, spelled.size)), spelled.size)
                    if (last.joinToString("") == "") { spelled.add(""); continue }
                } else { spelled.add(""); continue }
            }
            when (i) {
                2 -> name = (dn(digit) + "백").replace("일백", "백")
                3 -> name = (dn(digit) + "천").replace("일천", "천")
                4 -> name = (dn(digit) + "만").replace("일만", "만")
                5 -> name = (dn(digit) + "십").replace("일십", "십")
                6 -> name = (dn(digit) + "백").replace("일백", "백")
                7 -> name = (dn(digit) + "천").replace("일천", "천")
                8 -> name = dn(digit) + "억"
                9 -> name = dn(digit) + "십"
                10 -> name = dn(digit) + "백"
                11 -> name = dn(digit) + "천"
                12 -> name = dn(digit) + "조"
                13 -> name = dn(digit) + "십"
                14 -> name = dn(digit) + "백"
                15 -> name = dn(digit) + "천"
            }
            spelled.add(name)
        }
        return spelled.joinToString("")
    }
}
