package io.github.koreanvoice

import io.github.koreanvoice.text.KoreanNormalizer
import io.github.koreanvoice.text.KoreanNumbers
import io.github.koreanvoice.text.MeloFrontend
import io.github.koreanvoice.text.TextChunker
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertTrue

class NormalizerTest {
    private val n = KoreanNormalizer()

    @Test
    fun sinoNumbers() {
        assertEquals("영", KoreanNumbers.sino("0"))
        assertEquals("십", KoreanNumbers.sino("10"))
        assertEquals("백일", KoreanNumbers.sino("101"))
        assertEquals("천", KoreanNumbers.sino("1000"))
        assertEquals("만", KoreanNumbers.sino("10000"))
        assertEquals("십일만", KoreanNumbers.sino("110000"))
        assertEquals("일억", KoreanNumbers.sino("100000000"))
        assertEquals("이천이십사", KoreanNumbers.sino("2024"))
        assertEquals("일억이천삼백사십오만육천칠백팔십구", KoreanNumbers.sino("123456789"))
        assertEquals("스무", KoreanNumbers.native(20, true))
        assertEquals("스물한", KoreanNumbers.native(21, true))
        assertEquals("아흔아홉", KoreanNumbers.native(99, false))
    }

    private fun check(input: String, expected: String) = assertEquals(expected, n.normalizeLine(input), "input: $input")

    @Test
    fun counters() {
        check("사과 3개를 샀다.", "사과 세 개를 샀다.")
        check("학생 25명이 왔다.", "학생 스물다섯 명이 왔다.")
        check("3개월 동안", "삼 개월 동안")
        check("스무 살, 20살", "스무 살, 스무 살")
        check("오후 3시 30분에 만나요.", "오후 세 시 삼십 분에 만나요.")
        check("13시에 출발", "십삼 시에 출발")
        check("2024년 10월 6일", "이천이십사 년 시월 육 일")
        check("6월 25일", "유월 이십오 일")
        check("1시간 20분", "한 시간 이십 분")
        check("2번째 시도", "두 번째 시도")
        check("1번째", "첫 번째")
        check("3번 출구", "삼 번 출구")
        check("3~5개", "세에서 다섯 개")
        check("제3장", "제삼장")
        check("120개", "백이십 개")
    }

    @Test
    fun formats() {
        check("14:30", "십사 시 삼십 분")
        check("2024-03-01", "이천이십사 년 삼월 일 일")
        check("010-1234-5678", "공일공, 일이삼사, 오육칠팔")
        check("3.14", "삼점일사")
        check("50%", "오십 퍼센트")
        check("₩5,000", "오천 원")
        check("$20", "이십 달러")
        check("5km", "오 킬로미터")
        check("영하 -5°C", "영하 마이너스 오 도")
        check("1/2", "이분의 일")
        check("3:2로 이겼다", "삼 대 이로 이겼다")
        check("1,234,567명", "백이십삼만사천오백육십칠 명")
    }

    @Test
    fun englishAndSymbols() {
        check("AI 기술", "에이아이 기술")
        check("BTS 노래", "비티에스 노래")
        check("유튜브 YouTube", "유튜브 유튜브")
        check("iPhone 15", "아이폰 십오")
        check("GPU 성능", "지피유 성능")
        check("“안녕하세요”라고 말했다.", "안녕하세요 라고 말했다.")
        check("ㅋㅋㅋ 진짜?!", "크크크 진짜?")
        check("대한民國", "대한민국")
        check("(참고) 내용", "참고, 내용")
        check("A&B", "에이 앤드 비")
    }

    @Test
    fun outputAlphabetIsSafe() {
        val samples = listOf(
            "Hello, world! 오늘은 2024년 5월 5일 어린이날입니다 :) #행복 @friend",
            "가격은 $1,299.99이고 할인율은 15%p입니다.",
            "★☆♥ 이모지 😀 테스트 ~~ ... ??",
            "http://example.com 에 접속하세요. 메일: test.user@example.co.kr",
            "李舜臣 장군은 1545年에 태어났다.",
        )
        val fe = MeloFrontend()
        for (s in samples) {
            val out = n.normalize(s)
            assertTrue(out.all { it == ' ' || it in ",.!?…" || it in '가'..'힣' }, "unsafe chars in '$out'")
            fe.process(out) // must not throw
            println("$s\n  → $out")
        }
    }

    @Test
    fun chunking() {
        val text = n.normalize(
            "첫 문장입니다. 짧다. 이것은 아주 긴 문장으로서 여러 개의 절을 포함하고 있으며, 쉼표가 들어 있고, " +
                "중간에서 자연스럽게 나뉘어야 하는데 그렇지 않으면 모델이 한 번에 처리하기에는 너무 길어서 품질이 떨어질 수 있습니다.\n두 번째 문단!"
        )
        val chunks = TextChunker().split(text)
        chunks.forEach { println("${it.pauseMs}ms | ${it.text}") }
        assertTrue(chunks.all { it.text.length <= 90 })
        assertTrue(chunks.last().paragraphEnd)
        assertEquals(text.replace("\n", " ").replace(Regex("\\s+"), " ").replace(",", "").replace(" ", ""),
            chunks.joinToString(" ") { it.text }.replace(",", "").replace(" ", ""))
    }
}
