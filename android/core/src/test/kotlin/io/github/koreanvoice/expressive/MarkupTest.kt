package io.github.koreanvoice.expressive

import io.github.koreanvoice.audio.Dsp
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertTrue

class MarkupTest {
    private fun texts(r: Markup.ParseResult) = r.items.filterIsInstance<Markup.Text>()

    @Test
    fun parsesStylesPausesReadingsAndResets() {
        val r = Markup.parse("처음 문장. [기쁨:강]정말 좋아요! [쉼:1.5][속도:1.2][높이:-2][울림:홀:0.4][전화]여보세요? [/상태][/감정]보통. {NASA|나사}에 갔다. [기본]끝.")
        val t = texts(r)
        assertEquals(Emotion.NEUTRAL, t[0].style.emotion)
        assertEquals(Emotion.HAPPY, t[1].style.emotion); assertEquals(1.4f, t[1].style.intensity); assertTrue(t[1].style.explicitEmotion)
        assertTrue(r.items.any { it is Markup.Pause && it.ms == 1500 })
        val call = t.first { it.text.contains("여보세요") }.style
        assertEquals(1.2, call.tempo); assertEquals(-2.0, call.pitch); assertEquals(Dsp.Room.HALL, call.room); assertEquals(0.4, call.wet); assertEquals(Dsp.State.PHONE, call.state)
        val plain = t.first { it.text.contains("보통") }.style
        assertEquals(Dsp.State.NORMAL, plain.state); assertEquals(Emotion.NEUTRAL, plain.emotion); assertEquals(1.2, plain.tempo)
        assertTrue(t.any { it.text.contains("나사에 갔다") })
        assertEquals(Style(), t.last().style)
        assertTrue(r.warnings.isEmpty())
    }

    @Test
    fun unknownTagsWarnAndNonverbalMapsToModelTags() {
        val r = Markup.parse("안녕 [뭐지] 하하 [웃음] 그래.")
        assertEquals(1, r.warnings.size)
        assertTrue(texts(r).joinToString("") { it.text }.contains("[laughter]"))
    }

    @Test
    fun dictionaryAndEmotionGuess() {
        assertEquals("지피티는 일등", Markup.applyReadings("GPT는 1등", mapOf("GPT" to "지피티", "1등" to "일등")))
        val spans = EmotionGuesser.dialogueSpans("그가 말했다. “정말 기뻐요!” 그리고 떠났다.")
        assertEquals(3, spans.size); assertTrue(spans[1].second)
        assertEquals(Emotion.HAPPY, EmotionGuesser.guess("정말 기뻐요!", true).emotion)
        assertEquals(Emotion.SAD, EmotionGuesser.guess("눈물이 났다.", false).emotion)
        assertEquals(Emotion.NEUTRAL, EmotionGuesser.guess("회의는 세 시에 시작합니다.", false).emotion)
        val p = Style(Emotion.EXCITED, 1.4f).params(CbParams())
        assertTrue(p.exaggeration > 0.9f && p.cfgWeight < 0.5f)
    }

    @Test
    fun endingTags() {
        val r = Markup.parse("[말끝:올림]정말요? [말끝:길게]그래요. [/말끝]끝.")
        val t = texts(r)
        assertEquals(2.5, t[0].style.endShift); assertEquals(1.35, t[1].style.endStretch); assertEquals(2.5, t[1].style.endShift)
        assertEquals(0.0, t[2].style.endShift); assertEquals(1.0, t[2].style.endStretch)
        assertEquals(-1.5, texts(Markup.parse("[말끝:-1.5]네.")).first().style.endShift)
        assertTrue(r.warnings.isEmpty())
    }
}
