package io.github.koreanvoice.expressive

import io.github.koreanvoice.audio.Dsp

/** Emotions with their generation settings (Chatterbox exaggeration / CFG / temperature) and prosody offsets. */
enum class Emotion(
    val label: String,
    val exaggeration: Float, val cfg: Float, val temperature: Float,
    val pitch: Double = 0.0, val tempo: Double = 1.0, val tone: Dsp.Tone = Dsp.Tone.NORMAL,
    /** Desired expressiveness of the reference clip (−1 calm … +1 lively) for prompt routing. */
    val arousal: Double = 0.0,
) {
    NEUTRAL("기본", 0.5f, 0.5f, 0.8f),
    CALM("차분", 0.3f, 0.5f, 0.7f, -0.3, 0.97, arousal = -0.7),
    HAPPY("기쁨", 0.8f, 0.4f, 0.85f, 0.7, 1.02, Dsp.Tone.BRIGHT, 0.6),
    EXCITED("신남", 1.1f, 0.35f, 0.9f, 1.2, 1.05, Dsp.Tone.BRIGHT, 1.0),
    SAD("슬픔", 0.6f, 0.35f, 0.75f, -0.8, 0.92, Dsp.Tone.DARK, -0.6),
    ANGRY("화남", 1.0f, 0.45f, 0.85f, 0.3, 1.04, Dsp.Tone.CLEAR, 0.9),
    SERIOUS("진지", 0.35f, 0.55f, 0.7f, -0.4, 0.98, arousal = -0.4),
    TENDER("다정", 0.55f, 0.4f, 0.8f, 0.2, 0.97, Dsp.Tone.WARM, -0.2),
    SURPRISED("놀람", 1.0f, 0.4f, 0.85f, 1.5, 1.03, arousal = 0.9),
    TENSE("긴장", 0.8f, 0.5f, 0.9f, 0.4, 1.06, arousal = 0.5),
    FEAR("두려움", 0.75f, 0.45f, 0.85f, 0.6, 1.04, Dsp.Tone.DARK, 0.4);

    companion object {
        private val alias = mapOf("보통" to NEUTRAL, "평온" to CALM, "행복" to HAPPY, "기쁘게" to HAPPY, "흥분" to EXCITED, "신나게" to EXCITED,
            "슬프게" to SAD, "우울" to SAD, "분노" to ANGRY, "화난" to ANGRY, "화나게" to ANGRY, "엄숙" to SERIOUS, "따뜻" to TENDER, "부드럽게" to TENDER,
            "놀라서" to SURPRISED, "깜짝" to SURPRISED, "불안" to TENSE, "무서움" to FEAR, "공포" to FEAR, "겁" to FEAR)
        fun of(s: String): Emotion? = entries.firstOrNull { it.label == s || it.name.equals(s, true) } ?: alias[s]
    }
}

/** Speaking style in effect for a span of text. */
data class Style(
    val emotion: Emotion = Emotion.NEUTRAL,
    /** 0.5 (subtle) … 1.6 (strong); scales the emotion's offsets from neutral. */
    val intensity: Float = 1f,
    /** Emotion was chosen explicitly by markup (auto-emotion will not override it). */
    val explicitEmotion: Boolean = false,
    val tempo: Double = 1.0,
    val pitch: Double = 0.0,
    val room: Dsp.Room = Dsp.Room.NONE,
    val wet: Double? = null,
    val state: Dsp.State = Dsp.State.NORMAL,
    val tone: Dsp.Tone? = null,
    val volumeDb: Double = 0.0,
) {
    fun params(base: CbParams): CbParams {
        val e = emotion; val n = Emotion.NEUTRAL; val k = intensity
        fun mix(a: Float, b: Float) = b + (a - b) * k
        // personal tuning (base) is the neutral point; emotions move relative to it
        return base.copy(
            exaggeration = (base.exaggeration + (mix(e.exaggeration, n.exaggeration) - n.exaggeration)).coerceIn(0.25f, 2f),
            cfgWeight = (base.cfgWeight + (mix(e.cfg, n.cfg) - n.cfg)).coerceIn(0.1f, 1f),
            temperature = (base.temperature + (mix(e.temperature, n.temperature) - n.temperature)).coerceIn(0.3f, 1.2f),
        )
    }
    val totalPitch get() = pitch + emotion.pitch * intensity
    val totalTempo get() = tempo * (1 + (emotion.tempo - 1) * intensity)
    val effectiveTone get() = tone ?: emotion.tone
    /** Key used to group chunks that can be synthesised together. */
    val synthKey get() = listOf(emotion, intensity, totalPitch, totalTempo, room, wet, state, effectiveTone, volumeDb).joinToString("|")
}

/**
 * Professional markup for the script text (all optional, Korean keywords):
 *
 *  [기쁨] [슬픔] [화남] [차분] [신남] [진지] [다정] [놀람] [긴장] [두려움]   emotion (until changed)
 *  [기쁨:강] [슬픔:약] [감정:기쁨:1.3]                                    with intensity
 *  [속도:1.2] [속도:빠르게|느리게]   [높이:+2] [높이:높게|낮게]   [볼륨:-3]
 *  [울림:방|스튜디오|욕실|무대|홀|성당|동굴|없음] [울림:홀:0.4]
 *  [상태:전화|라디오|확성기|멀리|가까이|속삭임|물속|로봇|기본]   [속삭임]  [전화] …
 *  [톤:밝게|따뜻하게|선명하게|어둡게|기본]
 *  [쉼:0.8]  (pause in seconds)   [쉼]  (0.6 s)
 *  [/감정] [/속도] … resets one setting,  [기본] resets everything
 *  {표기|읽는법}  reads the text as written after the bar, e.g. {NASA|나사}, {1등|일 등}
 *  [웃음] [한숨] [헛기침] [숨] [흐느낌]  non-verbal sounds (experimental)
 */
object Markup {
    sealed class Item
    data class Text(val text: String, val style: Style) : Item()
    data class Pause(val ms: Int) : Item()

    // literal brackets/braces are escaped everywhere: Android's ICU regex engine is stricter than Java's
    private val TAG = Regex("""\[([^\[\]\n]{1,40})\]""")
    private val READING = Regex("""\{([^\{\}|\n]*)\|([^\{\}\n]*)\}""")
    /** Non-verbal tags of the Chatterbox multilingual tokenizer. */
    val NONVERBAL = mapOf("웃음" to "[laughter]", "한숨" to "[sigh]", "헛기침" to "[cough]", "숨" to "[gasp]", "흐느낌" to "[cry]",
        "기침" to "[cough]", "허밍" to "[humming]", "속삭이듯" to "[whisper]")

    data class ParseResult(val items: List<Item>, val warnings: List<String>)

    fun hasMarkup(s: String) = TAG.containsMatchIn(s) || READING.containsMatchIn(s)

    /** Apply {표기|읽는법} and the user's pronunciation dictionary. */
    fun applyReadings(s: String, dictionary: Map<String, String> = emptyMap()): String {
        var t = READING.replace(s) { it.groupValues[2] }
        for ((k, v) in dictionary.entries.sortedByDescending { it.key.length }) if (k.isNotBlank()) t = t.replace(k, v)
        return t
    }

    fun parse(src: String, base: Style = Style(), dictionary: Map<String, String> = emptyMap()): ParseResult {
        val text = applyReadings(src, dictionary)
        val items = ArrayList<Item>()
        val warn = ArrayList<String>()
        var st = base
        var last = 0
        fun emit(s: String) { if (s.isNotEmpty()) items.add(Text(s, st)) }
        for (m in TAG.findAll(text)) {
            emit(text.substring(last, m.range.first))
            last = m.range.last + 1
            val body = m.groupValues[1].trim()
            val nv = NONVERBAL[body]
            if (nv != null) { items.add(Text(nv, st)); continue }
            val next = apply(st, base, body)
            if (next == null) {
                if (body.startsWith("쉼") || body.startsWith("pause", true)) {
                    val sec = body.substringAfter(':', "0.6").trim().removeSuffix("초").toDoubleOrNull() ?: 0.6
                    items.add(Pause((sec.coerceIn(0.0, 10.0) * 1000).toInt()))
                } else { warn.add("알 수 없는 태그: [${body}] — 그대로 읽지 않고 무시했습니다."); }
            } else st = next
        }
        emit(text.substring(last))
        // merge adjacent texts with identical style
        val merged = ArrayList<Item>()
        for (it in items) {
            val prev = merged.lastOrNull()
            if (it is Text && prev is Text && prev.style == it.style) merged[merged.size - 1] = Text(prev.text + it.text, it.style)
            else merged.add(it)
        }
        return ParseResult(merged, warn)
    }

    private fun intensityOf(s: String?): Float? = when (s?.trim()) {
        null, "" -> 1f
        "약", "약하게", "살짝" -> 0.6f
        "중", "보통" -> 1f
        "강", "강하게", "매우" -> 1.4f
        "최강", "아주강하게" -> 1.7f
        else -> s.toFloatOrNull()?.coerceIn(0.2f, 2f)
    }

    /** Returns the new style, or null if [body] is not a style tag. */
    fun apply(st: Style, base: Style, body: String): Style? {
        if (body == "기본" || body == "초기화" || body == "reset") return base
        if (body.startsWith("/")) {
            return when (body.substring(1).trim()) {
                "감정", "emotion" -> st.copy(emotion = base.emotion, intensity = base.intensity, explicitEmotion = false)
                "속도" -> st.copy(tempo = base.tempo)
                "높이" -> st.copy(pitch = base.pitch)
                "울림" -> st.copy(room = base.room, wet = base.wet)
                "상태" -> st.copy(state = base.state)
                "톤" -> st.copy(tone = base.tone)
                "볼륨" -> st.copy(volumeDb = base.volumeDb)
                else -> Emotion.of(body.substring(1).trim())?.let { st.copy(emotion = base.emotion, intensity = base.intensity, explicitEmotion = false) }
                    ?: Dsp.State.of(body.substring(1).trim())?.let { st.copy(state = base.state) }
            }
        }
        val parts = body.split(':').map { it.trim() }
        val key = parts[0]
        Emotion.of(key)?.let { e -> val k = intensityOf(parts.getOrNull(1)) ?: return null; return st.copy(emotion = e, intensity = k, explicitEmotion = true) }
        Dsp.State.of(key)?.let { s -> if (s != Dsp.State.NORMAL) return st.copy(state = s) }
        val v = parts.getOrNull(1) ?: return null
        return when (key) {
            "감정", "emotion" -> Emotion.of(v)?.let { e -> st.copy(emotion = e, intensity = intensityOf(parts.getOrNull(2)) ?: 1f, explicitEmotion = true) }
            "속도", "speed" -> when (v) {
                "빠르게" -> 1.15; "아주빠르게" -> 1.3; "느리게" -> 0.88; "아주느리게" -> 0.75; "보통" -> 1.0
                else -> v.removeSuffix("배").toDoubleOrNull()
            }?.let { st.copy(tempo = it.coerceIn(0.5, 2.0)) }
            "높이", "pitch" -> when (v) {
                "높게" -> 2.0; "아주높게" -> 4.0; "낮게" -> -2.0; "아주낮게" -> -4.0; "보통" -> 0.0
                else -> v.removePrefix("+").toDoubleOrNull()
            }?.let { st.copy(pitch = it.coerceIn(-12.0, 12.0)) }
            "울림", "reverb" -> Dsp.Room.of(v)?.let { r -> st.copy(room = r, wet = parts.getOrNull(2)?.toDoubleOrNull()?.coerceIn(0.0, 1.0)) }
            "상태", "state" -> Dsp.State.of(v)?.let { st.copy(state = it) }
            "톤", "tone" -> Dsp.Tone.of(v)?.let { st.copy(tone = it) }
            "볼륨", "volume" -> v.removePrefix("+").removeSuffix("dB").toDoubleOrNull()?.let { st.copy(volumeDb = it.coerceIn(-24.0, 12.0)) }
            else -> null
        }
    }
}

/**
 * Automatic per-sentence emotion from the text (Korean lexicon + punctuation + dialogue detection).
 * Heuristic by design: it only nudges the delivery; explicit markup always wins.
 */
object EmotionGuesser {
    private val lexicon: List<Pair<Emotion, List<String>>> = listOf(
        Emotion.HAPPY to listOf("기쁘", "기뻐", "행복", "좋아", "좋았", "신나", "최고", "축하", "고마", "감사", "다행", "웃었", "웃으", "설레", "반가"),
        Emotion.EXCITED to listOf("대박", "와아", "야호", "만세", "짱", "드디어", "해냈"),
        Emotion.SAD to listOf("슬프", "슬퍼", "눈물", "울었", "울고", "그리워", "그립", "외로", "아프", "아팠", "이별", "안타깝", "미안", "죄송", "서럽", "쓸쓸", "떠나"),
        Emotion.ANGRY to listOf("화나", "화가", "짜증", "분노", "용서", "닥쳐", "미쳤", "그만해", "뭐하는", "어이없", "열받", "괘씸"),
        Emotion.SURPRISED to listOf("깜짝", "놀라", "놀랐", "세상에", "어머", "헐", "설마", "믿을 수 없", "말도 안"),
        Emotion.FEAR to listOf("무서", "두려", "겁이", "겁나", "떨리", "소름", "도망"),
        Emotion.TENDER to listOf("괜찮아", "걱정 마", "사랑해", "보고 싶", "힘내", "토닥", "수고했"),
        Emotion.TENSE to listOf("서둘러", "빨리", "위험", "조심", "급해", "큰일"),
    )

    data class Guess(val emotion: Emotion, val intensity: Float)

    fun guess(sentence: String, inDialogue: Boolean): Guess {
        val s = sentence.trim()
        var best: Emotion? = null; var score = 0.0
        for ((e, words) in lexicon) {
            val c = words.count { s.contains(it) }.toDouble()
            if (c > score) { score = c; best = e }
        }
        val bang = s.count { it == '!' }
        val question = s.endsWith("?")
        var emo = best ?: when {
            bang >= 2 -> Emotion.EXCITED
            bang == 1 && inDialogue -> Emotion.SURPRISED.takeIf { question } ?: Emotion.HAPPY
            else -> Emotion.NEUTRAL
        }
        if (emo == Emotion.HAPPY && bang >= 2) emo = Emotion.EXCITED
        var k = when {
            best == null && emo == Emotion.NEUTRAL -> 1f
            score >= 2 || bang >= 1 -> 1.15f
            else -> 0.85f
        }
        if (inDialogue) k *= 1.2f else k *= 0.75f     // narration is delivered with restraint, dialogue with more colour
        return Guess(emo, k.coerceIn(0.5f, 1.5f))
    }

    /** Splits text into (segment, isDialogue) by quotation marks. */
    fun dialogueSpans(text: String): List<Pair<String, Boolean>> {
        val out = ArrayList<Pair<String, Boolean>>()
        val sb = StringBuilder(); var inQ = false
        for (ch in text) {
            val open = ch == '“' || (ch == '"' && !inQ) || ch == '「' || ch == '『'
            val close = ch == '”' || (ch == '"' && inQ) || ch == '」' || ch == '』'
            if (!inQ && open) { if (sb.isNotBlank()) out.add(sb.toString() to false); sb.clear(); inQ = true; sb.append(ch); continue }
            if (inQ && close) { sb.append(ch); out.add(sb.toString() to true); sb.clear(); inQ = false; continue }
            sb.append(ch)
        }
        if (sb.isNotBlank()) out.add(sb.toString() to inQ)
        return out
    }
}
