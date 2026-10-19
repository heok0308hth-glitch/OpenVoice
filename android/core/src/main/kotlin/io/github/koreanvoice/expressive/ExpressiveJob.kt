package io.github.koreanvoice.expressive

import io.github.koreanvoice.audio.AudioTools
import io.github.koreanvoice.audio.Dsp
import io.github.koreanvoice.audio.Mp3Encoder
import io.github.koreanvoice.audio.Psola
import io.github.koreanvoice.text.KoreanNormalizer
import io.github.koreanvoice.text.TextChunker
import java.io.File
import java.io.OutputStream
import java.security.MessageDigest
import kotlin.math.max
import kotlin.math.pow

/**
 * Long-text expressive synthesis (up to 100,000 characters):
 *  markup → per-span style → dialogue/narration detection → (optional) automatic emotion per sentence
 *  → Korean normalisation → chunking → Chatterbox synthesis with emotion-matched reference clips
 *  (cached per chunk on disk, resumable, automatic retry of garbled chunks)
 *  → assembly on an overlap-add timeline (reverb tails ring into the pauses), personal tuning,
 *  tone/state/room effects, loudness levelling, background bed with ducking, limiter → MP3/WAV.
 */
class ExpressiveJob(
    private val engine: CbEngine,
    private val bank: VoiceBank,
    cacheRoot: File,
    private val text: String,
    private val opt: Options = Options(),
) {
    data class Options(
        val base: Style = Style(),
        val autoEmotion: Boolean = true,
        /** Use the reference clip whose expressiveness matches each emotion. */
        val emotionRouting: Boolean = true,
        val dictionary: Map<String, String> = emptyMap(),
        val pauseScale: Float = 1f,
        val background: FloatArray? = null,
        val backgroundDb: Double = -22.0,
        /** Gentle broadcast-style compression before the limiter. */
        val mastering: Boolean = false,
        /** Retry chunks whose length is implausible for their text (garbled / cut off). */
        val retries: Int = 2,
        val maxChunkChars: Int = 70,
        /** Apply the voice's auto-learned pitch / colour / tempo / room. */
        val useTuning: Boolean = true,
        /** Run T3 and the decoder concurrently (pays off on phones with many big cores). */
        val pipelined: Boolean = false,
        /** Reproduce the original recordings' microphone colour, room and background (from auto-learning). */
        val recreateEnvironment: Boolean = false,
    ) {
        fun key() = listOf(base, autoEmotion, emotionRouting, dictionary.toSortedMap(), maxChunkChars, useTuning).toString()
    }

    sealed class Entry
    data class Chunk(val text: String, val style: Style, val pauseMs: Int, val dialogue: Boolean) : Entry()
    data class Pause(val ms: Int) : Entry()

    data class Progress(val done: Int, val total: Int, val elapsedMs: Long, val etaMs: Long, val currentText: String)
    class CancelledException : RuntimeException("취소됨")

    private val normalizer = KoreanNormalizer()
    val warnings = ArrayList<String>()

    val plan: List<Entry> by lazy { buildPlan() }
    val chunks: List<Chunk> get() = plan.filterIsInstance<Chunk>()
    val jobDir: File by lazy { File(cacheRoot, "xjob_" + key()).apply { mkdirs() } }

    private fun key(): String {
        val md = MessageDigest.getInstance("SHA-256")
        md.update(text.toByteArray()); md.update(opt.key().toByteArray())
        md.update(bank.dir.name.toByteArray()); md.update(bank.generation.toString().toByteArray())
        md.update(VoiceBank.tuningJson(bank.tuning).toString().toByteArray())
        md.update("x2".toByteArray())
        return md.digest().take(10).joinToString("") { "%02x".format(it) }
    }

    // ------------------------------------------------------------------ planning

    private fun buildPlan(): List<Entry> {
        val parsed = Markup.parse(text, opt.base, opt.dictionary)
        warnings.addAll(parsed.warnings)
        val chunker = TextChunker(maxChars = opt.maxChunkChars, mergeUpTo = 55, minChars = 12)
        val out = ArrayList<Entry>()
        for (item in parsed.items) when (item) {
            is Markup.Pause -> out.add(Pause(item.ms))
            is Markup.Text -> {
                // paragraphs stay separate; within them, dialogue (in quotes) and narration are styled differently
                for ((span, dialogue) in EmotionGuesser.dialogueSpans(item.text)) {
                    val norm = normalizeKeepingTags(span)
                    if (norm.isBlank()) continue
                    for (c in chunker.split(norm)) {
                        if (c.text.replace(NV_TAG, "").none { it in '가'..'힣' }) {
                            if (NV_TAG.containsMatchIn(c.text)) out.add(Chunk(c.text, item.style, c.pauseMs, dialogue))
                            continue
                        }
                        var st = item.style
                        if (opt.autoEmotion && !st.explicitEmotion) {
                            val g = EmotionGuesser.guess(c.text, dialogue)
                            st = st.copy(emotion = g.emotion, intensity = g.intensity)
                        }
                        out.add(Chunk(c.text, st, c.pauseMs, dialogue))
                    }
                }
            }
        }
        return out
    }

    /** Normalise Korean text but keep Chatterbox non-verbal tags (the normaliser would strip brackets). */
    private fun normalizeKeepingTags(s: String): String {
        val sb = StringBuilder(); var last = 0
        for (m in NV_TAG.findAll(s)) {
            sb.append(normalizer.normalize(s.substring(last, m.range.first))).append(' ').append(m.value).append(' ')
            last = m.range.last + 1
        }
        sb.append(normalizer.normalize(s.substring(last)))
        return sb.toString().replace(Regex("[ \\t]+"), " ").trim()
    }

    // ------------------------------------------------------------------ rendering

    private fun chunkFile(i: Int) = File(jobDir, "c_%05d.pcm".format(i))
    fun completedChunks() = chunks.indices.count { chunkFile(it).exists() }

    private val voiceCache = HashMap<String, CbVoice>()
    @Synchronized private fun voiceFor(st: Style): CbVoice {
        val seg = if (opt.emotionRouting) bank.segmentForArousal(st.emotion.arousal * st.intensity) else null
        val key = seg?.id ?: "default"
        return voiceCache.getOrPut(key) { bank.compose(seg) }
    }

    fun render(isCancelled: () -> Boolean = { false }, onProgress: (Progress) -> Unit = {}) {
        jobDir.mkdirs()                                  // may have been removed by deleteCache()
        val cs = chunks
        val todo = cs.indices.filter { !chunkFile(it).exists() }
        val totalChars = cs.sumOf { it.text.length }.coerceAtLeast(1)
        val charsBefore = cs.indices.filter { chunkFile(it).exists() }.sumOf { cs[it].text.length }
        var charsDone = 0
        val start = System.currentTimeMillis()
        onProgress(Progress(cs.size - todo.size, cs.size, 0, -1, ""))
        var done = cs.size - todo.size
        fun finished(i: Int, audio: FloatArray) {
            AudioTools.writePcm16(chunkFile(i), audio)
            done++; charsDone += cs[i].text.length
            val el = System.currentTimeMillis() - start
            onProgress(Progress(done, cs.size, el, (el.toDouble() / charsDone * (totalChars - charsBefore - charsDone)).toLong(), cs[i].text))
        }
        if (!opt.pipelined) {
            for (i in todo) {
                if (isCancelled()) throw CancelledException()
                val (voice, toks) = tokensFor(cs[i], i, isCancelled)
                finished(i, engine.decode(toks, voice))
            }
            return
        }
        // two-stage pipeline: T3 (text → speech tokens) of chunk i+1 runs while S3Gen decodes chunk i
        val queue = java.util.concurrent.ArrayBlockingQueue<Triple<Int, CbVoice, LongArray>>(2)
        val error = java.util.concurrent.atomic.AtomicReference<Throwable?>(null)
        val producer = Thread({
            try {
                for (i in todo) {
                    if (isCancelled() || error.get() != null) break
                    val (v, t) = tokensFor(cs[i], i, isCancelled)
                    while (!queue.offer(Triple(i, v, t), 200, java.util.concurrent.TimeUnit.MILLISECONDS)) if (isCancelled() || error.get() != null) return@Thread
                }
            } catch (t: Throwable) { error.compareAndSet(null, t) }
            finally { while (!queue.offer(Triple(-1, CbVoice(FloatArray(0), LongArray(0), LongArray(0), FloatArray(0), FloatArray(0)), LongArray(0)), 200, java.util.concurrent.TimeUnit.MILLISECONDS)) if (error.get() != null) break }
        }, "t3-stage")
        producer.start()
        try {
            while (true) {
                val item = queue.poll(200, java.util.concurrent.TimeUnit.MILLISECONDS)
                if (item == null) { if (error.get() != null || isCancelled() || !producer.isAlive && queue.isEmpty()) break; continue }
                if (item.first < 0) break
                finished(item.first, engine.decode(item.third, item.second))
                if (isCancelled()) break
            }
        } finally { producer.join(30_000) }
        error.get()?.let { throw it }
        if (isCancelled()) throw CancelledException()
    }

    /**
     * T3 for one chunk, retrying with another random path when the token count is implausible for the
     * text (garbled / cut off / babbling) — judged before decoding, so bad tries cost no decoder time.
     */
    fun tokensFor(c: Chunk, index: Int, isCancelled: () -> Boolean = { false }): Pair<CbVoice, LongArray> {
        val voice = voiceFor(c.style)
        val p0 = c.style.params(bank.tuning.params)
        val syl = c.text.count { it in '가'..'힣' }.coerceAtLeast(1)
        // plausible Korean speaking rate: ~3–12 syllables per second (non-verbal tags add time); 25 tokens = 1 s
        val lo = syl / 12.0; val hi = syl / 2.8 + 1.5 * NV_TAG.findAll(c.text).count()
        var best = LongArray(0); var bestBad = Double.MAX_VALUE
        for (attempt in 0..opt.retries) {
            if (isCancelled()) throw CancelledException()
            val seed = (c.text.hashCode().toLong() shl 8) + index * 31L + attempt * 7919L
            val (toks, _) = engine.generate(c.text, voice, p0.copy(seed = seed, maxNewTokens = (syl * 12 + 60).coerceAtMost(1000)), isCancelled)
            val sec = toks.size / 25.0
            val bad = when { sec < lo -> (lo - sec) / lo; sec > hi -> (sec - hi) / hi; else -> 0.0 }
            if (bad < bestBad) { bestBad = bad; best = toks }
            if (bad == 0.0) break
        }
        return voice to best
    }

    /** Synthesise one chunk (T3 with retries + decoder). */
    fun synthChunk(c: Chunk, index: Int, isCancelled: () -> Boolean = { false }): FloatArray {
        val (v, t) = tokensFor(c, index, isCancelled)
        return engine.decode(t, v)
    }

    // ------------------------------------------------------------------ assembly

    fun assembleMp3(out: OutputStream, bitrateKbps: Int = 96, id3: Mp3Encoder.Id3Info? = null) {
        Mp3Encoder(out, SR, bitrateKbps, id3 = id3).use { enc -> assemble { enc.encode(it) } }
    }

    fun assembleWav(file: File) {
        val parts = ArrayList<FloatArray>()
        assemble { parts.add(it) }
        val all = FloatArray(parts.sumOf { it.size }); var p = 0
        for (a in parts) { System.arraycopy(a, 0, all, p, a.size); p += a.size }
        AudioTools.writeWav(file, all, SR)
    }

    /** Overlap-add timeline that streams finished audio to [sink]. */
    private class Timeline(private val sink: (FloatArray) -> Unit, private val post: (FloatArray) -> FloatArray) {
        private var buf = FloatArray(SR * 30)
        private var origin = 0L        // absolute sample index of buf[0]
        var cursor = 0L; private set
        fun add(x: FloatArray, at: Long = cursor) {
            val off = (at - origin).toInt()
            if (off + x.size > buf.size) buf = buf.copyOf(max(buf.size * 2, off + x.size + SR))
            for (i in x.indices) buf[off + i] += x[i]
        }
        fun advance(n: Int) { cursor += n; flush(cursor) }
        private fun flush(upTo: Long) {
            val n = (upTo - origin).toInt()
            if (n < SR * 2) return
            sink(post(buf.copyOf(n)))
            // a long pause can move the cursor past the end of the buffer
            buf = if (n >= buf.size) FloatArray(SR * 30) else buf.copyOfRange(n, buf.size).copyOf(max(buf.size - n, SR * 30))
            origin = upTo
        }
        fun finish(tail: Int) {
            val end = (cursor - origin).toInt() + tail
            val n = end.coerceAtMost(buf.size)
            if (n > 0) sink(post(buf.copyOf(n)))
        }
    }

    private fun assemble(sink: (FloatArray) -> Unit) {
        val t = if (opt.useTuning) bank.tuning else VoiceBank.Tuning()
        val bg = opt.background?.let { Dsp.BackgroundMixer(it, SR, opt.backgroundDb) }
        val env = if (opt.recreateEnvironment) t.env else null
        val envFir = env?.eq?.takeIf { e -> e.any { kotlin.math.abs(it) > 0.2 } }?.let { Dsp.designEq(it, SR) }
        val envRoom = env?.let { Dsp.Room.of(it.room) } ?: Dsp.Room.NONE
        val roomTone = env?.noiseDb?.let { db -> bank.roomTone()?.takeIf { it.size > SR }?.let { Dsp.BackgroundMixer(it, SR, db, 0.0) } }
        val post: (FloatArray) -> FloatArray = { x ->
            var y = roomTone?.mix(x) ?: x
            y = bg?.mix(y) ?: y
            if (opt.mastering) y = Dsp.compress(y, SR, -20.0, 2.5)
            AudioTools.limit(y); y
        }
        val tl = Timeline(sink, post)
        tl.add(AudioTools.silence(SR, 250)); tl.advance(SR / 4)
        var prevGain = -1f
        var ci = 0
        var maxTail = 0
        for (e in plan) {
            when (e) {
                is Pause -> tl.advance(e.ms * SR / 1000)
                is Chunk -> {
                    val f = chunkFile(ci); ci++
                    check(f.exists()) { "chunk ${ci - 1} not rendered" }
                    var x = AudioTools.trimSilence(AudioTools.readPcm16(f), SR)
                    if (x.isEmpty()) continue
                    val st = e.style
                    val tempo = st.totalTempo * t.tempo
                    val pitch = st.totalPitch + t.pitch
                    x = Psola.shapeEndings(x, SR, st.endShift + t.endShift, st.endStretch * t.endStretch)
                    x = Dsp.tempoPitch(x, SR, tempo, pitch)
                    x = Dsp.tilt(x, SR, t.tilt)
                    x = Dsp.tone(x, SR, st.effectiveTone)
                    var g = AudioTools.normalizeGain(x, SR)
                    if (prevGain > 0) g = 0.6f * g + 0.4f * prevGain
                    prevGain = g
                    val vol = (g * 10.0.pow(st.volumeDb / 20)).toFloat()
                    for (k in x.indices) x[k] *= vol
                    AudioTools.fade(x, SR)
                    val speechLen = x.size
                    if (t.wet > 0.005) x = Dsp.reverb(x, SR, Dsp.Room.ROOM, t.wet, 0.3)
                    if (env != null) x = Dsp.applyEnvironment(x, SR, null, envRoom, env.wet, envFir)
                    x = Dsp.state(x, SR, st.state)
                    if (st.room != Dsp.Room.NONE) x = Dsp.reverb(x, SR, st.room, st.wet ?: st.room.defaultWet, 2.5)
                    tl.add(x)
                    maxTail = max(0, x.size - speechLen)
                    tl.advance(speechLen + (e.pauseMs * opt.pauseScale / tempo).toInt() * SR / 1000)
                }
            }
        }
        tl.finish(maxTail + SR / 4)
    }

    fun deleteCache() { jobDir.deleteRecursively() }

    companion object {
        const val SR = 24000
        val NV_TAG = Regex("""\[(laughter|sigh|cough|gasp|cry|humming|whisper)\]""")
    }
}
