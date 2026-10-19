package io.github.koreanvoice.expressive

import io.github.koreanvoice.audio.AudioTools
import io.github.koreanvoice.audio.Resampler
import io.github.koreanvoice.audio.VoiceAnalysis
import org.json.JSONArray
import org.json.JSONObject
import java.io.DataInputStream
import java.io.DataOutputStream
import java.io.File
import kotlin.math.max
import kotlin.math.min
import kotlin.math.sqrt

/**
 * A learned voice that keeps growing: every audio/video added is cut into clean single-speaker
 * segments; the voice identity is the quality-weighted average of all segments (more material →
 * steadier timbre), and each segment stays available as a style reference so that emotions can be
 * delivered from the clip that best matches them. Auto-learning results (tuned settings and
 * identity corrections) are stored alongside and survive later additions.
 */
class VoiceBank(val dir: File) {
    class Segment(
        val id: String,
        val source: String,
        val seconds: Double,
        val quality: Int,
        val ve: FloatArray,
        val xvec: FloatArray,
        val profile: VoiceAnalysis.Profile,
    ) {
        val audioFile get() = "seg_$id.pcm"
        val voiceFile get() = "seg_$id.cbv"
    }

    /** Result of auto-learning (see [AutoTuner]). */
    data class Tuning(
        val params: CbParams = CbParams(),
        val pitch: Double = 0.0,
        val tilt: Double = 0.0,
        val tempo: Double = 1.0,
        val wet: Double = 0.0,
        val veBlend: Double = 0.0,
        val promptSegment: String? = null,
        val veCorrection: FloatArray? = null,
        val xvCorrection: FloatArray? = null,
        val history: List<Double> = emptyList(),
        val iterations: Int = 0,
        /** Sentence-ending (말끝) shaping: final pitch glide in semitones and lengthening factor. */
        val endShift: Double = 0.0,
        val endStretch: Double = 1.0,
        /** Recording-environment reproduction (applied only when the user enables it). */
        val env: Environment? = null,
    )

    /** How the original recordings sound as a recording: microphone colour, room and background. */
    data class Environment(
        /** EQ correction per LTAS band (dB). */
        val eq: DoubleArray,
        val room: String = "NONE",
        val wet: Double = 0.0,
        /** Room-tone level relative to the speech level (dB), or null for no background. */
        val noiseDb: Double? = null,
        /** Environment similarity (0..1) without / with reproduction, measured during auto-learning. */
        val before: Double = Double.NaN,
        val after: Double = Double.NaN,
    )

    /** One auto-learning round's best voice ("generation"), kept so it can be listened to and chosen. */
    data class Generation(val round: Int, val tuning: Tuning, val score: Double, val criteria: Double, val ecapa: Double, val sample: String)

    var name: String = dir.name
    val segments = ArrayList<Segment>()
    var tuning: Tuning = Tuning()
    /** Generations of the latest auto-learning run. */
    val generations = ArrayList<Generation>()
    /** Index into [generations] currently in use (-1 = the tuning was not taken from a generation list). */
    var selectedGeneration: Int = -1
    /** Number of learning sessions (additions + auto-learning runs). */
    var generation: Int = 0
    val createdAt: Long

    init {
        dir.mkdirs()
        val f = File(dir, "bank.json")
        if (f.exists()) {
            val o = JSONObject(f.readText())
            name = o.optString("name", name); generation = o.optInt("generation", 0); createdAt = o.optLong("createdAt", System.currentTimeMillis())
            val arr = o.getJSONArray("segments")
            for (i in 0 until arr.length()) {
                val s = arr.getJSONObject(i)
                segments.add(Segment(s.getString("id"), s.getString("source"), s.getDouble("seconds"), s.getInt("quality"),
                    floats(s.getJSONArray("ve")), floats(s.getJSONArray("xv")),
                    VoiceAnalysis.Profile.fromList(doubles(s.getJSONArray("profile")))))
            }
            o.optJSONObject("tuning")?.let { tuning = tuningFrom(it) }
            o.optJSONArray("generations")?.let { a ->
                for (i in 0 until a.length()) { val g = a.getJSONObject(i)
                    generations.add(Generation(g.getInt("round"), tuningFrom(g.getJSONObject("tuning")), g.optDouble("score"), g.optDouble("criteria"), g.optDouble("ecapa"), g.getString("sample"))) }
            }
            selectedGeneration = o.optInt("selectedGeneration", -1)
        } else createdAt = System.currentTimeMillis()
    }

    val totalSeconds get() = segments.sumOf { it.seconds }

    fun save() {
        val o = JSONObject()
        o.put("name", name); o.put("generation", generation); o.put("createdAt", createdAt); o.put("version", 2)
        o.put("segments", JSONArray().apply {
            for (s in segments) put(JSONObject().apply {
                put("id", s.id); put("source", s.source); put("seconds", s.seconds); put("quality", s.quality)
                put("ve", JSONArray(s.ve.map { it.toDouble() })); put("xv", JSONArray(s.xvec.map { it.toDouble() }))
                put("profile", JSONArray(s.profile.toList().map { if (it.isNaN()) 0.0 else it }))
            })
        })
        o.put("tuning", tuningJson(tuning))
        o.put("generations", JSONArray().apply { for (g in generations) put(JSONObject().apply {
            put("round", g.round); put("tuning", tuningJson(g.tuning)); put("score", finite(g.score)); put("criteria", finite(g.criteria))
            put("ecapa", finite(g.ecapa)); put("sample", g.sample) }) })
        o.put("selectedGeneration", selectedGeneration)
        val tmp = File(dir, "bank.json.tmp"); tmp.writeText(o.toString()); tmp.renameTo(File(dir, "bank.json"))
    }

    // ------------------------------------------------------------------ learning from audio / video

    data class IngestReport(val found: Int, val kept: Int, val rejectedNoise: Int, val rejectedSpeaker: Int, val keptSeconds: Double, val messages: List<String>)

    /** A clean speech piece found in a recording, before speaker selection. */
    class Candidate(val audio: ShortArray, val quality: Int, val ve: FloatArray, val xv: FloatArray)

    /** Collects candidates from a long recording window by window (bounded memory, e.g. video audio). */
    inner class Collector(private val engine: CbEngine) {
        val candidates = ArrayList<Candidate>()
        var pieces = 0; private set
        var noisy = 0; private set
        /** Background between words (room tone), for recording-environment reproduction. */
        val roomTone = ArrayList<FloatArray>()
        val roomToneSamples get() = roomTone.sumOf { it.size }

        /** Add one window of audio (any sample rate). */
        fun add(pcm: FloatArray, sampleRate: Int, cancel: () -> Boolean = { false }, onPiece: (Int) -> Unit = {}) {
            var x = if (sampleRate == SR) pcm.copyOf() else Resampler.resample(pcm, sampleRate, SR)
            x = AudioTools.highPass(x, SR, 60.0)
            if (roomToneSamples < ROOM_TONE_MAX * SR) roomTone.addAll(extractRoomTone(x))
            for (p in cut(x)) {
                if (cancel()) throw InterruptedException()
                pieces++; onPiece(pieces)
                val rep = AudioTools.analyze(p, SR)
                if (rep.snrDb < 12 || rep.clippedRatio > 0.01 || rep.voicedSec < 1.5) { noisy++; continue }
                val p16 = Resampler.resample(p, SR, 16000)
                val q = (100 - max(0.0, 30 - rep.snrDb) * 2.5 - rep.clippedRatio * 3000).coerceIn(10.0, 100.0).toInt()
                candidates.add(Candidate(AudioTools.toPcm16(p), q, engine.veEmbed(p16), unit(engine.xvector(p16))))
            }
        }
    }

    /**
     * Learn from a recording (any length; e.g. the audio track of a video). Speech is cut into 3–10 s
     * segments, noisy/clipped pieces are dropped, and only the dominant speaker (or, if the bank already
     * has material, the bank's speaker) is kept.
     */
    fun ingest(engine: CbEngine, pcm: FloatArray, sampleRate: Int, source: String,
               progress: (Float, String) -> Unit = { _, _ -> }, cancel: () -> Boolean = { false }): IngestReport {
        progress(0f, "오디오 분석 중")
        val c = Collector(engine)
        c.add(pcm, sampleRate, cancel) { n -> progress(0.4f * (1 - 1f / (1 + n * 0.2f)), "구간 분석 $n") }
        return commit(engine, c, source, progress, cancel)
    }

    /** Speaker selection + learning of the collected candidates. */
    fun commit(engine: CbEngine, c: Collector, source: String,
               progress: (Float, String) -> Unit = { _, _ -> }, cancel: () -> Boolean = { false }): IngestReport {
        val msgs = ArrayList<String>()
        val cands = c.candidates
        if (c.pieces == 0) return IngestReport(0, 0, 0, 0, 0.0, listOf("음성 구간을 찾지 못했습니다."))
        if (cands.isEmpty()) return IngestReport(c.pieces, 0, c.noisy, 0, 0.0, listOf("모든 구간이 잡음/클리핑으로 제외되었습니다. 더 깨끗한 음성을 사용해 주세요."))
        val ref: FloatArray = if (segments.size >= 2) identityXvec() else {
            // medoid of the largest group = dominant speaker of this recording
            var best = 0; var bestScore = -1.0
            for (i in cands.indices) {
                val sc = cands.indices.sumOf { j -> val cs = cos(cands[i].xv, cands[j].xv); if (cs > SAME_SPEAKER) 1.0 + cs else 0.0 }
                if (sc > bestScore) { bestScore = sc; best = i }
            }
            cands[best].xv
        }
        // best material first; at most what the bank can hold
        val keep = cands.filter { cos(it.xv, ref) > SAME_SPEAKER }
            .sortedByDescending { it.quality * min(it.audio.size.toDouble() / SR, 10.0) + 30 * cos(it.xv, ref) }
            .take(MAX_SEGMENTS)
        val otherSpk = cands.count { cos(it.xv, ref) <= SAME_SPEAKER }
        if (otherSpk > 0) msgs.add("다른 화자로 판단된 구간 ${otherSpk}개를 제외했습니다.")
        if (c.noisy > 0) msgs.add("잡음/클리핑/너무 짧은 구간 ${c.noisy}개를 제외했습니다.")
        var kept = 0; var keptSec = 0.0
        for ((i, cd) in keep.withIndex()) {
            if (cancel()) throw InterruptedException()
            progress(0.4f + 0.58f * i / keep.size, "목소리 학습 ${i + 1}/${keep.size}")
            val audio = FloatArray(cd.audio.size) { cd.audio[it] / 32767f }
            val id = java.lang.Long.toString(System.nanoTime() and 0xffffffffffL, 36)
            val v = engine.buildVoice(audio, SR)
            File(dir, "seg_$id.cbv").outputStream().buffered().use { v.write(DataOutputStream(it)) }
            AudioTools.writePcm16(File(dir, "seg_$id.pcm"), audio)
            segments.add(Segment(id, source, audio.size.toDouble() / SR, cd.quality, cd.ve, cd.xv, VoiceAnalysis.analyze(audio, SR)))
            kept++; keptSec += audio.size.toDouble() / SR
        }
        // keep the bank bounded: drop the lowest-quality segments beyond the cap
        if (segments.size > MAX_SEGMENTS) {
            val drop = segments.sortedBy { it.quality * it.seconds }.take(segments.size - MAX_SEGMENTS).toSet()
            for (s in drop) { File(dir, s.audioFile).delete(); File(dir, s.voiceFile).delete() }
            segments.removeAll(drop)
            if (tuning.promptSegment != null && segments.none { it.id == tuning.promptSegment }) tuning = tuning.copy(promptSegment = null)
        }
        if (kept > 0) generation++
        if (kept > 0 && c.roomToneSamples > SR) appendRoomTone(c.roomTone)
        save()
        progress(1f, "완료")
        return IngestReport(c.pieces, kept, c.noisy, otherSpk, keptSec, msgs)
    }

    /** VAD → segments of 3–10 s (merging nearby speech, splitting long runs at the quietest point). */
    private fun cut(x: FloatArray): List<FloatArray> {
        val segs = AudioTools.vad(x, SR, minSpeechMs = 300, mergeGapMs = 600, padMs = 120)
        val out = ArrayList<FloatArray>()
        var curS = -1; var curE = -1
        fun flush() {
            if (curS < 0) return
            var s = curS
            while (curE - s > MAX_SEC * SR) {
                // split at the quietest 20 ms frame between 5 s and 10 s
                val lo = s + 5 * SR; val hi = min(curE - 2 * SR, s + MAX_SEC * SR)
                var cutAt = s + MAX_SEC * SR
                if (hi > lo) {
                    var bestE = Double.MAX_VALUE
                    var p = lo
                    while (p + 480 < hi) { var e = 0.0; for (k in p until p + 480) e += x[k] * x[k]; if (e < bestE) { bestE = e; cutAt = p + 240 }; p += 240 }
                }
                out.add(x.copyOfRange(s, cutAt)); s = cutAt
            }
            if (curE - s >= MIN_SEC * SR) out.add(x.copyOfRange(s, curE))
            curS = -1
        }
        for (g in segs) {
            if (curS >= 0 && g.start - curE < SR * 0.6 && g.end - curS <= MAX_SEC * SR) curE = g.end
            else { flush(); curS = g.start; curE = g.end }
        }
        flush()
        return out
    }

    // ------------------------------------------------------------------ auto-learning results

    /** Applies an auto-learning result and keeps its generations (with listenable samples). */
    fun applyAutoLearning(res: AutoTuner.Result) {
        dir.listFiles { f -> f.name.startsWith("gen_") }?.forEach { it.delete() }
        generations.clear()
        for (g in res.generations) { AudioTools.writePcm16(File(dir, g.generation.sample), g.audio); generations.add(g.generation) }
        tuning = res.tuning
        selectedGeneration = if (res.accepted) generations.lastIndex else -1
        generation++
        save()
    }

    /** Switch to generation [i] (or the defaults for i = −1), keeping learning history and environment. */
    fun useGeneration(i: Int) {
        val keep = tuning
        tuning = if (i in generations.indices) generations[i].tuning.copy(history = keep.history, iterations = keep.iterations, env = keep.env)
        else Tuning(history = keep.history, iterations = keep.iterations, env = keep.env)
        selectedGeneration = i
        save()
    }

    fun generationAudio(g: Generation): FloatArray? = File(dir, g.sample).takeIf { it.exists() }?.let { AudioTools.readPcm16(it) }

    // ------------------------------------------------------------------ room tone

    val roomToneFile get() = File(dir, "roomtone.pcm")
    fun roomTone(): FloatArray? = if (roomToneFile.exists()) AudioTools.readPcm16(roomToneFile) else null

    /** Add background pieces (joined with short cross-fades), keeping at most [ROOM_TONE_MAX] seconds. */
    fun appendRoomTone(pieces: List<FloatArray>) {
        val all = ArrayList<FloatArray>(); roomTone()?.let { all.add(it) }; all.addAll(pieces)
        val xf = SR / 200
        var acc = FloatArray(0)
        for (p in all) {
            if (acc.isEmpty()) { acc = p.copyOf(); continue }
            val n = minOf(xf, p.size, acc.size)
            val joined = acc.copyOf(acc.size + p.size - n)
            for (k in 0 until n) { val w = k.toFloat() / n; joined[acc.size - n + k] = acc[acc.size - n + k] * (1 - w) + p[k] * w }
            System.arraycopy(p, n, joined, acc.size, p.size - n)
            acc = joined
            if (acc.size >= ROOM_TONE_MAX * SR) break
        }
        AudioTools.writePcm16(roomToneFile, acc.copyOf(minOf(acc.size, ROOM_TONE_MAX * SR)))
    }

    // ------------------------------------------------------------------ composing the voice

    private fun weight(s: Segment) = s.quality / 100.0 * min(s.seconds, 10.0)

    /** Quality-weighted mean VoiceEncoder embedding of all segments (unit length). */
    fun identityVe(): FloatArray {
        val m = FloatArray(256)
        for (s in segments) { val w = weight(s).toFloat(); for (k in 0 until 256) m[k] += s.ve[k] * w }
        return unit(m)
    }

    /** Mean (unit) CAMPPlus x-vector of all segments. */
    fun identityXvec(): FloatArray {
        val m = FloatArray(192)
        for (s in segments) { val w = weight(s).toFloat(); for (k in 0 until 192) m[k] += s.xvec[k] * w }
        return unit(m)
    }

    /** Average voice profile of the real recordings (target for auto-learning). */
    fun targetProfile(): VoiceAnalysis.Profile {
        val ps = segments.map { it.profile }
        val w = segments.map { weight(it) }; val ws = w.sum()
        fun avg(f: (VoiceAnalysis.Profile) -> Double) = ps.indices.sumOf { f(ps[it]).let { v -> if (v.isNaN()) 0.0 else v } * w[it] } / ws
        val ltas = DoubleArray(VoiceAnalysis.LTAS_BANDS) { b -> ps.indices.sumOf { ps[it].ltas[b] * w[it] } / ws }
        return VoiceAnalysis.Profile(avg { it.f0Median }, avg { it.f0Std }, avg { it.voicedRatio }, avg { it.energyStd }, avg { it.onsetRate }, ltas, avg { it.tailRatio })
    }

    /** Representative segment: high quality, long enough, close to the speaker centroid. */
    fun bestSegment(): Segment {
        val c = identityXvec()
        return segments.maxBy { (it.quality / 100.0) * min(1.0, it.seconds / 6.0) + 2 * cos(it.xvec, c) }
    }

    /** Segment whose expressiveness best matches [arousal] (−1 calm … +1 lively), among good ones. */
    fun segmentForArousal(arousal: Double): Segment {
        if (segments.size < 3 || kotlin.math.abs(arousal) < 0.2) return tuning.promptSegment?.let { id -> segments.firstOrNull { it.id == id } } ?: bestSegment()
        val c = identityXvec()
        val good = segments.filter { it.seconds >= 3 && cos(it.xvec, c) > SAME_SPEAKER + 0.1 }.ifEmpty { segments }
        val a = good.map { it.profile.arousal }
        val mean = a.average(); val sd = sqrt(a.sumOf { (it - mean) * (it - mean) } / a.size).coerceAtLeast(1e-6)
        return good.minBy { kotlin.math.abs((it.profile.arousal - mean) / sd - arousal * 1.2) - it.quality / 400.0 }
    }

    fun segmentVoice(s: Segment): CbVoice = File(dir, s.voiceFile).inputStream().buffered().use { CbVoice.read(DataInputStream(it)) }

    fun segmentAudio(s: Segment): FloatArray = AudioTools.readPcm16(File(dir, s.audioFile))

    /**
     * The voice used for synthesis: identity from all material (+ learned correction), style reference
     * from [ref] (default: tuned or best segment).
     */
    fun compose(ref: Segment? = null, t: Tuning = tuning): CbVoice {
        val seg = ref ?: t.promptSegment?.let { id -> segments.firstOrNull { it.id == id } } ?: bestSegment()
        val sv = segmentVoice(seg)
        var ve = identityVe()
        if (t.veBlend > 0) ve = unit(FloatArray(256) { ((1 - t.veBlend) * ve[it] + t.veBlend * sv.ve[it]).toFloat() })
        t.veCorrection?.let { c -> ve = unit(FloatArray(256) { ve[it] + c[it] }) }
        // S3Gen normalises the x-vector internally; keep the reference's scale for safety
        val scale = norm(sv.xvec)
        var xv = identityXvec()
        t.xvCorrection?.let { c -> xv = unit(FloatArray(192) { xv[it] + c[it] }) }
        return CbVoice(ve, sv.t3Tokens, sv.genTokens, sv.genMel, FloatArray(192) { (xv[it] * scale).toFloat() })
    }

    fun removeSegment(id: String) {
        segments.firstOrNull { it.id == id }?.let { s -> File(dir, s.audioFile).delete(); File(dir, s.voiceFile).delete(); segments.remove(s) }
        if (tuning.promptSegment == id) tuning = tuning.copy(promptSegment = null)
        save()
    }

    companion object {
        const val SR = 24000
        const val MIN_SEC = 3
        const val MAX_SEC = 10
        const val MAX_SEGMENTS = 60
        /** CAMPPlus cosine above which two segments are treated as the same speaker. */
        const val SAME_SPEAKER = 0.5
        const val ROOM_TONE_MAX = 20

        /**
         * Quiet stretches between speech (20 ms frames within 4 dB of the noise floor and ≥ 100 ms away from
         * speech), at least 200 ms long — the recording's own background.
         */
        fun extractRoomTone(x: FloatArray): List<FloatArray> {
            val fr = SR / 50
            val db = AudioTools.frameDb(x, fr)
            if (db.size < 20) return emptyList()
            val sorted = db.sorted(); val floor = sorted[sorted.size / 10]; val speech = sorted[(sorted.size * 0.9).toInt()]
            if (speech - floor < 15) return emptyList()           // no clear speech/background contrast
            val quiet = BooleanArray(db.size) { db[it] < floor + 4 }
            val ok = BooleanArray(db.size) { i -> (maxOf(0, i - 5)..minOf(db.size - 1, i + 5)).all { quiet[it] } }
            val out = ArrayList<FloatArray>()
            var i = 0
            while (i < ok.size) {
                if (!ok[i]) { i++; continue }
                var j = i; while (j < ok.size && ok[j]) j++
                if (j - i >= 10) out.add(x.copyOfRange(i * fr, minOf(x.size, j * fr)))
                i = j
            }
            return out
        }

        fun cos(a: FloatArray, b: FloatArray): Double {
            var d = 0.0; var na = 0.0; var nb = 0.0
            for (i in a.indices) { d += a[i] * b[i]; na += a[i] * a[i]; nb += b[i] * b[i] }
            return d / sqrt(na * nb + 1e-20)
        }
        fun norm(a: FloatArray): Double { var n = 0.0; for (v in a) n += v * v; return sqrt(n) }
        fun unit(a: FloatArray): FloatArray { val n = norm(a).coerceAtLeast(1e-12); return FloatArray(a.size) { (a[it] / n).toFloat() } }
        private fun floats(a: JSONArray) = FloatArray(a.length()) { a.getDouble(it).toFloat() }
        private fun doubles(a: JSONArray) = List(a.length()) { a.getDouble(it) }

        fun tuningJson(t: VoiceBank.Tuning) = JSONObject().apply {
            put("exaggeration", t.params.exaggeration.toDouble()); put("cfg", t.params.cfgWeight.toDouble()); put("temperature", t.params.temperature.toDouble())
            put("seed", t.params.seed)
            put("pitch", t.pitch); put("tilt", t.tilt); put("tempo", t.tempo); put("wet", t.wet); put("veBlend", t.veBlend)
            t.promptSegment?.let { put("prompt", it) }
            t.veCorrection?.let { put("veCorr", JSONArray(it.map { v -> v.toDouble() })) }
            t.xvCorrection?.let { put("xvCorr", JSONArray(it.map { v -> v.toDouble() })) }
            put("history", JSONArray(t.history.map { finite(it) })); put("iterations", t.iterations)
            put("endShift", t.endShift); put("endStretch", t.endStretch)
            t.env?.let { e -> put("env", JSONObject().apply {
                put("eq", JSONArray(e.eq.map { finite(it) })); put("room", e.room); put("wet", e.wet); e.noiseDb?.let { put("noiseDb", it) }
                put("before", finite(e.before)); put("after", finite(e.after)) }) }
        }

        /** Android's org.json rejects NaN/Infinity. */
        fun finite(v: Double) = if (v.isFinite()) v else 0.0

        fun tuningFrom(o: JSONObject) = VoiceBank.Tuning(
            CbParams(exaggeration = o.optDouble("exaggeration", 0.5).toFloat(), cfgWeight = o.optDouble("cfg", 0.5).toFloat(),
                temperature = o.optDouble("temperature", 0.8).toFloat(), seed = o.optLong("seed", 0)),
            o.optDouble("pitch", 0.0), o.optDouble("tilt", 0.0), o.optDouble("tempo", 1.0), o.optDouble("wet", 0.0), o.optDouble("veBlend", 0.0),
            if (o.has("prompt")) o.getString("prompt") else null,
            o.optJSONArray("veCorr")?.let { floats(it) }, o.optJSONArray("xvCorr")?.let { floats(it) },
            o.optJSONArray("history")?.let { doubles(it) } ?: emptyList(), o.optInt("iterations", 0),
            o.optDouble("endShift", 0.0), o.optDouble("endStretch", 1.0),
            o.optJSONObject("env")?.let { e -> Environment(doubles(e.getJSONArray("eq")).toDoubleArray(), e.optString("room", "NONE"), e.optDouble("wet", 0.0),
                if (e.has("noiseDb")) e.getDouble("noiseDb") else null, e.optDouble("before", Double.NaN), e.optDouble("after", Double.NaN)) })
    }
}
