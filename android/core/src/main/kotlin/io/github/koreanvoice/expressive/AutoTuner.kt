package io.github.koreanvoice.expressive

import io.github.koreanvoice.audio.AudioTools
import io.github.koreanvoice.audio.Dsp
import io.github.koreanvoice.audio.Psola
import io.github.koreanvoice.audio.Resampler
import io.github.koreanvoice.audio.VoiceAnalysis
import io.github.koreanvoice.audio.VoiceCriteria
import kotlin.math.abs
import kotlin.math.min

/**
 * Auto-learning: repeatedly tries many different ways of speaking and keeps the one that sounds most like the
 * real recordings, then builds the next "generation" of the voice from it and learns again (default 5 rounds).
 *
 * Tried per round (≈100 candidates = a few freshly generated deliveries × decoder/colour/ending variants):
 *  - delivery: reference clip (style source), emotion intensity, pacing (CFG), randomness (temperature),
 *    blend between the all-material identity and the reference clip's own identity;
 *  - decoder: strength of the learned x-vector correction (decoder only → intelligibility untouched);
 *  - colour: gentle spectral tilt;
 *  - sentence endings (말끝): final pitch glide and lengthening (formant-preserving PSOLA).
 *
 * Judged by: ECAPA speaker similarity (synthesis is never conditioned on it, so it cannot be gamed) plus
 * VoiceEncoder/CAMPPlus similarity and ~100 voice criteria ([VoiceCriteria]: timbre, pitch, endings, rhythm &
 * style, emphasis, voice quality, mood, environment) measured against the speaker's own natural variation.
 * Comparisons are paired (same probe sentences and sampling seeds for every candidate).
 *
 * "Learn again": after each round the mean embedding gap of its best results becomes the next x-vector
 * correction, the round's best voice is saved as a generation, and the next round searches around the elites
 * with a narrower spread. The final voice is applied only if it beats the defaults on held-out sentences
 * (ECAPA +0.01, or criteria +0.03 without losing ECAPA); every generation stays available to listen to and pick.
 * Finally the recording environment of the originals (microphone colour, room, background) is measured and a
 * reproduction is prepared (used only when the user enables it).
 */
class AutoTuner(private val engine: CbEngine, private val bank: VoiceBank, private val cfg: Config = Config()) {
    data class Config(
        val rounds: Int = 5,
        val candidatesPerRound: Int = 100,
        val generativePerRound: Int = 8,
        val sentences: List<String> = listOf("오늘은 날씨가 정말 좋네요.", "잠깐만요, 제 이야기 좀 들어 보실래요?"),
        val verifySentences: List<String> = listOf("어제 서점에 들렀다가 오래 못 본 친구를 우연히 만났어요.",
            "내일 아침 일곱 시까지 역 앞으로 와 주세요.", "그 영화 정말 재미있었어! 너도 꼭 봐.", "비가 그치자 공기가 한결 맑아졌습니다."),
        val seed: Long = 1,
        val learningRate: Double = 0.7,
        val acceptMargin: Double = 0.01,
        val criteriaMargin: Double = 0.03,
    ) {
        companion object {
            val QUICK = Config(rounds = 3, candidatesPerRound = 30, generativePerRound = 5)
            val STANDARD = Config(rounds = 5, candidatesPerRound = 100, generativePerRound = 8)
            val PRECISE = Config(rounds = 5, candidatesPerRound = 100, generativePerRound = 14)
        }
    }

    data class Gen(val seg: Int, val ex: Double, val cfg: Double, val temp: Double, val blend: Double)
    /** Decoder / colour / ending variant. */
    data class Variant(val xs: Double = 0.0, val tilt: Double = 0.0, val endShift: Double = 0.0, val endStretch: Double = 1.0)

    class Scored(val gen: Gen, val v: Variant, val score: Double, val parts: Map<String, Double>,
                 val veOut: FloatArray, val xvOut: FloatArray, val xvDir: FloatArray?, val criteria: VoiceCriteria.Report?)

    data class Progress(val round: Int, val rounds: Int, val done: Int, val total: Int, val baseline: Double, val best: Double,
                        val message: String, val groups: Map<VoiceCriteria.Group, Double> = emptyMap())

    class GenerationResult(val generation: VoiceBank.Generation, val audio: FloatArray)

    class Result(
        val tuning: VoiceBank.Tuning, val baseline: Double, val roundBest: List<Double>, val best: Scored,
        val evaluated: Int, val accepted: Boolean,
        /** Held-out ECAPA (tuned, default) and criteria similarity (tuned, default). */
        val heldOut: Pair<Double, Double>, val heldOutCriteria: Pair<Double, Double>,
        /** Criteria groups on held-out sentences: default → tuned. */
        val groupsBefore: Map<VoiceCriteria.Group, Double>, val groupsAfter: Map<VoiceCriteria.Group, Double>,
        val generations: List<GenerationResult>,
    )

    private val rnd = java.util.Random(cfg.seed)
    private val segs = bank.segments.filter { it.seconds >= 3 }.sortedByDescending { it.quality * min(it.seconds, 10.0) }.take(8).ifEmpty { bank.segments }
    private val veT = bank.identityVe()
    private val xvT = bank.identityXvec()
    /** ECAPA centroid of the real recordings (the primary, conditioning-independent target). */
    private val ecT: FloatArray? by lazy {
        if (!engine.hasEcapa()) null else {
            val m = FloatArray(192)
            for (s in bank.segments) { val e = engine.ecapa(Resampler.resample(bank.segmentAudio(s), SR, 16000)); for (i in 0 until 192) m[i] += e[i] }
            VoiceBank.unit(m)
        }
    }
    /** ~100 criteria of the real recordings and their natural spread. */
    val criteriaTarget: VoiceCriteria.Target by lazy { VoiceCriteria.Target.of(bank.segments.map { VoiceCriteria.measure(bank.segmentAudio(it), SR) }) }
    private var evaluated = 0

    private fun clamp(v: Double, lo: Double, hi: Double) = v.coerceIn(lo, hi)

    private fun randomGen() = Gen(rnd.nextInt(segs.size), 0.25 + rnd.nextDouble() * 0.75, 0.25 + rnd.nextDouble() * 0.5, 0.6 + rnd.nextDouble() * 0.3, rnd.nextDouble() * 0.5)

    private fun mutate(g: Gen, s: Double) = Gen(
        if (rnd.nextDouble() < 0.1 + 0.3 * s) rnd.nextInt(segs.size) else g.seg,
        clamp(g.ex + rnd.nextGaussian() * 0.2 * s, 0.25, 1.1), clamp(g.cfg + rnd.nextGaussian() * 0.12 * s, 0.2, 0.8),
        clamp(g.temp + rnd.nextGaussian() * 0.1 * s, 0.55, 0.95), clamp(g.blend + rnd.nextGaussian() * 0.15 * s, 0.0, 0.6))

    private fun mutateVariant(v: Variant, s: Double) = Variant(
        clamp(v.xs + rnd.nextGaussian() * 0.5 * s, 0.0, 1.5), clamp(v.tilt + rnd.nextGaussian() * 1.2 * s, -3.0, 3.0),
        clamp(v.endShift + rnd.nextGaussian() * 1.0 * s, -2.5, 2.5), clamp(v.endStretch + rnd.nextGaussian() * 0.12 * s, 0.9, 1.4))

    private fun xvCorrection(dir: FloatArray?, xs: Double) = if (dir == null || xs < 1e-3) null else FloatArray(192) { (dir[it] * xs).toFloat() }

    fun tuningOf(g: Gen, v: Variant, dir: FloatArray?) = VoiceBank.Tuning(
        CbParams(exaggeration = g.ex.toFloat(), cfgWeight = g.cfg.toFloat(), temperature = g.temp.toFloat()),
        tilt = v.tilt, veBlend = g.blend, promptSegment = segs[g.seg].id, xvCorrection = xvCorrection(dir, v.xs),
        endShift = v.endShift, endStretch = v.endStretch)

    /** T3 for each sentence (paired seeds), regenerating implausible lengths like the production job does. */
    private fun generate(g: Gen, sentences: List<String>, seedBase: Long): Pair<List<LongArray>, Double> {
        val voice = bank.compose(segs[g.seg], tuningOf(g, Variant(), null))
        val p = CbParams(exaggeration = g.ex.toFloat(), cfgWeight = g.cfg.toFloat(), temperature = g.temp.toFloat(), maxNewTokens = 400)
        var penalty = 0.0
        val toks = sentences.mapIndexed { k, s ->
            val syl = s.count { it in '가'..'힣' }.coerceAtLeast(1)
            var best = LongArray(0); var bestBad = Double.MAX_VALUE
            for (attempt in 0..2) {
                val t = engine.generate(s, voice, p.copy(seed = seedBase + k * 31 + attempt * 7919L)).first
                val sec = t.size / 25.0; val lo = syl / 12.0; val hi = syl / 2.8
                val bad = when { sec < lo -> (lo - sec) / lo; sec > hi -> (sec - hi) / hi; else -> 0.0 }
                if (bad < bestBad) { bestBad = bad; best = t }
                if (bad == 0.0) break
            }
            if (bestBad > 0) penalty += 0.3
            best
        }
        return toks to penalty
    }

    /** Decodes each sentence; returns the per-sentence audio (silence-trimmed). */
    private fun decode(g: Gen, toks: List<LongArray>, xvc: FloatArray?): List<FloatArray> {
        val voice = bank.compose(segs[g.seg], tuningOf(g, Variant(), null).copy(xvCorrection = xvc))
        return toks.map { AudioTools.trimSilence(engine.decode(it, voice), SR) }
    }

    /** Colour and endings per sentence, then joined with 300 ms pauses. */
    fun render(parts: List<FloatArray>, v: Variant): FloatArray {
        val out = ArrayList<FloatArray>()
        for (a in parts) {
            var y = Psola.shapeEndings(a, SR, v.endShift, v.endStretch)
            y = Dsp.tilt(y, SR, v.tilt)
            out.add(y); out.add(AudioTools.silence(SR, 300))
        }
        val all = FloatArray(out.sumOf { it.size }); var o = 0
        for (a in out) { System.arraycopy(a, 0, all, o, a.size); o += a.size }
        return all
    }

    /** Similarity of [x] to the real recordings (higher is better). */
    fun score(x: FloatArray, penalty: Double = 0.0): Triple<Double, Map<String, Double>, Triple<FloatArray, FloatArray, VoiceCriteria.Report>> {
        val x16 = Resampler.resample(x, SR, 16000)
        val ve = engine.veEmbed(x16)
        val xv = VoiceBank.unit(engine.xvector(x16))
        val sVe = VoiceBank.cos(ve, veT); val sXv = VoiceBank.cos(xv, xvT)
        // ECAPA dominates: VoiceEncoder/CAMPPlus are what T3/S3Gen are conditioned on (and what corrections
        // edit), so on their own they would reward "gaming" the conditioning instead of real similarity
        val sEc = ecT?.let { VoiceBank.cos(engine.ecapa(x16), it) }
        val spk = if (sEc != null) 0.8 * sEc + 0.1 * sVe + 0.1 * sXv else 0.5 * sVe + 0.5 * sXv
        val crit = VoiceCriteria.compare(VoiceCriteria.measure(x, SR, x16), criteriaTarget)
        val total = spk + 0.15 * (crit.overall - 0.5) - penalty
        evaluated++
        return Triple(total, mapOf("ecapa" to (sEc ?: Double.NaN), "ve" to sVe, "xv" to sXv, "criteria" to crit.overall, "penalty" to penalty), Triple(ve, xv, crit))
    }

    fun run(progress: (Progress) -> Unit = {}, cancel: () -> Boolean = { false }): Result {
        require(bank.segments.isNotEmpty()) { "학습된 음성이 없습니다." }
        fun check() { if (cancel()) throw InterruptedException("cancelled") }
        val perGen = (cfg.candidatesPerRound / cfg.generativePerRound).coerceAtLeast(1)
        val nDec = maxOf(1, minOf(4, perGen / 3)); val nVar = maxOf(1, perGen / nDec)
        val total = cfg.rounds * cfg.generativePerRound * nDec * nVar + 1
        var done = 0
        val searchSeed = cfg.seed * 7919L
        progress(Progress(0, cfg.rounds, 0, total, Double.NaN, Double.NaN, "원본 음성의 특징 ${VoiceCriteria.SIZE}가지 측정 중"))
        criteriaTarget
        val bestSeg = segs.indexOf(segs.maxBy { s -> (s.quality / 100.0) * min(1.0, s.seconds / 6.0) + 2 * VoiceBank.cos(s.xvec, xvT) })
        val base = Gen(maxOf(0, bestSeg), 0.5, 0.5, 0.8, 0.0)
        progress(Progress(0, cfg.rounds, 0, total, Double.NaN, Double.NaN, "기준 음성 측정 중"))
        val (t0, pen0) = generate(base, cfg.sentences, searchSeed)
        val s0 = score(render(decode(base, t0, null), Variant()), pen0)
        val baseline = Scored(base, Variant(), s0.first, s0.second, s0.third.first, s0.third.second, null, s0.third.third)
        done++
        val all = ArrayList<Scored>(); all.add(baseline)
        var best = baseline
        var dir: FloatArray? = null
        // warm start: a previous auto-learning result competes as an elite, so repeated runs keep improving
        val prev = bank.tuning
        val prevSeg = segs.indexOfFirst { it.id == prev.promptSegment }
        if (prev.iterations > 0 && prevSeg >= 0) {
            progress(Progress(0, cfg.rounds, done, total, baseline.score, baseline.score, "이전 학습 결과 측정 중"))
            val g = Gen(prevSeg, prev.params.exaggeration.toDouble(), prev.params.cfgWeight.toDouble(), prev.params.temperature.toDouble(), prev.veBlend)
            val v = Variant(if (prev.xvCorrection != null) 1.0 else 0.0, prev.tilt, prev.endShift, prev.endStretch)
            val (t, pen) = generate(g, cfg.sentences, searchSeed)
            val s = score(render(decode(g, t, prev.xvCorrection), v), pen)
            val sc = Scored(g, v, s.first, s.second, s.third.first, s.third.second, prev.xvCorrection, s.third.third)
            all.add(sc)
            if (sc.score > best.score) { best = sc; dir = prev.xvCorrection }
        }
        val roundBest = ArrayList<Double>()
        val generations = ArrayList<GenerationResult>()
        for (r in 1..cfg.rounds) {
            val sigma = Math.pow(0.65, (r - 1).toDouble())
            val elites = all.sortedByDescending { it.score }.distinctBy { it.gen }.take(3)
            val gens = ArrayList<Gen>()
            gens.add(best.gen)
            while (gens.size < cfg.generativePerRound) gens.add(if (r == 1 && gens.size % 2 == 0) randomGen() else mutate(elites[rnd.nextInt(elites.size)].gen, sigma))
            var rb: Scored? = null; var rbAudio: FloatArray? = null
            val roundRaw = ArrayList<Scored>()
            for (g in gens) {
                check()
                progress(Progress(r, cfg.rounds, done, total, baseline.score, best.score,
                    "${r}회차: 새 방식으로 말하는 중 (감정 ${"%.2f".format(g.ex)}, 속도감 ${"%.2f".format(g.cfg)}, 다양성 ${"%.2f".format(g.temp)})", best.criteria?.groups ?: emptyMap()))
                val (toks, pen) = generate(g, cfg.sentences, searchSeed)
                val strengths = LinkedHashSet<Double>().apply { add(0.0); if (dir != null) { add(best.v.xs.coerceAtLeast(0.5)); var tries = 0; while (size < nDec && tries++ < 20) add(mutateVariant(best.v, sigma).xs) } }.take(nDec)
                for (xs in strengths) {
                    check()
                    val parts = decode(g, toks, xvCorrection(dir, xs))
                    val vars = LinkedHashSet<Variant>().apply {
                        add(Variant(xs)); add(best.v.copy(xs = xs)); var tries = 0
                        while (size < nVar && tries++ < 20) add(mutateVariant(best.v, sigma).copy(xs = xs))
                    }.take(nVar)
                    for (v in vars) {
                        val audio = render(parts, v)
                        val s = score(audio, pen)
                        val sc = Scored(g, v, s.first, s.second, s.third.first, s.third.second, dir, s.third.third)
                        all.add(sc); done++
                        if (v == Variant(xs)) roundRaw.add(sc)
                        if (rb == null || sc.score > rb.score) { rb = sc; rbAudio = audio }
                    }
                }
                progress(Progress(r, cfg.rounds, done, total, baseline.score, maxOf(best.score, rb?.score ?: best.score), "${r}회차 진행 중",
                    (if ((rb?.score ?: -9.0) > best.score) rb?.criteria else best.criteria)?.groups ?: emptyMap()))
            }
            rb!!
            roundBest.add(rb.score)
            if (rb.score > best.score) best = rb
            // generation r = the best voice found so far, kept with a sample so the user can listen and choose
            val gt = tuningOf(best.gen, best.v, best.xvDir)
            generations.add(GenerationResult(VoiceBank.Generation(r, gt, best.score, best.criteria?.overall ?: Double.NaN,
                best.parts["ecapa"] ?: Double.NaN, "gen_$r.pcm"), if (best === rb) rbAudio!! else generations.lastOrNull()?.audio ?: rbAudio!!))
            // learn again from its own output: the mean x-vector gap of this round's best plain results becomes
            // the next correction direction (decoder only), whose strength the next round searches
            val top = roundRaw.sortedByDescending { it.score }.take(3)
            val baseDir = best.xvDir?.let { d -> FloatArray(192) { (d[it] * best.v.xs).toFloat() } } ?: FloatArray(192)
            dir = FloatArray(192) { i -> (baseDir[i] + cfg.learningRate * sigma * (xvT[i] - top.sumOf { it.xvOut[i].toDouble() } / top.size)).toFloat() }
            progress(Progress(r, cfg.rounds, done, total, baseline.score, best.score, "${r}세대 완성: 점수 ${"%.3f".format(best.score)} (기준 ${"%.3f".format(baseline.score)})", best.criteria?.groups ?: emptyMap()))
        }

        // held-out check: different sentences, fresh seeds; the search maximum is optimistic
        progress(Progress(cfg.rounds, cfg.rounds, done, total, baseline.score, best.score, "새 문장으로 최종 검증 중"))
        fun verify(g: Gen, v: Variant, xvc: FloatArray?): Pair<Double, VoiceCriteria.Report> {
            check()
            val (toks, _) = generate(g, cfg.verifySentences, cfg.seed * 104729L + 17)
            val audio = render(decode(g, toks, xvc), v)
            val x16 = Resampler.resample(audio, SR, 16000)
            val ec = ecT?.let { VoiceBank.cos(engine.ecapa(x16), it) } ?: (VoiceBank.cos(engine.veEmbed(x16), veT) + VoiceBank.cos(VoiceBank.unit(engine.xvector(x16)), xvT)) / 2
            return ec to VoiceCriteria.compare(VoiceCriteria.measure(audio, SR, x16), criteriaTarget)
        }
        val (vBase, cBase) = verify(baseline.gen, Variant(), null)
        val tunedCheck = if (best === baseline) null else verify(best.gen, best.v, best.xvDir?.let { xvCorrection(it, best.v.xs) })
        val vBest = tunedCheck?.first ?: Double.NaN; val cBest = tunedCheck?.second
        val accepted = tunedCheck != null && (vBest - vBase >= cfg.acceptMargin ||
            (vBest - vBase >= -0.005 && cBest!!.overall - cBase.overall >= cfg.criteriaMargin))
        val win = if (accepted) best else baseline
        var t = tuningOf(win.gen, win.v, win.xvDir).copy(history = listOf(baseline.score) + roundBest, iterations = bank.tuning.iterations + 1)

        // recording environment of the originals (applied only if the user turns it on)
        progress(Progress(cfg.rounds, cfg.rounds, done, total, baseline.score, best.score, "원본 녹음 환경(마이크 색깔·울림·배경) 분석 중"))
        t = t.copy(env = matchEnvironment(win, t))
        return Result(t, baseline.score, roundBest, win, evaluated, accepted, vBest to vBase,
            (cBest?.overall ?: Double.NaN) to cBase.overall, cBase.groups, (if (accepted) cBest!! else cBase).groups, generations)
    }

    /** Environment fit for an already learned voice (e.g. after adding material, or for older voices). */
    fun fitEnvironment(t: VoiceBank.Tuning): VoiceBank.Environment? {
        val seg = maxOf(0, segs.indexOfFirst { it.id == t.promptSegment })
        val g = Gen(seg, t.params.exaggeration.toDouble(), t.params.cfgWeight.toDouble(), t.params.temperature.toDouble(), t.veBlend)
        val v = Variant(if (t.xvCorrection != null) 1.0 else 0.0, t.tilt, t.endShift, t.endStretch)
        return matchEnvironment(Scored(g, v, 0.0, emptyMap(), FloatArray(256), FloatArray(192), t.xvCorrection, null), t)
    }

    /**
     * Fits the original recordings' environment: an EQ curve towards their average spectrum, the room (preset +
     * amount) and the background level, by a small grid search on the environment + timbre criteria. Candidates
     * that cost more than 0.01 ECAPA speaker similarity are rejected (in tests, heavy reverb chosen without this
     * guard made voices less recognisable).
     */
    private fun matchEnvironment(win: Scored, t: VoiceBank.Tuning): VoiceBank.Environment? {
        val (toks, _) = generate(win.gen, cfg.sentences, cfg.seed * 7919L)
        val dry = render(decode(win.gen, toks, t.xvCorrection), win.v)
        // production levels speech to −20 dBFS before effects; do the same so background levels are comparable
        val gain = AudioTools.normalizeGain(dry, SR); for (i in dry.indices) dry[i] *= gain
        val target = criteriaTarget
        val ltasIdx = VoiceCriteria.DEFS.indices.filter { VoiceCriteria.DEFS[it].name.startsWith("주파수 대역") }
        val dryLtas = VoiceAnalysis.ltas(dry, SR)
        val raw = DoubleArray(dryLtas.size) { b -> (target.mean[ltasIdx[b]] - dryLtas[b]).let { if (it.isNaN()) 0.0 else it } }
        val curve = DoubleArray(raw.size) { b -> ((maxOf(0, b - 1)..minOf(raw.size - 1, b + 1)).map { raw[it] }.average()).coerceIn(-6.0, 6.0) }
        val tone = bank.roomTone()
        fun ecapa(x: FloatArray) = ecT?.let { VoiceBank.cos(engine.ecapa(Resampler.resample(x, SR, 16000)), it) }
        fun envScore(x: FloatArray): Double {
            val r = VoiceCriteria.compare(VoiceCriteria.measure(x, SR), target)
            return 0.6 * (r.groups[VoiceCriteria.Group.ENVIRONMENT] ?: 0.0) + 0.4 * (r.groups[VoiceCriteria.Group.TIMBRE] ?: 0.0)
        }
        val noiseIdx = VoiceCriteria.DEFS.indexOfFirst { it.name == "배경 잡음 수준(상대)" }
        val noiseDb = target.mean[noiseIdx].takeIf { !it.isNaN() && tone != null && tone.size > SR }?.coerceIn(-60.0, -15.0)
        val before = envScore(dry)
        val ecDry = ecapa(dry)
        var bestEnv: VoiceBank.Environment? = null; var bestScore = before
        for (eqS in listOf(0.0, 0.5, 1.0)) for (room in listOf(Dsp.Room.NONE, Dsp.Room.STUDIO, Dsp.Room.ROOM, Dsp.Room.BATH)) {
            for (wet in if (room == Dsp.Room.NONE) listOf(0.0) else listOf(0.04, 0.08, 0.14)) for (useNoise in listOf(false, true)) {
                if (useNoise && noiseDb == null) continue
                val eq = DoubleArray(curve.size) { curve[it] * eqS }
                var y = Dsp.applyEnvironment(dry, SR, eq, room, wet)
                if (useNoise) y = Dsp.BackgroundMixer(tone!!, SR, noiseDb!!, 0.0).mix(y)   // level relative to −20 dBFS speech
                val sc = envScore(y)
                if (sc <= bestScore + 0.005) continue
                if (ecDry != null && ecapa(y)!! < ecDry - 0.01) continue           // keep the voice recognisable
                bestScore = sc; bestEnv = VoiceBank.Environment(eq, room.name, wet, if (useNoise) noiseDb else null)
            }
        }
        return (bestEnv ?: VoiceBank.Environment(DoubleArray(curve.size), "NONE", 0.0, null)).copy(before = before, after = bestScore)
    }

    companion object { const val SR = 24000 }
}
