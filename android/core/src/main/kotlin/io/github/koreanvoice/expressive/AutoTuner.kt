package io.github.koreanvoice.expressive

import io.github.koreanvoice.audio.AudioTools
import io.github.koreanvoice.audio.Dsp
import io.github.koreanvoice.audio.Resampler
import io.github.koreanvoice.audio.VoiceAnalysis
import kotlin.math.abs
import kotlin.math.min

/**
 * Auto-learning: repeatedly tries many different ways of speaking and keeps the one that sounds most like
 * the real recordings.
 *
 * Search space (per round: a few freshly generated deliveries × decoder/colour variants of each):
 *  - generation: reference clip, emotion intensity (exaggeration), pacing (CFG), randomness (temperature),
 *    blend between the all-material identity and the reference clip's own identity;
 *  - decoder: strength of the learned x-vector correction (decoder-only, so intelligibility is untouched);
 *  - colour: gentle spectral tilt (microphone match).
 * Pitch/tempo shifting and reverb are deliberately not searched: they change formants / room and reduced
 * similarity on unseen sentences in our tests.
 *
 * Scoring: ECAPA similarity (synthesis is never conditioned on it, so it cannot be "gamed") dominates;
 * VoiceEncoder / CAMPPlus similarity and pitch/rhythm/spectrum statistics act as tie-breakers.
 * Comparisons are paired (common random numbers: the same probe sentences and sampling seeds for every
 * candidate). After each round the mean embedding gap of the best results updates the x-vector correction
 * ("learning from its own output"), and the next round searches around the elites with a narrower spread.
 *
 * The result is accepted only if it beats the default voice by a clear margin on held-out sentences that
 * were never used in the search, with fresh seeds — otherwise the voice keeps its defaults.
 */
class AutoTuner(private val engine: CbEngine, private val bank: VoiceBank, private val cfg: Config = Config()) {
    data class Config(
        val rounds: Int = 5,
        /** Candidates evaluated per round (generations × variants). */
        val candidatesPerRound: Int = 100,
        /** Freshly generated deliveries per round (each gets candidatesPerRound / generativePerRound variants). */
        val generativePerRound: Int = 8,
        val sentences: List<String> = listOf("오늘은 날씨가 정말 좋네요.", "잠깐만요, 제 이야기 좀 들어 보실래요?"),
        val verifySentences: List<String> = listOf("어제 서점에 들렀다가 오래 못 본 친구를 우연히 만났어요.",
            "내일 아침 일곱 시까지 역 앞으로 와 주세요.", "그 영화 정말 재미있었어! 너도 꼭 봐.", "비가 그치자 공기가 한결 맑아졌습니다."),
        val seed: Long = 1,
        /** Step size of the x-vector correction update. */
        val learningRate: Double = 0.7,
        /** Minimum ECAPA gain on the held-out check for the result to be applied. */
        val acceptMargin: Double = 0.01,
    ) {
        companion object {
            val QUICK = Config(rounds = 3, candidatesPerRound = 30, generativePerRound = 5)
            val STANDARD = Config(rounds = 5, candidatesPerRound = 100, generativePerRound = 8)
            val PRECISE = Config(rounds = 5, candidatesPerRound = 100, generativePerRound = 14)
        }
    }

    data class Gen(val seg: Int, val ex: Double, val cfg: Double, val temp: Double, val blend: Double)
    /** Decoder/colour variant: x-vector correction strength and spectral tilt (dB). */
    data class Variant(val xs: Double = 0.0, val tilt: Double = 0.0)

    class Scored(val gen: Gen, val v: Variant, val score: Double, val parts: Map<String, Double>,
                 val veOut: FloatArray, val xvOut: FloatArray, val xvDir: FloatArray?)

    data class Progress(val round: Int, val rounds: Int, val done: Int, val total: Int, val baseline: Double, val best: Double, val message: String)

    data class Result(val tuning: VoiceBank.Tuning, val baseline: Double, val roundBest: List<Double>, val best: Scored,
                      val evaluated: Int, val accepted: Boolean)

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
    private val prof = bank.targetProfile()
    private var evaluated = 0
    /** (tuned, default) mean ECAPA similarity on the held-out check. */
    var verified: Pair<Double, Double>? = null; private set

    private fun clamp(v: Double, lo: Double, hi: Double) = v.coerceIn(lo, hi)

    private fun randomGen() = Gen(rnd.nextInt(segs.size), 0.25 + rnd.nextDouble() * 0.75, 0.25 + rnd.nextDouble() * 0.5, 0.6 + rnd.nextDouble() * 0.3, rnd.nextDouble() * 0.5)

    private fun mutate(g: Gen, s: Double) = Gen(
        if (rnd.nextDouble() < 0.1 + 0.3 * s) rnd.nextInt(segs.size) else g.seg,
        clamp(g.ex + rnd.nextGaussian() * 0.2 * s, 0.25, 1.1), clamp(g.cfg + rnd.nextGaussian() * 0.12 * s, 0.2, 0.8),
        clamp(g.temp + rnd.nextGaussian() * 0.1 * s, 0.55, 0.95), clamp(g.blend + rnd.nextGaussian() * 0.15 * s, 0.0, 0.6))

    private fun mutateVariant(v: Variant, s: Double) = Variant(clamp(v.xs + rnd.nextGaussian() * 0.5 * s, 0.0, 1.5), clamp(v.tilt + rnd.nextGaussian() * 1.2 * s, -3.0, 3.0))

    private fun xvCorrection(dir: FloatArray?, xs: Double) = if (dir == null || xs < 1e-3) null else FloatArray(192) { (dir[it] * xs).toFloat() }

    fun tuningOf(g: Gen, v: Variant, dir: FloatArray?) = VoiceBank.Tuning(
        CbParams(exaggeration = g.ex.toFloat(), cfgWeight = g.cfg.toFloat(), temperature = g.temp.toFloat()),
        tilt = v.tilt, veBlend = g.blend, promptSegment = segs[g.seg].id, xvCorrection = xvCorrection(dir, v.xs))

    /**
     * T3 for each probe sentence (paired seeds), regenerating implausibly short/long results like the
     * production job does. Returns tokens per sentence and an integrity penalty if retries all failed.
     */
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

    private fun decode(g: Gen, toks: List<LongArray>, xvc: FloatArray?): FloatArray {
        val voice = bank.compose(segs[g.seg], tuningOf(g, Variant(), null).copy(xvCorrection = xvc))
        val parts = ArrayList<FloatArray>()
        for (t in toks) { parts.add(AudioTools.trimSilence(engine.decode(t, voice), SR)); parts.add(AudioTools.silence(SR, 300)) }
        val all = FloatArray(parts.sumOf { it.size }); var o = 0
        for (a in parts) { System.arraycopy(a, 0, all, o, a.size); o += a.size }
        return all
    }

    /** Similarity of [x] to the real recordings (higher is better). */
    fun score(x: FloatArray, penalty: Double = 0.0): Triple<Double, Map<String, Double>, Pair<FloatArray, FloatArray>> {
        val x16 = Resampler.resample(x, SR, 16000)
        val ve = engine.veEmbed(x16)
        val xv = VoiceBank.unit(engine.xvector(x16))
        val sVe = VoiceBank.cos(ve, veT); val sXv = VoiceBank.cos(xv, xvT)
        // ECAPA dominates: VoiceEncoder/CAMPPlus are what T3/S3Gen are conditioned on (and what corrections
        // edit), so on their own they would reward "gaming" the conditioning instead of real similarity
        val sEc = ecT?.let { VoiceBank.cos(engine.ecapa(x16), it) }
        val spk = if (sEc != null) 0.8 * sEc + 0.1 * sVe + 0.1 * sXv else 0.5 * sVe + 0.5 * sXv
        val p = VoiceAnalysis.analyze(x, SR, x16, hopMs = 20)
        fun d(a: Double, b: Double, scale: Double) = if (a.isNaN() || b.isNaN()) 1.0 else min(1.0, abs(a - b) / scale)
        val pros = d(p.f0Median, prof.f0Median, 4.0) + d(p.f0Std, prof.f0Std, 3.0) + d(p.energyStd, prof.energyStd, 5.0) + d(p.onsetRate, prof.onsetRate, 2.0)
        val chan = min(1.0, VoiceAnalysis.ltasDistance(p.ltas, prof.ltas) / 8.0)
        val total = spk - 0.03 * pros - 0.03 * chan - penalty
        evaluated++
        return Triple(total, mapOf("ecapa" to (sEc ?: Double.NaN), "ve" to sVe, "xv" to sXv, "prosody" to pros, "channel" to chan, "penalty" to penalty), ve to xv)
    }

    fun run(progress: (Progress) -> Unit = {}, cancel: () -> Boolean = { false }): Result {
        require(bank.segments.isNotEmpty()) { "학습된 음성이 없습니다." }
        fun check() { if (cancel()) throw InterruptedException("cancelled") }
        val perGen = (cfg.candidatesPerRound / cfg.generativePerRound).coerceAtLeast(1)
        val nDec = maxOf(1, minOf(4, perGen / 3)); val nTilt = maxOf(1, perGen / nDec)
        val total = cfg.rounds * cfg.generativePerRound * nDec * nTilt + 1
        var done = 0
        val searchSeed = cfg.seed * 7919L
        val bestSeg = segs.indexOf(segs.maxBy { s -> (s.quality / 100.0) * min(1.0, s.seconds / 6.0) + 2 * VoiceBank.cos(s.xvec, xvT) })
        val base = Gen(maxOf(0, bestSeg), 0.5, 0.5, 0.8, 0.0)
        progress(Progress(0, cfg.rounds, 0, total, Double.NaN, Double.NaN, "기준 음성 측정 중"))
        val (t0, pen0) = generate(base, cfg.sentences, searchSeed)
        val s0 = score(decode(base, t0, null), pen0)
        val baseline = Scored(base, Variant(), s0.first, s0.second, s0.third.first, s0.third.second, null)
        done++
        val all = ArrayList<Scored>(); all.add(baseline)
        var best = baseline
        var dir: FloatArray? = null                       // current x-vector correction direction
        // warm start: a previous auto-learning result competes as an elite, so repeated runs keep improving
        val prev = bank.tuning
        val prevSeg = segs.indexOfFirst { it.id == prev.promptSegment }
        if (prev.iterations > 0 && prevSeg >= 0) {
            progress(Progress(0, cfg.rounds, done, total, baseline.score, baseline.score, "이전 학습 결과 측정 중"))
            val g = Gen(prevSeg, prev.params.exaggeration.toDouble(), prev.params.cfgWeight.toDouble(), prev.params.temperature.toDouble(), prev.veBlend)
            val v = Variant(if (prev.xvCorrection != null) 1.0 else 0.0, prev.tilt)
            val (t, pen) = generate(g, cfg.sentences, searchSeed)
            val s = score(Dsp.tilt(decode(g, t, prev.xvCorrection), SR, v.tilt), pen)
            val sc = Scored(g, v, s.first, s.second, s.third.first, s.third.second, prev.xvCorrection)
            all.add(sc)
            if (sc.score > best.score) { best = sc; dir = prev.xvCorrection }
        }
        val roundBest = ArrayList<Double>()
        for (r in 1..cfg.rounds) {
            val sigma = Math.pow(0.65, (r - 1).toDouble())
            val elites = all.sortedByDescending { it.score }.distinctBy { it.gen }.take(3)
            val gens = ArrayList<Gen>()
            gens.add(best.gen)                                   // the champion, re-checked with the updated correction
            while (gens.size < cfg.generativePerRound) gens.add(if (r == 1 && gens.size % 2 == 0) randomGen() else mutate(elites[rnd.nextInt(elites.size)].gen, sigma))
            var rb: Scored? = null
            val roundRaw = ArrayList<Scored>()
            for (g in gens) {
                check()
                progress(Progress(r, cfg.rounds, done, total, baseline.score, best.score, "${r}회차: 새 방식으로 말하는 중 (감정 ${"%.2f".format(g.ex)}, 속도감 ${"%.2f".format(g.cfg)}, 다양성 ${"%.2f".format(g.temp)})"))
                val (toks, pen) = generate(g, cfg.sentences, searchSeed)
                val strengths = LinkedHashSet<Double>().apply { add(0.0); if (dir != null) { add(best.v.xs.coerceAtLeast(0.5)); while (size < nDec) add(mutateVariant(best.v, sigma).xs) } }.take(nDec)
                for (xs in strengths) {
                    check()
                    val audio = decode(g, toks, xvCorrection(dir, xs))
                    val tilts = LinkedHashSet<Double>().apply { add(0.0); add(best.v.tilt); while (size < nTilt) add(mutateVariant(best.v, sigma).tilt) }.take(nTilt)
                    for (tilt in tilts) {
                        val s = score(Dsp.tilt(audio, SR, tilt), pen)
                        val sc = Scored(g, Variant(xs, tilt), s.first, s.second, s.third.first, s.third.second, dir)
                        all.add(sc); done++
                        if (tilt == 0.0) roundRaw.add(sc)
                        if (rb == null || sc.score > rb.score) rb = sc
                    }
                }
                progress(Progress(r, cfg.rounds, done, total, baseline.score, maxOf(best.score, rb?.score ?: best.score), "${r}회차 진행 중"))
            }
            rb!!
            roundBest.add(rb.score)
            if (rb.score > best.score) best = rb
            // learn from its own output: the mean x-vector gap of this round's best uncoloured results becomes the
            // new correction direction (decoder only), whose strength the next round searches
            val top = roundRaw.sortedByDescending { it.score }.take(3)
            val baseDir = best.xvDir?.let { d -> FloatArray(192) { (d[it] * best.v.xs).toFloat() } } ?: FloatArray(192)
            dir = FloatArray(192) { i -> (baseDir[i] + cfg.learningRate * sigma * (xvT[i] - top.sumOf { it.xvOut[i].toDouble() } / top.size)).toFloat() }
            progress(Progress(r, cfg.rounds, done, total, baseline.score, best.score, "${r}회차 완료: 최고 ${"%.3f".format(best.score)} (기준 ${"%.3f".format(baseline.score)})"))
        }
        // held-out check: different sentences, fresh seeds, ECAPA only; the search maximum is optimistic
        progress(Progress(cfg.rounds, cfg.rounds, done, total, baseline.score, best.score, "새 문장으로 최종 검증 중"))
        fun verify(g: Gen, v: Variant, xvc: FloatArray?): Double {
            check()
            val (toks, _) = generate(g, cfg.verifySentences, cfg.seed * 104729L + 17)
            val x16 = Resampler.resample(Dsp.tilt(decode(g, toks, xvc), SR, v.tilt), SR, 16000)
            return ecT?.let { VoiceBank.cos(engine.ecapa(x16), it) } ?: (VoiceBank.cos(engine.veEmbed(x16), veT) + VoiceBank.cos(VoiceBank.unit(engine.xvector(x16)), xvT)) / 2
        }
        val bestXvc = best.xvDir?.let { xvCorrection(it, best.v.xs) }
        val vBest = if (best === baseline) Double.NaN else verify(best.gen, best.v, bestXvc)
        val vBase = verify(baseline.gen, Variant(), null)
        verified = vBest to vBase
        val accepted = !vBest.isNaN() && vBest - vBase >= cfg.acceptMargin
        val win = if (accepted) best else baseline
        val t = tuningOf(win.gen, win.v, win.xvDir).copy(history = listOf(baseline.score) + roundBest, iterations = bank.tuning.iterations + 1)
        return Result(t, baseline.score, roundBest, win, evaluated, accepted)
    }

    companion object { const val SR = 24000 }
}
