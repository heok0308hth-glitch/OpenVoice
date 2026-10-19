package io.github.koreanvoice.expressive

import io.github.koreanvoice.audio.AudioTools
import io.github.koreanvoice.audio.Dsp
import io.github.koreanvoice.audio.Resampler
import io.github.koreanvoice.audio.VoiceAnalysis
import kotlin.math.abs
import kotlin.math.min

/**
 * Auto-learning: repeatedly tries many different ways of speaking (reference clip, emotion intensity,
 * pacing/guidance, randomness, identity blend, pitch, spectral colour, tempo, room) and keeps the one
 * that sounds most like the real recordings — measured on-device by two independent speaker encoders
 * plus pitch/intonation/dynamics/rhythm/spectrum statistics.
 *
 * After every round the best result is fed back: the gap between its speaker embeddings and the real
 * voice is added to the voice identity ("learn from your own output"), and the next round searches
 * around the elite candidates with a narrower spread. Corrections that do not improve the next round
 * are rolled back, so learning can only keep or raise the score.
 */
class AutoTuner(private val engine: CbEngine, private val bank: VoiceBank, private val cfg: Config = Config()) {
    data class Config(
        val rounds: Int = 5,
        /** Candidates evaluated per round (generative × DSP variants). */
        val candidatesPerRound: Int = 100,
        /** How many of them are freshly synthesised (the rest are DSP variants of those). */
        val generativePerRound: Int = 12,
        val sentences: List<String> = listOf("오늘은 날씨가 정말 좋네요.", "잠깐만요, 제 이야기 좀 들어 보실래요?"),
        val seed: Long = 1,
        /** Step size of the identity correction. */
        val learningRate: Double = 0.6,
    ) {
        companion object {
            val QUICK = Config(rounds = 3, candidatesPerRound = 30, generativePerRound = 6)
            val STANDARD = Config(rounds = 5, candidatesPerRound = 100, generativePerRound = 10)
            val PRECISE = Config(rounds = 5, candidatesPerRound = 100, generativePerRound = 20)
        }
    }

    data class Gen(val seg: Int, val ex: Double, val cfg: Double, val temp: Double, val blend: Double)
    data class Fx(val pitch: Double = 0.0, val tilt: Double = 0.0, val tempo: Double = 1.0, val wet: Double = 0.0)

    class Scored(val gen: Gen, val fx: Fx, val score: Double, val parts: Map<String, Double>,
                 val veOut: FloatArray, val xvOut: FloatArray, val veCorr: FloatArray?, val xvCorr: FloatArray?)

    data class Progress(val round: Int, val rounds: Int, val done: Int, val total: Int, val baseline: Double, val best: Double, val message: String)

    data class Result(val tuning: VoiceBank.Tuning, val baseline: Double, val roundBest: List<Double>, val best: Scored, val evaluated: Int)

    private val rnd = java.util.Random(cfg.seed)
    private val segs = bank.segments.filter { it.seconds >= 3 }.sortedByDescending { it.quality * min(it.seconds, 10.0) }.take(8).ifEmpty { bank.segments }
    private val veT = bank.identityVe()
    private val xvT = bank.identityXvec()
    private val prof = bank.targetProfile()
    private var evaluated = 0
    /** (tuned, baseline) mean scores of the final re-check with fresh seeds. */
    var verified: Pair<Double, Double>? = null; private set

    private fun clamp(v: Double, lo: Double, hi: Double) = v.coerceIn(lo, hi)

    private fun randomGen() = Gen(rnd.nextInt(segs.size), 0.25 + rnd.nextDouble() * 0.9, 0.2 + rnd.nextDouble() * 0.6, 0.55 + rnd.nextDouble() * 0.4, rnd.nextDouble() * 0.6)

    private fun mutate(g: Gen, s: Double) = Gen(
        if (rnd.nextDouble() < 0.1 + 0.3 * s) rnd.nextInt(segs.size) else g.seg,
        clamp(g.ex + rnd.nextGaussian() * 0.25 * s, 0.25, 1.4), clamp(g.cfg + rnd.nextGaussian() * 0.15 * s, 0.15, 0.9),
        clamp(g.temp + rnd.nextGaussian() * 0.12 * s, 0.5, 1.0), clamp(g.blend + rnd.nextGaussian() * 0.2 * s, 0.0, 0.8))

    private fun mutateFx(f: Fx, s: Double) = Fx(
        clamp(f.pitch + rnd.nextGaussian() * 1.0 * s, -3.0, 3.0), clamp(f.tilt + rnd.nextGaussian() * 2.0 * s, -6.0, 6.0),
        clamp(f.tempo + rnd.nextGaussian() * 0.06 * s, 0.85, 1.15), clamp(f.wet + rnd.nextGaussian() * 0.08 * s, 0.0, 0.3))

    fun tuningOf(g: Gen, veCorr: FloatArray?, xvCorr: FloatArray?) = VoiceBank.Tuning(
        CbParams(exaggeration = g.ex.toFloat(), cfgWeight = g.cfg.toFloat(), temperature = g.temp.toFloat()),
        veBlend = g.blend, promptSegment = segs[g.seg].id, veCorrection = veCorr, xvCorrection = xvCorr)

    /** Synthesise the probe sentences with a candidate voice. Returns 24 kHz audio and an integrity penalty. */
    private fun synth(g: Gen, veCorr: FloatArray?, xvCorr: FloatArray?): Pair<FloatArray, Double> {
        val voice = bank.compose(segs[g.seg], tuningOf(g, veCorr, xvCorr))
        val p = CbParams(exaggeration = g.ex.toFloat(), cfgWeight = g.cfg.toFloat(), temperature = g.temp.toFloat(), seed = rnd.nextLong(), maxNewTokens = 400)
        val parts = ArrayList<FloatArray>(); var penalty = 0.0
        for (s in cfg.sentences) {
            val r = engine.synthesize(s, voice, p)
            val a = AudioTools.trimSilence(r.audio, SR)
            val syl = s.count { it in '가'..'힣' }.coerceAtLeast(1)
            val sec = a.size.toDouble() / SR
            if (sec < syl / 14.0 || sec > syl / 2.5) penalty += 0.3       // garbled / truncated / babbling output
            parts.add(a); parts.add(AudioTools.silence(SR, 300))
        }
        val all = FloatArray(parts.sumOf { it.size }); var o = 0
        for (a in parts) { System.arraycopy(a, 0, all, o, a.size); o += a.size }
        return all to penalty
    }

    fun applyFx(x: FloatArray, f: Fx): FloatArray {
        var y = Dsp.tempoPitch(x, SR, f.tempo, f.pitch)
        y = Dsp.tilt(y, SR, f.tilt)
        if (f.wet > 0.005) y = Dsp.reverb(y, SR, Dsp.Room.ROOM, f.wet, 0.3)
        return y
    }

    /** Similarity of [x] to the real recordings (higher is better). */
    fun score(x: FloatArray, penalty: Double = 0.0): Triple<Double, Map<String, Double>, Pair<FloatArray, FloatArray>> {
        val x16 = Resampler.resample(x, SR, 16000)
        val ve = engine.veEmbed(x16)
        val xv = VoiceBank.unit(engine.xvector(x16))
        val sVe = VoiceBank.cos(ve, veT); val sXv = VoiceBank.cos(xv, xvT)
        val p = VoiceAnalysis.analyze(x, SR)
        fun d(a: Double, b: Double, scale: Double) = if (a.isNaN() || b.isNaN()) 1.0 else min(1.0, abs(a - b) / scale)
        val pros = d(p.f0Median, prof.f0Median, 4.0) + d(p.f0Std, prof.f0Std, 3.0) + d(p.energyStd, prof.energyStd, 5.0) + d(p.onsetRate, prof.onsetRate, 2.0)
        val chan = min(1.0, VoiceAnalysis.ltasDistance(p.ltas, prof.ltas) / 8.0) + min(1.0, abs(p.tailRatio - prof.tailRatio) * 2)
        val total = 0.5 * sVe + 0.5 * sXv - 0.06 * pros - 0.05 * chan - penalty
        evaluated++
        return Triple(total, mapOf("ve" to sVe, "xv" to sXv, "prosody" to pros, "channel" to chan, "penalty" to penalty), ve to xv)
    }

    fun run(progress: (Progress) -> Unit = {}, cancel: () -> Boolean = { false }): Result {
        require(bank.segments.isNotEmpty()) { "학습된 음성이 없습니다." }
        val perGen = (cfg.candidatesPerRound / cfg.generativePerRound).coerceAtLeast(1)
        val total = cfg.rounds * cfg.generativePerRound * perGen + 1
        var done = 0
        val bestSeg = segs.indexOf(segs.maxBy { s -> (s.quality / 100.0) * min(1.0, s.seconds / 6.0) + 2 * VoiceBank.cos(s.xvec, xvT) })
        val base = Gen(maxOf(0, bestSeg), 0.5, 0.5, 0.8, 0.0)
        progress(Progress(0, cfg.rounds, 0, total, Double.NaN, Double.NaN, "기준 음성 측정 중"))
        val (a0, pen0) = synth(base, null, null)
        val s0 = score(a0, pen0)
        val baseline = Scored(base, Fx(), s0.first, s0.second, s0.third.first, s0.third.second, null, null)
        done++
        val all = ArrayList<Scored>(); all.add(baseline)
        var best = baseline
        var veCorr: FloatArray? = null; var xvCorr: FloatArray? = null
        // warm start: a previous auto-learning result competes as an elite, so repeated runs keep improving
        val prev = bank.tuning
        val prevSeg = segs.indexOfFirst { it.id == prev.promptSegment }
        if (prev.iterations > 0 && prevSeg >= 0) {
            progress(Progress(0, cfg.rounds, done, total, baseline.score, baseline.score, "이전 학습 결과 측정 중"))
            val g = Gen(prevSeg, prev.params.exaggeration.toDouble(), prev.params.cfgWeight.toDouble(), prev.params.temperature.toDouble(), prev.veBlend)
            val f = Fx(prev.pitch, prev.tilt, prev.tempo, prev.wet)
            val (a, pen) = synth(g, prev.veCorrection, prev.xvCorrection)
            val s = score(if (f == Fx()) a else applyFx(a, f), pen)
            val sc = Scored(g, f, s.first, s.second, s.third.first, s.third.second, prev.veCorrection, prev.xvCorrection)
            all.add(sc)
            if (sc.score > best.score) { best = sc; veCorr = sc.veCorr; xvCorr = sc.xvCorr }
        }
        val roundBest = ArrayList<Double>()
        for (r in 1..cfg.rounds) {
            val sigma = Math.pow(0.65, (r - 1).toDouble())
            val elites = all.sortedByDescending { it.score }.distinctBy { it.gen }.take(3)
            val gens = ArrayList<Gen>()
            gens.add(best.gen)                                   // re-check the champion under the new identity correction
            while (gens.size < cfg.generativePerRound) {
                gens.add(if (r == 1 && gens.size % 2 == 0) randomGen() else mutate(elites[rnd.nextInt(elites.size)].gen, sigma))
            }
            var rb: Scored? = null
            for (g in gens) {
                if (cancel()) throw InterruptedException("cancelled")
                progress(Progress(r, cfg.rounds, done, total, baseline.score, best.score, "${r}회차: 새 방식으로 합성 중 (감정 ${"%.2f".format(g.ex)}, 속도감 ${"%.2f".format(g.cfg)})"))
                val (audio, pen) = synth(g, veCorr, xvCorr)
                val fxs = ArrayList<Fx>(); fxs.add(Fx()); fxs.add(best.fx)
                while (fxs.size < perGen) fxs.add(mutateFx(if (rnd.nextBoolean()) best.fx else Fx(), sigma))
                for (f in fxs.distinct().take(perGen)) {
                    if (cancel()) throw InterruptedException("cancelled")
                    val y = if (f == Fx()) audio else applyFx(audio, f)
                    val s = score(y, pen)
                    val sc = Scored(g, f, s.first, s.second, s.third.first, s.third.second, veCorr, xvCorr)
                    all.add(sc); done++
                    if (rb == null || sc.score > rb.score) rb = sc
                }
                progress(Progress(r, cfg.rounds, done, total, baseline.score, maxOf(best.score, rb?.score ?: best.score), "${r}회차 진행 중"))
            }
            rb!!
            roundBest.add(rb.score)
            if (rb.score > best.score) best = rb
            else { veCorr = best.veCorr; xvCorr = best.xvCorr }    // roll back a correction that did not help
            // learn from the round's best output: move the identity by the remaining embedding gap
            val lr = cfg.learningRate * sigma
            val vb = best.veCorr ?: FloatArray(256); val xb = best.xvCorr ?: FloatArray(192)
            veCorr = FloatArray(256) { (vb[it] + lr * (veT[it] - best.veOut[it])).toFloat() }
            xvCorr = FloatArray(192) { (xb[it] + lr * (xvT[it] - best.xvOut[it])).toFloat() }
            progress(Progress(r, cfg.rounds, done, total, baseline.score, best.score, "${r}회차 완료: 최고 ${"%.3f".format(best.score)} (기준 ${"%.3f".format(baseline.score)})"))
        }
        // final check with fresh random seeds (the search maximum is optimistic): keep the tuned voice
        // only if it still beats the starting point on average
        progress(Progress(cfg.rounds, cfg.rounds, done, total, baseline.score, best.score, "최종 검증 중"))
        fun verify(g: Gen, f: Fx, vc: FloatArray?, xc: FloatArray?): Double = (0 until VERIFY_RUNS).map {
            if (cancel()) throw InterruptedException("cancelled")
            val (a, pen) = synth(g, vc, xc); score(if (f == Fx()) a else applyFx(a, f), pen).first
        }.average()
        val vBest = verify(best.gen, best.fx, best.veCorr, best.xvCorr)
        val vBase = verify(baseline.gen, Fx(), null, null)
        verified = vBest to vBase
        if (vBest < vBase) best = baseline
        val t = tuningOf(best.gen, best.veCorr, best.xvCorr).copy(
            pitch = best.fx.pitch, tilt = best.fx.tilt, tempo = best.fx.tempo, wet = best.fx.wet,
            history = listOf(baseline.score) + roundBest, iterations = bank.tuning.iterations + 1)
        return Result(t, baseline.score, roundBest, best, evaluated)
    }

    companion object { const val SR = 24000; const val VERIFY_RUNS = 2 }
}
