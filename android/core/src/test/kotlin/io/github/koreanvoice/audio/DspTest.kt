package io.github.koreanvoice.audio

import kotlin.math.PI
import kotlin.math.abs
import kotlin.math.pow
import kotlin.math.sin
import kotlin.test.Test
import kotlin.test.assertTrue

class DspTest {
    private val sr = 24000

    /** Glottal-ish harmonic signal with syllable-like amplitude envelope. */
    private fun voice(f0: Double, sec: Double = 2.0): FloatArray = FloatArray((sr * sec).toInt()) { i ->
        val t = i.toDouble() / sr
        var v = 0.0
        for (h in 1..20) v += sin(2 * PI * f0 * h * t) / h
        val env = 0.5 + 0.5 * sin(2 * PI * 3.0 * t)
        (0.15 * v * env).toFloat()
    }

    private fun medianF0(x: FloatArray): Double {
        val f = VoiceAnalysis.yin(Resampler.resample(x, sr, 16000), 16000).filter { it > 0 }.sorted()
        return f[f.size / 2]
    }

    @Test
    fun convolverMatchesDirectConvolution() {
        val rnd = java.util.Random(1)
        val ir = FloatArray(5000) { (rnd.nextGaussian() * 0.1).toFloat() }
        val x = FloatArray(20000) { (rnd.nextGaussian()).toFloat() }
        val c = StreamingConvolver(ir, 1024)
        val y = c.process(x.copyOf(x.size + c.latency)).copyOfRange(c.latency, c.latency + x.size)
        var maxErr = 0.0
        for (n in 0 until x.size step 97) {
            var acc = 0.0; for (k in ir.indices) if (n - k >= 0) acc += ir[k] * x[n - k]
            maxErr = maxOf(maxErr, abs(acc - y[n]))
        }
        println("convolver max err $maxErr")
        assertTrue(maxErr < 1e-3)
    }

    @Test
    fun pitchAndTempo() {
        val x = voice(150.0)
        val base = medianF0(x)
        val up = medianF0(Dsp.pitchShift(x, sr, 2.0))
        val down = medianF0(Dsp.pitchShift(x, sr, -3.0))
        val fast = Dsp.timeStretch(x, sr, 1.25)
        val slow = Dsp.timeStretch(x, sr, 0.8)
        println("f0 base=$base up2=$up (exp ${base * 2.0.pow(2 / 12.0)}) down3=$down fastLen=${fast.size.toDouble() / x.size} slowLen=${slow.size.toDouble() / x.size} fastF0=${medianF0(fast)}")
        assertTrue(abs(up / base - 2.0.pow(2 / 12.0)) < 0.01)
        assertTrue(abs(down / base - 2.0.pow(-3 / 12.0)) < 0.01)
        assertTrue(abs(fast.size.toDouble() / x.size - 0.8) < 0.01 && abs(slow.size.toDouble() / x.size - 1.25) < 0.01)
        assertTrue(abs(medianF0(fast) / base - 1) < 0.01)
    }

    @Test
    fun effectsChangeMeasuredCharacter() {
        val x = voice(140.0, 3.0)
        val dry = VoiceAnalysis.analyze(x, sr)
        val hall = VoiceAnalysis.analyze(Dsp.reverb(x, sr, Dsp.Room.HALL, 0.5), sr)
        val wh = VoiceAnalysis.analyze(Dsp.whisper(x, sr), sr)
        val phone = VoiceAnalysis.analyze(Dsp.state(x, sr, Dsp.State.PHONE), sr)
        println("tail dry=${dry.tailRatio} hall=${hall.tailRatio}; voiced dry=${dry.voicedRatio} whisper=${wh.voicedRatio}; ltasDist phone=${VoiceAnalysis.ltasDistance(dry.ltas, phone.ltas)}")
        assertTrue(hall.tailRatio > dry.tailRatio)
        assertTrue(wh.voicedRatio < dry.voicedRatio * 0.5)
        assertTrue(VoiceAnalysis.ltasDistance(dry.ltas, phone.ltas) > 5)
        for (s in Dsp.State.entries) { val y = Dsp.state(x, sr, s); assertTrue(y.all { it.isFinite() }, s.name) }
        for (r in Dsp.Room.entries) { val y = Dsp.reverb(x, sr, r); assertTrue(y.all { it.isFinite() } && y.size >= x.size, r.name) }
        for (t in Dsp.Tone.entries) assertTrue(Dsp.tone(x, sr, t).all { it.isFinite() })
        val bg = Dsp.BackgroundMixer(Dsp.noiseBed("빗소리", sr), sr).mix(x)
        assertTrue(bg.all { it.isFinite() })
    }
}

class EqTest {
    @Test
    fun designedEqFollowsBandCurve() {
        val sr = 24000
        val rnd = java.util.Random(3)
        val noise = FloatArray(sr * 4) { (rnd.nextGaussian() * 0.1).toFloat() }
        val curve = DoubleArray(VoiceAnalysis.LTAS_BANDS) { if (it in 14..17) 6.0 else 0.0 }
        val y = Dsp.fir(noise, Dsp.designEq(curve, sr))
        val a = VoiceAnalysis.ltas(noise, sr); val b = VoiceAnalysis.ltas(y, sr)
        val diff = DoubleArray(a.size) { b[it] - a[it] }
        val boost = (15..16).map { diff[it] }.average() - (0..8).map { diff[it] }.average()
        println("eq boost in target bands ${"%.2f".format(boost)} dB, length ${y.size == noise.size}")
        kotlin.test.assertTrue(boost in 4.5..7.5 && y.size == noise.size)
    }
}

class PsolaTest {
    private val sr = 24000

    /** Vowel-like signal: glottal pulses at f0 through two fixed formant resonators (envelope independent of f0). */
    private fun vowel(f0: Double, sec: Double = 1.5): FloatArray {
        val n = (sr * sec).toInt(); val x = FloatArray(n)
        var phase = 0.0
        for (i in 0 until n) { phase += f0 / sr; if (phase >= 1) { phase -= 1; x[i] = 1f } }
        var y = Biquad.peak(700.0, sr, 18.0, 6.0).process(x)
        y = Biquad.peak(1800.0, sr, 14.0, 6.0).process(y)
        y = Biquad.lowPass(4000.0, sr).process(y)
        val m = y.maxOf { kotlin.math.abs(it) }; for (i in y.indices) y[i] *= 0.5f / m
        return y
    }

    private fun f0(x: FloatArray): Double {
        val f = VoiceAnalysis.yin(Resampler.resample(x, sr, 16000), 16000).filter { it > 0 }.sorted(); return f[f.size / 2]
    }

    @Test
    fun pitchShiftKeepsFormants() {
        val x = vowel(140.0)
        val p = Psola.pitchShift(x, sr, 3.0)
        val f = Math.pow(2.0, 3 / 12.0)          // resampling-based shift (formants move), for comparison
        val r = Resampler.resample(Dsp.timeStretch(x, sr, 1 / f), (sr * f / 100).toInt() * 100, sr)
        val ratio = f0(p) / f0(x)
        val dP = VoiceAnalysis.ltasDistance(VoiceAnalysis.ltas(x, sr), VoiceAnalysis.ltas(p, sr))
        val dR = VoiceAnalysis.ltasDistance(VoiceAnalysis.ltas(x, sr), VoiceAnalysis.ltas(r, sr))
        println("psola f0 ratio=$ratio (exp ${Math.pow(2.0, 3 / 12.0)}) len=${p.size.toDouble() / x.size} envelope dist psola=$dP resample=$dR")
        kotlin.test.assertTrue(kotlin.math.abs(ratio - Math.pow(2.0, 3 / 12.0)) < 0.02)
        kotlin.test.assertTrue(kotlin.math.abs(p.size.toDouble() / x.size - 1) < 0.03)
        kotlin.test.assertTrue(dP < dR)
    }

    @Test
    fun endingShapingRaisesFinalPitch() {
        // two "phrases" of 0.8 s separated by 0.4 s silence
        val a = vowel(130.0, 0.8); val gap = FloatArray(sr * 2 / 5)
        val x = a + gap + a + gap
        val y = Psola.shapeEndings(x, sr, 3.0, 1.2)
        val tr = Psola.track(x, sr)
        println("phrase ends=${Psola.phraseEnds(tr, x.size, sr)} len ratio=${y.size.toDouble() / x.size}")
        val endY = VoiceAnalysis.yin(Resampler.resample(y, sr, 16000), 16000)
        // pitch in the last voiced 100 ms of the first phrase must be ≥ 1.5 st higher than its start
        val voiced = endY.withIndex().filter { it.value > 0 }
        val firstRun = voiced.zipWithNext().takeWhile { (a, b) -> b.index - a.index <= 5 }.map { it.second }.let { listOf(voiced.first()) + it }
        val startF = firstRun.take(20).map { it.value }.average(); val endF = firstRun.takeLast(8).map { it.value }.average()
        println("first phrase f0 start=$startF end=$endF run=${firstRun.size} tail=${firstRun.takeLast(40).map { it.value.toInt() }}")
        kotlin.test.assertEquals(2, Psola.phraseEnds(tr, x.size, sr).size)
        kotlin.test.assertTrue(12 * Math.log(endF / startF) / Math.log(2.0) > 1.5)
        kotlin.test.assertTrue(y.size > x.size * 1.02)
    }
}
