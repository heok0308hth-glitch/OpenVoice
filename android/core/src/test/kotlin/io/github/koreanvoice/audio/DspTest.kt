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
