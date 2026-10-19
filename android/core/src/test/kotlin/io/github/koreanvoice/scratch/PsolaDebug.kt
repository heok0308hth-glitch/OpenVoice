package io.github.koreanvoice.scratch

import io.github.koreanvoice.audio.*
import kotlin.test.Test

class PsolaDebug {
    val sr = 24000
    fun vowel(f0: Double, sec: Double): FloatArray {
        val n = (sr * sec).toInt(); val x = FloatArray(n); var ph = 0.0
        for (i in 0 until n) { ph += f0 / sr; if (ph >= 1) { ph -= 1; x[i] = 1f } }
        var y = Biquad.peak(700.0, sr, 18.0, 6.0).process(x); y = Biquad.peak(1800.0, sr, 14.0, 6.0).process(y); y = Biquad.lowPass(4000.0, sr).process(y)
        val m = y.maxOf { kotlin.math.abs(it) }; for (i in y.indices) y[i] *= 0.5f / m; return y
    }
    fun f0s(x: FloatArray) = VoiceAnalysis.yin(Resampler.resample(x, sr, 16000), 16000).map { it.toInt() }
    @Test fun dbg() {
        if (System.getProperty("study") != "psola") return
        val a = vowel(130.0, 0.8)
        println("A const+3: " + f0s(Psola.modify(a, sr, { 3.0 }, { 1.0 })).takeLast(40))
        println("B ramp: " + f0s(Psola.modify(a, sr, { i -> 3.0 * i / a.size }, { 1.0 })).takeLast(40))
        val x = a + FloatArray(9600)
        println("X orig: " + f0s(x).drop(40).take(50))
        println("C shape no stretch: " + f0s(Psola.shapeEndings(x, sr, 3.0, 1.0)).drop(40).take(50))
        println("E manual no silence: " + f0s(Psola.modify(a, sr, { i -> if (i in 11280 until 19440) 3.0 * ((i - 11280) / 7680.0).let { it * it } else 0.0 }, { 1.0 })).drop(40).take(50))
        println("F linear whole with silence: " + f0s(Psola.modify(x, sr, { i -> 3.0 * i / a.size }, { 1.0 })).drop(40).take(50))
        AudioTools.writeWav(java.io.File("/tmp/claude-0/-home-user-OpenVoice/59aa00c6-12e1-5485-9e43-86c42d67ad88/scratchpad/psola_e.wav"), Psola.modify(a, sr, { i -> if (i in 11280 until 19440) 3.0 * ((i - 11280) / 7680.0).let { it * it } else 0.0 }, { 1.0 }), sr)
        println("D manual: " + f0s(Psola.modify(x, sr, { i -> if (i in 11280 until 19440) 3.0 * ((i - 11280) / 7680.0).let { it * it } else 0.0 }, { 1.0 })).drop(40).take(50))
    }
}
