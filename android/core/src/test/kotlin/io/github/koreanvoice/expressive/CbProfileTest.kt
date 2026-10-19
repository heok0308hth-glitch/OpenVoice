package io.github.koreanvoice.expressive

import io.github.koreanvoice.audio.Dsp
import io.github.koreanvoice.audio.Resampler
import io.github.koreanvoice.audio.VoiceAnalysis
import java.io.File
import kotlin.test.Test

/** Timing of the auto-learning building blocks. -Dstudy=profile */
class CbProfileTest {
    private val modelDir = File(System.getProperty("cbModelDir").orEmpty())
    private val out = File(System.getProperty("outDir").orEmpty())

    @Test
    fun profileScoring() {
        if (System.getProperty("study") != "profile") return
        CbEngine(modelDir, 4).use { eng ->
            val bank = VoiceBank(File(out, "bank_spk0"))
            val a = bank.segmentAudio(bank.segments.first()).copyOf(24000 * 8)
            fun t(label: String, n: Int = 3, f: () -> Unit) { f(); val t0 = System.nanoTime(); repeat(n) { f() }; println("PROF $label ${"%.1f".format((System.nanoTime() - t0) / 1e6 / n)} ms") }
            val a16 = Resampler.resample(a, 24000, 16000)
            t("resample") { Resampler.resample(a, 24000, 16000) }
            t("veEmbed") { eng.veEmbed(a16) }
            t("xvector") { eng.xvector(a16) }
            t("ecapa") { eng.ecapa(a16) }
            t("analyze") { VoiceAnalysis.analyze(a, 24000) }
            t("tempoPitch") { Dsp.tempoPitch(a, 24000, 1.05, 1.0) }
            t("tilt") { Dsp.tilt(a, 24000, 2.0) }
            t("reverb") { Dsp.reverb(a, 24000, Dsp.Room.ROOM, 0.1, 0.3) }
            val v = bank.compose()
            t("synth 1 sentence", 2) { eng.synthesize("어제 서점에 들렀다가 오래된 친구를 우연히 만났어요.", v, CbParams(seed = 1)) }
        }
    }
}
