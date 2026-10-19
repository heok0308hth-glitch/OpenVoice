package io.github.koreanvoice.scratch

import io.github.koreanvoice.audio.*
import io.github.koreanvoice.expressive.*
import java.io.File
import kotlin.test.Test

class PsolaIdentity {
    @Test fun identity() {
        if (System.getProperty("study") != "psolaid") return
        val W = File(System.getProperty("outDir"))
        CbEngine(File(System.getProperty("cbModelDir")), 4).use { eng ->
            for (spk in listOf("spk0", "spk1", "spk2", "spk3")) {
                val bank = VoiceBank(File(W, "bank_$spk"))
                val ref = bank.segmentAudio(bank.segments.first())
                fun e(x: FloatArray) = eng.ecapa(Resampler.resample(x, 24000, 16000))
                val base = e(ref)
                val row = StringBuilder("$spk")
                for (st in listOf(-2.0, -1.0, 1.0, 2.0)) {
                    val p = VoiceBank.cos(e(Psola.pitchShift(ref, 24000, st)), base)
                    val r = VoiceBank.cos(e(Dsp.pitchShift(ref, 24000, st)), base)
                    row.append("  ${st}st psola=${"%.3f".format(p)} resample=${"%.3f".format(r)}")
                }
                val t = VoiceBank.cos(e(Psola.tempoPitch(ref, 24000, 1.15, 0.0)), base); val w = VoiceBank.cos(e(Dsp.timeStretch(ref, 24000, 1.15)), base)
                row.append("  tempo1.15 psola=${"%.3f".format(t)} wsola=${"%.3f".format(w)}")
                val en = VoiceBank.cos(e(Psola.shapeEndings(ref, 24000, 2.0, 1.2)), base)
                row.append("  endings(+2st,1.2)=${"%.3f".format(en)}")
                println("PSID $row")
            }
        }
    }
}
