package io.github.koreanvoice.expressive

import io.github.koreanvoice.audio.AudioTools
import io.github.koreanvoice.audio.Dsp
import io.github.koreanvoice.audio.Psola
import java.io.File
import kotlin.test.Test

/** Re-fits the environment of the validated banks with the ECAPA guard and renders D2. -Dstudy=envrefit */
class CbEnvRefit {
    val sents = CbValidationTest().sents
    @Test fun refit() {
        if (System.getProperty("study") != "envrefit") return
        val out = File(System.getProperty("outDir")); val dir = File(out, "validate")
        CbEngine(File(System.getProperty("cbModelDir")), 4).use { eng ->
            for (spk in listOf("spk0", "spk1", "spk2", "spk3")) {
                val bank = VoiceBank(File(out, "bank_$spk"))
                val e = AutoTuner(eng, bank, AutoTuner.Config.STANDARD).fitEnvironment(bank.tuning)
                bank.tuning = bank.tuning.copy(env = e); bank.save()
                println("ENV $spk room=${e?.room} wet=${e?.wet} noise=${e?.noiseDb} eqMax=${e?.eq?.maxOf { kotlin.math.abs(it) }} before=${e?.before} after=${e?.after}")
                val v = bank.compose(); val t = bank.tuning
                val tone = e?.noiseDb?.let { db -> bank.roomTone()?.takeIf { it.size > 24000 }?.let { Dsp.BackgroundMixer(it, 24000, db, 0.0) } }
                for ((k, s) in sents.withIndex()) {
                    var y = AudioTools.trimSilence(eng.synthesize(s, v, t.params.copy(seed = 4242L + k)).audio, 24000)
                    y = Psola.shapeEndings(y, 24000, t.endShift, t.endStretch)
                    y = Dsp.tilt(Dsp.tempoPitch(y, 24000, t.tempo, t.pitch), 24000, t.tilt)
                    val g = AudioTools.normalizeGain(y, 24000); for (i in y.indices) y[i] *= g
                    if (e != null) y = Dsp.applyEnvironment(y, 24000, e.eq, Dsp.Room.of(e.room) ?: Dsp.Room.NONE, e.wet)
                    if (tone != null) y = tone.mix(AudioTools.silence(24000, 200) + y + AudioTools.silence(24000, 300))
                    AudioTools.writeWav(File(dir, "${spk}_D2_$k.wav"), y, 24000)
                }
            }
        }
    }
}
