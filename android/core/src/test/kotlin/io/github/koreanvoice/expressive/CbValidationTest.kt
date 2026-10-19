package io.github.koreanvoice.expressive

import io.github.koreanvoice.audio.AudioTools
import io.github.koreanvoice.audio.Dsp
import io.github.koreanvoice.audio.Psola
import io.github.koreanvoice.audio.Resampler
import io.github.koreanvoice.audio.VoiceCriteria
import java.io.File
import kotlin.test.Test

/**
 * Auto-learning validation on unseen sentences with identical seeds:
 *  B = voice bank defaults, C = after auto-learning, D = C + recording-environment reproduction.
 * -Dstudy=validate -Dtuner=quick|standard|precise -Dspeakers=spk0,... -DkrefDir ; log in <outDir>/validate/log.txt
 */
class CbValidationTest {
    private val modelDir = File(System.getProperty("cbModelDir").orEmpty())
    private val out = File(System.getProperty("outDir").orEmpty().ifEmpty { "build/test-audio/v2" })
    private val kref = File(System.getProperty("krefDir").orEmpty())
    val sents = listOf("그날 밤, 우리는 오래된 기차역에서 처음 만났습니다.", "정말요? 그게 사실이라면 너무 기쁜 소식이네요!",
        "회의는 다음 주 화요일 오후 세 시로 변경되었습니다.", "괜찮아, 천천히 해도 돼. 내가 옆에 있을게.",
        "창밖으로 비가 내리기 시작하자 사람들은 서둘러 우산을 폈다.", "이번 여름에는 가족과 함께 제주도로 여행을 가려고 합니다.")

    @Test
    fun validate() {
        if (System.getProperty("study") != "validate" || !File(modelDir, "t3_lm.onnx").exists()) return
        val dir = File(out, "validate").apply { mkdirs() }
        val log = File(dir, "log.txt")
        fun say(s: String) { println(s); log.appendText(s + "\n") }
        val cfg = when (System.getProperty("tuner")) { "quick" -> AutoTuner.Config.QUICK; "precise" -> AutoTuner.Config.PRECISE; else -> AutoTuner.Config.STANDARD }
        val speakers = System.getProperty("speakers").orEmpty().ifEmpty { "spk0,spk1,spk2,spk3" }.split(',')
        fun f(v: Double) = "%.3f".format(v)
        CbEngine(modelDir, 4).use { eng ->
            for (spk in speakers) {
                val bank = VoiceBank(File(out, "bank_$spk"))
                bank.tuning = VoiceBank.Tuning(); bank.generations.clear()
                if (bank.roomTone() == null && File(kref, "$spk.wav").exists()) {
                    val (x, sr) = File(kref, "$spk.wav").inputStream().use { AudioTools.readWav(it) }
                    bank.appendRoomTone(VoiceBank.extractRoomTone(Resampler.resample(x, sr, 24000)))
                }
                say("$spk room tone ${"%.1f".format((bank.roomTone()?.size ?: 0) / 24000.0)} s")
                fun render(tag: String, env: Boolean) {
                    val v = bank.compose(); val t = bank.tuning
                    val e = if (env) t.env else null
                    val tone = e?.noiseDb?.let { db -> bank.roomTone()?.takeIf { it.size > 24000 }?.let { Dsp.BackgroundMixer(it, 24000, db, 0.0) } }
                    for ((k, s) in sents.withIndex()) {
                        var y = AudioTools.trimSilence(eng.synthesize(s, v, t.params.copy(seed = 4242L + k)).audio, 24000)
                        y = Psola.shapeEndings(y, 24000, t.endShift, t.endStretch)
                        y = Dsp.tilt(Dsp.tempoPitch(y, 24000, t.tempo, t.pitch), 24000, t.tilt)
                        val g = AudioTools.normalizeGain(y, 24000); for (i in y.indices) y[i] *= g
                        if (e != null) y = Dsp.applyEnvironment(y, 24000, e.eq, Dsp.Room.of(e.room) ?: Dsp.Room.NONE, e.wet)
                        if (tone != null) y = tone.mix(AudioTools.silence(24000, 200) + y + AudioTools.silence(24000, 300))
                        AudioTools.writeWav(File(dir, "${spk}_${tag}_$k.wav"), y, 24000)
                    }
                }
                render("B", false)
                val t0 = System.nanoTime()
                val tuner = AutoTuner(eng, bank, cfg)
                val res = tuner.run(progress = { p -> if (p.message.contains("세대 완성")) say("  $spk ${p.message}") })
                say("$spk tuner ${"%.0f".format((System.nanoTime() - t0) / 1e9)}s evaluated=${res.evaluated} accepted=${res.accepted} " +
                    "heldout ECAPA tuned/default=${f(res.heldOut.first)}/${f(res.heldOut.second)} criteria=${f(res.heldOutCriteria.first)}/${f(res.heldOutCriteria.second)} " +
                    "best=${res.best.gen} ${res.best.v}")
                say("  groups default=${res.groupsBefore.mapValues { f(it.value) }}")
                say("  groups tuned  =${res.groupsAfter.mapValues { f(it.value) }}")
                say("  env=${res.tuning.env?.let { "room=${it.room} wet=${it.wet} noiseDb=${it.noiseDb} eqMax=${"%.1f".format(it.eq.maxOf { v -> kotlin.math.abs(v) })} before=${f(it.before)} after=${f(it.after)}" }}")
                say("  generations=${res.generations.map { "${it.generation.round}:${f(it.generation.score)}/${f(it.generation.criteria)}" }}")
                bank.applyAutoLearning(res)
                render("C", false)
                render("D", true)
            }
        }
    }
}
