package io.github.koreanvoice.expressive

import io.github.koreanvoice.audio.AudioTools
import io.github.koreanvoice.audio.Dsp
import java.io.File
import kotlin.test.Test

/**
 * Auto-learning validation: B = voice bank defaults, C = after auto-learning, same unseen sentences and seeds.
 * -Dstudy=validate -Dtuner=quick|standard|precise -Dspeakers=spk0,... ; log in <outDir>/validate/log.txt
 */
class CbValidationTest {
    private val modelDir = File(System.getProperty("cbModelDir").orEmpty())
    private val out = File(System.getProperty("outDir").orEmpty().ifEmpty { "build/test-audio/v2" })
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
        CbEngine(modelDir, 4).use { eng ->
            for (spk in speakers) {
                val bank = VoiceBank(File(out, "bank_$spk"))
                bank.tuning = VoiceBank.Tuning()
                fun render(tag: String) {
                    val v = bank.compose(); val t = bank.tuning
                    for ((k, s) in sents.withIndex()) {
                        var y = eng.synthesize(s, v, t.params.copy(seed = 4242L + k)).audio
                        y = Dsp.tilt(Dsp.tempoPitch(y, 24000, t.tempo, t.pitch), 24000, t.tilt)
                        if (t.wet > 0.005) y = Dsp.reverb(y, 24000, Dsp.Room.ROOM, t.wet, 0.3)
                        AudioTools.writeWav(File(dir, "${spk}_${tag}_$k.wav"), y, 24000)
                    }
                }
                render("B")
                val t0 = System.nanoTime()
                val tuner = AutoTuner(eng, bank, cfg)
                val res = tuner.run(progress = { p -> if (p.message.contains("완료")) say("  $spk ${p.message}") })
                say("$spk tuner ${"%.0f".format((System.nanoTime() - t0) / 1e9)}s evaluated=${res.evaluated} baseline=${"%.4f".format(res.baseline)} rounds=${res.roundBest.map { "%.4f".format(it) }} heldout(tuned,default)=${tuner.verified} accepted=${res.accepted} best=${res.best.gen} ${res.best.v} parts=${res.best.parts}")
                bank.tuning = res.tuning
                render("C")
            }
        }
    }
}
