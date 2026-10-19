package io.github.koreanvoice.expressive

import io.github.koreanvoice.audio.AudioTools
import java.io.File
import kotlin.test.Test
import kotlin.test.assertTrue

/**
 * Learning experiment (outputs are scored independently by tools/score_v2.py):
 *  A = single clip (first 10 s, like stock Chatterbox), B = voice bank of all segments,
 *  C = bank + auto-learning. -DcbModelDir -DkrefDir -DoutDir [-Dspeakers=spk0,spk1] [-Dtuner=quick|standard|precise]
 */
class CbLearningTest {
    private val modelDir = File(System.getProperty("cbModelDir").orEmpty())
    private val kref = File(System.getProperty("krefDir").orEmpty())
    private val out = File(System.getProperty("outDir").orEmpty().ifEmpty { "build/test-audio/v2" })
    private fun enabled() = File(modelDir, "t3_lm.onnx").exists() && kref.isDirectory

    private val evalSentences = listOf(
        "그날 밤, 우리는 오래된 기차역에서 처음 만났습니다.",
        "정말요? 그게 사실이라면 너무 기쁜 소식이네요!",
        "회의는 다음 주 화요일 오후 세 시로 변경되었습니다.",
        "괜찮아, 천천히 해도 돼. 내가 옆에 있을게.",
    )

    private fun wav(f: File) = f.inputStream().use { AudioTools.readWav(it) }

    @Test
    fun speakerFilterKeepsDominantSpeaker() {
        if (!enabled()) return
        CbEngine(modelDir, 4).use { eng ->
            val (a, sr) = wav(File(kref, "spk0.wav")); val (b, _) = wav(File(kref, "spk1_held0.wav"))
            val mixed = a + FloatArray(sr) + b
            val dir = File(out, "bank_mix").apply { deleteRecursively() }
            val bank = VoiceBank(dir)
            val rep = bank.ingest(eng, mixed, sr, "mix")
            println("mix ingest: $rep")
            val bank1 = VoiceBank(File(out, "bank_spk1tmp").apply { deleteRecursively() })
            bank1.ingest(eng, b, sr, "spk1")
            val sims = bank.segments.map { s -> VoiceBank.cos(s.xvec, bank1.identityXvec()) }
            println("kept segment sims to spk1: $sims")
            assertTrue(rep.rejectedSpeaker >= 1 && sims.all { it < VoiceBank.SAME_SPEAKER })
        }
    }

    @Test
    fun learnAndSynthesize() {
        if (!enabled()) return
        val speakers = System.getProperty("speakers").orEmpty().ifEmpty { "spk0,spk1,spk2,spk3" }.split(',')
        val tunerCfg = when (System.getProperty("tuner")) { "standard" -> AutoTuner.Config.STANDARD; "precise" -> AutoTuner.Config.PRECISE; else -> AutoTuner.Config.QUICK }
        out.mkdirs()
        CbEngine(modelDir, 4).use { eng ->
            for (spk in speakers) {
                val (pcm, sr) = wav(File(kref, "$spk.wav"))
                // A: stock-style single reference
                val vA = eng.buildVoice(pcm, sr)
                // B: voice bank
                val bank = VoiceBank(File(out, "bank_$spk").apply { deleteRecursively() })
                val t0 = System.nanoTime()
                val rep = bank.ingest(eng, pcm, sr, "$spk.wav")
                println("$spk ingest ${(System.nanoTime() - t0) / 1e9}s: $rep")
                val vB = bank.compose()
                for ((i, s) in evalSentences.withIndex()) {
                    val p = CbParams(seed = 100L + i)
                    AudioTools.writeWav(File(out, "${spk}_A_$i.wav"), eng.synthesize(s, vA, p).audio, 24000)
                    AudioTools.writeWav(File(out, "${spk}_B_$i.wav"), eng.synthesize(s, vB, p).audio, 24000)
                }
                // C: auto-learning
                val tuner = AutoTuner(eng, bank, tunerCfg)
                val t1 = System.nanoTime()
                val res = tuner.run(progress = { pr -> if (pr.message.contains("완료") || pr.message.contains("검증")) println("  ${pr.message}") })
                println("$spk tuner ${(System.nanoTime() - t1) / 1e9}s evaluated=${res.evaluated} baseline=${res.baseline} rounds=${res.roundBest} verified=${tuner.verified} best=${res.best.gen} ${res.best.fx} parts=${res.best.parts}")
                bank.tuning = res.tuning; bank.generation++; bank.save()
                val job = ExpressiveJob(eng, bank, File(out, "cache"), "", ExpressiveJob.Options(autoEmotion = false, emotionRouting = false))
                for ((i, s) in evalSentences.withIndex()) {
                    val vC = bank.compose()
                    val raw = eng.synthesize(s, vC, bank.tuning.params.copy(seed = 100L + i)).audio
                    val t = bank.tuning
                    var y = io.github.koreanvoice.audio.Dsp.tempoPitch(raw, 24000, t.tempo, t.pitch)
                    y = io.github.koreanvoice.audio.Dsp.tilt(y, 24000, t.tilt)
                    if (t.wet > 0.005) y = io.github.koreanvoice.audio.Dsp.reverb(y, 24000, io.github.koreanvoice.audio.Dsp.Room.ROOM, t.wet, 0.3)
                    AudioTools.writeWav(File(out, "${spk}_C_$i.wav"), y, 24000)
                }
                job.deleteCache()
            }
        }
    }
}
