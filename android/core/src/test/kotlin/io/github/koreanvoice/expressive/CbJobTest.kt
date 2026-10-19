package io.github.koreanvoice.expressive

import io.github.koreanvoice.audio.AudioTools
import io.github.koreanvoice.audio.Dsp
import java.io.File
import kotlin.test.Test
import kotlin.test.assertTrue

/** End-to-end long-text job with markup on a learned bank. -DcbModelDir -DoutDir (uses <outDir>/bank_spk0). */
class CbJobTest {
    private val modelDir = File(System.getProperty("cbModelDir").orEmpty())
    private val out = File(System.getProperty("outDir").orEmpty().ifEmpty { "build/test-audio/v2" })

    val script = """
        [차분]옛날 옛적, 깊은 산속에 작은 마을이 있었습니다. 마을 사람들은 매일 아침 2km 떨어진 우물까지 물을 길으러 갔어요.
        어느 날 한 아이가 소리쳤습니다. “우와, 저기 좀 보세요! 하늘에서 별이 떨어져요!”
        [쉼:0.8][슬픔]하지만 그 별은 곧 사라져 버렸습니다.[/감정] [울림:성당]그날 밤, 종소리가 세 번 울렸습니다.[울림:없음]
        [전화]여보세요? 거기 누구 없어요?[/상태] [속삭임]이건 비밀이야.[/상태] [속도:1.3][높이:+2]빨리 와!
        [기본]{NASA|나사}에서 발표한 내용은 2024년 3월 15일 오후 3시 30분에 공개됐습니다.
    """.trimIndent()

    @Test
    fun registrationMemory() {
        if (!File(modelDir, "t3_lm.onnx").exists() || !File(out, "bank_spk0/bank.json").exists()) return
        fun rss() = File("/proc/self/status").readLines().first { it.startsWith("VmRSS") }.filter { it.isDigit() }.toLong() / 1024
        CbEngine(modelDir, 4).use { eng ->
            val bank = VoiceBank(File(out, "bank_spk0"))
            val a = bank.segmentAudio(bank.segments.first())
            eng.buildVoice(a, 24000)
            println("RSS with registration models ${rss()} MB")
            eng.releaseRegistrationModels(); System.gc()
            println("RSS after releasing registration models ${rss()} MB")
        }
    }

    @Test
    fun markupJobRendersAndAssembles() {
        if (!File(modelDir, "t3_lm.onnx").exists() || !File(out, "bank_spk0/bank.json").exists()) return
        fun rss() = File("/proc/self/status").readLines().first { it.startsWith("VmRSS") }.filter { it.isDigit() }.toLong() / 1024
        println("RSS before engine ${rss()} MB")
        CbEngine(modelDir, 4).use { eng ->
            println("RSS after engine load ${rss()} MB")
            val bank = VoiceBank(File(out, "bank_spk0"))
            for (pipelined in listOf(false, true)) {
                val job = ExpressiveJob(eng, bank, File(out, "cache_job"), script,
                    ExpressiveJob.Options(background = Dsp.noiseBed("빗소리", 24000), backgroundDb = -26.0, pipelined = pipelined))
                job.deleteCache()
                val plan = job.plan
                plan.forEach { println("  $it") }
                val t0 = System.nanoTime()
                job.render { p -> if (p.done == p.total) println("rendered ${p.total} chunks in ${p.elapsedMs} ms") }
                val wav = File(out, "job_markup${if (pipelined) "_pipe" else ""}.wav")
                job.assembleWav(wav)
                File(out, "job_markup.mp3").outputStream().use { job.assembleMp3(it, 96) }
                val (x, sr) = wav.inputStream().use { AudioTools.readWav(it) }
                val sec = x.size.toDouble() / sr
                println("RSS after render ${rss()} MB")
                println("pipelined=$pipelined total ${(System.nanoTime() - t0) / 1e9}s for ${sec}s audio; warnings=${job.warnings}")
                assertTrue(sec > 20 && x.all { it.isFinite() && kotlin.math.abs(it) <= 1f })
                assertTrue(plan.filterIsInstance<ExpressiveJob.Chunk>().any { it.style.state == Dsp.State.PHONE })
                assertTrue(plan.filterIsInstance<ExpressiveJob.Chunk>().any { it.dialogue && it.style.emotion != Emotion.NEUTRAL })
                job.deleteCache()
            }
        }
    }
}

/** Writes raw vs Korean-normalised renderings of mixed text (numbers/English/units) for an ASR comparison. */
class CbNormalizationProbe {
    private val modelDir = File(System.getProperty("cbModelDir").orEmpty())
    private val out = File(System.getProperty("outDir").orEmpty().ifEmpty { "build/test-audio/v2" })

    @Test
    fun rawVersusNormalized() {
        if (!File(modelDir, "t3_lm.onnx").exists() || !File(out, "bank_spk0/bank.json").exists()) return
        val sents = listOf("오늘 회의는 3시 30분에 Zoom으로 진행합니다.", "이 제품의 가격은 12,900원이고 무게는 1.5kg입니다.",
            "AI 기술이 빠르게 발전하면서 ChatGPT 같은 서비스가 인기를 끌고 있어요.", "2024년 5월 8일, 서울 기온은 25도였습니다.",
            "그는 [laughter] 크게 웃으며 말했다.", "아, 정말 [sigh] 너무 피곤하다.")
        val norm = io.github.koreanvoice.text.KoreanNormalizer()
        CbEngine(modelDir, 4).use { eng ->
            val v = VoiceBank(File(out, "bank_spk0")).compose()
            for ((i, s) in sents.withIndex()) {
                val n = if (s.contains("[")) s.replace(Regex("\\[[a-z]+\\] ?"), "") else norm.normalize(s)
                println("$i raw='$s' norm='$n'")
                AudioTools.writeWav(File(out, "probe_raw_$i.wav"), eng.synthesize(s, v, CbParams(seed = 5L + i)).audio, 24000)
                AudioTools.writeWav(File(out, "probe_norm_$i.wav"), eng.synthesize(n, v, CbParams(seed = 5L + i)).audio, 24000)
            }
        }
    }
}
