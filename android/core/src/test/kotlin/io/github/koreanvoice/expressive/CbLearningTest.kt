package io.github.koreanvoice.expressive

import io.github.koreanvoice.audio.AudioTools
import java.io.File
import kotlin.test.Test
import kotlin.test.assertTrue

/** Voice bank learning: speaker filtering on mixed recordings. -DcbModelDir -DkrefDir -DoutDir */
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
}
