package io.github.koreanvoice.expressive

import io.github.koreanvoice.audio.AudioTools
import io.github.koreanvoice.audio.Resampler
import java.io.File
import kotlin.test.Test

/** Controlled study of what moves speaker similarity. Outputs <outDir>/study/<spk>_<cond>_<k>.wav. -Dstudy=1 */
class CbStudyTest {
    private val modelDir = File(System.getProperty("cbModelDir").orEmpty())
    private val out = File(System.getProperty("outDir").orEmpty().ifEmpty { "build/test-audio/v2" })
    val sents = listOf("그날 밤, 우리는 오래된 기차역에서 처음 만났습니다.", "정말요? 그게 사실이라면 너무 기쁜 소식이네요!",
        "회의는 다음 주 화요일 오후 세 시로 변경되었습니다.", "괜찮아, 천천히 해도 돼. 내가 옆에 있을게.",
        "창밖으로 비가 내리기 시작하자 사람들은 서둘러 우산을 폈다.", "이번 여름에는 가족과 함께 제주도로 여행을 가려고 합니다.")

    @Test
    fun study() {
        if (System.getProperty("study") != "1" || !File(modelDir, "t3_lm.onnx").exists()) return
        val dir = File(out, "study").apply { mkdirs() }
        CbEngine(modelDir, 4).use { eng ->
            for (spk in listOf("spk0", "spk1", "spk2", "spk3")) {
                val bank = VoiceBank(File(out, "bank_$spk"))
                val veT = bank.identityVe(); val xvT = bank.identityXvec()
                fun gen(cond: String, v: CbVoice, p: CbParams = CbParams()): Pair<FloatArray, FloatArray> {
                    val ves = FloatArray(256); val xvs = FloatArray(192); var sv = 0.0; var sx = 0.0
                    for ((k, s) in sents.withIndex()) {
                        val a = eng.synthesize(s, v, p.copy(seed = 1000L + k)).audio
                        AudioTools.writeWav(File(dir, "${spk}_${cond}_$k.wav"), a, 24000)
                        val a16 = Resampler.resample(AudioTools.trimSilence(a, 24000), 24000, 16000)
                        val ve = eng.veEmbed(a16); val xv = VoiceBank.unit(eng.xvector(a16))
                        for (i in 0 until 256) ves[i] += ve[i] / sents.size; for (i in 0 until 192) xvs[i] += xv[i] / sents.size
                        sv += VoiceBank.cos(ve, veT) / sents.size; sx += VoiceBank.cos(xv, xvT) / sents.size
                    }
                    println("$spk $cond onDevice ve=${"%.4f".format(sv)} xv=${"%.4f".format(sx)}")
                    return ves to xvs
                }
                val base = bank.compose()
                val (mv, mx) = gen("base", base)
                // identity correction from the mean gap over all outputs (systematic drift, low noise)
                var vc = FloatArray(256) { veT[it] - mv[it] }; var xc = FloatArray(192) { xvT[it] - mx[it] }
                val t1 = bank.tuning.copy(veCorrection = vc, xvCorrection = xc, promptSegment = bank.bestSegment().id, veBlend = 0.0)
                val (mv2, mx2) = gen("corr1", bank.compose(t = t1))
                val vc2 = FloatArray(256) { vc[it] + veT[it] - mv2[it] }; val xc2 = FloatArray(192) { xc[it] + xvT[it] - mx2[it] }
                gen("corr2", bank.compose(t = t1.copy(veCorrection = vc2, xvCorrection = xc2)))
                gen("xvonly", bank.compose(t = t1.copy(veCorrection = null)))
                gen("ex03", base, CbParams(exaggeration = 0.3f))
                gen("ex08", base, CbParams(exaggeration = 0.8f, cfgWeight = 0.4f))
                val others = bank.segments.filter { it.id != bank.bestSegment().id }.take(2)
                others.forEachIndexed { i, s -> gen("seg$i", bank.compose(ref = s)) }
            }
        }
    }
}
