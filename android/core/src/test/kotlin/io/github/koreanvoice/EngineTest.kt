package io.github.koreanvoice

import io.github.koreanvoice.audio.AudioTools
import io.github.koreanvoice.audio.Mp3Encoder
import io.github.koreanvoice.audio.Resampler
import io.github.koreanvoice.engine.SynthParams
import io.github.koreanvoice.engine.SynthesisJob
import io.github.koreanvoice.engine.ThreadConfig
import io.github.koreanvoice.engine.VoiceEngine
import org.json.JSONArray
import java.io.File
import kotlin.math.log10
import kotlin.math.sqrt
import kotlin.test.Test
import kotlin.test.assertTrue

/**
 * End-to-end tests with the real ONNX model pack.
 * -DmodelDir=<model pack>  -DgoldenDir=<dir with golden_audio/>  -DoutDir=<where to write audio>
 */
class EngineTest {
    private val modelDir = System.getProperty("modelDir").orEmpty()
    private val golden = File(System.getProperty("goldenDir").orEmpty(), "golden_audio")
    private val outDir = File(System.getProperty("outDir").orEmpty().ifEmpty { "build/test-audio" }).apply { mkdirs() }
    private fun enabled() = modelDir.isNotEmpty() && File(modelDir, "melo_kr.onnx").exists()

    private fun snr(a: FloatArray, b: FloatArray): Double {
        val n = minOf(a.size, b.size)
        var s = 0.0; var e = 0.0
        for (i in 0 until n) { s += a[i].toDouble() * a[i]; val d = a[i] - b[i]; e += d.toDouble() * d }
        return 10 * log10(s / (e + 1e-20))
    }

    private fun cos(a: FloatArray, b: FloatArray): Double {
        var d = 0.0; var na = 0.0; var nb = 0.0
        for (i in a.indices) { d += a[i] * b[i]; na += a[i] * a[i]; nb += b[i] * b[i] }
        return d / sqrt(na * nb)
    }

    @Test
    fun parityWithPythonPipeline() {
        if (!enabled() || !golden.exists()) { println("skip parity"); return }
        VoiceEngine(File(modelDir), ThreadConfig(4, 4)).use { eng ->
            val texts = JSONArray(File(golden, "texts.json").readText())
            val p = SynthParams(noiseScale = 0f, noiseScaleW = 0f)
            for (i in 0 until texts.length()) {
                val kt = eng.baseSpeech(texts.getString(i), p)
                val (py44, sr) = File(golden, "base44_$i.wav").inputStream().use { AudioTools.readWav(it) }
                check(sr == 44100)
                val py = Resampler.decimate2(py44)
                val s = snr(py, kt)
                println("base speech parity #$i: len kt=${kt.size} py=${py.size} SNR=${"%.1f".format(s)} dB")
                assertTrue(kotlin.math.abs(kt.size - py.size) <= 1 && s > 30, "Kotlin output differs from Python pipeline")
            }
            val (ref, rsr) = File(golden, "ref_speaker0.wav").inputStream().use { AudioTools.readWav(it) }
            val se = eng.embed(Resampler.resample(ref, rsr, 22050))
            val pySe = readNpy(File(golden, "se_speaker0_whole.npy"))
            val c = cos(se, pySe)
            println("speaker embedding cosine vs python: ${"%.5f".format(c)}")
            assertTrue(c > 0.995)
        }
    }

    @Test
    fun synthesizeCorpusAndLongText() {
        if (!enabled()) { println("skip e2e"); return }
        val threads = ThreadConfig(2, 2)
        VoiceEngine(File(modelDir), threads).use { eng ->
            val (ref, rsr) = File(golden, "ref_speaker0.wav").inputStream().use { AudioTools.readWav(it) }
            val t0 = System.nanoTime()
            val (voice, report) = eng.extractVoice(ref, rsr)
            println("voice extraction: ${(System.nanoTime() - t0) / 1e6} ms, report=$report score=${report.score}")

            // corpus sentences → wav for ASR scoring
            val corpus = File(golden, "../ref/corpus.txt").takeIf { it.exists() }?.readLines()?.filter { it.isNotBlank() } ?: emptyList()
            val p = SynthParams()
            var audioSec = 0.0; var ttsMs = 0.0; var convMs = 0.0
            for ((i, line) in corpus.withIndex()) {
                val norm = eng.normalizer.normalize(line)
                val a = System.nanoTime(); val base = eng.baseSpeech(norm, p); val b = System.nanoTime()
                val out = eng.convert(base, voice, p); val c = System.nanoTime()
                ttsMs += (b - a) / 1e6; convMs += (c - b) / 1e6; audioSec += out.size / 22050.0
                AudioTools.writeWav(File(outDir, "kt_%02d.wav".format(i)), out, 22050)
            }
            if (audioSec > 0) println("corpus: audio ${"%.1f".format(audioSec)} s, TTS RTF ${"%.3f".format(ttsMs / 1000 / audioSec)}, " +
                "convert RTF ${"%.3f".format(convMs / 1000 / audioSec)} (threads ${threads.ttsThreads}+${threads.convertThreads}, sequential)")

            // long text through the pipelined job
            val longText = (corpus + corpus).joinToString(" ") + "\n" +
                "제3장. 2024년 10월 6일 오후 3시 30분, 우리는 서울역 3번 출구에서 만났다. 커피 2잔은 ₩9,000이었다."
            val cache = File(outDir, "cache").apply { deleteRecursively() }
            val job = SynthesisJob(eng, cache, longText, voice, p)
            val start = System.nanoTime()
            job.render(onProgress = { pr -> if (pr.done % 10 == 0 || pr.done == pr.total) println("progress ${pr.done}/${pr.total} eta=${pr.etaMs}ms") })
            val renderSec = (System.nanoTime() - start) / 1e9
            val mp3 = File(outDir, "long.mp3")
            mp3.outputStream().buffered().use {
                job.assembleMp3(it, 64, Mp3Encoder.Id3Info("테스트", "KoreanVoice", "AI 생성 음성 (OpenVoice V2 + MeloTTS)"))
            }
            val wav = File(outDir, "long.wav"); job.assembleWav(wav)
            val sec = (wav.length() - 44) / 2 / 22050.0
            println("long text: ${longText.length} chars, ${job.chunks.size} chunks, render ${"%.1f".format(renderSec)} s for " +
                "${"%.1f".format(sec)} s audio (pipelined RTF ${"%.3f".format(renderSec / sec)}), mp3 ${mp3.length() / 1024} KB")
            assertTrue(mp3.length() > 10_000 && sec > 30)
        }
    }

    private fun readNpy(f: File): FloatArray {
        val b = f.readBytes()
        val headerLen = (b[8].toInt() and 0xff) or ((b[9].toInt() and 0xff) shl 8)
        val off = 10 + headerLen
        val bb = java.nio.ByteBuffer.wrap(b, off, b.size - off).order(java.nio.ByteOrder.LITTLE_ENDIAN)
        return FloatArray((b.size - off) / 4) { bb.getFloat() }
    }
}
