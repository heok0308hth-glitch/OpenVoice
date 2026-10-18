package io.github.koreanvoice

import io.github.koreanvoice.audio.AudioTools
import io.github.koreanvoice.audio.Resampler
import io.github.koreanvoice.engine.ThreadConfig
import io.github.koreanvoice.engine.VoiceEngine
import java.io.File
import kotlin.math.sqrt
import kotlin.test.Test
import kotlin.test.assertTrue

/** Speaker embeddings are sensitive to the resampler's high band; compare against soxr (librosa default). */
class ResamplerTuningTest {
    private val modelDir = System.getProperty("modelDir").orEmpty()
    private val golden = File(System.getProperty("goldenDir").orEmpty(), "golden_audio")

    @Test
    fun embeddingMatchesSoxr() {
        if (modelDir.isEmpty() || !File(golden, "ref_speaker0_soxr22k.wav").exists()) { println("skip"); return }
        VoiceEngine(File(modelDir), ThreadConfig(4, 4)).use { eng ->
            val (x44, sr) = File(golden, "ref_speaker0.wav").inputStream().use { AudioTools.readWav(it) }
            val (soxr, _) = File(golden, "ref_speaker0_soxr22k.wav").inputStream().use { AudioTools.readWav(it) }
            val ref = eng.embed(soxr)
            fun cos(a: FloatArray, b: FloatArray): Double {
                var d = 0.0; var na = 0.0; var nb = 0.0
                for (i in a.indices) { d += a[i] * b[i]; na += a[i] * a[i]; nb += b[i] * b[i] }
                return d / sqrt(na * nb)
            }
            var best = 0.0
            for (zc in listOf(24, 64)) for (cut in listOf(0.95, 0.97)) {
                val y = Resampler.resample(x44, sr, 22050, cut, zc)
                val c = cos(eng.embed(y), ref)
                println("zc=$zc cutoff=$cut cos=${"%.5f".format(c)}")
                if (zc == Resampler.DEFAULT_ZERO_CROSSINGS && cut == Resampler.DEFAULT_CUTOFF) best = c
            }
            println("default setting cos=$best")
            assertTrue(best > 0.99)
        }
    }
}
