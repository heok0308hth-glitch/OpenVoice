package io.github.koreanvoice.scratch

import io.github.koreanvoice.audio.AudioTools
import io.github.koreanvoice.expressive.CbEngine
import io.github.koreanvoice.expressive.VoiceBank
import java.io.File
import kotlin.test.Test

/** Peak native memory of the voice-learning path. -Dstudy=learnmem */
class LearnMemory {
    private fun rss() = File("/proc/self/status").readLines().first { it.startsWith("VmRSS") }.filter { it.isDigit() }.toLong() / 1024
    private fun hwm() = File("/proc/self/status").readLines().first { it.startsWith("VmHWM") }.filter { it.isDigit() }.toLong() / 1024
    @Test fun learn() {
        if (System.getProperty("study") != "learnmem") return
        val kref = File(System.getProperty("krefDir"))
        println("MEM start rss=${rss()}")
        CbEngine(File(System.getProperty("cbModelDir")), 4).use { eng ->
            println("MEM after engine init rss=${rss()} peak=${hwm()}")
            eng.releaseSynthesisModels()
            val (x, sr) = File(kref, "spk1.wav").inputStream().use { AudioTools.readWav(it) }
            val dir = File(System.getProperty("outDir"), "bank_memtest").apply { deleteRecursively() }
            val t0 = System.nanoTime()
            val rep = VoiceBank(dir).ingest(eng, x + x, sr, "test")
            println("MEM after ingest rss=${rss()} peak=${hwm()} ${(System.nanoTime() - t0) / 1e9}s kept=${rep.kept}")
        }
    }

    @Test fun synth() {
        if (System.getProperty("study") != "synthmem") return
        val kref = File(System.getProperty("krefDir"))
        if (System.getProperty("variant") == "np") System.setProperty("cb.noprepack", "1")
        CbEngine(File(System.getProperty("cbModelDir")), 4).use { eng ->
            val (x, sr) = File(kref, "spk1.wav").inputStream().use { AudioTools.readWav(it) }
            val v = eng.buildVoice(x, sr)
            eng.releaseRegistrationModels()
            println("MEM after buildVoice rss=${rss()} peak=${hwm()}")
            for (i in 0 until 3) {
                val t0 = System.nanoTime()
                val g = eng.generate("안녕하세요. 오늘은 날씨가 정말 좋네요. 산책하러 나가 볼까요?", v, io.github.koreanvoice.expressive.CbParams(seed = 7L))
                val t1 = System.nanoTime()
                val a = eng.decode(g.first, v)
                println("MEM synth rss=${rss()} peak=${hwm()} T3=${"%.2f".format((t1 - t0) / 1e9)}s dec=${"%.2f".format((System.nanoTime() - t1) / 1e9)}s tok=${g.first.size} hash=${g.first.contentHashCode()} sec=${a.size / 24000.0}")
            }
        }
    }
}
