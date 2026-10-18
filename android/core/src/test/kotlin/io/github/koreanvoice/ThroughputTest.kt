package io.github.koreanvoice

import io.github.koreanvoice.engine.SynthParams
import io.github.koreanvoice.engine.SynthesisJob
import io.github.koreanvoice.engine.ThreadConfig
import io.github.koreanvoice.engine.VoiceEngine
import java.io.File
import kotlin.test.Test

/** Compares sequential vs pipelined scheduling on this machine (-DmodelDir=...). */
class ThroughputTest {
    private val modelDir = System.getProperty("modelDir").orEmpty()

    @Test
    fun compareSchedules() {
        if (modelDir.isEmpty()) { println("skip"); return }
        val text = File(System.getProperty("goldenDir"), "ref/corpus.txt").readText()
        val voice = FloatArray(256) { (it % 7 - 3) * 0.1f }
        val cores = Runtime.getRuntime().availableProcessors()
        for ((name, cfg, pipe) in listOf(
            Triple("sequential ${cores}t", ThreadConfig(cores, cores), false),
            Triple("pipelined ${cores / 2}+${cores / 2}t", ThreadConfig(cores / 2, cores / 2), true),
            Triple("pipelined ${cores}+${cores}t", ThreadConfig(cores, cores), true),
        )) {
            VoiceEngine(File(modelDir), cfg).use { eng ->
                eng.baseSpeech("워밍업 문장입니다.", SynthParams())
                val cache = File("build/tp_cache").apply { deleteRecursively() }
                val job = SynthesisJob(eng, cache, text, voice, SynthParams(), pipelined = pipe)
                val t0 = System.nanoTime()
                job.render()
                val sec = (System.nanoTime() - t0) / 1e9
                val audio = cache.walk().filter { it.name.endsWith(".pcm") }.sumOf { it.length() } / 2 / 22050.0
                println("$name: ${"%.1f".format(sec)} s for ${"%.1f".format(audio)} s audio → RTF ${"%.3f".format(sec / audio)}")
            }
        }
    }
}
