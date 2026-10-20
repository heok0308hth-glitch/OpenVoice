package io.github.koreanvoice.scratch

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import io.github.koreanvoice.audio.AudioTools
import io.github.koreanvoice.audio.Resampler
import java.io.File
import java.nio.FloatBuffer
import kotlin.test.Test

/** s3 tokenizer memory/speed per optimizer setting. -Dstudy=modelmem -Dvariant=all|nocf|basic|none */
class ModelMemory {
    private fun st(k: String) = File("/proc/self/status").readLines().first { it.startsWith(k) }.filter { it.isDigit() }.toLong() / 1024
    @Test fun tokenizer() {
        if (System.getProperty("study") != "modelmem") return
        val v = System.getProperty("variant", "all")
        val dir = File(System.getProperty("cbModelDir"))
        val env = OrtEnvironment.getEnvironment()
        val o = OrtSession.SessionOptions().apply {
            setIntraOpNumThreads(4); setInterOpNumThreads(1)
            when (v) {
                "all" -> setOptimizationLevel(OrtSession.SessionOptions.OptLevel.ALL_OPT)
                "nocf" -> { setOptimizationLevel(OrtSession.SessionOptions.OptLevel.ALL_OPT); addConfigEntry("optimization.disable_specified_optimizers", "ConstantFolding") }
                "basic" -> setOptimizationLevel(OrtSession.SessionOptions.OptLevel.BASIC_OPT)
                "none" -> setOptimizationLevel(OrtSession.SessionOptions.OptLevel.NO_OPT)
            }
        }
        val r0 = st("VmRSS")
        val s = env.createSession(File(dir, "s3_tokenizer.onnx").path, o)
        val r1 = st("VmRSS")
        val (x, sr) = File(System.getProperty("krefDir"), "spk1.wav").inputStream().use { AudioTools.readWav(it) }
        val w = Resampler.resample(x, sr, 24000).copyOf(24000 * 10)
        var toks = LongArray(0); val t0 = System.nanoTime()
        repeat(3) {
            OnnxTensor.createTensor(env, FloatBuffer.wrap(w), longArrayOf(1, w.size.toLong())).use { t ->
                s.run(mapOf("audio_values" to t)).use { r -> val b = (r[0] as OnnxTensor).longBuffer; toks = LongArray(b.remaining()).also { b.get(it) } }
            }
        }
        println("TOK $v load+=${r1 - r0}MB rss=${st("VmRSS")} peak=${st("VmHWM")} t=${"%.2f".format((System.nanoTime() - t0) / 3e9)}s n=${toks.size} hash=${toks.contentHashCode()}")
    }

    @Test fun load() {
        if (System.getProperty("study") != "loadmem") return
        val (m, v) = System.getProperty("variant").split(":")
        val o = OrtSession.SessionOptions().apply {
            setIntraOpNumThreads(4)
            setOptimizationLevel(if (v.startsWith("none")) OrtSession.SessionOptions.OptLevel.NO_OPT else OrtSession.SessionOptions.OptLevel.ALL_OPT)
            if (v.endsWith("np")) addConfigEntry("session.disable_prepacking", "1")
            if (v.endsWith("ar")) setCPUArenaAllocator(false)
        }
        val r0 = st("VmRSS")
        OrtEnvironment.getEnvironment().createSession(File(System.getProperty("cbModelDir"), m).path, o).use {
            println("LOAD $m $v load+=${st("VmRSS") - r0}MB peak=${st("VmHWM")}")
        }
    }
}
