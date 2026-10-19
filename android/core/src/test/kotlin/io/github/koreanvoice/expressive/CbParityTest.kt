package io.github.koreanvoice.expressive

import io.github.koreanvoice.audio.AudioTools
import org.json.JSONArray
import java.io.File
import kotlin.math.sqrt
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertTrue

/**
 * Parity of the Kotlin Chatterbox engine with the Python/ONNX reference (cb/make_golden.py).
 * -DcbModelDir=<model dir> -DcbGoldenDir=<golden dir> -DkrefDir=<reference wavs>
 */
class CbParityTest {
    private val modelDir = File(System.getProperty("cbModelDir").orEmpty())
    private val goldDir = File(System.getProperty("cbGoldenDir").orEmpty())
    private val krefDir = File(System.getProperty("krefDir").orEmpty())
    private fun enabled() = File(modelDir, "t3_lm.onnx").exists() && File(goldDir, "cb_tok_golden.json").exists()

    @Test
    fun tokenizerMatchesPython() {
        if (!enabled()) return
        val tok = CbTokenizer(File(modelDir, "tokenizer.json").readText())
        val g = JSONArray(File(goldDir, "cb_tok_golden.json").readText())
        var bad = 0; var badPunc = 0
        for (i in 0 until g.length()) {
            val o = g.getJSONObject(i)
            val pn = CbTokenizer.puncNorm(o.getString("text"))
            if (pn != o.getString("punc")) { if (badPunc++ < 5) println("PUNC ${o.getString("text")} -> '$pn' vs '${o.getString("punc")}'") }
            val ids = tok.encodeKorean(o.getString("punc")).toList()
            val exp = o.getJSONArray("ids").let { a -> List(a.length()) { a.getInt(it) } }
            if (ids != exp) { if (bad++ < 5) println("TOK ${o.getString("punc")}\n  kt=$ids\n  py=$exp") }
        }
        println("tokenizer mismatches $bad / ${g.length()}, punc mismatches $badPunc")
        assertEquals(0, bad); assertEquals(0, badPunc)
    }

    private fun cos(a: FloatArray, b: FloatArray): Double {
        var d = 0.0; var na = 0.0; var nb = 0.0
        for (i in a.indices) { d += a[i] * b[i]; na += a[i] * a[i]; nb += b[i] * b[i] }
        return d / sqrt(na * nb)
    }

    @Test
    fun voiceAndGenerationMatchPython() {
        if (!enabled()) return
        val npz = Npz.read(File(goldDir, "voice_spk0.npz"))
        CbEngine(modelDir, 4).use { eng ->
            val (pcm, sr) = File(krefDir, "spk0.wav").inputStream().use { AudioTools.readWav(it) }
            val t0 = System.nanoTime()
            val v = eng.buildVoice(pcm, sr)
            println("buildVoice ${(System.nanoTime() - t0) / 1e6} ms")
            val pyGen = npz.getValue("gen_tokens").longs!!; val pyT3 = npz.getValue("t3_tokens").longs!!
            val genAgree = v.genTokens.zip(pyGen.toList()).count { it.first == it.second }.toDouble() / pyGen.size
            val t3Agree = v.t3Tokens.zip(pyT3.toList()).count { it.first == it.second }.toDouble() / pyT3.size
            val veCos = cos(v.ve, npz.getValue("ve").floats!!); val xvCos = cos(v.xvec, npz.getValue("xvec").floats!!)
            val melCos = cos(v.genMel, npz.getValue("gen_mel").floats!!)
            println("genTokens ${v.genTokens.size}/${pyGen.size} agree=$genAgree t3 ${v.t3Tokens.size} agree=$t3Agree ve=$veCos xv=$xvCos mel=$melCos")
            assertTrue(veCos > 0.99 && xvCos > 0.99 && melCos > 0.999 && genAgree > 0.9 && t3Agree > 0.9)

            // generation with the Python voice (isolates T3 parity from registration resampling differences)
            val pv = CbVoice(npz.getValue("ve").floats!!, pyT3, pyGen, npz.getValue("gen_mel").floats!!, npz.getValue("xvec").floats!!)
            val g = JSONArray(File(goldDir, "greedy_spk0.json").readText())
            for (i in 0 until g.length()) {
                val o = g.getJSONObject(i)
                val exp = o.getJSONArray("tokens").let { a -> LongArray(a.length()) { a.getLong(it) } }
                val t1 = System.nanoTime()
                val (toks, forced) = eng.generate(o.getString("text"), pv, CbParams(temperature = 1e-4f))
                val ms = (System.nanoTime() - t1) / 1e6
                var prefix = 0; while (prefix < minOf(toks.size, exp.size) && toks[prefix] == exp[prefix]) prefix++
                println("kt=${toks.take(14)}\npy=${exp.take(14)}")
                println("gen '${o.getString("text")}' kt=${toks.size} py=${exp.size} commonPrefix=$prefix forced=$forced ${ms}ms (${ms / toks.size} ms/token)")
                // argmax near-ties (|Δlogit| ~1e-5) may flip single tokens; the rest of the sequence must still agree
                val n = minOf(40, toks.size, exp.size)
                val same = (0 until n).count { toks[it] == exp[it] }
                println("position agreement over first $n: $same")
                assertTrue(same >= n * 0.85)
                val t2 = System.nanoTime()
                val audio = eng.decode(toks, pv)
                println("decode ${audio.size / 24000.0}s audio in ${(System.nanoTime() - t2) / 1e6} ms")
                File("build/test-audio").mkdirs()
                AudioTools.writeWav(File("build/test-audio/cb_kt_$i.wav"), audio, 24000)
            }
        }
    }
}

/** Minimal .npz reader (uncompressed, little-endian float32 / int64). */
object Npz {
    class Arr(val shape: IntArray, val floats: FloatArray?, val longs: LongArray?)
    fun read(f: File): Map<String, Arr> {
        val out = HashMap<String, Arr>()
        java.util.zip.ZipFile(f).use { z ->
            for (e in z.entries()) {
                val b = z.getInputStream(e).readBytes()
                val hl = (b[8].toInt() and 0xff) or ((b[9].toInt() and 0xff) shl 8)
                val hdr = String(b, 10, hl)
                val descr = Regex("'descr': '([^']+)'").find(hdr)!!.groupValues[1]
                val shape = Regex("'shape': \\(([^)]*)\\)").find(hdr)!!.groupValues[1].split(',').mapNotNull { it.trim().toIntOrNull() }.toIntArray()
                val bb = java.nio.ByteBuffer.wrap(b, 10 + hl, b.size - 10 - hl).order(java.nio.ByteOrder.LITTLE_ENDIAN)
                val n = shape.fold(1) { a, x -> a * x }
                val arr = when (descr) {
                    "<f4" -> Arr(shape, FloatArray(n) { bb.float }, null)
                    "<i8" -> Arr(shape, null, LongArray(n) { bb.long })
                    else -> error("dtype $descr")
                }
                out[e.name.removeSuffix(".npy")] = arr
            }
        }
        return out
    }
}
