package io.github.koreanvoice

import io.github.koreanvoice.text.MeloFrontend
import org.json.JSONArray
import java.io.File
import kotlin.test.Test
import kotlin.test.assertTrue

/**
 * Compares the Kotlin MeloTTS frontend with outputs of the original Python code
 * (tools/make_golden.py). Uses the bundled subset, or -DgoldenDir=<dir containing golden_*.json> for the full sets.
 */
class FrontendGoldenTest {
    private val dir = System.getProperty("goldenDir").orEmpty()

    private fun compare(name: String): Pair<Int, Int> {
        // full sets via -DgoldenDir, otherwise the 620-sentence subset bundled in test resources
        val f = File(dir, name)
        val textData = if (dir.isNotEmpty() && f.exists()) f.readText()
            else javaClass.getResourceAsStream("/$name")?.readBytes()?.toString(Charsets.UTF_8) ?: run { println("skip: $name"); return 0 to 0 }
        val arr = JSONArray(textData)
        val fe = MeloFrontend()
        var ok = 0; var shown = 0
        val t0 = System.nanoTime()
        for (i in 0 until arr.length()) {
            val o = arr.getJSONObject(i)
            val r = fe.process(o.getString("text"))
            val ids = o.getJSONArray("ids").let { a -> LongArray(a.length()) { a.getLong(it) } }
            val w2p = o.getJSONArray("word2ph").let { a -> IntArray(a.length()) { a.getInt(it) } }
            val bert = o.getJSONArray("bert_ids").let { a -> LongArray(a.length()) { a.getLong(it) } }
            val same = r.ids.contentEquals(ids) && r.word2ph.contentEquals(w2p) && r.bertIds.contentEquals(bert) &&
                r.normText == o.getString("norm")
            if (same) ok++ else if (shown++ < 15) {
                val exp = o.getJSONArray("phones").joinToString("") { it.toString() }
                println("MISMATCH [$name] ${o.getString("text")}\n  py: $exp\n  kt: ${r.phones.joinToString("")}" +
                    "\n  w2p equal=${r.word2ph.contentEquals(w2p)} bert equal=${r.bertIds.contentEquals(bert)}")
            }
        }
        val ms = (System.nanoTime() - t0) / 1e6
        println("[$name] identical: $ok / ${arr.length()}  (${"%.1f".format(ms / arr.length())} ms/sentence)")
        return ok to arr.length()
    }

    @Test
    fun distilledAnnotatorMatchesExactly() {
        val (ok, n) = compare("golden_distilled.json")
        if (n > 0) assertTrue(ok == n, "Kotlin port must equal Python g2pkk+distilled annotator exactly ($ok/$n)")
    }

    @Test
    fun agreementWithMecabReference() {
        val (ok, n) = compare("golden_mecab.json")
        if (n > 0) assertTrue(ok.toDouble() / n > 0.97, "agreement with mecab-based original too low ($ok/$n)")
    }
}
