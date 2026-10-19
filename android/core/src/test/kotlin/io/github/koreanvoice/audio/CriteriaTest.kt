package io.github.koreanvoice.audio

import java.io.File
import kotlin.test.Test
import kotlin.test.assertTrue

/** -DkrefDir=<dir with spkN.wav / spkN_heldK.wav> */
class CriteriaTest {
    private val kref = File(System.getProperty("krefDir").orEmpty())

    @Test
    fun criteriaCountAndDiscrimination() {
        println("criteria: ${VoiceCriteria.SIZE}")
        assertTrue(VoiceCriteria.SIZE in 95..105)
        if (!File(kref, "spk0.wav").exists()) return
        fun load(f: File) = f.inputStream().use { AudioTools.readWav(it) }
        val spks = listOf("spk0", "spk1", "spk2", "spk3")
        // target from 4 s pieces of the training recording
        val targets = spks.associateWith { s ->
            val (x, sr) = load(File(kref, "$s.wav"))
            val pieces = (0 until x.size / (sr * 4)).map { x.copyOfRange(it * sr * 4, (it + 1) * sr * 4) }
            val t0 = System.nanoTime()
            val ms = pieces.map { VoiceCriteria.measure(it, sr) }
            println("$s ${pieces.size} pieces measured in ${(System.nanoTime() - t0) / 1e6 / pieces.size} ms each, NaN=${ms.sumOf { m -> m.count { it.isNaN() } }}")
            VoiceCriteria.Target.of(ms)
        }
        var own = 0.0; var other = 0.0; var correct = 0; var total = 0
        for (s in spks) for (k in 0 until 5) {
            val (x, sr) = load(File(kref, "${s}_held$k.wav"))
            val c = VoiceCriteria.measure(x, sr)
            val sims = spks.associateWith { VoiceCriteria.compare(c, targets.getValue(it)).overall }
            own += sims.getValue(s); other += spks.filter { it != s }.sumOf { sims.getValue(it) } / 3
            total++; if (sims.maxBy { it.value }.key == s) correct++
            if (k == 0) { val r = VoiceCriteria.compare(c, targets.getValue(s)); println("$s held0 groups=${r.groups.mapValues { "%.2f".format(it.value) }} worst=${r.worst(3)}") }
        }
        println("own=${own / total} other=${other / total} identified $correct/$total")
        assertTrue(own > other)
    }
}
