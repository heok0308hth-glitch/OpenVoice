package io.github.koreanvoice.expressive

import io.github.koreanvoice.audio.AudioTools
import io.github.koreanvoice.audio.VoiceCriteria
import java.io.File
import kotlin.test.Test

/** 99-criteria similarity of validation outputs (B/C/D) vs each bank's real recordings. -Dstudy=criteria */
class CbCriteriaEval {
    @Test
    fun evaluate() {
        if (System.getProperty("study") != "criteria") return
        val out = File(System.getProperty("outDir")); val dir = File(out, "validate")
        val tot = HashMap<String, MutableList<VoiceCriteria.Report>>()
        for (spk in listOf("spk0", "spk1", "spk2", "spk3")) {
            val bank = VoiceBank(File(out, "bank_$spk"))
            val target = VoiceCriteria.Target.of(bank.segments.map { VoiceCriteria.measure(bank.segmentAudio(it), 24000) })
            for (tag in listOf("B", "C", "D")) {
                val files = (0 until 6).map { File(dir, "${spk}_${tag}_$it.wav") }.filter { it.exists() }
                if (files.isEmpty()) continue
                // all sentences of a condition joined (as the tuner does) with 300 ms pauses
                val parts = files.flatMap { listOf(it.inputStream().use { s -> AudioTools.readWav(s) }.first, AudioTools.silence(24000, 300)) }
                val all = FloatArray(parts.sumOf { it.size }); var o = 0; for (p in parts) { System.arraycopy(p, 0, all, o, p.size); o += p.size }
                val r = VoiceCriteria.compare(VoiceCriteria.measure(all, 24000), target)
                tot.getOrPut(tag) { ArrayList() }.add(r)
                println("CRIT $spk $tag overall=${"%.3f".format(r.overall)} " + r.groups.entries.joinToString(" ") { "${it.key.label}=${"%.2f".format(it.value)}" })
            }
        }
        for ((tag, rs) in tot.toSortedMap()) println("CRIT MEAN $tag overall=${"%.3f".format(rs.map { it.overall }.average())} " +
            VoiceCriteria.Group.entries.joinToString(" ") { g -> "${g.label}=${"%.2f".format(rs.mapNotNull { it.groups[g]?.takeIf { v -> !v.isNaN() } }.average())}" })
    }
}
