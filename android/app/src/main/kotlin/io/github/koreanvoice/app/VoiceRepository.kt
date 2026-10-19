package io.github.koreanvoice.app

import android.content.Context
import io.github.koreanvoice.expressive.VoiceBank
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import java.io.File
import java.util.UUID

/** Summary of a learned voice for the UI. */
data class VoiceInfo(
    val id: String,
    val name: String,
    val createdAt: Long,
    val seconds: Double,
    val segments: Int,
    val sources: List<String>,
    val generation: Int,
    val autoLearnRuns: Int,
    val scoreHistory: List<Double>,
)

/** Learned voices, each a [VoiceBank] directory under files/voices2/. */
class VoiceRepository(context: Context) {
    private val root = File(context.filesDir, "voices2").apply { mkdirs() }
    private val banks = HashMap<String, VoiceBank>()
    private val _voices = MutableStateFlow(load())
    val voices: StateFlow<List<VoiceInfo>> = _voices

    @Synchronized
    fun bank(id: String): VoiceBank? {
        banks[id]?.let { return it }
        val d = File(root, id)
        if (!File(d, "bank.json").exists()) return null
        return VoiceBank(d).also { banks[id] = it }
    }

    private fun info(id: String, b: VoiceBank) = VoiceInfo(id, b.name, b.createdAt, b.totalSeconds, b.segments.size,
        b.segments.map { it.source }.distinct(), b.generation, b.tuning.iterations, b.tuning.history)

    private fun load(): List<VoiceInfo> = root.listFiles { f -> f.isDirectory && File(f, "bank.json").exists() }.orEmpty()
        .mapNotNull { d -> runCatching { info(d.name, bank(d.name)!!) }.getOrNull() }
        .sortedByDescending { it.createdAt }

    fun refresh() { _voices.value = load() }

    /** Creates an empty voice (material is added by a Learn task). */
    @Synchronized
    fun create(name: String): String {
        val id = UUID.randomUUID().toString().take(12)
        val b = VoiceBank(File(root, id)); b.name = name.ifBlank { "내 목소리" }; b.save()
        banks[id] = b
        refresh()
        return id
    }

    @Synchronized
    fun rename(id: String, name: String) { bank(id)?.let { it.name = name; it.save() }; refresh() }

    @Synchronized
    fun delete(id: String) { banks.remove(id); File(root, id).deleteRecursively(); refresh() }

    /** Removes voices that never received any usable material (e.g. a cancelled first learning run). */
    @Synchronized
    fun pruneEmpty(exceptId: String? = null) {
        for (v in load()) if (v.segments == 0 && v.id != exceptId) delete(v.id)
    }

    fun get(id: String?): VoiceInfo? = _voices.value.find { it.id == id }
}
