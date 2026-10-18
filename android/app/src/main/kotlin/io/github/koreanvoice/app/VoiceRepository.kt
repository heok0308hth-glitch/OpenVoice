package io.github.koreanvoice.app

import android.content.Context
import io.github.koreanvoice.audio.AudioTools
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import org.json.JSONArray
import org.json.JSONObject
import java.io.File
import java.util.UUID

/** A cloned voice: averaged OpenVoice speaker embedding plus metadata. */
data class VoiceProfile(
    val id: String,
    val name: String,
    val embedding: FloatArray,
    val createdAt: Long,
    val sources: List<String>,
    val voicedSec: Double,
    val score: Int,
    val warnings: List<String>,
) {
    fun toJson(): JSONObject = JSONObject().apply {
        put("id", id); put("name", name); put("createdAt", createdAt); put("voicedSec", voicedSec); put("score", score)
        put("embedding", JSONArray().apply { embedding.forEach { put(it.toDouble()) } })
        put("sources", JSONArray(sources)); put("warnings", JSONArray(warnings))
    }

    companion object {
        fun fromJson(o: JSONObject) = VoiceProfile(
            o.getString("id"), o.getString("name"),
            o.getJSONArray("embedding").let { a -> FloatArray(a.length()) { a.getDouble(it).toFloat() } },
            o.getLong("createdAt"),
            o.optJSONArray("sources")?.let { a -> List(a.length()) { a.getString(it) } } ?: emptyList(),
            o.optDouble("voicedSec", 0.0), o.optInt("score", 0),
            o.optJSONArray("warnings")?.let { a -> List(a.length()) { a.getString(it) } } ?: emptyList(),
        )
    }
}

class VoiceRepository(context: Context) {
    private val dir = File(context.filesDir, "voices").apply { mkdirs() }
    private val _voices = MutableStateFlow(load())
    val voices: StateFlow<List<VoiceProfile>> = _voices

    private fun load(): List<VoiceProfile> = dir.listFiles { f -> f.extension == "json" }.orEmpty()
        .mapNotNull { runCatching { VoiceProfile.fromJson(JSONObject(it.readText())) }.getOrNull() }
        .sortedByDescending { it.createdAt }

    fun sampleFile(id: String) = File(dir, "$id.wav")

    /**
     * Combine one or more analysed samples into a profile. Embeddings are averaged weighted by the
     * amount of voiced speech, so longer / cleaner samples count more.
     */
    fun create(name: String, parts: List<Triple<String, FloatArray, AudioTools.SampleReport>>, preview: FloatArray?): VoiceProfile {
        val total = parts.sumOf { it.third.voicedSec.coerceAtLeast(0.5) }
        val emb = FloatArray(256)
        for ((_, e, r) in parts) { val w = (r.voicedSec.coerceAtLeast(0.5) / total).toFloat(); for (i in 0 until 256) emb[i] += e[i] * w }
        val voiced = parts.sumOf { it.third.voicedSec }
        val score = parts.maxOf { it.third.score }
        val warnings = parts.flatMap { it.third.warnings }.distinct()
        val p = VoiceProfile(UUID.randomUUID().toString(), name.ifBlank { "내 목소리" }, emb, System.currentTimeMillis(),
            parts.map { it.first }, voiced, score, warnings)
        File(dir, "${p.id}.json").writeText(p.toJson().toString())
        if (preview != null) AudioTools.writeWav(sampleFile(p.id), preview, 22050)
        _voices.value = load()
        return p
    }

    fun rename(id: String, name: String) {
        val p = _voices.value.find { it.id == id } ?: return
        File(dir, "$id.json").writeText(p.copy(name = name).toJson().toString())
        _voices.value = load()
    }

    fun delete(id: String) {
        File(dir, "$id.json").delete(); sampleFile(id).delete()
        _voices.value = load()
    }

    fun get(id: String?): VoiceProfile? = _voices.value.find { it.id == id }
}
