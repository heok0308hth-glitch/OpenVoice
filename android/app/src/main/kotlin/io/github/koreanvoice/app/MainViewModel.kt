package io.github.koreanvoice.app

import android.app.Application
import android.net.Uri
import android.provider.OpenableColumns
import androidx.lifecycle.AndroidViewModel
import androidx.lifecycle.viewModelScope
import io.github.koreanvoice.audio.AudioTools
import io.github.koreanvoice.audio.Resampler
import io.github.koreanvoice.engine.SynthParams
import io.github.koreanvoice.text.TextChunker
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import java.io.File

class MainViewModel(app: Application) : AndroidViewModel(app) {

    // ------------------------------------------------------------------ preview
    sealed interface PreviewState {
        data object Idle : PreviewState
        data object Working : PreviewState
        data class Ready(val file: File, val rtf: Double) : PreviewState
        data class Error(val message: String) : PreviewState
    }

    private val _preview = MutableStateFlow<PreviewState>(PreviewState.Idle)
    val preview: StateFlow<PreviewState> = _preview

    /** Synthesize the first sentence(s) only, so settings / voice can be checked quickly. */
    fun preview(text: String, voiceId: String?, params: SynthParams) {
        if (_preview.value is PreviewState.Working) return
        _preview.value = PreviewState.Working
        viewModelScope.launch(Dispatchers.Default) {
            _preview.value = try {
                val eng = AppGraph.engine()
                val norm = eng.normalizer.normalize(text)
                val chunks = TextChunker().split(norm)
                if (chunks.isEmpty()) error("읽을 수 있는 한국어 텍스트가 없습니다.")
                val picked = ArrayList<TextChunker.Chunk>()
                for (c in chunks) { picked.add(c); if (picked.sumOf { it.text.length } > 60) break }
                val voice = AppGraph.voices.get(voiceId)?.embedding
                val t0 = System.nanoTime()
                val parts = ArrayList<FloatArray>()
                for (c in picked) {
                    var a = eng.synthesizeChunk(c.text, voice, params)
                    a = AudioTools.trimSilence(a, 22050)
                    val g = AudioTools.normalizeGain(a, 22050)
                    for (i in a.indices) a[i] *= g
                    AudioTools.fade(a, 22050); AudioTools.limit(a)
                    parts.add(a); parts.add(AudioTools.silence(22050, (c.pauseMs / params.speed).toInt()))
                }
                val all = FloatArray(parts.sumOf { it.size }); var p = 0
                for (x in parts) { System.arraycopy(x, 0, all, p, x.size); p += x.size }
                val sec = (System.nanoTime() - t0) / 1e9
                val f = Sharing.writePreview(getApplication(), all)
                withContext(Dispatchers.Main) { Player.stop(); Player.toggle(f) }
                PreviewState.Ready(f, sec / (all.size / 22050.0))
            } catch (t: Throwable) {
                PreviewState.Error(t.message ?: t.toString())
            }
        }
    }

    // ------------------------------------------------------------------ voice creation
    sealed interface VoiceBuildState {
        data object Idle : VoiceBuildState
        data class Working(val message: String) : VoiceBuildState
        data class Analyzed(val items: List<Analyzed.Item>) : VoiceBuildState {
            data class Item(val label: String, val embedding: FloatArray, val report: AudioTools.SampleReport, val preview: FloatArray)
        }
        data class Error(val message: String) : VoiceBuildState
        data class Saved(val profile: VoiceProfile) : VoiceBuildState
    }

    private val _voiceBuild = MutableStateFlow<VoiceBuildState>(VoiceBuildState.Idle)
    val voiceBuild: StateFlow<VoiceBuildState> = _voiceBuild

    fun resetVoiceBuild() { _voiceBuild.value = VoiceBuildState.Idle }

    private fun displayName(uri: Uri): String = runCatching {
        getApplication<Application>().contentResolver.query(uri, arrayOf(OpenableColumns.DISPLAY_NAME), null, null, null)?.use { c ->
            if (c.moveToFirst()) c.getString(0) else null
        }
    }.getOrNull() ?: uri.lastPathSegment ?: "샘플"

    fun analyzeUris(uris: List<Uri>) = analyze(uris.map { uri ->
        displayName(uri) to {
            val d = AudioDecoder.decode(getApplication(), uri)
            d.samples to d.sampleRate
        }
    })

    fun analyzeRecording(samples: FloatArray, sr: Int) = analyze(listOf("직접 녹음" to { samples to sr }))

    private fun analyze(sources: List<Pair<String, () -> Pair<FloatArray, Int>>>) {
        val previous = (_voiceBuild.value as? VoiceBuildState.Analyzed)?.items.orEmpty()
        viewModelScope.launch(Dispatchers.Default) {
            try {
                val eng = run {
                    _voiceBuild.value = VoiceBuildState.Working("모델을 불러오는 중…")
                    AppGraph.engine()
                }
                val items = ArrayList(previous)
                for ((i, src) in sources.withIndex()) {
                    _voiceBuild.value = VoiceBuildState.Working("샘플 분석 중 (${i + 1}/${sources.size}): ${src.first}")
                    val (pcm, sr) = src.second()
                    val (emb, report) = eng.extractVoice(pcm, sr)
                    var x = if (sr == 22050) pcm else Resampler.resample(pcm, sr, 22050)
                    if (x.size > 22050 * 8) x = x.copyOfRange(0, 22050 * 8)
                    items.add(VoiceBuildState.Analyzed.Item(src.first, emb, report, x))
                }
                _voiceBuild.value = VoiceBuildState.Analyzed(items)
            } catch (t: Throwable) {
                _voiceBuild.value = VoiceBuildState.Error("샘플을 처리할 수 없습니다: ${t.message ?: t}")
            }
        }
    }

    fun saveVoice(name: String) {
        val st = _voiceBuild.value as? VoiceBuildState.Analyzed ?: return
        viewModelScope.launch(Dispatchers.IO) {
            val p = AppGraph.voices.create(name, st.items.map { Triple(it.label, it.embedding, it.report) }, st.items.first().preview)
            AppGraph.settings.lastVoiceId = p.id
            _voiceBuild.value = VoiceBuildState.Saved(p)
        }
    }

    /** Short test sentence in a saved voice. */
    fun testVoice(profile: VoiceProfile) =
        preview("안녕하세요. 이 목소리로 원하는 글을 읽어 드릴게요.", profile.id, AppGraph.settings.params)

    fun loadTextFile(uri: Uri, onLoaded: (String) -> Unit) {
        viewModelScope.launch(Dispatchers.IO) {
            val bytes = getApplication<Application>().contentResolver.openInputStream(uri)?.use { it.readBytes() } ?: return@launch
            val text = decodeText(bytes)
            withContext(Dispatchers.Main) { onLoaded(text) }
        }
    }

    /** UTF-8 (with/without BOM), UTF-16 or legacy Korean (EUC-KR/CP949) text files. */
    private fun decodeText(b: ByteArray): String {
        if (b.size >= 2 && b[0] == 0xFF.toByte() && b[1] == 0xFE.toByte()) return String(b, Charsets.UTF_16LE).drop(1)
        if (b.size >= 2 && b[0] == 0xFE.toByte() && b[1] == 0xFF.toByte()) return String(b, Charsets.UTF_16BE).drop(1)
        val utf8 = Charsets.UTF_8.newDecoder()
        return try {
            utf8.decode(java.nio.ByteBuffer.wrap(b)).toString().removePrefix("\uFEFF")
        } catch (e: java.nio.charset.CharacterCodingException) {
            String(b, charset("MS949"))
        }
    }
}
