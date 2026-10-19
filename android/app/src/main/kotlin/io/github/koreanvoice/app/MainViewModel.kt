package io.github.koreanvoice.app

import android.app.Application
import android.net.Uri
import android.provider.OpenableColumns
import androidx.lifecycle.AndroidViewModel
import androidx.lifecycle.viewModelScope
import io.github.koreanvoice.audio.AudioTools
import io.github.koreanvoice.audio.Dsp
import io.github.koreanvoice.expressive.ExpressiveJob
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
        data class Working(val message: String) : PreviewState
        data class Ready(val file: File, val rtf: Double) : PreviewState
        data class Error(val message: String) : PreviewState
    }

    private val _preview = MutableStateFlow<PreviewState>(PreviewState.Idle)
    val preview: StateFlow<PreviewState> = _preview

    /** First sentence(s) of the script with all delivery settings, to check voice and style quickly. */
    fun preview(text: String, voiceId: String?, delivery: DeliverySettings, maxChars: Int = 80) {
        if (_preview.value is PreviewState.Working) return
        if (voiceId == null) { _preview.value = PreviewState.Error("먼저 목소리를 선택하거나 만들어 주세요."); return }
        _preview.value = PreviewState.Working("모델을 불러오는 중…")
        viewModelScope.launch(Dispatchers.Default) {
            _preview.value = try {
                val eng = AppGraph.engine()
                val bank = AppGraph.voices.bank(voiceId) ?: error("목소리를 찾을 수 없습니다.")
                require(bank.segments.isNotEmpty()) { "이 목소리에는 아직 학습된 음성이 없습니다." }
                val snippet = cutSnippet(text, maxChars)
                _preview.value = PreviewState.Working("미리듣기 생성 중…")
                val bed = if (delivery.background != "없음" && delivery.background != "파일") Dsp.noiseBed(delivery.background, 24000) else null
                val job = ExpressiveJob(eng, bank, File(getApplication<Application>().cacheDir, "preview"), snippet, ExpressiveJob.Options(
                    base = delivery.style(), autoEmotion = delivery.autoEmotion, emotionRouting = delivery.emotionRouting,
                    dictionary = AppGraph.settings.dictionary, pauseScale = delivery.pauseScale, background = bed,
                    backgroundDb = delivery.backgroundDb.toDouble(), mastering = delivery.mastering, useTuning = delivery.useTuning))
                if (job.chunks.isEmpty()) error("읽을 수 있는 한국어 텍스트가 없습니다.")
                val t0 = System.nanoTime()
                job.render()
                val f = File(getApplication<Application>().cacheDir, "preview.wav")
                job.assembleWav(f)
                job.deleteCache()
                val sec = (System.nanoTime() - t0) / 1e9
                val audioSec = (f.length() - 44) / 2.0 / 24000
                withContext(Dispatchers.Main) { Player.stop(); Player.toggle(f) }
                PreviewState.Ready(f, sec / audioSec.coerceAtLeast(0.1))
            } catch (t: Throwable) {
                PreviewState.Error(t.message ?: t.toString())
            }
        }
    }

    /** Leading sentences up to ~[maxChars] characters, keeping markup tags intact. */
    private fun cutSnippet(text: String, maxChars: Int): String {
        val t = text.trim()
        if (t.length <= maxChars) return t
        val ends = Regex("[.!?…](\\s|$)|\\n").findAll(t).map { it.range.first + 1 }.toList()
        val cut = ends.lastOrNull { it <= maxChars } ?: ends.firstOrNull() ?: maxChars
        return t.substring(0, cut.coerceAtMost(maxOf(maxChars * 2, cut)))
    }

    fun testVoice(id: String) = preview("[기본]안녕하세요. 이 목소리로 원하는 글을 읽어 드릴게요. 오늘 하루도 정말 수고 많으셨어요!", id, DeliverySettings())

    // ------------------------------------------------------------------ voices
    fun displayName(uri: Uri): String = runCatching {
        getApplication<Application>().contentResolver.query(uri, arrayOf(OpenableColumns.DISPLAY_NAME), null, null, null)?.use { c ->
            if (c.moveToFirst()) c.getString(0) else null
        }
    }.getOrNull() ?: uri.lastPathSegment ?: "자료"

    /** Creates a voice (if [voiceId] is null) and starts learning from files / a recording in the background service. */
    fun learn(voiceId: String?, name: String, uris: List<Uri>, recording: Pair<FloatArray, Int>?): String {
        val id = voiceId ?: AppGraph.voices.create(name)
        AppGraph.settings.lastVoiceId = id
        val title = (AppGraph.voices.get(id)?.name ?: name) + " 학습"
        SynthesisService.start(getApplication(), TaskRequest.Learn(id, title, uris.map { it to displayName(it) }, recording))
        return id
    }

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
            utf8.decode(java.nio.ByteBuffer.wrap(b)).toString().removePrefix("﻿")
        } catch (e: java.nio.charset.CharacterCodingException) {
            String(b, charset("MS949"))
        }
    }

    /** Plays a stored segment of a voice (the real recording). */
    fun playSegment(voiceId: String, segId: String) {
        viewModelScope.launch(Dispatchers.IO) {
            val bank = AppGraph.voices.bank(voiceId) ?: return@launch
            val s = bank.segments.firstOrNull { it.id == segId } ?: return@launch
            val f = File(getApplication<Application>().cacheDir, "seg_play.wav")
            AudioTools.writeWav(f, bank.segmentAudio(s), 24000)
            withContext(Dispatchers.Main) { Player.stop(); Player.toggle(f) }
        }
    }
}
