package io.github.koreanvoice.app

import android.content.Context
import io.github.koreanvoice.engine.VoiceEngine
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.withContext
import org.json.JSONObject
import java.io.File
import java.io.FileOutputStream
import java.net.HttpURLConnection
import java.net.URL
import java.security.MessageDigest

/**
 * Makes the ONNX model pack available as plain files (ONNX Runtime loads sessions from paths, which
 * keeps the weights in native memory instead of the Java heap).
 *  - "full" flavor: models are bundled in the APK assets and copied out once.
 *  - "lite" flavor: models are downloaded once (free, from the project's GitHub repository),
 *    verified with SHA-256, then everything runs offline.
 */
class ModelManager(private val context: Context) {

    sealed interface State {
        data object Checking : State
        data class NeedsDownload(val totalBytes: Long) : State
        data class Working(val label: String, val progress: Float) : State
        data object Ready : State
        data class Failed(val message: String) : State
    }

    private val _state = MutableStateFlow<State>(State.Checking)
    val state: StateFlow<State> = _state

    val modelDir: File = File(context.filesDir, "models")

    private val manifest: JSONObject by lazy {
        JSONObject(context.assets.open("model_manifest.json").bufferedReader().use { it.readText() })
    }

    private fun entry(name: String) = manifest.getJSONObject("files").getJSONObject(name)

    private fun isInstalled(name: String): Boolean {
        val f = File(modelDir, name)
        val marker = File(modelDir, "$name.ok")
        return f.exists() && marker.exists() && f.length() == entry(name).getLong("size") &&
            marker.readText() == entry(name).getString("sha256")
    }

    suspend fun check() = withContext(Dispatchers.IO) {
        modelDir.mkdirs()
        val missing = VoiceEngine.MODEL_FILES.filterNot { isInstalled(it) }
        when {
            missing.isEmpty() -> _state.value = State.Ready
            BuildConfig.BUNDLED_MODELS -> install()
            else -> _state.value = State.NeedsDownload(missing.sumOf { entry(it).getLong("size") })
        }
    }

    /** Copy bundled assets (full) or download (lite). */
    suspend fun install() = withContext(Dispatchers.IO) {
        try {
            modelDir.mkdirs()
            val files = VoiceEngine.MODEL_FILES.filterNot { isInstalled(it) }
            val total = files.sumOf { entry(it).getLong("size") }.coerceAtLeast(1)
            var done = 0L
            for (name in files) {
                val e = entry(name)
                val dst = File(modelDir, name)
                val tmp = File(modelDir, "$name.part")
                val md = MessageDigest.getInstance("SHA-256")
                if (BuildConfig.BUNDLED_MODELS) {
                    tmp.delete()
                    context.assets.open("models/$name").use { input ->
                        FileOutputStream(tmp).use { out ->
                            val buf = ByteArray(1 shl 20)
                            while (true) {
                                val n = input.read(buf); if (n < 0) break
                                out.write(buf, 0, n); md.update(buf, 0, n); done += n
                                _state.value = State.Working("모델 준비 중 (최초 1회)", done.toFloat() / total)
                            }
                        }
                    }
                } else {
                    tmp.delete()
                    FileOutputStream(tmp).use { out ->
                        val parts = e.getJSONArray("parts")
                        for (p in 0 until parts.length()) {
                            val partName = parts.getString(p)
                            val partFile = File(modelDir, "$partName.dl")
                            val before = done
                            download(partName, partFile) { got ->
                                _state.value = State.Working("모델 다운로드 중 (최초 1회, 약 ${total / 1_000_000} MB)", (before + got).toFloat() / total)
                            }
                            partFile.inputStream().use { input ->
                                val buf = ByteArray(1 shl 20)
                                while (true) { val n = input.read(buf); if (n < 0) break; out.write(buf, 0, n); md.update(buf, 0, n); done += n }
                            }
                            partFile.delete()
                        }
                    }
                }
                val sha = md.digest().joinToString("") { "%02x".format(it) }
                if (sha != e.getString("sha256")) {
                    tmp.delete()
                    error("$name 무결성 검사 실패 (손상된 파일). 다시 시도해 주세요.")
                }
                dst.delete()
                check(tmp.renameTo(dst)) { "파일 저장 실패: $name" }
                File(modelDir, "$name.ok").writeText(sha)
            }
            _state.value = State.Ready
        } catch (t: Throwable) {
            _state.value = State.Failed(t.message ?: t.toString())
        }
    }

    /** Download one part to [dst], resuming a previous partial download with an HTTP Range request. */
    private fun download(partName: String, dst: File, onProgress: (Long) -> Unit) {
        val bases = BuildConfig.MODEL_BASE_URLS.split('|').filter { it.isNotBlank() }
        var last: Exception? = null
        for (base in bases) {
            for (attempt in 0 until 3) {
                try {
                    val have = if (dst.exists()) dst.length() else 0L
                    val conn = URL(base.trimEnd('/') + "/" + partName).openConnection() as HttpURLConnection
                    conn.connectTimeout = 20_000; conn.readTimeout = 60_000
                    conn.instanceFollowRedirects = true
                    if (have > 0) conn.setRequestProperty("Range", "bytes=$have-")
                    val code = conn.responseCode
                    if (code == 416 && have > 0) return // already complete
                    if (code != 200 && code != 206) { last = Exception("HTTP $code ($partName)"); break }
                    val append = code == 206
                    var got = if (append) have else 0L
                    conn.inputStream.use { input ->
                        FileOutputStream(dst, append).use { out ->
                            val buf = ByteArray(1 shl 16)
                            while (true) { val n = input.read(buf); if (n < 0) break; out.write(buf, 0, n); got += n; onProgress(got) }
                        }
                    }
                    return
                } catch (e: Exception) { last = e; Thread.sleep(1500L * (attempt + 1)) }
            }
        }
        throw last ?: Exception("다운로드 주소가 설정되지 않았습니다")
    }
}
