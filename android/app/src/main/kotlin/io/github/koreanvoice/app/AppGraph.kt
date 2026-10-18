package io.github.koreanvoice.app

import android.app.Application
import android.app.NotificationChannel
import android.app.NotificationManager
import android.content.Context
import io.github.koreanvoice.engine.SynthParams
import io.github.koreanvoice.engine.SynthesisJob
import io.github.koreanvoice.engine.ThreadConfig
import io.github.koreanvoice.engine.VoiceEngine
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import java.io.File

class KoreanVoiceApp : Application() {
    override fun onCreate() {
        super.onCreate()
        AppGraph.init(this)
        val nm = getSystemService(NotificationManager::class.java)
        nm.createNotificationChannel(NotificationChannel(SynthesisService.CHANNEL, "음성 생성", NotificationManager.IMPORTANCE_LOW))
        nm.createNotificationChannel(NotificationChannel(SynthesisService.CHANNEL_DONE, "생성 완료", NotificationManager.IMPORTANCE_DEFAULT))
    }
}

enum class PerformanceMode(val label: String, val description: String) {
    SAVER("절전", "발열·배터리 최소화 (느림)"),
    BALANCED("균형", "대부분의 기기에 권장"),
    MAX("최대", "모든 코어 사용 (발열 증가)"),
}

/** Process-wide singletons. */
object AppGraph {
    lateinit var app: Context; private set
    lateinit var models: ModelManager; private set
    lateinit var voices: VoiceRepository; private set
    lateinit var settings: Settings; private set
    val jobs = JobManager()

    fun init(context: Context) {
        app = context.applicationContext
        models = ModelManager(app)
        voices = VoiceRepository(app)
        settings = Settings(app)
    }

    val outputDir: File get() = File(app.getExternalFilesDir(null) ?: app.filesDir, "outputs").apply { mkdirs() }
    val jobCacheDir: File get() = File(app.filesDir, "jobs").apply { mkdirs() }

    private var engine: VoiceEngine? = null
    private var engineMode: PerformanceMode? = null

    fun scheduleFor(mode: PerformanceMode): Pair<ThreadConfig, Boolean> {
        val cores = Runtime.getRuntime().availableProcessors()
        return when (mode) {
            PerformanceMode.SAVER -> ThreadConfig(2, 2) to false
            PerformanceMode.BALANCED -> if (cores >= 6) ThreadConfig(3, 3) to true else ThreadConfig(cores.coerceAtLeast(2), cores.coerceAtLeast(2)) to false
            PerformanceMode.MAX -> if (cores >= 6) ThreadConfig(cores / 2, cores / 2) to true else ThreadConfig(cores, cores) to false
        }
    }

    /** Lazily create the (heavy) inference engine; recreated when the performance mode changes. */
    @Synchronized
    fun engine(): VoiceEngine {
        val mode = settings.performanceMode
        engine?.let { if (engineMode == mode) return it; it.close() }
        val (threads, pipelined) = scheduleFor(mode)
        return VoiceEngine(models.modelDir, threads, allowSpinning = !pipelined).also { engine = it; engineMode = mode }
    }

    @Synchronized
    fun releaseEngine() { engine?.close(); engine = null; engineMode = null }
}

class Settings(context: Context) {
    private val prefs = context.getSharedPreferences("settings", Context.MODE_PRIVATE)
    var performanceMode: PerformanceMode
        get() = runCatching { PerformanceMode.valueOf(prefs.getString("perf", "BALANCED")!!) }.getOrDefault(PerformanceMode.BALANCED)
        set(v) = prefs.edit().putString("perf", v.name).apply()
    var lastVoiceId: String?
        get() = prefs.getString("voice", null)
        set(v) = prefs.edit().putString("voice", v).apply()
    var draftText: String
        get() = prefs.getString("draft", "") ?: ""
        set(v) = prefs.edit().putString("draft", v).apply()
    var bitrate: Int
        get() = prefs.getInt("bitrate", 64)
        set(v) = prefs.edit().putInt("bitrate", v).apply()
    var params: SynthParams
        get() = SynthParams(
            speed = prefs.getFloat("speed", 1f), noiseScale = prefs.getFloat("noise", 0.6f), noiseScaleW = prefs.getFloat("noiseW", 0.8f),
            sdpRatio = prefs.getFloat("sdp", 0.2f), tau = prefs.getFloat("tau", 0.3f), timbreStrength = prefs.getFloat("timbre", 1.2f), adaptiveSource = prefs.getBoolean("adaptive", true),
        )
        set(p) = prefs.edit().putFloat("speed", p.speed).putFloat("noise", p.noiseScale).putFloat("noiseW", p.noiseScaleW)
            .putFloat("sdp", p.sdpRatio).putFloat("tau", p.tau).putFloat("timbre", p.timbreStrength).putBoolean("adaptive", p.adaptiveSource).apply()
    var pauseScale: Float
        get() = prefs.getFloat("pause", 1f)
        set(v) = prefs.edit().putFloat("pause", v).apply()
    var consentAccepted: Boolean
        get() = prefs.getBoolean("consent", false)
        set(v) = prefs.edit().putBoolean("consent", v).apply()
}

data class SynthesisRequest(
    val text: String,
    val title: String,
    val voiceId: String?,
    val params: SynthParams,
    val bitrate: Int,
    val pauseScale: Float,
)

sealed interface JobState {
    data object Idle : JobState
    data class Preparing(val message: String) : JobState
    data class Running(val title: String, val progress: SynthesisJob.Progress) : JobState
    data class Finalizing(val title: String) : JobState
    data class Done(val title: String, val file: File, val durationSec: Double, val elapsedMs: Long, val savedToMusic: Boolean) : JobState
    data class Failed(val message: String) : JobState
    data object Cancelled : JobState
}

class JobManager {
    private val _state = MutableStateFlow<JobState>(JobState.Idle)
    val state: StateFlow<JobState> = _state
    @Volatile var pending: SynthesisRequest? = null
    @Volatile var cancelRequested = false

    fun update(s: JobState) { _state.value = s }
    val isBusy: Boolean get() = _state.value.let { it is JobState.Preparing || it is JobState.Running || it is JobState.Finalizing }
    fun reset() { if (!isBusy) _state.value = JobState.Idle }
}
