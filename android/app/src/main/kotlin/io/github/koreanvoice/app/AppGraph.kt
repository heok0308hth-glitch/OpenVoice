package io.github.koreanvoice.app

import android.app.Application
import android.app.NotificationChannel
import android.app.NotificationManager
import android.content.Context
import android.net.Uri
import io.github.koreanvoice.audio.Dsp
import io.github.koreanvoice.expressive.AutoTuner
import io.github.koreanvoice.expressive.CbEngine
import io.github.koreanvoice.expressive.Emotion
import io.github.koreanvoice.expressive.Style
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import org.json.JSONObject
import java.io.File

class KoreanVoiceApp : Application() {
    override fun onCreate() {
        super.onCreate()
        AppGraph.init(this)
        val nm = getSystemService(NotificationManager::class.java)
        nm.createNotificationChannel(NotificationChannel(SynthesisService.CHANNEL, "음성 생성·학습", NotificationManager.IMPORTANCE_LOW))
        nm.createNotificationChannel(NotificationChannel(SynthesisService.CHANNEL_DONE, "완료 알림", NotificationManager.IMPORTANCE_DEFAULT))
    }
}

enum class PerformanceMode(val label: String, val description: String) {
    SAVER("절전", "발열·배터리 최소화 (느림)"),
    BALANCED("균형", "대부분의 기기에 권장"),
    MAX("최대", "모든 고성능 코어 사용 (발열 증가)"),
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

    private var engine: CbEngine? = null
    private var engineKey: String? = null

    fun threadsFor(mode: PerformanceMode): Int {
        val cores = Runtime.getRuntime().availableProcessors()
        return when (mode) {
            PerformanceMode.SAVER -> 2
            PerformanceMode.BALANCED -> minOf(4, cores)
            PerformanceMode.MAX -> minOf(6, maxOf(2, cores - 2))
        }
    }

    /** In MAX mode on 8+ core phones, T3 and the decoder run concurrently (each with half the cores). */
    // measured slower on a 4-core machine (2 sessions oversubscribe the CPU); benefit on 8-core phones unverified → off
    fun pipelined(): Boolean = false

    /** Lazily create the (heavy) inference engine; recreated when performance mode / decoder change. */
    @Synchronized
    fun engine(): CbEngine {
        val mode = settings.performanceMode
        val dec = if (settings.highQualityDecoder && models.hasHqDecoder()) CbEngine.DECODER_HQ else CbEngine.DECODER_FAST
        val key = "$mode|$dec"
        engine?.let { if (engineKey == key) return it; it.close() }
        return CbEngine(models.modelDir, threadsFor(mode), dec).also { engine = it; engineKey = key }
    }

    @Synchronized
    fun releaseEngine() { engine?.close(); engine = null; engineKey = null }
}

/** Document-level delivery settings chosen in the generation screen. */
data class DeliverySettings(
    val emotion: Emotion = Emotion.NEUTRAL,
    val intensity: Float = 1f,
    val autoEmotion: Boolean = true,
    val emotionRouting: Boolean = true,
    val tempo: Float = 1f,
    val pitch: Float = 0f,
    val room: Dsp.Room = Dsp.Room.NONE,
    val wet: Float = -1f,
    val state: Dsp.State = Dsp.State.NORMAL,
    val tone: Dsp.Tone = Dsp.Tone.NORMAL,
    val background: String = "없음",
    val backgroundDb: Float = -22f,
    val mastering: Boolean = false,
    val pauseScale: Float = 1f,
    val useTuning: Boolean = true,
    val recreateEnv: Boolean = false,
) {
    fun style() = Style(
        emotion = emotion, intensity = intensity, explicitEmotion = !autoEmotion && emotion != Emotion.NEUTRAL,
        tempo = tempo.toDouble(), pitch = pitch.toDouble(), room = room, wet = if (wet < 0) null else wet.toDouble(),
        state = state, tone = if (tone == Dsp.Tone.NORMAL) null else tone,
    )

    fun toJson() = JSONObject().apply {
        put("emotion", emotion.name); put("intensity", intensity.toDouble()); put("auto", autoEmotion); put("routing", emotionRouting)
        put("tempo", tempo.toDouble()); put("pitch", pitch.toDouble()); put("room", room.name); put("wet", wet.toDouble())
        put("state", state.name); put("tone", tone.name); put("bg", background); put("bgDb", backgroundDb.toDouble())
        put("master", mastering); put("pause", pauseScale.toDouble()); put("tuning", useTuning); put("env", recreateEnv)
    }

    companion object {
        fun fromJson(o: JSONObject) = DeliverySettings(
            runCatching { Emotion.valueOf(o.getString("emotion")) }.getOrDefault(Emotion.NEUTRAL), o.optDouble("intensity", 1.0).toFloat(),
            o.optBoolean("auto", true), o.optBoolean("routing", true), o.optDouble("tempo", 1.0).toFloat(), o.optDouble("pitch", 0.0).toFloat(),
            runCatching { Dsp.Room.valueOf(o.getString("room")) }.getOrDefault(Dsp.Room.NONE), o.optDouble("wet", -1.0).toFloat(),
            runCatching { Dsp.State.valueOf(o.getString("state")) }.getOrDefault(Dsp.State.NORMAL),
            runCatching { Dsp.Tone.valueOf(o.getString("tone")) }.getOrDefault(Dsp.Tone.NORMAL),
            o.optString("bg", "없음"), o.optDouble("bgDb", -22.0).toFloat(), o.optBoolean("master", false), o.optDouble("pause", 1.0).toFloat(),
            o.optBoolean("tuning", true), o.optBoolean("env", false))
    }
}

class Settings(context: Context) {
    private val prefs = context.getSharedPreferences("settings", Context.MODE_PRIVATE)
    var performanceMode: PerformanceMode
        get() = runCatching { PerformanceMode.valueOf(prefs.getString("perf", "BALANCED")!!) }.getOrDefault(PerformanceMode.BALANCED)
        set(v) = prefs.edit().putString("perf", v.name).apply()
    var highQualityDecoder: Boolean
        get() = prefs.getBoolean("hqdec", false)
        set(v) = prefs.edit().putBoolean("hqdec", v).apply()
    var lastVoiceId: String?
        get() = prefs.getString("voice2", null)
        set(v) = prefs.edit().putString("voice2", v).apply()
    var draftText: String
        get() = prefs.getString("draft", "") ?: ""
        set(v) = prefs.edit().putString("draft", v).apply()
    var bitrate: Int
        get() = prefs.getInt("bitrate", 96)
        set(v) = prefs.edit().putInt("bitrate", v).apply()
    var delivery: DeliverySettings
        get() = prefs.getString("delivery", null)?.let { runCatching { DeliverySettings.fromJson(JSONObject(it)) }.getOrNull() } ?: DeliverySettings()
        set(v) = prefs.edit().putString("delivery", v.toJson().toString()).apply()
    /** User pronunciation dictionary: one "표기=읽는법" per line. */
    var dictionaryText: String
        get() = prefs.getString("dict", "") ?: ""
        set(v) = prefs.edit().putString("dict", v).apply()
    val dictionary: Map<String, String>
        get() = dictionaryText.lines().mapNotNull { l -> l.split('=', limit = 2).takeIf { it.size == 2 && it[0].isNotBlank() }?.let { it[0].trim() to it[1].trim() } }.toMap()
    var consentAccepted: Boolean
        get() = prefs.getBoolean("consent", false)
        set(v) = prefs.edit().putBoolean("consent", v).apply()
}

/** Long-running work executed by [SynthesisService]. */
sealed interface TaskRequest {
    val title: String
    data class Synthesis(val text: String, override val title: String, val voiceId: String, val delivery: DeliverySettings,
                         val bitrate: Int, val backgroundUri: Uri?, val dictionary: Map<String, String>) : TaskRequest
    data class Learn(val voiceId: String, override val title: String, val uris: List<Pair<Uri, String>>,
                     val recording: Pair<FloatArray, Int>? = null) : TaskRequest
    data class AutoLearn(val voiceId: String, override val title: String, val config: AutoTuner.Config, val modeLabel: String) : TaskRequest
}

enum class TaskKind(val label: String) { SYNTHESIS("음성 생성"), LEARN("목소리 학습"), AUTOLEARN("자동 학습") }

sealed interface JobState {
    data object Idle : JobState
    data class Preparing(val kind: TaskKind, val message: String) : JobState
    data class Running(val kind: TaskKind, val title: String, val fraction: Float, val primary: String, val secondary: String = "",
                       val etaMs: Long = -1, val scores: List<Double> = emptyList(), val baseline: Double = Double.NaN,
                       /** Auto-learning: similarity per criteria group (label → 0..1) of the current best voice. */
                       val groups: List<Pair<String, Double>> = emptyList()) : JobState
    data class Done(val kind: TaskKind, val title: String, val summary: String, val file: File? = null, val voiceId: String? = null) : JobState
    data class Failed(val kind: TaskKind, val message: String) : JobState
    data class Cancelled(val kind: TaskKind) : JobState
}

class JobManager {
    private val _state = MutableStateFlow<JobState>(JobState.Idle)
    val state: StateFlow<JobState> = _state
    @Volatile var pending: TaskRequest? = null
    @Volatile var cancelRequested = false

    fun update(s: JobState) { _state.value = s }
    val isBusy: Boolean get() = _state.value.let { it is JobState.Preparing || it is JobState.Running }
    fun reset() { if (!isBusy) _state.value = JobState.Idle }
}
