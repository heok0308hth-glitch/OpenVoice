package io.github.koreanvoice.app

import android.app.Notification
import android.app.PendingIntent
import android.app.Service
import android.content.ContentValues
import android.content.Context
import android.content.Intent
import android.content.pm.ServiceInfo
import android.os.Build
import android.os.Environment
import android.os.IBinder
import android.os.PowerManager
import android.provider.MediaStore
import androidx.core.app.NotificationCompat
import androidx.core.app.NotificationManagerCompat
import androidx.core.app.ServiceCompat
import io.github.koreanvoice.audio.AudioTools
import io.github.koreanvoice.audio.Dsp
import io.github.koreanvoice.audio.Mp3Encoder
import io.github.koreanvoice.audio.Resampler
import io.github.koreanvoice.audio.VoiceCriteria
import io.github.koreanvoice.expressive.AutoTuner
import io.github.koreanvoice.expressive.ExpressiveJob
import io.github.koreanvoice.expressive.VoiceBank
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.SupervisorJob
import kotlinx.coroutines.cancel
import kotlinx.coroutines.launch
import java.io.File
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale

/**
 * Foreground service for everything that takes minutes: long-text synthesis, learning a voice from
 * audio/video, and auto-learning. Work survives leaving the app / screen off, progress is shown in a
 * notification, and synthesis resumes from the last finished sentence after a cancel or a kill.
 */
class SynthesisService : Service() {
    private val scope = CoroutineScope(SupervisorJob() + Dispatchers.Default)
    private var wakeLock: PowerManager.WakeLock? = null

    override fun onBind(intent: Intent?): IBinder? = null

    override fun onStartCommand(intent: Intent?, flags: Int, startId: Int): Int {
        if (intent?.action == ACTION_CANCEL) {
            AppGraph.jobs.cancelRequested = true
            return START_NOT_STICKY
        }
        val req = AppGraph.jobs.pending ?: run { stopSelf(); return START_NOT_STICKY }
        AppGraph.jobs.pending = null
        AppGraph.jobs.cancelRequested = false
        val type = if (Build.VERSION.SDK_INT >= 35) ServiceInfo.FOREGROUND_SERVICE_TYPE_MEDIA_PROCESSING
        else if (Build.VERSION.SDK_INT >= 29) ServiceInfo.FOREGROUND_SERVICE_TYPE_DATA_SYNC else 0
        ServiceCompat.startForeground(this, NOTIF_ID, progressNotification(req.title, "준비 중…", 0f), type)
        wakeLock = (getSystemService(Context.POWER_SERVICE) as PowerManager)
            .newWakeLock(PowerManager.PARTIAL_WAKE_LOCK, "KoreanVoice:work").apply { acquire(12 * 60 * 60 * 1000L) }
        scope.launch { run(req) }
        return START_NOT_STICKY
    }

    private val cancelled get() = AppGraph.jobs.cancelRequested
    private var lastNotif = 0L

    private fun report(s: JobState.Running) {
        AppGraph.jobs.update(s)
        val now = System.currentTimeMillis()
        if (now - lastNotif > 1000) {
            lastNotif = now
            val eta = if (s.etaMs > 0) " · 약 ${formatDuration(s.etaMs)} 남음" else ""
            notify(progressNotification(s.title, s.primary + eta, s.fraction))
        }
    }

    private fun kindOf(r: TaskRequest) = when (r) {
        is TaskRequest.Synthesis -> TaskKind.SYNTHESIS
        is TaskRequest.Learn -> TaskKind.LEARN
        is TaskRequest.AutoLearn -> TaskKind.AUTOLEARN
    }

    private fun run(req: TaskRequest) {
        val jobs = AppGraph.jobs
        val kind = kindOf(req)
        try {
            jobs.update(JobState.Preparing(kind, "모델을 불러오는 중…"))
            val done = when (req) {
                is TaskRequest.Synthesis -> synthesize(req)
                is TaskRequest.Learn -> learn(req)
                is TaskRequest.AutoLearn -> autoLearn(req)
            }
            jobs.update(done)
            doneNotification(done.title, done.summary.lineSequence().first())
        } catch (e: ExpressiveJob.CancelledException) {
            jobs.update(JobState.Cancelled(kind))
        } catch (e: InterruptedException) {
            jobs.update(JobState.Cancelled(kind))
        } catch (e: OutOfMemoryError) {
            AppGraph.releaseEngine()
            jobs.update(JobState.Failed(kind, "메모리가 부족합니다. 다른 앱을 종료하거나 '절전' 성능 모드를 사용해 주세요."))
        } catch (t: Throwable) {
            jobs.update(JobState.Failed(kind, t.message ?: t.toString()))
        } finally {
            if (req is TaskRequest.Learn) AppGraph.voices.pruneEmpty()
            AppGraph.voices.refresh()
            runCatching { AppGraph.engine().releaseRegistrationModels() }
            wakeLock?.let { if (it.isHeld) it.release() }
            ServiceCompat.stopForeground(this, ServiceCompat.STOP_FOREGROUND_REMOVE)
            stopSelf()
        }
    }

    // ------------------------------------------------------------------ synthesis
    private fun synthesize(req: TaskRequest.Synthesis): JobState.Done {
        val t0 = System.currentTimeMillis()
        val engine = AppGraph.engine()
        val bank = AppGraph.voices.bank(req.voiceId) ?: error("목소리를 찾을 수 없습니다.")
        require(bank.segments.isNotEmpty()) { "이 목소리에는 아직 학습된 음성이 없습니다." }
        val d = req.delivery
        val bed: FloatArray? = when {
            req.backgroundUri != null -> AudioDecoder.decode(this, req.backgroundUri, maxSeconds = 180).let { Resampler.resample(it.samples, it.sampleRate, 24000) }
            d.background != "없음" && d.background != "파일" -> Dsp.noiseBed(d.background, 24000)
            else -> null
        }
        AppGraph.jobs.update(JobState.Preparing(TaskKind.SYNTHESIS, "텍스트 분석 중…"))
        val job = ExpressiveJob(engine, bank, AppGraph.jobCacheDir, req.text, ExpressiveJob.Options(
            base = d.style(), autoEmotion = d.autoEmotion, emotionRouting = d.emotionRouting, dictionary = req.dictionary,
            pauseScale = d.pauseScale, background = bed, backgroundDb = d.backgroundDb.toDouble(), mastering = d.mastering, useTuning = d.useTuning, pipelined = AppGraph.pipelined(), recreateEnvironment = d.recreateEnv))
        if (job.chunks.isEmpty()) error("읽을 수 있는 한국어 텍스트가 없습니다.")
        job.render(isCancelled = { cancelled }) { p ->
            report(JobState.Running(TaskKind.SYNTHESIS, req.title, if (p.total == 0) 0f else p.done.toFloat() / p.total,
                "${p.done}/${p.total} 문장", p.currentText, p.etaMs))
        }
        AppGraph.jobs.update(JobState.Preparing(TaskKind.SYNTHESIS, "효과 적용 및 MP3 저장 중…"))
        notify(progressNotification(req.title, "MP3 파일로 저장 중…", 0f))
        val stamp = SimpleDateFormat("yyyyMMdd_HHmmss", Locale.KOREA).format(Date())
        val safeTitle = req.title.replace(Regex("[^0-9A-Za-z가-힣 _-]"), "").trim().take(40).ifEmpty { "음성" }
        val out = File(AppGraph.outputDir, "${safeTitle}_$stamp.mp3")
        val id3 = Mp3Encoder.Id3Info(req.title, bank.name, "AI 생성 음성 · Chatterbox Multilingual (KoreanVoice 앱)")
        out.outputStream().buffered(1 shl 16).use { job.assembleMp3(it, req.bitrate, id3) }
        val saved = runCatching { copyToMusic(out) }.getOrDefault(false)
        job.deleteCache()
        val dur = out.length() * 8.0 / (req.bitrate * 1000.0)
        val warn = if (job.warnings.isEmpty()) "" else "\n" + job.warnings.distinct().take(3).joinToString("\n")
        return JobState.Done(TaskKind.SYNTHESIS, req.title,
            "길이 ${formatDuration((dur * 1000).toLong())} · 소요 ${formatDuration(System.currentTimeMillis() - t0)}" +
                (if (saved) "\n음악/KoreanVoice 폴더에도 저장되었습니다." else "") + warn, file = out)
    }

    // ------------------------------------------------------------------ learning from audio / video
    private fun learn(req: TaskRequest.Learn): JobState.Done {
        val engine = AppGraph.engine()
        val bank = AppGraph.voices.bank(req.voiceId) ?: error("목소리를 찾을 수 없습니다.")
        val before = bank.segments.size
        val collector = bank.Collector(engine)
        val n = req.uris.size + (if (req.recording != null) 1 else 0)
        var idx = 0
        val sources = ArrayList<String>()
        fun step(label: String, extra: String) = report(JobState.Running(TaskKind.LEARN, req.title,
            0.45f * idx / n, "자료 분석 ${idx + 1}/$n: $label", extra))
        req.recording?.let { (pcm, sr) ->
            step("직접 녹음", ""); collector.add(pcm, sr, { cancelled }); sources.add("직접 녹음"); idx++
        }
        var truncatedAny = false
        for ((uri, label) in req.uris) {
            step(label, "오디오 추출 중")
            var sec = 0.0
            AudioDecoder.decodeWindows(this, uri, windowSeconds = 60, maxSeconds = 20 * 60) { w, sr, trunc ->
                if (cancelled) throw InterruptedException()
                collector.add(w, sr, { cancelled }) { pieces -> step(label, "${"%.0f".format(sec / 60)}분 지점 · 음성 구간 ${pieces}개 발견") }
                sec += w.size.toDouble() / sr
                if (trunc) truncatedAny = true
            }
            sources.add(label); idx++
        }
        val rep = bank.commit(engine, collector, sources.joinToString(", ").take(80), { f, msg ->
            report(JobState.Running(TaskKind.LEARN, req.title, 0.45f + 0.55f * f, msg, "목소리 특징·말투 기준 구간 저장"))
        }, { cancelled })
        if (rep.kept == 0 && before == 0) error(rep.messages.joinToString("\n").ifEmpty { "사용할 수 있는 음성을 찾지 못했습니다." })
        val sb = StringBuilder()
        sb.append("음성 구간 ${rep.kept}개(${"%.0f".format(rep.keptSeconds)}초) 학습 · 전체 ${bank.segments.size}개, ${"%.0f".format(bank.totalSeconds)}초")
        rep.messages.forEach { sb.append("\n• ").append(it) }
        if (truncatedAny) sb.append("\n• 20분을 넘는 부분은 건너뛰었습니다.")
        if (bank.totalSeconds < 20) sb.append("\n• 20초 이상의 음성이 있으면 더 닮은 목소리가 됩니다.")
        if (bank.tuning.iterations > 0) sb.append("\n• 자료가 늘었으니 '자동 학습'을 다시 실행하면 더 좋아질 수 있습니다.")
        return JobState.Done(TaskKind.LEARN, req.title, sb.toString(), voiceId = req.voiceId)
    }

    // ------------------------------------------------------------------ auto-learning
    private fun autoLearn(req: TaskRequest.AutoLearn): JobState.Done {
        val engine = AppGraph.engine()
        val bank = AppGraph.voices.bank(req.voiceId) ?: error("목소리를 찾을 수 없습니다.")
        require(bank.segments.isNotEmpty()) { "먼저 음성 자료를 학습시켜 주세요." }
        val t0 = System.currentTimeMillis()
        val tuner = AutoTuner(engine, bank, req.config)
        val history = ArrayList<Double>()
        var lastRound = 0
        val res = tuner.run(progress = { p ->
            if (p.round > lastRound && lastRound > 0 && !p.best.isNaN()) history.add(p.best)
            lastRound = p.round
            val el = System.currentTimeMillis() - t0
            val eta = if (p.done > 1) (el.toDouble() / p.done * (p.total - p.done)).toLong() else -1
            report(JobState.Running(TaskKind.AUTOLEARN, req.title, p.done.toFloat() / p.total,
                "${p.round}/${p.rounds}세대 · 시도 ${p.done}/${p.total}", p.message, eta, history.toList() + listOf(p.best).filter { !it.isNaN() }, p.baseline,
                p.groups.filter { !it.value.isNaN() }.map { it.key.label to it.value }))
        }, cancel = { cancelled })
        bank.applyAutoLearning(res)
        val (vt, vb) = res.heldOut; val (ct, cb) = res.heldOutCriteria
        val g = res.best.gen; val v = res.best.v
        fun pct(x: Double) = "${"%.0f".format(x * 100)}%"
        val summary = buildString {
            append("${req.modeLabel} 자동 학습 완료 · ${res.generations.size}세대 · 시도 ${res.evaluated}가지 · ${formatDuration(System.currentTimeMillis() - t0)}\n")
            if (res.accepted) {
                append("새 문장 검증: 화자 유사도(ECAPA) ${"%.3f".format(vb)} → ${"%.3f".format(vt)}, ${VoiceCriteria.SIZE}가지 특징 유사도 ${pct(cb)} → ${pct(ct)} — 마지막 세대를 적용했습니다.\n")
                append("찾은 설정: 감정 ${"%.2f".format(g.ex)}, 속도감 ${"%.2f".format(g.cfg)}, 다양성 ${"%.2f".format(g.temp)}, 음색 혼합 ${"%.2f".format(g.blend)}, " +
                    "음색 보정 ${"%.1f".format(v.xs)}, 기울기 ${"%+.1f".format(v.tilt)}dB, 말끝 ${"%+.1f".format(v.endShift)}반음·길이 ${"%.2f".format(v.endStretch)}배\n")
            } else {
                append("새 문장 검증에서 기본 설정보다 분명히 나은 방식을 찾지 못해 기본 설정을 유지했습니다" +
                    (if (!vt.isNaN()) " (화자 유사도 기본 ${"%.3f".format(vb)} / 후보 ${"%.3f".format(vt)})." else ".") +
                    " 목소리 화면에서 각 세대를 직접 들어 보고 고를 수 있습니다.\n")
            }
            append("특징별 유사도(기본 → 결과): " + res.groupsBefore.keys.joinToString(", ") { k ->
                "${k.label} ${pct(res.groupsBefore[k] ?: Double.NaN)}→${pct(res.groupsAfter[k] ?: Double.NaN)}" }.replace("NaN%", "-"))
            res.tuning.env?.let { e ->
                if (e.after > e.before + 0.01) append("\n원본 녹음 환경 재현 준비됨(울림·마이크 색깔${if (e.noiseDb != null) "·배경 잡음" else ""}): 환경 유사도 ${pct(e.before)} → ${pct(e.after)}. 생성 화면의 '원본 녹음 환경 재현'을 켜면 적용됩니다.")
            }
        }
        return JobState.Done(TaskKind.AUTOLEARN, req.title, summary, voiceId = req.voiceId)
    }

    /** Also publish the MP3 to the shared Music/KoreanVoice folder (visible to music players). */
    private fun copyToMusic(file: File): Boolean {
        if (Build.VERSION.SDK_INT < 29) return false
        val values = ContentValues().apply {
            put(MediaStore.Audio.Media.DISPLAY_NAME, file.name)
            put(MediaStore.Audio.Media.MIME_TYPE, "audio/mpeg")
            put(MediaStore.Audio.Media.RELATIVE_PATH, Environment.DIRECTORY_MUSIC + "/KoreanVoice")
            put(MediaStore.Audio.Media.IS_PENDING, 1)
        }
        val uri = contentResolver.insert(MediaStore.Audio.Media.EXTERNAL_CONTENT_URI, values) ?: return false
        contentResolver.openOutputStream(uri)?.use { o -> file.inputStream().use { it.copyTo(o) } } ?: return false
        values.clear(); values.put(MediaStore.Audio.Media.IS_PENDING, 0)
        contentResolver.update(uri, values, null, null)
        return true
    }

    private fun progressNotification(title: String, text: String, fraction: Float): Notification {
        val cancel = PendingIntent.getService(this, 1, Intent(this, SynthesisService::class.java).setAction(ACTION_CANCEL),
            PendingIntent.FLAG_IMMUTABLE or PendingIntent.FLAG_UPDATE_CURRENT)
        return NotificationCompat.Builder(this, CHANNEL)
            .setSmallIcon(R.drawable.ic_notification)
            .setContentTitle(title)
            .setContentText(text)
            .setOnlyAlertOnce(true)
            .setOngoing(true)
            .setContentIntent(openApp())
            .setProgress(1000, (fraction * 1000).toInt(), fraction <= 0f)
            .addAction(0, "취소", cancel)
            .build()
    }

    private fun openApp(): PendingIntent = PendingIntent.getActivity(this, 0,
        Intent(this, MainActivity::class.java).addFlags(Intent.FLAG_ACTIVITY_SINGLE_TOP), PendingIntent.FLAG_IMMUTABLE)

    private fun canNotify(): Boolean = Build.VERSION.SDK_INT < 33 ||
        checkSelfPermission(android.Manifest.permission.POST_NOTIFICATIONS) == android.content.pm.PackageManager.PERMISSION_GRANTED

    @android.annotation.SuppressLint("MissingPermission")
    private fun notify(n: Notification, id: Int = NOTIF_ID) {
        if (canNotify()) runCatching { NotificationManagerCompat.from(this).notify(id, n) }
    }

    private fun doneNotification(title: String, text: String) {
        val n = NotificationCompat.Builder(this, CHANNEL_DONE)
            .setSmallIcon(R.drawable.ic_notification).setContentTitle("완료: $title").setContentText(text)
            .setContentIntent(openApp()).setAutoCancel(true).build()
        notify(n, NOTIF_DONE_ID)
    }

    override fun onDestroy() {
        scope.cancel()
        super.onDestroy()
    }

    companion object {
        const val CHANNEL = "synthesis"
        const val CHANNEL_DONE = "synthesis_done"
        const val NOTIF_ID = 11
        const val NOTIF_DONE_ID = 12
        const val ACTION_CANCEL = "io.github.koreanvoice.CANCEL"

        fun start(context: Context, req: TaskRequest) {
            AppGraph.jobs.pending = req
            AppGraph.jobs.update(JobState.Preparing(when (req) {
                is TaskRequest.Synthesis -> TaskKind.SYNTHESIS; is TaskRequest.Learn -> TaskKind.LEARN; is TaskRequest.AutoLearn -> TaskKind.AUTOLEARN
            }, "시작하는 중…"))
            androidx.core.content.ContextCompat.startForegroundService(context, Intent(context, SynthesisService::class.java))
        }

        fun cancel(context: Context) {
            AppGraph.jobs.cancelRequested = true
        }
    }
}

fun formatDuration(ms: Long): String {
    val s = ms / 1000
    val h = s / 3600; val m = (s % 3600) / 60; val sec = s % 60
    return when {
        h > 0 -> "${h}시간 ${m}분"
        m > 0 -> "${m}분 ${sec}초"
        else -> "${sec}초"
    }
}
