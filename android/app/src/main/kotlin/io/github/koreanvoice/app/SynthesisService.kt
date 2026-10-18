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
import io.github.koreanvoice.audio.Mp3Encoder
import io.github.koreanvoice.engine.SynthesisJob
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
 * Foreground service that renders long texts. The work survives leaving the app / turning off the
 * screen, shows progress in a notification, and — because chunks are cached on disk — a job that
 * was killed resumes from the last finished sentence when started again with the same input.
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
        ServiceCompat.startForeground(this, NOTIF_ID, progressNotification(req.title, "준비 중…", 0, 0), type)
        wakeLock = (getSystemService(Context.POWER_SERVICE) as PowerManager)
            .newWakeLock(PowerManager.PARTIAL_WAKE_LOCK, "KoreanVoice:synthesis").apply { acquire(8 * 60 * 60 * 1000L) }
        scope.launch { run(req) }
        return START_NOT_STICKY
    }

    private fun run(req: SynthesisRequest) {
        val jobs = AppGraph.jobs
        val t0 = System.currentTimeMillis()
        try {
            jobs.update(JobState.Preparing("모델을 불러오는 중…"))
            val engine = AppGraph.engine()
            val voice = AppGraph.voices.get(req.voiceId)
            val (_, pipelined) = AppGraph.scheduleFor(AppGraph.settings.performanceMode)
            jobs.update(JobState.Preparing("텍스트 분석 중…"))
            val job = SynthesisJob(engine, AppGraph.jobCacheDir, req.text, voice?.embedding, req.params, pipelined = pipelined)
            if (job.chunks.isEmpty()) error("읽을 수 있는 한국어 텍스트가 없습니다.")
            var lastNotif = 0L
            job.render(isCancelled = { jobs.cancelRequested }) { p ->
                jobs.update(JobState.Running(req.title, p))
                val now = System.currentTimeMillis()
                if (now - lastNotif > 1000 || p.done == p.total) {
                    lastNotif = now
                    val eta = if (p.etaMs > 0) " · 약 ${formatDuration(p.etaMs)} 남음" else ""
                    notify(progressNotification(req.title, "${p.done}/${p.total} 문장$eta", p.done, p.total))
                }
            }
            jobs.update(JobState.Finalizing(req.title))
            notify(progressNotification(req.title, "MP3 파일로 저장 중…", 0, 0))
            val stamp = SimpleDateFormat("yyyyMMdd_HHmmss", Locale.KOREA).format(Date())
            val safeTitle = req.title.replace(Regex("[^0-9A-Za-z가-힣 _-]"), "").trim().take(40).ifEmpty { "음성" }
            val out = File(AppGraph.outputDir, "${safeTitle}_$stamp.mp3")
            val id3 = Mp3Encoder.Id3Info(req.title, voice?.name ?: "기본 음성",
                "AI 생성 음성 · OpenVoice V2 + MeloTTS (KoreanVoice 앱)")
            out.outputStream().buffered(1 shl 16).use { job.assembleMp3(it, req.bitrate, id3, req.pauseScale) }
            val saved = runCatching { copyToMusic(out) }.getOrDefault(false)
            job.deleteCache()
            val dur = estimateDuration(out, req.bitrate)
            jobs.update(JobState.Done(req.title, out, dur, System.currentTimeMillis() - t0, saved))
            doneNotification(req.title, "완료 · ${formatDuration((dur * 1000).toLong())} 분량" + if (saved) " · 음악/KoreanVoice에 저장됨" else "")
        } catch (e: SynthesisJob.CancelledException) {
            jobs.update(JobState.Cancelled)
        } catch (e: OutOfMemoryError) {
            AppGraph.releaseEngine()
            jobs.update(JobState.Failed("메모리가 부족합니다. 다른 앱을 종료하거나 '절전' 성능 모드를 사용해 주세요."))
        } catch (t: Throwable) {
            jobs.update(JobState.Failed(t.message ?: t.toString()))
        } finally {
            wakeLock?.let { if (it.isHeld) it.release() }
            ServiceCompat.stopForeground(this, ServiceCompat.STOP_FOREGROUND_REMOVE)
            stopSelf()
        }
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

    private fun estimateDuration(f: File, kbps: Int): Double = f.length() * 8.0 / (kbps * 1000.0)

    private fun progressNotification(title: String, text: String, done: Int, total: Int): Notification {
        val cancel = PendingIntent.getService(this, 1, Intent(this, SynthesisService::class.java).setAction(ACTION_CANCEL),
            PendingIntent.FLAG_IMMUTABLE or PendingIntent.FLAG_UPDATE_CURRENT)
        return NotificationCompat.Builder(this, CHANNEL)
            .setSmallIcon(R.drawable.ic_notification)
            .setContentTitle("음성 생성: $title")
            .setContentText(text)
            .setOnlyAlertOnce(true)
            .setOngoing(true)
            .setContentIntent(openApp())
            .setProgress(total, done, total == 0)
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
            .setSmallIcon(R.drawable.ic_notification).setContentTitle("생성 완료: $title").setContentText(text)
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

        fun start(context: Context, req: SynthesisRequest) {
            AppGraph.jobs.pending = req
            AppGraph.jobs.update(JobState.Preparing("시작하는 중…"))
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
