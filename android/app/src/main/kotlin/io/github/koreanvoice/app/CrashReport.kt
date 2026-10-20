package io.github.koreanvoice.app

import android.app.ActivityManager
import android.app.ApplicationExitInfo
import android.content.Context
import android.os.Build
import java.io.File
import java.io.PrintWriter
import java.io.StringWriter
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale

/**
 * Explains, on the next launch, why the previous process ended abnormally: an uncaught exception
 * (stack trace written here), or — Android 11+ — what the system reports (low-memory kill, native
 * crash, ANR). A low-memory kill or native crash cannot be caught in-process, so the step a task was
 * at is kept as a breadcrumb and shown alongside.
 */
object CrashReport {
    class Report(val title: String, val advice: String, val details: String)

    private lateinit var app: Context
    private val prefs by lazy { app.getSharedPreferences("crash", Context.MODE_PRIVATE) }
    private val crashFile get() = File(app.filesDir, "last_crash.txt")
    private var lastStep: String? = null
    private var lastStepAt = 0L
    private var consumed = false
    private var current: Report? = null

    fun install(context: Context) {
        app = context.applicationContext
        val prev = Thread.getDefaultUncaughtExceptionHandler()
        Thread.setDefaultUncaughtExceptionHandler { t, e ->
            runCatching {
                val sw = StringWriter(); e.printStackTrace(PrintWriter(sw))
                crashFile.writeText("${stamp(System.currentTimeMillis())} · 스레드 ${t.name}\n" +
                    (prefs.getString("step", null)?.let { "작업: $it\n" } ?: "") + memLine() + "\n\n" + sw.toString().take(6000))
            }
            prev?.uncaughtException(t, e)
        }
    }

    /** Remembers what a background task is doing; null when it ends normally. */
    fun step(s: String?) {
        if (!::app.isInitialized || s == lastStep) return
        val now = System.currentTimeMillis()
        if (s != null && lastStep != null && now - lastStepAt < 1000) return
        lastStep = s; lastStepAt = now
        val e = prefs.edit()
        if (s == null) e.remove("step") else e.putString("step", "$s · ${memLine()}")
        e.apply()
    }

    /** The report about the previous process (kept until [dismiss]); null if it ended normally. */
    fun pending(): Report? {
        if (!consumed && ::app.isInitialized) { consumed = true; current = runCatching { read() }.getOrNull() }
        return current
    }

    fun dismiss() { current = null }

    private fun read(): Report? {
        val step = prefs.getString("step", null)
        val crash = crashFile.takeIf { it.exists() }?.let { f -> runCatching { f.readText() }.getOrNull().also { f.delete() } }
        val exit = lastAbnormalExit()
        prefs.edit().remove("step").apply()
        val stepLine = step?.let { "진행 중이던 작업: $it\n" } ?: ""
        val env = "기기: ${Build.MANUFACTURER} ${Build.MODEL} · Android ${Build.VERSION.RELEASE} · 앱 ${BuildConfig.VERSION_NAME}\n" + totalMemLine()
        return when {
            crash != null -> Report("지난번에 앱에서 오류가 발생해 종료되었습니다",
                "아래 내용을 복사해 알려 주시면 원인을 고칠 수 있습니다.", "$env\n$stepLine\n$crash")
            exit != null -> {
                val (title, advice) = explain(exit)
                Report(title, advice, "$env\n$stepLine\n${describe(exit)}")
            }
            step != null -> Report("지난번 작업 중 앱이 종료되었습니다",
                "휴대폰 메모리가 부족했을 가능성이 큽니다. 다른 앱을 닫고 다시 시도해 주세요. 반복되면 '설정 → 성능 모드'를 '절전'으로 바꿔 보세요.",
                "$env\n$stepLine")
            else -> null
        }
    }

    private fun lastAbnormalExit(): ApplicationExitInfo? {
        if (Build.VERSION.SDK_INT < 30) return null
        val am = app.getSystemService(ActivityManager::class.java) ?: return null
        val seen = prefs.getLong("seenExit", 0L)
        val list = runCatching { am.getHistoricalProcessExitReasons(app.packageName, 0, 8) }.getOrNull() ?: return null
        list.firstOrNull()?.let { prefs.edit().putLong("seenExit", maxOf(seen, it.timestamp)).apply() }
        val weekAgo = System.currentTimeMillis() - 7L * 24 * 3600 * 1000
        return list.firstOrNull { e ->
            e.timestamp > seen && e.timestamp > weekAgo && when (e.reason) {
                ApplicationExitInfo.REASON_CRASH, ApplicationExitInfo.REASON_CRASH_NATIVE, ApplicationExitInfo.REASON_ANR -> true
                // killed while visible or running a task (not an ordinary background clean-up)
                ApplicationExitInfo.REASON_LOW_MEMORY, ApplicationExitInfo.REASON_SIGNALED, ApplicationExitInfo.REASON_EXCESSIVE_RESOURCE_USAGE ->
                    e.importance <= ActivityManager.RunningAppProcessInfo.IMPORTANCE_FOREGROUND_SERVICE
                else -> false
            }
        }
    }

    private fun explain(e: ApplicationExitInfo): Pair<String, String> = when (e.reason) {
        ApplicationExitInfo.REASON_LOW_MEMORY, ApplicationExitInfo.REASON_SIGNALED, ApplicationExitInfo.REASON_EXCESSIVE_RESOURCE_USAGE ->
            "지난번에 메모리 부족으로 앱이 강제 종료되었습니다" to
                "휴대폰 메모리가 부족해 안드로이드가 앱을 종료했습니다. 이 버전은 목소리 학습 중 메모리 사용을 크게 줄였습니다. " +
                "그래도 반복되면 다른 앱을 닫고, '설정 → 성능 모드'를 '절전'으로 바꿔 주세요."
        ApplicationExitInfo.REASON_ANR -> "지난번에 앱이 응답하지 않아 종료되었습니다" to
            "아래 내용을 복사해 알려 주시면 원인을 고칠 수 있습니다."
        else -> "지난번에 앱이 비정상 종료되었습니다" to "아래 내용을 복사해 알려 주시면 원인을 고칠 수 있습니다."
    }

    private fun describe(e: ApplicationExitInfo): String {
        val reason = when (e.reason) {
            ApplicationExitInfo.REASON_LOW_MEMORY -> "LOW_MEMORY (메모리 부족)"
            ApplicationExitInfo.REASON_SIGNALED -> "SIGNALED (시스템 강제 종료, signal ${e.status})"
            ApplicationExitInfo.REASON_CRASH -> "CRASH"
            ApplicationExitInfo.REASON_CRASH_NATIVE -> "CRASH_NATIVE (네이티브 오류)"
            ApplicationExitInfo.REASON_ANR -> "ANR (응답 없음)"
            ApplicationExitInfo.REASON_EXCESSIVE_RESOURCE_USAGE -> "EXCESSIVE_RESOURCE_USAGE"
            else -> "reason ${e.reason}"
        }
        return "${stamp(e.timestamp)} · 종료 원인: $reason\n" +
            "당시 메모리: PSS ${e.pss / 1024} MB, RSS ${e.rss / 1024} MB · 중요도 ${e.importance}\n" +
            (e.description?.let { "설명: $it\n" } ?: "")
    }

    private fun memLine(): String = runCatching {
        val rss = File("/proc/self/status").readLines().first { it.startsWith("VmRSS") }.filter { it.isDigit() }.toLong() / 1024
        "메모리 ${rss} MB"
    }.getOrDefault("")

    private fun totalMemLine(): String = runCatching {
        val mi = ActivityManager.MemoryInfo()
        app.getSystemService(ActivityManager::class.java).getMemoryInfo(mi)
        "기기 메모리: 전체 ${mi.totalMem shr 20} MB, 여유 ${mi.availMem shr 20} MB"
    }.getOrDefault("")

    private fun stamp(t: Long) = SimpleDateFormat("yyyy-MM-dd HH:mm:ss", Locale.KOREA).format(Date(t))
}
