package io.github.koreanvoice.app

import android.annotation.SuppressLint
import android.content.Context
import android.content.Intent
import android.media.AudioFormat
import android.media.AudioRecord
import android.media.MediaPlayer
import android.media.MediaRecorder
import android.net.Uri
import androidx.core.content.FileProvider
import io.github.koreanvoice.audio.AudioTools
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import java.io.File
import kotlin.math.abs
import kotlin.math.max

/** Single shared player: one file plays at a time. */
object Player {
    private var mp: MediaPlayer? = null
    private val _playing = MutableStateFlow<String?>(null)
    val playing: StateFlow<String?> = _playing

    fun toggle(file: File) {
        if (_playing.value == file.path) { stop(); return }
        stop()
        runCatching {
            mp = MediaPlayer().apply {
                setDataSource(file.path)
                setOnCompletionListener { stop() }
                prepare(); start()
            }
            _playing.value = file.path
        }
    }

    fun stop() {
        mp?.runCatching { stop(); release() }
        mp = null
        _playing.value = null
    }
}

/** Mono 16-bit microphone recorder (records at a supported rate; resampled later). */
class VoiceRecorder {
    @Volatile private var running = false
    private val _level = MutableStateFlow(0f)
    val level: StateFlow<Float> = _level
    private val _seconds = MutableStateFlow(0.0)
    val seconds: StateFlow<Double> = _seconds

    var sampleRate = 44100; private set

    /** Blocks until [stop] is called or [maxSec] elapsed; returns the recorded samples. */
    @SuppressLint("MissingPermission")
    fun record(maxSec: Int = 120): FloatArray {
        val rate = intArrayOf(44100, 48000, 22050, 16000).first {
            AudioRecord.getMinBufferSize(it, AudioFormat.CHANNEL_IN_MONO, AudioFormat.ENCODING_PCM_16BIT) > 0
        }
        sampleRate = rate
        val minBuf = AudioRecord.getMinBufferSize(rate, AudioFormat.CHANNEL_IN_MONO, AudioFormat.ENCODING_PCM_16BIT)
        val rec = AudioRecord(MediaRecorder.AudioSource.VOICE_RECOGNITION, rate, AudioFormat.CHANNEL_IN_MONO,
            AudioFormat.ENCODING_PCM_16BIT, max(minBuf, rate))
        val out = ArrayList<ShortArray>()
        val buf = ShortArray(rate / 10)
        var total = 0
        running = true
        rec.startRecording()
        try {
            while (running && total < maxSec * rate) {
                val n = rec.read(buf, 0, buf.size)
                if (n <= 0) continue
                out.add(buf.copyOf(n)); total += n
                var peak = 0
                for (i in 0 until n) peak = max(peak, abs(buf[i].toInt()))
                _level.value = peak / 32768f
                _seconds.value = total.toDouble() / rate
            }
        } finally {
            rec.stop(); rec.release(); running = false
        }
        val res = FloatArray(total)
        var p = 0
        for (s in out) for (v in s) res[p++] = v / 32768f
        return res
    }

    fun stop() { running = false }
}

object Sharing {
    fun uri(context: Context, file: File): Uri = FileProvider.getUriForFile(context, context.packageName + ".files", file)

    fun share(context: Context, file: File) {
        val i = Intent(Intent.ACTION_SEND).apply {
            type = if (file.extension == "wav") "audio/wav" else "audio/mpeg"
            putExtra(Intent.EXTRA_STREAM, uri(context, file))
            addFlags(Intent.FLAG_GRANT_READ_URI_PERMISSION)
        }
        context.startActivity(Intent.createChooser(i, "음성 파일 공유").addFlags(Intent.FLAG_ACTIVITY_NEW_TASK))
    }

    fun exportTo(context: Context, file: File, target: Uri) {
        context.contentResolver.openOutputStream(target)?.use { o -> file.inputStream().use { it.copyTo(o) } }
    }

    fun writePreview(context: Context, samples: FloatArray): File {
        val f = File(context.cacheDir, "preview.wav")
        AudioTools.writeWav(f, samples, 22050)
        return f
    }
}

/** Phonetically varied Korean sentences for recording a voice sample (~20–30 s). */
const val RECORDING_SCRIPT = """
안녕하세요. 지금부터 제 목소리를 녹음하겠습니다.
오늘 아침에는 따뜻한 커피 한 잔과 함께 신문을 읽었어요.
푸른 하늘 아래 넓은 들판에서 아이들이 즐겁게 뛰어놀고 있습니다.
책을 읽으면 새로운 세상을 만날 수 있다는 것이 정말 신기하지 않나요?
빠르게 변하는 세상 속에서도 우리는 서로를 배려하며 살아가야 합니다.
"""
