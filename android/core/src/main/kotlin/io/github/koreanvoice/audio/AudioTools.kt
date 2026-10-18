package io.github.koreanvoice.audio

import java.io.DataInputStream
import java.io.File
import java.io.InputStream
import java.io.OutputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder
import kotlin.math.abs
import kotlin.math.log10
import kotlin.math.max
import kotlin.math.min
import kotlin.math.pow
import kotlin.math.sqrt
import kotlin.math.tanh

/** Voice-activity detection, loudness and edge processing for 22.05 kHz mono float audio. */
object AudioTools {

    data class Segment(val start: Int, val end: Int)

    data class SampleReport(
        val totalSec: Double,
        val voicedSec: Double,
        val snrDb: Double,
        val clippedRatio: Double,
        val peakDb: Double,
    ) {
        /** 0..100 heuristic quality score shown to the user. */
        val score: Int
            get() {
                var s = 100.0
                if (voicedSec < 10) s -= (10 - voicedSec) * 4
                if (voicedSec < 3) s -= 30
                if (snrDb < 25) s -= (25 - snrDb) * 2
                s -= clippedRatio * 2000
                if (peakDb < -30) s -= 15
                return s.coerceIn(0.0, 100.0).toInt()
            }
        val warnings: List<String>
            get() = buildList {
                if (voicedSec < 3) add("음성 구간이 너무 짧습니다 (최소 3초, 권장 10~30초).")
                else if (voicedSec < 8) add("음성이 조금 짧습니다. 10~30초 분량이면 더 정확해집니다.")
                if (snrDb < 18) add("배경 소음이 큽니다. 조용한 곳에서 녹음한 샘플을 권장합니다.")
                if (clippedRatio > 0.002) add("소리가 찢어진(클리핑) 구간이 있습니다. 녹음 볼륨을 낮춰 주세요.")
                if (peakDb < -30) add("음량이 매우 작습니다.")
            }
    }

    fun frameDb(x: FloatArray, frame: Int): DoubleArray {
        val n = x.size / frame
        return DoubleArray(n) { i ->
            var e = 0.0
            for (k in i * frame until (i + 1) * frame) e += x[k].toDouble() * x[k]
            10 * log10(e / frame + 1e-12)
        }
    }

    private fun percentile(v: DoubleArray, p: Double): Double {
        if (v.isEmpty()) return -120.0
        val s = v.sortedArray()
        return s[((s.size - 1) * p).toInt()]
    }

    /** Energy VAD with adaptive noise floor; returns voiced segments (sample indices). */
    fun vad(x: FloatArray, sr: Int, minSpeechMs: Int = 200, mergeGapMs: Int = 300, padMs: Int = 100): List<Segment> {
        val frame = sr / 50 // 20 ms
        val db = frameDb(x, frame)
        if (db.isEmpty()) return emptyList()
        val floor = percentile(db, 0.10)
        val peak = percentile(db, 0.98)
        val thr = max(floor + 10.0, peak - 40.0)
        val voiced = BooleanArray(db.size) { db[it] > thr }
        val segs = ArrayList<Segment>()
        var i = 0
        while (i < voiced.size) {
            if (!voiced[i]) { i++; continue }
            var j = i
            while (j < voiced.size && voiced[j]) j++
            segs.add(Segment(i, j)); i = j
        }
        val gap = mergeGapMs / 20
        val merged = ArrayList<Segment>()
        for (s in segs) {
            if (merged.isNotEmpty() && s.start - merged.last().end <= gap) merged[merged.size - 1] = Segment(merged.last().start, s.end)
            else merged.add(s)
        }
        val pad = padMs / 20
        return merged.filter { (it.end - it.start) * 20 >= minSpeechMs }
            .map { Segment(max(0, (it.start - pad)) * frame, min(x.size, (it.end + pad) * frame)) }
    }

    fun analyze(x: FloatArray, sr: Int): SampleReport {
        val segs = vad(x, sr)
        val voiced = segs.sumOf { it.end - it.start }
        val db = frameDb(x, sr / 50)
        val floor = percentile(db, 0.10)
        val speech = percentile(db, 0.90)
        var clipped = 0; var peak = 0f
        for (v in x) { val a = abs(v); if (a >= 0.99f) clipped++; if (a > peak) peak = a }
        return SampleReport(x.size.toDouble() / sr, voiced.toDouble() / sr, speech - floor,
            if (x.isEmpty()) 0.0 else clipped.toDouble() / x.size, 20 * log10(peak + 1e-9))
    }

    /** Concatenate voiced parts and cut into pieces of at most [maxSec] (OpenVoice se_extractor style). */
    fun voicedPieces(x: FloatArray, sr: Int, maxSec: Double = 10.0, minSec: Double = 1.0): List<FloatArray> {
        val segs = vad(x, sr)
        val total = segs.sumOf { it.end - it.start }
        val joined = FloatArray(total)
        var p = 0
        for (s in segs) { System.arraycopy(x, s.start, joined, p, s.end - s.start); p += s.end - s.start }
        val src = if (total >= sr * minSec) joined else x
        val maxLen = (maxSec * sr).toInt()
        val n = max(1, (src.size + maxLen - 1) / maxLen)
        val len = src.size / n
        return (0 until n).map { k -> src.copyOfRange(k * len, if (k == n - 1) src.size else (k + 1) * len) }
            .filter { it.size >= sr * minSec || n == 1 }
    }

    fun highPass(x: FloatArray, sr: Int, cutoffHz: Double = 60.0): FloatArray {
        val rc = 1.0 / (2 * Math.PI * cutoffHz); val dt = 1.0 / sr; val a = rc / (rc + dt)
        val y = FloatArray(x.size)
        var prevX = 0.0; var prevY = 0.0
        for (i in x.indices) { val v = a * (prevY + x[i] - prevX); y[i] = v.toFloat(); prevX = x[i].toDouble(); prevY = v }
        return y
    }

    /** RMS (dBFS) of the active (non-silent) frames. */
    fun activeRmsDb(x: FloatArray, sr: Int): Double {
        val db = frameDb(x, sr / 50)
        if (db.isEmpty()) return -120.0
        val peak = db.max()
        val act = db.filter { it > peak - 30 }
        if (act.isEmpty()) return -120.0
        val meanPow = act.sumOf { 10.0.pow(it / 10) } / act.size
        return 10 * log10(meanPow + 1e-12)
    }

    /** Trim leading/trailing silence (relative threshold), keeping [marginMs]. */
    fun trimSilence(x: FloatArray, sr: Int, relDb: Double = -45.0, marginMs: Int = 40): FloatArray {
        if (x.isEmpty()) return x
        val frame = sr / 100
        val db = frameDb(x, frame)
        if (db.isEmpty()) return x
        val thr = db.max() + relDb
        val first = db.indexOfFirst { it > thr }
        val last = db.indexOfLast { it > thr }
        if (first < 0) return FloatArray(0)
        val m = marginMs * sr / 1000
        val s = max(0, first * frame - m)
        val e = min(x.size, (last + 1) * frame + m)
        return x.copyOfRange(s, e)
    }

    fun fade(x: FloatArray, sr: Int, ms: Int = 8) {
        val n = min(x.size / 2, ms * sr / 1000)
        for (i in 0 until n) { val g = i.toFloat() / n; x[i] *= g; x[x.size - 1 - i] *= g }
    }

    /** Gain to bring the active RMS to [targetDb], limited to ±[maxGainDb]. */
    fun normalizeGain(x: FloatArray, sr: Int, targetDb: Double = -20.0, maxGainDb: Double = 12.0): Float {
        val cur = activeRmsDb(x, sr)
        if (cur < -90) return 1f
        val g = (targetDb - cur).coerceIn(-maxGainDb, maxGainDb)
        return 10.0.pow(g / 20).toFloat()
    }

    /** Smooth peak limiter: linear below 0.9, tanh knee above (never exceeds 1.0). */
    fun limit(x: FloatArray) {
        for (i in x.indices) {
            val v = x[i]; val a = abs(v)
            if (a > 0.9f) x[i] = (Math.signum(v) * (0.9 + 0.1 * tanh((a - 0.9) / 0.1))).toFloat()
        }
    }

    fun toPcm16(x: FloatArray): ShortArray = ShortArray(x.size) { (x[it].coerceIn(-1f, 1f) * 32767f).toInt().toShort() }

    // ---------------------------------------------------------------- raw / WAV IO
    fun writePcm16(file: File, x: FloatArray) {
        val bb = ByteBuffer.allocate(x.size * 2).order(ByteOrder.LITTLE_ENDIAN)
        for (v in x) bb.putShort((v.coerceIn(-1f, 1f) * 32767f).toInt().toShort())
        val tmp = File(file.path + ".tmp")
        tmp.writeBytes(bb.array())
        if (!tmp.renameTo(file)) { file.delete(); tmp.renameTo(file) }
    }

    fun readPcm16(file: File): FloatArray {
        val b = file.readBytes()
        val bb = ByteBuffer.wrap(b).order(ByteOrder.LITTLE_ENDIAN)
        return FloatArray(b.size / 2) { bb.getShort().toFloat() / 32768f }
    }

    fun writeWav(out: OutputStream, x: FloatArray, sr: Int) {
        val data = x.size * 2
        val bb = ByteBuffer.allocate(44 + data).order(ByteOrder.LITTLE_ENDIAN)
        bb.put("RIFF".toByteArray()).putInt(36 + data).put("WAVE".toByteArray())
        bb.put("fmt ".toByteArray()).putInt(16).putShort(1).putShort(1).putInt(sr).putInt(sr * 2).putShort(2).putShort(16)
        bb.put("data".toByteArray()).putInt(data)
        for (v in x) bb.putShort((v.coerceIn(-1f, 1f) * 32767f).toInt().toShort())
        out.write(bb.array())
    }

    fun writeWav(file: File, x: FloatArray, sr: Int) = file.outputStream().use { writeWav(it, x, sr) }

    /** Minimal RIFF/WAVE reader (PCM 8/16/24/32-bit and float32), downmixed to mono. */
    fun readWav(input: InputStream): Pair<FloatArray, Int> {
        val all = DataInputStream(input).readBytes()
        val bb = ByteBuffer.wrap(all).order(ByteOrder.LITTLE_ENDIAN)
        require(all.size >= 12 && String(all, 0, 4) == "RIFF" && String(all, 8, 4) == "WAVE") { "WAV 파일이 아닙니다" }
        var pos = 12
        var fmt = 1; var ch = 1; var sr = 22050; var bits = 16
        while (pos + 8 <= all.size) {
            val id = String(all, pos, 4); val len = bb.getInt(pos + 4)
            val body = pos + 8
            if (id == "fmt ") {
                fmt = bb.getShort(body).toInt() and 0xffff; ch = bb.getShort(body + 2).toInt(); sr = bb.getInt(body + 4)
                bits = bb.getShort(body + 14).toInt()
                if (fmt == 0xFFFE && len >= 40) fmt = bb.getShort(body + 24).toInt()
            } else if (id == "data") {
                val end = min(all.size, body + len)
                val bps = bits / 8
                val frames = (end - body) / (bps * ch)
                val out = FloatArray(frames)
                for (f in 0 until frames) {
                    var acc = 0f
                    for (c in 0 until ch) {
                        val o = body + (f * ch + c) * bps
                        acc += when {
                            fmt == 3 && bits == 32 -> bb.getFloat(o)
                            bits == 16 -> bb.getShort(o) / 32768f
                            bits == 8 -> ((all[o].toInt() and 0xff) - 128) / 128f
                            bits == 24 -> ((all[o].toInt() and 0xff) or ((all[o + 1].toInt() and 0xff) shl 8) or (all[o + 2].toInt() shl 16)) / 8388608f
                            bits == 32 -> bb.getInt(o) / 2147483648f
                            else -> 0f
                        }
                    }
                    out[f] = acc / ch
                }
                return out to sr
            }
            pos = body + len + (len and 1)
        }
        error("WAV data 청크가 없습니다")
    }

    fun silence(sr: Int, ms: Int): FloatArray = FloatArray(max(0, ms) * sr / 1000)

    fun rms(x: FloatArray): Double = if (x.isEmpty()) 0.0 else sqrt(x.sumOf { it.toDouble() * it } / x.size)
}
