package io.github.koreanvoice.audio

import kotlin.math.PI
import kotlin.math.abs
import kotlin.math.cos
import kotlin.math.max
import kotlin.math.min
import kotlin.math.pow
import kotlin.math.roundToInt

/**
 * TD-PSOLA: pitch and duration changes that keep the spectral envelope (formants) — unlike resampling-based
 * shifting, the voice does not get a "chipmunk"/"giant" timbre, so speaker identity is preserved.
 * Supports time-varying pitch (semitones) and duration factors, used for sentence-ending shaping (말끝).
 */
object Psola {
    /** Pitch track at the signal rate: period in samples per analysis frame (0 = unvoiced), hop in samples. */
    class Track(val period: DoubleArray, val hop: Int, val offset: Int) {
        fun periodAt(i: Int): Double {
            val f = ((i - offset).toDouble() / hop).roundToInt().coerceIn(0, max(0, period.size - 1))
            return if (period.isEmpty()) 0.0 else period[f]
        }
    }

    fun track(x: FloatArray, sr: Int): Track {
        val x16 = if (sr == 16000) x else Resampler.resample(x, sr, 16000)
        val f0 = VoiceAnalysis.yin(x16, 16000)
        // smooth isolated octave jumps with a 3-point median
        val m = DoubleArray(f0.size) { i ->
            if (f0[i] <= 0) 0.0 else {
                val w = listOfNotNull(f0.getOrNull(i - 1)?.takeIf { it > 0 }, f0[i], f0.getOrNull(i + 1)?.takeIf { it > 0 }).sorted()
                w[w.size / 2]
            }
        }
        val hop = sr / 100
        return Track(DoubleArray(m.size) { if (m[it] > 0) sr / m[it] else 0.0 }, hop, (0.02 * sr).toInt())
    }

    /** Analysis pitch marks: one per period at waveform peaks in voiced parts, every 5 ms elsewhere. */
    private fun marks(x: FloatArray, t: Track, sr: Int): Pair<IntArray, DoubleArray> {
        val ms = ArrayList<Int>(); val ps = ArrayList<Double>()
        val uv = sr / 200
        var i = 0
        while (i < x.size) {
            val p = t.periodAt(i)
            if (p > 0) {
                // peak of |x| within the next period
                val end = min(x.size, i + p.toInt())
                var best = i; var bv = -1f
                for (k in i until end) { val v = abs(x[k]); if (v > bv) { bv = v; best = k } }
                if (ms.isNotEmpty() && best - ms.last() < p * 0.5) { i = ms.last() + p.toInt(); continue }
                ms.add(best); ps.add(p); i = best + max(1, (p * 0.75).toInt())
            } else { ms.add(i); ps.add(uv.toDouble()); i += uv }
        }
        // use the real distance between consecutive voiced marks as the local period, so an unmodified
        // signal is rebuilt from its own marks (no jitter); fall back to the track period at irregular marks
        for (k in 0 until ms.size - 1) {
            val d = (ms[k + 1] - ms[k]).toDouble()
            if (t.periodAt(ms[k]) > 0 && d in ps[k] * 0.7..ps[k] * 1.3) ps[k] = d
        }
        return ms.toIntArray() to ps.toDoubleArray()
    }

    /**
     * General modification. [semitones](sample) gives the pitch change, [stretch](sample) the local duration
     * factor (>1 = slower/longer), both as functions of the input position. Unvoiced parts are only stretched.
     */
    fun modify(x: FloatArray, sr: Int, semitones: (Int) -> Double, stretch: (Int) -> Double, tr: Track = track(x, sr)): FloatArray {
        if (x.size < sr / 20) return x
        val (am, ap) = marks(x, tr, sr)
        if (am.isEmpty()) return x
        val est = (x.size * 2.2).toInt() + sr
        val out = DoubleArray(est); val norm = DoubleArray(est)
        var ts = am[0].toDouble()          // synthesis time
        var ta = am[0].toDouble()          // corresponding analysis time
        var k = 0
        while (ta < x.size) {
            while (k + 1 < am.size && abs(am[k + 1] - ta) < abs(am[k] - ta)) k++
            val voiced = tr.periodAt(am[k]) > 0
            val p = ap[k]
            val beta = if (voiced) 2.0.pow(semitones(am[k]) / 12) else 1.0
            val alpha = stretch(am[k]).coerceIn(0.25, 4.0)
            val half = p.toInt().coerceAtLeast(16)
            val c = ts.roundToInt()
            for (j in -half until half) {
                val src = am[k] + j; val dst = c + j
                if (src < 0 || src >= x.size || dst < 0 || dst >= est) continue
                val w = 0.5 + 0.5 * cos(PI * j / half)
                out[dst] += x[src] * w; norm[dst] += w
            }
            val sp = p / beta
            ts += sp
            ta += sp / alpha
        }
        val len = min(est, ts.roundToInt())
        return FloatArray(len) { if (norm[it] > 1e-6) (out[it] / max(norm[it], 0.5)).toFloat() else 0f }
    }

    fun pitchShift(x: FloatArray, sr: Int, semitones: Double): FloatArray =
        if (abs(semitones) < 0.05) x else modify(x, sr, { semitones }, { 1.0 })

    fun tempoPitch(x: FloatArray, sr: Int, tempo: Double, semitones: Double): FloatArray =
        if (abs(semitones) < 0.05 && abs(tempo - 1) < 0.005) x else modify(x, sr, { semitones }, { 1.0 / tempo })

    /** Phrase ends: end sample of voiced runs (gaps ≤ 30 ms bridged) followed by ≥150 ms without voicing. */
    fun phraseEnds(t: Track, n: Int, sr: Int): List<Int> {
        val v = t.period.map { it > 0 }
        val ends = ArrayList<Int>()
        var i = 0
        while (i < v.size) {
            if (!v[i]) { i++; continue }
            val start = i; var last = i; var j = i + 1
            while (j < v.size && j - last <= 3) { if (v[j]) last = j; j++ }
            val after = (last + 1 until min(v.size, last + 16)).count { v[it] }
            if (last - start >= 8 && after == 0) ends.add(min(n, t.offset + (last + 1) * t.hop))
            i = last + 1
        }
        return ends
    }

    /**
     * Sentence-ending shaping (말끝): over the last [endMs] of every phrase the pitch glides by [shiftSt]
     * semitones (+ rising, − falling) and the last part is lengthened by [stretch]. Only the ending regions
     * are resynthesised; everything else is copied untouched and the joins are cross-faded.
     */
    fun shapeEndings(x: FloatArray, sr: Int, shiftSt: Double, stretch: Double, endMs: Int = 320): FloatArray {
        if (abs(shiftSt) < 0.05 && abs(stretch - 1) < 0.01) return x
        val tr = track(x, sr)
        val ends = phraseEnds(tr, x.size, sr)
        if (ends.isEmpty()) return x
        val len = endMs * sr / 1000; val pad = sr / 25; val tail = sr / 8; val xf = sr / 100
        val out = ArrayList<FloatArray>()
        var cursor = 0
        for (e in ends) {
            val s0 = max(cursor, e - len - pad); val s1 = min(x.size, e + tail)
            if (s1 - s0 < len / 2) continue
            val seg = x.copyOfRange(s0, s1)
            val endInSeg = e - s0; val startInSeg = endInSeg - len
            fun p(i: Int) = ((i - startInSeg).toDouble() / len).coerceIn(0.0, 1.0)
            val mod = modify(seg, sr, { i -> if (i < startInSeg) 0.0 else shiftSt * p(i) * p(i) },
                { i -> if (i < startInSeg || i > endInSeg + sr / 50) 1.0 else 1 + (stretch - 1) * min(1.0, p(i) * 1.6) })
            // keep the untouched part before the region, then cross-fade into the modified region
            val head = x.copyOfRange(cursor, s0 + xf)
            val n = min(xf, mod.size)
            for (k in 0 until n) { val w = k.toFloat() / n; head[head.size - n + k] = head[head.size - n + k] * (1 - w) + mod[k] * w }
            out.add(head); out.add(mod.copyOfRange(n, mod.size))
            cursor = s1
        }
        if (cursor < x.size) out.add(x.copyOfRange(cursor, x.size))
        val res = FloatArray(out.sumOf { it.size }); var o = 0
        for (a in out) { System.arraycopy(a, 0, res, o, a.size); o += a.size }
        return res
    }
}
