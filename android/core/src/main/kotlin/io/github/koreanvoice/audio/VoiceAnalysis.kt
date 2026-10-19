package io.github.koreanvoice.audio

import kotlin.math.PI
import kotlin.math.abs
import kotlin.math.cos
import kotlin.math.ln
import kotlin.math.log10
import kotlin.math.max
import kotlin.math.min
import kotlin.math.pow
import kotlin.math.sqrt

/**
 * Measurable voice characteristics: pitch contour (YIN), intonation range, loudness dynamics,
 * speaking rhythm and the long-term average spectrum (timbre + microphone/room colouration).
 * Used to score auto-learning candidates and to tag reference clips by expressiveness.
 */
object VoiceAnalysis {
    data class Profile(
        /** Median F0 in semitones relative to 100 Hz (NaN if no voiced frames). */
        val f0Median: Double,
        /** Intonation range: std of F0 in semitones. */
        val f0Std: Double,
        val voicedRatio: Double,
        /** Std of active-frame loudness (dB): dynamics / emphasis. */
        val energyStd: Double,
        /** Voiced onsets per second (rhythm / speaking-rate proxy). */
        val onsetRate: Double,
        /** Level-normalised long-term average spectrum, dB in [LTAS_BANDS] log bands. */
        val ltas: DoubleArray,
        /** Gap fill between syllables, 0..1 (reverberance / background proxy). */
        val tailRatio: Double,
    ) {
        /** Expressiveness ("arousal") estimate used to route emotions to matching reference clips. */
        val arousal: Double get() = 0.5 * f0Std + 0.25 * energyStd + 0.6 * onsetRate

        fun toList(): List<Double> = listOf(f0Median, f0Std, voicedRatio, energyStd, onsetRate, tailRatio) + ltas.toList()

        companion object {
            fun fromList(v: List<Double>) = Profile(v[0], v[1], v[2], v[3], v[4], v.subList(6, v.size).toDoubleArray(), v[5])
        }
    }

    const val LTAS_BANDS = 24

    fun analyze(x: FloatArray, sr: Int): Profile {
        val x16 = if (sr == 16000) x else Resampler.resample(x, sr, 16000)
        val f0 = yin(x16, 16000)
        val voiced = f0.filter { it > 0 }.map { 12 * ln(it / 100.0) / ln(2.0) }
        val med = if (voiced.isEmpty()) Double.NaN else voiced.sorted()[voiced.size / 2]
        // robust std (exclude octave errors beyond ±12 st from the median)
        val core = voiced.filter { abs(it - med) < 12 }
        val mean = core.average()
        val std = if (core.size < 2) 0.0 else sqrt(core.sumOf { (it - mean) * (it - mean) } / core.size)
        val db = AudioTools.frameDb(x16, 160)
        val peak = db.maxOrNull() ?: -120.0
        val act = db.filter { it > peak - 35 }
        val am = act.average()
        val eStd = if (act.size < 2) 0.0 else sqrt(act.sumOf { (it - am) * (it - am) } / act.size)
        var onsets = 0
        val vflags = f0.map { it > 0 }
        for (i in 3 until vflags.size) if (vflags[i] && !vflags[i - 1] && !vflags[i - 2]) onsets++
        val activeSec = act.size * 0.01
        return Profile(med, std, if (f0.isEmpty()) 0.0 else voiced.size.toDouble() / f0.size, eStd,
            if (activeSec > 0.3) onsets / activeSec else 0.0, ltas(x, sr), tailRatio(db))
    }

    /** YIN F0 tracker (10 ms hop, 40 ms window, 60–500 Hz); 0 = unvoiced. */
    fun yin(x: FloatArray, sr: Int, fmin: Double = 60.0, fmax: Double = 500.0, threshold: Double = 0.15): DoubleArray {
        val w = (0.04 * sr).toInt(); val hop = sr / 100
        val tauMin = (sr / fmax).toInt(); val tauMax = min(w - 1, (sr / fmin).toInt())
        val n = max(0, (x.size - w - tauMax) / hop)
        val out = DoubleArray(n)
        val d = DoubleArray(tauMax + 1)
        val db = AudioTools.frameDb(x, hop)
        val peak = db.maxOrNull() ?: -120.0
        for (f in 0 until n) {
            val s = f * hop
            if (f < db.size && db[f] < peak - 40) continue
            for (tau in 1..tauMax) {
                var acc = 0.0
                for (j in 0 until w) { val v = x[s + j] - x[s + j + tau]; acc += v * v }
                d[tau] = acc
            }
            var running = 0.0
            var tauSel = -1
            val cmnd = DoubleArray(tauMax + 1); cmnd[0] = 1.0
            for (tau in 1..tauMax) { running += d[tau]; cmnd[tau] = if (running > 0) d[tau] * tau / running else 1.0 }
            var tau = tauMin
            while (tau <= tauMax) {
                if (cmnd[tau] < threshold) {
                    while (tau + 1 <= tauMax && cmnd[tau + 1] < cmnd[tau]) tau++
                    tauSel = tau; break
                }
                tau++
            }
            if (tauSel <= 0) continue
            // parabolic interpolation
            val t = tauSel.toDouble()
            val refined = if (tauSel in 1 until tauMax) {
                val a = cmnd[tauSel - 1]; val b = cmnd[tauSel]; val c = cmnd[tauSel + 1]
                val den = a - 2 * b + c
                if (abs(den) > 1e-12) t + 0.5 * (a - c) / den else t
            } else t
            out[f] = sr / refined
        }
        return out
    }

    /** Level-normalised LTAS in log-spaced bands (100 Hz – 10 kHz, capped at Nyquist). */
    fun ltas(x: FloatArray, sr: Int): DoubleArray {
        val nfft = 1024; val hop = 512
        val win = DoubleArray(nfft) { 0.5 - 0.5 * cos(2 * PI * it / nfft) }
        val acc = DoubleArray(nfft / 2 + 1)
        val db = AudioTools.frameDb(x, hop)
        val peak = db.maxOrNull() ?: -120.0
        var frames = 0
        var s = 0
        val re = DoubleArray(nfft); val im = DoubleArray(nfft)
        while (s + nfft <= x.size) {
            val fi = s / hop
            if (fi < db.size && db[fi] > peak - 30) {
                for (i in 0 until nfft) { re[i] = x[s + i] * win[i]; im[i] = 0.0 }
                Fft.transform(re, im)
                for (k in acc.indices) acc[k] += re[k] * re[k] + im[k] * im[k]
                frames++
            }
            s += hop
        }
        val hi = min(10000.0, sr / 2.0 * 0.95); val lo = 100.0
        val bands = DoubleArray(LTAS_BANDS)
        for (b in 0 until LTAS_BANDS) {
            val f0 = lo * (hi / lo).pow(b.toDouble() / LTAS_BANDS); val f1 = lo * (hi / lo).pow((b + 1.0) / LTAS_BANDS)
            val k0 = (f0 * nfft / sr).toInt(); val k1 = max(k0 + 1, (f1 * nfft / sr).toInt())
            var e = 0.0; for (k in k0 until min(k1, acc.size)) e += acc[k]
            bands[b] = 10 * log10(e / max(1, frames) / (k1 - k0) + 1e-12)
        }
        val m = bands.average()
        for (b in bands.indices) bands[b] -= m
        return bands
    }

    /**
     * Gap fill: how far the quiet frames inside the speech span sit below the median loudness
     * (as 0..1, higher = the pauses between syllables are filled by reverberation / noise).
     */
    private fun tailRatio(db: DoubleArray): Double {
        if (db.size < 10) return 0.0
        val peak = db.max()
        val first = db.indexOfFirst { it > peak - 30 }; val last = db.indexOfLast { it > peak - 30 }
        if (last - first < 10) return 0.0
        val span = db.copyOfRange(first, last + 1).sortedArray()
        val p10 = span[span.size / 10]; val med = span[span.size / 2]
        return (1 + (p10 - med) / 40).coerceIn(0.0, 1.0)
    }

    /** RMS distance between two LTAS curves (dB). */
    fun ltasDistance(a: DoubleArray, b: DoubleArray): Double {
        var s = 0.0; for (i in a.indices) { val d = a[i] - b[i]; s += d * d }
        return sqrt(s / a.size)
    }

    /** Spectral tilt (dB per band index, regression slope) – used to choose a matching EQ. */
    fun tilt(l: DoubleArray): Double {
        val n = l.size; val mx = (n - 1) / 2.0
        var num = 0.0; var den = 0.0
        for (i in 0 until n) { num += (i - mx) * l[i]; den += (i - mx) * (i - mx) }
        return num / den
    }
}
