package io.github.koreanvoice.audio

import kotlin.math.PI
import kotlin.math.abs
import kotlin.math.ceil
import kotlin.math.floor
import kotlin.math.min
import kotlin.math.sin
import kotlin.math.sqrt

/**
 * Band-limited resampler (Kaiser-windowed sinc, interpolated from a finely sampled kernel table).
 * Works for any rate pair; anti-aliasing cutoff is 0.95 × the lower Nyquist frequency with a steep
 * transition (64 zero crossings): the OpenVoice speaker encoder is sensitive to the 8–11 kHz band, and
 * this setting reproduces librosa/soxr embeddings (cosine 0.998) where a 24-tap kernel gave 0.958.
 */
object Resampler {
    const val DEFAULT_ZERO_CROSSINGS = 64
    const val DEFAULT_CUTOFF = 0.95
    private const val TABLE_RES = 512
    private const val BETA = 8.6
    private val tables = HashMap<Int, FloatArray>()

    private fun table(zc: Int): FloatArray = synchronized(tables) {
        tables.getOrPut(zc) {
            val n = zc * TABLE_RES
            val t = FloatArray(n + 2)
            val i0b = besselI0(BETA)
            for (i in 0..n) {
                val x = i.toDouble() / TABLE_RES
                val sinc = if (x == 0.0) 1.0 else sin(PI * x) / (PI * x)
                val r = x / zc
                val w = if (r >= 1.0) 0.0 else besselI0(BETA * sqrt(1 - r * r)) / i0b
                t[i] = (sinc * w).toFloat()
            }
            t
        }
    }

    private fun besselI0(x: Double): Double {
        var sum = 1.0; var term = 1.0; var k = 1
        while (term > 1e-12 * sum) { term *= (x / (2 * k)) * (x / (2 * k)); sum += term; k++ }
        return sum
    }

    fun resample(
        input: FloatArray, fromRate: Int, toRate: Int,
        cutoff: Double = DEFAULT_CUTOFF, zeroCrossings: Int = DEFAULT_ZERO_CROSSINGS,
    ): FloatArray {
        if (fromRate == toRate || input.isEmpty()) return input.copyOf()
        val g = gcd(fromRate, toRate)
        if (toRate / g <= 1024) return polyphase(input, fromRate / g, toRate / g, cutoff, zeroCrossings)
        val tab = table(zeroCrossings)
        val limit = zeroCrossings * TABLE_RES
        val ratio = toRate.toDouble() / fromRate
        val outLen = floor(input.size * ratio).toLong().toInt()
        val out = FloatArray(outLen)
        val scale = min(1.0, ratio) * cutoff          // cutoff relative to input Nyquist
        val halfWidth = zeroCrossings / scale         // in input samples
        for (n in 0 until outLen) {
            val t = n / ratio
            val start = maxOf(0, ceil(t - halfWidth).toInt())
            val end = min(input.size - 1, floor(t + halfWidth).toInt())
            var acc = 0.0
            for (k in start..end) {
                val ax = abs(t - k) * scale * TABLE_RES
                val i = ax.toInt()
                if (i >= limit) continue
                val f = (ax - i).toFloat()
                acc += input[k] * (tab[i] + (tab[i + 1] - tab[i]) * f)
            }
            out[n] = (acc * scale).toFloat()
        }
        return out
    }

    private fun gcd(a: Int, b: Int): Int = if (b == 0) a else gcd(b, a % b)

    private class Bank(val taps: Array<FloatArray>, val first: IntArray)
    private val banks = HashMap<String, Bank>()

    /**
     * Same kernel as the generic path, but for a rational ratio L/M the filter has only L distinct phases,
     * so the taps are computed once (≈20× faster, e.g. 24 kHz ↔ 16 kHz).
     */
    private fun polyphase(input: FloatArray, m: Int, l: Int, cutoff: Double, zc: Int): FloatArray {
        val ratio = l.toDouble() / m
        val scale = min(1.0, ratio) * cutoff
        val halfWidth = zc / scale
        val bank = synchronized(banks) {
            banks.getOrPut("$m/$l/$cutoff/$zc") {
                val tab = table(zc); val limit = zc * TABLE_RES
                val taps = arrayOfNulls<FloatArray>(l); val first = IntArray(l)
                for (p in 0 until l) {
                    val frac = p.toDouble() / l            // t = base + frac
                    val j0 = ceil(frac - halfWidth).toInt(); val j1 = floor(frac + halfWidth).toInt()
                    first[p] = j0
                    taps[p] = FloatArray(j1 - j0 + 1) { j ->
                        val ax = abs(frac - (j0 + j)) * scale * TABLE_RES
                        val i = ax.toInt()
                        if (i >= limit) 0f else ((tab[i] + (tab[i + 1] - tab[i]) * (ax - i).toFloat()) * scale).toFloat()
                    }
                }
                @Suppress("UNCHECKED_CAST") Bank(taps as Array<FloatArray>, first)
            }
        }
        val outLen = floor(input.size * ratio).toLong().toInt()
        val out = FloatArray(outLen)
        for (n in 0 until outLen) {
            val num = n.toLong() * m
            val base = (num / l).toInt(); val p = (num % l).toInt()
            val w = bank.taps[p]; val k0 = base + bank.first[p]
            var acc = 0.0
            val lo = maxOf(0, -k0); val hi = minOf(w.size, input.size - k0)
            for (j in lo until hi) acc += input[k0 + j] * w[j]
            out[n] = acc.toFloat()
        }
        return out
    }

    /** Exact 2:1 decimation (44.1 kHz MeloTTS output → 22.05 kHz converter input). */
    fun decimate2(input: FloatArray): FloatArray = resample(input, 2, 1)
}
