package io.github.koreanvoice.audio

import kotlin.math.PI
import kotlin.math.abs
import kotlin.math.cos
import kotlin.math.exp
import kotlin.math.ln
import kotlin.math.max
import kotlin.math.min
import kotlin.math.pow
import kotlin.math.roundToInt
import kotlin.math.sin
import kotlin.math.sqrt
import kotlin.math.tanh

/** In-place iterative radix-2 complex FFT (re/im arrays, size power of two). */
object Fft {
    fun transform(re: DoubleArray, im: DoubleArray, inverse: Boolean = false) {
        val n = re.size
        var j = 0
        for (i in 1 until n) {
            var bit = n shr 1
            while (j and bit != 0) { j = j xor bit; bit = bit shr 1 }
            j = j xor bit
            if (i < j) { var t = re[i]; re[i] = re[j]; re[j] = t; t = im[i]; im[i] = im[j]; im[j] = t }
        }
        var len = 2
        while (len <= n) {
            val ang = 2 * PI / len * (if (inverse) 1 else -1)
            val wr = cos(ang); val wi = sin(ang)
            var i = 0
            while (i < n) {
                var cr = 1.0; var ci = 0.0
                for (k in 0 until len / 2) {
                    val a = i + k; val b = a + len / 2
                    val xr = re[b] * cr - im[b] * ci; val xi = re[b] * ci + im[b] * cr
                    re[b] = re[a] - xr; im[b] = im[a] - xi; re[a] += xr; im[a] += xi
                    val ncr = cr * wr - ci * wi; ci = cr * wi + ci * wr; cr = ncr
                }
                i += len
            }
            len = len shl 1
        }
        if (inverse) for (i in 0 until n) { re[i] /= n; im[i] /= n }
    }

    fun nextPow2(n: Int): Int { var p = 1; while (p < n) p = p shl 1; return p }
}

/** RBJ biquad (Direct Form I), stateful so it can stream. */
class Biquad private constructor(private val b0: Double, private val b1: Double, private val b2: Double, private val a1: Double, private val a2: Double) {
    private var x1 = 0.0; private var x2 = 0.0; private var y1 = 0.0; private var y2 = 0.0

    fun process(x: FloatArray): FloatArray {
        for (i in x.indices) {
            val v = x[i].toDouble()
            val y = b0 * v + b1 * x1 + b2 * x2 - a1 * y1 - a2 * y2
            x2 = x1; x1 = v; y2 = y1; y1 = y
            x[i] = y.toFloat()
        }
        return x
    }

    companion object {
        private fun make(b0: Double, b1: Double, b2: Double, a0: Double, a1: Double, a2: Double) = Biquad(b0 / a0, b1 / a0, b2 / a0, a1 / a0, a2 / a0)
        private fun w(f: Double, sr: Int) = 2 * PI * f / sr
        fun lowPass(f: Double, sr: Int, q: Double = 0.707): Biquad { val w0 = w(f, sr); val al = sin(w0) / (2 * q); val c = cos(w0)
            return make((1 - c) / 2, 1 - c, (1 - c) / 2, 1 + al, -2 * c, 1 - al) }
        fun highPass(f: Double, sr: Int, q: Double = 0.707): Biquad { val w0 = w(f, sr); val al = sin(w0) / (2 * q); val c = cos(w0)
            return make((1 + c) / 2, -(1 + c), (1 + c) / 2, 1 + al, -2 * c, 1 - al) }
        fun peak(f: Double, sr: Int, gainDb: Double, q: Double = 1.0): Biquad { val w0 = w(f, sr); val al = sin(w0) / (2 * q); val c = cos(w0); val a = 10.0.pow(gainDb / 40)
            return make(1 + al * a, -2 * c, 1 - al * a, 1 + al / a, -2 * c, 1 - al / a) }
        fun lowShelf(f: Double, sr: Int, gainDb: Double): Biquad { val w0 = w(f, sr); val c = cos(w0); val a = 10.0.pow(gainDb / 40); val al = sin(w0) / 2 * sqrt(2.0); val sa = 2 * sqrt(a) * al
            return make(a * ((a + 1) - (a - 1) * c + sa), 2 * a * ((a - 1) - (a + 1) * c), a * ((a + 1) - (a - 1) * c - sa), (a + 1) + (a - 1) * c + sa, -2 * ((a - 1) + (a + 1) * c), (a + 1) + (a - 1) * c - sa) }
        fun highShelf(f: Double, sr: Int, gainDb: Double): Biquad { val w0 = w(f, sr); val c = cos(w0); val a = 10.0.pow(gainDb / 40); val al = sin(w0) / 2 * sqrt(2.0); val sa = 2 * sqrt(a) * al
            return make(a * ((a + 1) + (a - 1) * c + sa), -2 * a * ((a - 1) + (a + 1) * c), a * ((a + 1) + (a - 1) * c - sa), (a + 1) - (a - 1) * c + sa, 2 * ((a - 1) - (a + 1) * c), (a + 1) - (a - 1) * c - sa) }
    }
}

/**
 * Uniformly partitioned overlap-save convolution (streaming): output has the same length as the input,
 * the reverb tail carries over into the next call (so effects span sentence boundaries naturally).
 */
class StreamingConvolver(ir: FloatArray, private val block: Int = 2048) {
    private val n = 2 * block
    private val parts: Int = max(1, (ir.size + block - 1) / block)
    private val hRe = Array(parts) { DoubleArray(n) }; private val hIm = Array(parts) { DoubleArray(n) }
    private val fdlRe = Array(parts) { DoubleArray(n) }; private val fdlIm = Array(parts) { DoubleArray(n) }
    private var head = 0
    private val inBuf = DoubleArray(n)             // [previous block | current block]
    private val pending = FloatArray(block); private var fill = 0
    private val lastOut = FloatArray(block)

    init {
        for (p in 0 until parts) {
            for (k in 0 until block) { val i = p * block + k; if (i < ir.size) hRe[p][k] = ir[i].toDouble() }
            Fft.transform(hRe[p], hIm[p])
        }
    }

    val latency get() = block

    private fun runBlock() {
        System.arraycopy(inBuf, block, inBuf, 0, block)
        for (k in 0 until block) inBuf[block + k] = pending[k].toDouble()
        head = (head - 1 + parts) % parts
        val xr = fdlRe[head]; val xi = fdlIm[head]
        System.arraycopy(inBuf, 0, xr, 0, n); java.util.Arrays.fill(xi, 0.0)
        Fft.transform(xr, xi)
        val accR = DoubleArray(n); val accI = DoubleArray(n)
        for (p in 0 until parts) {
            val s = (head + p) % parts
            val ar = fdlRe[s]; val ai = fdlIm[s]; val br = hRe[p]; val bi = hIm[p]
            for (k in 0 until n) { accR[k] += ar[k] * br[k] - ai[k] * bi[k]; accI[k] += ar[k] * bi[k] + ai[k] * br[k] }
        }
        Fft.transform(accR, accI, inverse = true)
        for (k in 0 until block) lastOut[k] = accR[block + k].toFloat()
    }

    /** Push samples, get the same number of (delayed by [latency]) output samples. */
    fun process(x: FloatArray): FloatArray {
        val out = FloatArray(x.size)
        for (i in x.indices) {
            out[i] = lastOut[fill]                 // fixed latency of one block
            pending[fill++] = x[i]
            if (fill == block) { runBlock(); fill = 0 }
        }
        return out
    }
}

/** Effect presets and processors used by the professional controls and the auto-learning search. */
object Dsp {
    enum class Room(val label: String, val rt60: Double, val preDelayMs: Double, val brightness: Double, val defaultWet: Double) {
        NONE("없음", 0.0, 0.0, 0.0, 0.0),
        ROOM("방", 0.45, 5.0, 0.7, 0.16),
        STUDIO("스튜디오", 0.25, 3.0, 0.8, 0.10),
        BATH("욕실", 0.9, 4.0, 1.0, 0.24),
        STAGE("무대", 1.3, 18.0, 0.75, 0.22),
        HALL("홀", 2.1, 25.0, 0.65, 0.26),
        CHURCH("성당", 3.6, 35.0, 0.55, 0.30),
        CAVE("동굴", 2.8, 60.0, 0.45, 0.34);
        companion object { fun of(s: String) = entries.firstOrNull { it.label == s || it.name.equals(s, true) } }
    }

    enum class State(val label: String) {
        NORMAL("기본"), PHONE("전화"), RADIO("라디오"), MEGAPHONE("확성기"), DISTANT("멀리"), CLOSE("가까이"),
        WHISPER("속삭임"), UNDERWATER("물속"), ROBOT("로봇");
        companion object { fun of(s: String) = entries.firstOrNull { it.label == s || it.name.equals(s, true) } }
    }

    enum class Tone(val label: String) {
        NORMAL("기본"), BRIGHT("밝게"), WARM("따뜻하게"), CLEAR("선명하게"), DARK("어둡게");
        companion object { fun of(s: String) = entries.firstOrNull { it.label == s || it.name.equals(s, true) } }
    }

    /** Synthetic room impulse response: early reflections + exponentially decaying two-band noise tail. */
    fun impulseResponse(room: Room, sr: Int, seed: Long = 7): FloatArray {
        if (room == Room.NONE) return floatArrayOf(1f)
        val rnd = java.util.Random(seed)
        val len = (sr * min(4.0, room.rt60 * 1.1 + 0.05)).toInt()
        val ir = FloatArray(len)
        val pre = (room.preDelayMs * sr / 1000).toInt()
        // early reflections (image-source like, decreasing gain)
        val erCount = 10
        for (k in 0 until erCount) {
            val t = pre + ((k + 1) * (3 + rnd.nextDouble() * 9) * sr / 1000).toInt()
            if (t < len) ir[t] += ((if (rnd.nextBoolean()) 1 else -1) * 0.55 * exp(-k * 0.25)).toFloat()
        }
        val lowDecay = 6.91 / (room.rt60 * sr)            // 60 dB over RT60
        val highDecay = 6.91 / (room.rt60 * (0.35 + 0.5 * room.brightness) * sr)
        val lp = Biquad.lowPass(1800.0, sr)
        val noise = FloatArray(len) { (rnd.nextGaussian() * 0.5).toFloat() }
        val low = lp.process(noise.copyOf())
        val onset = pre + (0.012 * sr).toInt()
        for (i in onset until len) {
            val t = (i - onset).toDouble()
            val ramp = min(1.0, t / (0.02 * sr))
            val l = low[i] * exp(-lowDecay * t); val h = (noise[i] - low[i]) * exp(-highDecay * t)
            ir[i] += ((l + h) * ramp * 0.35).toFloat()
        }
        var e = 0.0; for (v in ir) e += v * v
        val g = (1 / sqrt(e)).toFloat()
        for (i in ir.indices) ir[i] *= g
        return ir
    }

    /** Reverb on a whole buffer (tail appended, [tailSec] at most). */
    fun reverb(x: FloatArray, sr: Int, room: Room, wet: Double = room.defaultWet, tailSec: Double = 1.5): FloatArray {
        if (room == Room.NONE || wet <= 0) return x
        val ir = impulseResponse(room, sr)
        val conv = StreamingConvolver(ir)
        val tail = (min(tailSec, room.rt60) * sr).toInt()
        val input = x.copyOf(x.size + tail + conv.latency)
        val w = conv.process(input)
        val out = FloatArray(x.size + tail)
        val dry = (1 - wet * 0.5).toFloat(); val wg = wet.toFloat()
        for (i in out.indices) out[i] = (if (i < x.size) x[i] * dry else 0f) + w[i + conv.latency] * wg
        return out
    }

    fun tone(x: FloatArray, sr: Int, t: Tone): FloatArray = when (t) {
        Tone.NORMAL -> x
        Tone.BRIGHT -> Biquad.highShelf(4000.0, sr, 4.0).process(Biquad.lowShelf(180.0, sr, -1.5).process(x.copyOf()))
        Tone.WARM -> Biquad.highShelf(6000.0, sr, -2.5).process(Biquad.lowShelf(250.0, sr, 3.0).process(x.copyOf()))
        Tone.CLEAR -> Biquad.peak(3000.0, sr, 3.0, 0.8).process(Biquad.highPass(90.0, sr).process(x.copyOf()))
        Tone.DARK -> Biquad.highShelf(3000.0, sr, -5.0).process(x.copyOf())
    }

    /** Spectral tilt EQ (dB/octave around 1 kHz, ±), used by the auto-learning channel match. */
    fun tilt(x: FloatArray, sr: Int, db: Double): FloatArray {
        if (abs(db) < 0.05) return x
        return Biquad.highShelf(1500.0, sr, db).process(Biquad.lowShelf(500.0, sr, -db).process(x.copyOf()))
    }

    fun state(x: FloatArray, sr: Int, s: State): FloatArray = when (s) {
        State.NORMAL -> x
        State.PHONE -> {
            var y = Resampler.resample(x, sr, 8000)
            y = Biquad.highPass(320.0, 8000).process(Biquad.highPass(320.0, 8000).process(y))
            y = Biquad.lowPass(3300.0, 8000).process(y)
            y = saturate(y, 2.0)
            Resampler.resample(y, 8000, sr)
        }
        State.RADIO -> {
            var y = Biquad.highPass(250.0, sr).process(Biquad.highPass(250.0, sr).process(x.copyOf()))
            y = Biquad.lowPass(5000.0, sr).process(Biquad.lowPass(5000.0, sr).process(y))
            y = Biquad.peak(1800.0, sr, 3.0).process(y)
            y = compress(saturate(y, 1.6), sr, -24.0, 4.0)
            val rnd = java.util.Random(3); for (i in y.indices) y[i] += (rnd.nextGaussian() * 0.002).toFloat()
            y
        }
        State.MEGAPHONE -> {
            var y = Biquad.highPass(600.0, sr).process(Biquad.highPass(600.0, sr).process(x.copyOf()))
            y = Biquad.lowPass(3800.0, sr).process(Biquad.lowPass(3800.0, sr).process(y))
            y = Biquad.peak(2000.0, sr, 8.0, 1.5).process(y)
            y = saturate(y, 4.0)
            reverb(y, sr, Room.STAGE, 0.15, 0.6).copyOf(x.size)
        }
        State.DISTANT -> {
            var y = Biquad.lowPass(2800.0, sr).process(x.copyOf())
            for (i in y.indices) y[i] *= 0.45f
            y = reverb(y, sr, Room.HALL, 0.55, 1.2)
            y
        }
        State.CLOSE -> Biquad.peak(5000.0, sr, 1.5).process(Biquad.lowShelf(200.0, sr, 4.5).process(x.copyOf()))
        State.WHISPER -> Biquad.highPass(350.0, sr).process(whisper(x, sr))
        State.UNDERWATER -> {
            var y = Biquad.lowPass(700.0, sr).process(Biquad.lowPass(700.0, sr).process(x.copyOf()))
            for (i in y.indices) y[i] *= (1 + 0.15 * sin(2 * PI * 5.0 * i / sr)).toFloat()
            reverb(y, sr, Room.CAVE, 0.3, 0.8)
        }
        State.ROBOT -> robot(x, sr)
    }

    fun saturate(x: FloatArray, drive: Double): FloatArray {
        val n = tanh(drive)
        for (i in x.indices) x[i] = (tanh(x[i] * drive) / n).toFloat()
        return x
    }

    /** Feed-forward RMS compressor (threshold dBFS, ratio), with make-up gain. */
    fun compress(x: FloatArray, sr: Int, thresholdDb: Double, ratio: Double, attackMs: Double = 8.0, releaseMs: Double = 120.0): FloatArray {
        val a = exp(-1.0 / (attackMs * sr / 1000)); val r = exp(-1.0 / (releaseMs * sr / 1000))
        var env = 0.0
        val makeup = 10.0.pow(-thresholdDb * (1 - 1 / ratio) / 40)
        for (i in x.indices) {
            val v = x[i].toDouble() * x[i]
            env = if (v > env) a * env + (1 - a) * v else r * env + (1 - r) * v
            val db = 10 * kotlin.math.log10(env + 1e-12)
            val gainDb = if (db > thresholdDb) (thresholdDb - db) * (1 - 1 / ratio) else 0.0
            x[i] = (x[i] * 10.0.pow(gainDb / 20) * makeup).toFloat()
        }
        return x
    }

    /** LPC whisperisation: keep the vocal-tract envelope, replace the (voiced) excitation by noise. */
    fun whisper(x: FloatArray, sr: Int, order: Int = 22): FloatArray {
        val frame = (0.032 * sr).toInt(); val hop = frame / 2
        val win = DoubleArray(frame) { 0.5 - 0.5 * cos(2 * PI * it / frame) }
        val out = DoubleArray(x.size + frame)
        val rnd = java.util.Random(11)
        val pre = FloatArray(x.size); var p = 0f
        for (i in x.indices) { pre[i] = x[i] - 0.9f * p; p = x[i] }
        var s = 0
        while (s + frame <= pre.size) {
            val f = DoubleArray(frame) { pre[s + it] * win[it] }
            val a = lpc(f, order)
            var resE = 0.0
            for (n in 0 until frame) { var e = f[n]; for (k in 1..order) if (n - k >= 0) e += a[k] * f[n - k]; resE += e * e }
            val g = sqrt(resE / frame)
            val y = DoubleArray(frame)
            for (n in 0 until frame) {
                var v = rnd.nextGaussian() * g
                for (k in 1..order) if (n - k >= 0) v -= a[k] * y[n - k]
                y[n] = v
            }
            for (n in 0 until frame) out[s + n] += y[n] * win[n]
            s += hop
        }
        val res = FloatArray(x.size); var q = 0.0
        for (i in x.indices) { q = out[i] + 0.9 * q; res[i] = q.toFloat() }
        // match loudness to the input
        val gi = AudioTools.rms(x); val go = AudioTools.rms(res)
        if (go > 1e-9) { val g = (gi / go * 0.8).toFloat(); for (i in res.indices) res[i] *= g }
        return res
    }

    /** Levinson–Durbin on the autocorrelation; returns a[0..order] with a[0] = 1 (prediction e[n] = Σ a_k x[n-k]). */
    fun lpc(f: DoubleArray, order: Int): DoubleArray {
        val r = DoubleArray(order + 1) { k -> var acc = 0.0; for (n in k until f.size) acc += f[n] * f[n - k]; acc }
        r[0] *= 1.0 + 1e-9; r[0] += 1e-12
        val a = DoubleArray(order + 1); a[0] = 1.0
        var err = r[0]
        for (i in 1..order) {
            var acc = r[i]; for (j in 1 until i) acc += a[j] * r[i - j]
            val k = -acc / err
            val tmp = a.copyOf()
            for (j in 1 until i) a[j] = tmp[j] + k * tmp[i - j]
            a[i] = k
            err *= (1 - k * k)
            if (err <= 0) break
        }
        return a
    }

    /** Monotone "robot": LPC envelope driven by a fixed 110 Hz pulse train. */
    fun robot(x: FloatArray, sr: Int, order: Int = 22): FloatArray {
        val frame = (0.032 * sr).toInt(); val hop = frame / 2
        val win = DoubleArray(frame) { 0.5 - 0.5 * cos(2 * PI * it / frame) }
        val out = DoubleArray(x.size + frame)
        val period = sr / 110
        var s = 0
        while (s + frame <= x.size) {
            val f = DoubleArray(frame) { x[s + it] * win[it] }
            val a = lpc(f, order)
            var resE = 0.0
            for (n in 0 until frame) { var e = f[n]; for (k in 1..order) if (n - k >= 0) e += a[k] * f[n - k]; resE += e * e }
            val g = sqrt(resE / frame * period)
            val y = DoubleArray(frame)
            for (n in 0 until frame) {
                var v = if ((s + n) % period == 0) g else 0.0
                for (k in 1..order) if (n - k >= 0) v -= a[k] * y[n - k]
                y[n] = v
            }
            for (n in 0 until frame) out[s + n] += y[n] * win[n]
            s += hop
        }
        val res = FloatArray(x.size) { out[it].toFloat() }
        val gi = AudioTools.rms(x); val go = AudioTools.rms(res)
        if (go > 1e-9) { val g = (gi / go).toFloat(); for (i in res.indices) res[i] *= g }
        return res
    }

    /**
     * WSOLA time-stretch. [rate] > 1 = faster/shorter. Pitch is preserved.
     */
    fun timeStretch(x: FloatArray, sr: Int, rate: Double): FloatArray {
        if (abs(rate - 1.0) < 0.005 || x.size < sr / 10) return x
        val n = (0.030 * sr).toInt().let { it + (it and 1) }
        val hs = n / 2
        val ha = hs * rate
        val tol = (0.008 * sr).toInt()
        val win = FloatArray(n) { (0.5 - 0.5 * cos(2 * PI * it / n)).toFloat() }
        val outLen = (x.size / rate).toInt() + n
        val out = FloatArray(outLen)
        val norm = FloatArray(outLen)
        fun at(i: Int) = if (i in x.indices) x[i] else 0f
        var prevPos = 0
        var k = 0
        while (true) {
            val ideal = (k * ha).roundToInt()
            if (ideal >= x.size) break
            var pos = ideal
            if (k > 0) {
                val nat = prevPos + hs     // natural continuation of the previous frame
                var best = Double.NEGATIVE_INFINITY
                var d = -tol
                while (d <= tol) {
                    val c = ideal + d
                    var acc = 0.0
                    var m = 0
                    while (m < n) { acc += at(c + m) * at(nat + m); m += 2 }
                    if (acc > best) { best = acc; pos = c }
                    d++
                }
            }
            val o = k * hs
            if (o + n > outLen) break
            for (m in 0 until n) { out[o + m] += at(pos + m) * win[m]; norm[o + m] += win[m] }
            prevPos = pos; k++
        }
        val len = min(outLen, (x.size / rate).toInt())
        return FloatArray(len) { if (norm[it] > 1e-3f) out[it] / norm[it] else out[it] }
    }

    /** Pitch shift by semitones (WSOLA stretch + resampling), duration preserved. */
    fun pitchShift(x: FloatArray, sr: Int, semitones: Double): FloatArray {
        if (abs(semitones) < 0.05) return x
        val f = 2.0.pow(semitones / 12)
        val stretched = timeStretch(x, sr, 1 / f)
        return Resampler.resample(stretched, (sr * f).roundToInt(), sr)
    }

    /** Tempo and pitch together (one resampling pass). */
    fun tempoPitch(x: FloatArray, sr: Int, tempo: Double, semitones: Double): FloatArray {
        if (abs(semitones) < 0.05) return timeStretch(x, sr, tempo)
        val f = 2.0.pow(semitones / 12)
        val stretched = timeStretch(x, sr, tempo / f)
        return Resampler.resample(stretched, (sr * f).roundToInt(), sr)
    }

    /**
     * Streaming background mixer: loops [bed] under the voice at [levelDb] (relative to the voice
     * level), ducking it by [duckDb] while speech is present.
     */
    class BackgroundMixer(private val bed: FloatArray, private val sr: Int, levelDb: Double = -20.0, private val duckDb: Double = 9.0) {
        private var pos = 0
        private var env = 0.0
        private val gain = 10.0.pow(levelDb / 20) * 0.1 / max(1e-6, AudioTools.rms(bed))
        private val att = exp(-1.0 / (0.03 * sr)); private val rel = exp(-1.0 / (0.45 * sr))
        private var duck = 1.0
        fun mix(voice: FloatArray): FloatArray {
            if (bed.isEmpty()) return voice
            val out = FloatArray(voice.size)
            val duckLin = 10.0.pow(-duckDb / 20)
            for (i in voice.indices) {
                val a = abs(voice[i]).toDouble()
                env = if (a > env) att * env + (1 - att) * a else rel * env + (1 - rel) * a
                val target = if (env > 0.01) duckLin else 1.0
                duck += (target - duck) * (if (target < duck) 0.002 else 0.0003)
                out[i] = (voice[i] + bed[pos] * gain * duck).toFloat()
                pos++; if (pos >= bed.size) pos = 0
            }
            return out
        }
    }

    /** Noise beds that need no assets. */
    fun noiseBed(kind: String, sr: Int, seconds: Double = 8.0): FloatArray {
        val rnd = java.util.Random(5)
        val n = (sr * seconds).toInt()
        val w = FloatArray(n) { rnd.nextGaussian().toFloat() }
        return when (kind) {
            "빗소리" -> { var y = Biquad.highPass(400.0, sr).process(w); y = Biquad.lowPass(6000.0, sr).process(y)
                for (i in y.indices) if (rnd.nextDouble() < 0.0008) y[i] += (rnd.nextGaussian() * 6).toFloat(); y }
            "바람" -> { var y = Biquad.lowPass(500.0, sr).process(w); for (i in y.indices) y[i] *= (0.6 + 0.4 * sin(2 * PI * 0.15 * i / sr)).toFloat(); y }
            "백색소음" -> w
            else -> { // brown-ish "실내 웅웅"
                var acc = 0.0; FloatArray(n) { acc = 0.995 * acc + w[it] * 0.1; acc.toFloat() }.let { Biquad.highPass(40.0, sr).process(it) }
            }
        }
    }
}
