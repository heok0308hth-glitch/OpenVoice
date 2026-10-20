package io.github.koreanvoice.expressive

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import io.github.koreanvoice.audio.Resampler
import java.io.Closeable
import java.io.DataInputStream
import java.io.DataOutputStream
import java.io.File
import java.io.RandomAccessFile
import java.nio.ByteOrder
import java.nio.FloatBuffer
import java.nio.LongBuffer
import java.nio.ShortBuffer
import java.nio.channels.FileChannel
import java.util.concurrent.atomic.AtomicInteger
import kotlin.math.exp
import kotlin.math.max
import kotlin.math.sqrt

/** Everything Chatterbox needs to speak in one voice (computed once from reference audio). */
class CbVoice(
    /** VoiceEncoder speaker embedding (256, L2-normalized) → T3 conditioning. */
    val ve: FloatArray,
    /** S3 speech tokens of the first ≤6 s (≤150) → T3 prompt. */
    val t3Tokens: LongArray,
    /** S3 tokens of the ≤10 s S3Gen reference. */
    val genTokens: LongArray,
    /** 80-bin log-mel of the S3Gen reference, [frames, 80] row-major. */
    val genMel: FloatArray,
    /** CAMPPlus x-vector (192). */
    val xvec: FloatArray,
) {
    val genMelFrames get() = genMel.size / 80

    fun write(out: DataOutputStream) {
        out.writeInt(MAGIC)
        for (a in listOf(ve, genMel, xvec)) { out.writeInt(a.size); a.forEach { out.writeFloat(it) } }
        for (a in listOf(t3Tokens, genTokens)) { out.writeInt(a.size); a.forEach { out.writeShort(it.toInt()) } }
    }

    companion object {
        private const val MAGIC = 0x43425631 // "CBV1"
        fun read(inp: DataInputStream): CbVoice {
            check(inp.readInt() == MAGIC) { "bad voice file" }
            fun f() = FloatArray(inp.readInt()) { inp.readFloat() }
            fun l() = LongArray(inp.readInt()) { inp.readShort().toLong() and 0xFFFF }
            val ve = f(); val mel = f(); val xv = f(); val t3 = l(); val gen = l()
            return CbVoice(ve, t3, gen, mel, xv)
        }
    }
}

/** Generation controls (defaults = ChatterboxMultilingualTTS.generate). */
data class CbParams(
    /** Emotion intensity ("exaggeration"), 0.25 – 2.0. */
    val exaggeration: Float = 0.5f,
    /** Classifier-free guidance weight; lower = slower, more relaxed pacing. */
    val cfgWeight: Float = 0.5f,
    val temperature: Float = 0.8f,
    val repetitionPenalty: Float = 2.0f,
    val minP: Float = 0.05f,
    val topP: Float = 1.0f,
    val maxNewTokens: Int = 1000,
    val seed: Long = 0L,
)

/**
 * On-device Chatterbox Multilingual (Resemble AI, MIT): T3 (0.5B Llama, 4-bit) with CFG + the
 * alignment-stream integrity checks of the original, and the S3Gen flow-matching decoder.
 */
class CbEngine(private val dir: File, threads: Int = 4, decoderFile: String = DECODER_FAST) : Closeable {
    private val env = OrtEnvironment.getEnvironment()
    private val opts = OrtSession.SessionOptions().apply {
        setIntraOpNumThreads(threads); setInterOpNumThreads(1)
        setOptimizationLevel(OrtSession.SessionOptions.OptLevel.ALL_OPT)
    }
    // ORT's graph optimizers fold the fp16→fp32 weight casts of the fp16 tokenizer into fp32 constants,
    // growing it from ≈250 MB to ≈600 MB resident. Unoptimized it gives identical tokens, ~15% slower.
    private val plainOpts = OrtSession.SessionOptions().apply {
        setIntraOpNumThreads(threads); setInterOpNumThreads(1)
        setOptimizationLevel(OrtSession.SessionOptions.OptLevel.NO_OPT)
    }
    val tokenizer = CbTokenizer(File(dir, "tokenizer.json").readText())

    // synthesis models are loaded on first use so that learning a voice never holds them (≈1 GB)
    private var lmS: OrtSession? = null; private var condS: OrtSession? = null; private var decS: OrtSession? = null
    private var tablesS: EmbeddingTables? = null
    private var decName = decoderFile
    private val lm: OrtSession @Synchronized get() = lmS ?: session("t3_lm.onnx").also { lmS = it }
    private val cond: OrtSession @Synchronized get() = condS ?: session("t3_cond.onnx").also { condS = it }
    private val dec: OrtSession @Synchronized get() = decS ?: session(decName).also { decS = it }
    private val tables: EmbeddingTables @Synchronized get() = tablesS
        ?: EmbeddingTables(File(dir, "t3_embed.bin"), File(dir, "t3_embed.json").readText()).also { tablesS = it }
    private val lmOut: List<String> by lazy { lm.outputNames.toList() }
    // calls running now on any thread; a model group is only released while its counter is zero
    private val synBusy = AtomicInteger(); private val regBusy = AtomicInteger()
    private inline fun <T> busy(c: AtomicInteger, block: () -> T): T {
        c.incrementAndGet()
        try { return block() } finally { c.decrementAndGet() }
    }

    // voice-registration / analysis models are loaded on demand (only needed when learning a voice)
    private var tokS: OrtSession? = null; private var campS: OrtSession? = null; private var melS: OrtSession? = null
    private var veMelS: OrtSession? = null; private var veLstmS: OrtSession? = null
    private var ecapaS: OrtSession? = null

    private fun session(name: String, o: OrtSession.SessionOptions = opts): OrtSession {
        val f = File(dir, name); require(f.exists()) { "모델 파일이 없습니다: $name" }
        return env.createSession(f.path, o)
    }

    @Synchronized
    fun useDecoder(name: String) {
        if (name == decName) return
        decS?.close(); decS = null; decName = name
    }

    /**
     * Release the synthesis models; they are reloaded on the next generate/decode. Skipped (returns
     * false) while a generate/decode is running on another thread, so a session is never closed mid-run.
     */
    @Synchronized
    fun releaseSynthesisModels(): Boolean {
        if (synBusy.get() > 0) return false
        closeSynthesis(); return true
    }

    private fun closeSynthesis() {
        listOf(lmS, condS, decS).forEach { runCatching { it?.close() } }; runCatching { tablesS?.close() }
        lmS = null; condS = null; decS = null; tablesS = null
    }

    fun hasDecoder(name: String) = File(dir, name).exists()

    // ------------------------------------------------------------------ voice registration
    @Synchronized private fun tok() = tokS ?: session("s3_tokenizer.onnx", plainOpts).also { tokS = it }
    @Synchronized private fun camp() = campS ?: session("campplus.onnx").also { campS = it }
    @Synchronized private fun mel() = melS ?: session("s3_mel.onnx").also { melS = it }
    @Synchronized private fun veMel() = veMelS ?: session("ve_mel.onnx").also { veMelS = it }
    @Synchronized private fun veLstm() = veLstmS ?: session("ve_lstm.onnx").also { veLstmS = it }
    @Synchronized private fun ecapaSession() = ecapaS ?: session("ecapa.onnx").also { ecapaS = it }

    /**
     * SpeechBrain ECAPA-TDNN speaker embedding (192, unit length) of 16 kHz audio. Synthesis is never
     * conditioned on it, so it serves as an independent judge of voice similarity for auto-learning.
     */
    fun ecapa(wav16: FloatArray): FloatArray = busy(regBusy) { normalize(run1(ecapaSession(), "wav", wav16)) }
    fun hasEcapa() = File(dir, "ecapa.onnx").exists()

    /**
     * Release the registration models (≈450 MB) once voices are built / learned; reloaded on demand.
     * Skipped (returns false) while one of them is running on another thread.
     */
    @Synchronized
    fun releaseRegistrationModels(): Boolean {
        if (regBusy.get() > 0) return false
        closeRegistration(); return true
    }

    private fun closeRegistration() {
        listOf(tokS, campS, melS, veMelS, veLstmS, ecapaS).forEach { runCatching { it?.close() } }
        tokS = null; campS = null; melS = null; veMelS = null; veLstmS = null; ecapaS = null
    }

    fun speechTokens(wav24: FloatArray): LongArray = busy(regBusy) {
        tensor(wav24, longArrayOf(1, wav24.size.toLong())).use { t ->
            tok().run(mapOf("audio_values" to t)).use { r -> (r[0] as OnnxTensor).longBuffer.toArr() }
        }
    }

    fun xvector(wav16: FloatArray): FloatArray = busy(regBusy) { run1(camp(), "wav", wav16) }

    fun s3Mel(wav24: FloatArray): FloatArray = busy(regBusy) { run1(mel(), "wav", wav24) }

    /** VoiceEncoder embedding: trim(top_db=20) → mel → 160-frame partials (50% overlap) → LSTM → mean → L2. */
    fun veEmbed(wav16: FloatArray): FloatArray = busy(regBusy) { veEmbedImpl(wav16) }

    private fun veEmbedImpl(wav16: FloatArray): FloatArray {
        val y = trimDb(wav16, 20.0)
        val mel = run1(veMel(), "wav", y)
        val n = mel.size / 40
        val step = 80; val win = 160
        var nw = if (n >= win) (n - win + step) / step else 0
        val rem = if (n >= win) (n - win + step) % step else n
        if (nw == 0 || (rem + (win - step)).toDouble() / win >= 0.8) nw++
        val target = win + step * (nw - 1)
        val padded = if (n < target) mel.copyOf(target * 40) else mel
        val parts = FloatArray(nw * win * 40)
        for (i in 0 until nw) System.arraycopy(padded, i * step * 40, parts, i * win * 40, win * 40)
        val e = tensor(parts, longArrayOf(nw.toLong(), win.toLong(), 40)).use { t ->
            veLstm().run(mapOf("partials" to t)).use { r -> (r[0] as OnnxTensor).floatBuffer.toArr() }
        }
        val m = FloatArray(256)
        for (i in 0 until nw) for (k in 0 until 256) m[k] += e[i * 256 + k] / nw
        return normalize(m)
    }

    /** Build a voice from mono audio at any sample rate (uses the first 6 s / 10 s like the original). */
    fun buildVoice(pcm: FloatArray, sampleRate: Int, veSource: FloatArray? = null): CbVoice {
        val wav24 = if (sampleRate == SR) pcm else Resampler.resample(pcm, sampleRate, SR)
        val ref10 = wav24.copyOf(minOf(wav24.size, 10 * SR))
        var gen = speechTokens(ref10)
        val mel = s3Mel(ref10)
        val frames = mel.size / 80
        if (frames != 2 * gen.size) gen = gen.copyOf(minOf(gen.size, frames / 2))
        val t3 = speechTokens(wav24.copyOf(minOf(wav24.size, 6 * SR))).let { it.copyOf(minOf(it.size, 150)) }
        val ref16 = Resampler.resample(ref10, SR, 16000)
        val ve = veEmbed(veSource ?: Resampler.resample(wav24, SR, 16000))
        return CbVoice(ve, t3, gen, mel, xvector(ref16))
    }

    // ------------------------------------------------------------------ synthesis
    class Result(val audio: FloatArray, val tokens: Int, val forcedStop: Boolean)

    fun synthesize(text: String, voice: CbVoice, p: CbParams, cancel: () -> Boolean = { false }): Result {
        val toks = generate(text, voice, p, cancel)
        return Result(decode(toks.first, voice), toks.first.size, toks.second)
    }

    /** T3: text → speech tokens. Returns tokens and whether the alignment checks forced the stop. */
    fun generate(text: String, voice: CbVoice, p: CbParams, cancel: () -> Boolean = { false }): Pair<LongArray, Boolean> {
        return busy(synBusy) { generateImpl(text, voice, p, cancel) }
    }

    private fun generateImpl(text: String, voice: CbVoice, p: CbParams, cancel: () -> Boolean): Pair<LongArray, Boolean> {
        val body = tokenizer.encodeKorean(CbTokenizer.puncNorm(text))
        val tt = IntArray(body.size + 2); tt[0] = START_TEXT; System.arraycopy(body, 0, tt, 1, body.size); tt[tt.size - 1] = STOP_TEXT
        val condEmb = tensor(voice.ve, longArrayOf(1, 256)).use { ve ->
            OnnxTensor.createTensor(env, LongBuffer.wrap(voice.t3Tokens), longArrayOf(1, voice.t3Tokens.size.toLong())).use { pt ->
                tensor(floatArrayOf(p.exaggeration), longArrayOf(1)).use { ex ->
                    cond.run(mapOf("speaker_emb" to ve, "prompt_tokens" to pt, "exaggeration" to ex)).use { r -> (r[0] as OnnxTensor).floatBuffer.toArr() }
                }
            }
        }
        val lc = condEmb.size / D
        val L = lc + tt.size + 1
        val emb = FloatArray(2 * L * D)
        for (b in 0 until 2) {
            val base = b * L * D
            System.arraycopy(condEmb, 0, emb, base, lc * D)
            for (j in tt.indices) {
                val o = base + (lc + j) * D
                if (b == 0) tables.add(EmbeddingTables.TEXT, tt[j], emb, o)          // CFG uncond keeps only the position embedding
                tables.add(EmbeddingTables.TEXT_POS, j, emb, o)
            }
            val o = base + (lc + tt.size) * D
            tables.add(EmbeddingTables.SPEECH, START_SPEECH, emb, o); tables.add(EmbeddingTables.SPEECH_POS, 0, emb, o)
        }
        val analyzer = AlignmentAnalyzer(lc, lc + tt.size)
        val sampler = Sampler(p)
        val generated = ArrayList<Int>(); generated.add(START_SPEECH)
        var forced = false
        var inputs = HashMap<String, OnnxTensor>()
        val zeros = ArrayList<OnnxTensor>()
        for (i in 0 until LAYERS) for (kv in arrayOf("k", "v")) {
            val z = OnnxTensor.createTensor(env, FloatBuffer.allocate(0), longArrayOf(2, HEADS.toLong(), 0, HEAD_DIM.toLong()))
            zeros.add(z); inputs["past_${kv}_$i"] = z
        }
        var embT = tensor(emb, longArrayOf(2, L.toLong(), D.toLong()))
        inputs["inputs_embeds"] = embT
        var prev: OrtSession.Result? = null
        try {
            for (step in 0 until p.maxNewTokens) {
                if (cancel()) throw InterruptedException("cancelled")
                val r = lm.run(inputs)
                embT.close(); prev?.close(); prev = r
                if (step == 0) zeros.forEach { it.close() }
                val logitsT = r[0] as OnnxTensor
                val lg = logitsT.floatBuffer.toArr()          // [2, V]
                val att = r[1] as OnnxTensor
                val shape = att.info.shape                     // [3, Tq, Ttot]
                val logits = FloatArray(VOCAB) { lg[it] + p.cfgWeight * (lg[it] - lg[VOCAB + it]) }
                val verdict = analyzer.step(att.floatBuffer.toArr(), shape[1].toInt(), shape[2].toInt(), step == 0, generated)
                if (verdict.suppressEos) logits[STOP_SPEECH] = -32768f
                if (verdict.forceEos) { java.util.Arrays.fill(logits, -32768f); logits[STOP_SPEECH] = 32768f; forced = true }
                val tok = sampler.sample(logits, generated)
                generated.add(tok)
                if (tok == STOP_SPEECH) break
                val e = FloatArray(2 * D)
                tables.add(EmbeddingTables.SPEECH, tok, e, 0); tables.add(EmbeddingTables.SPEECH_POS, step + 1, e, 0)
                System.arraycopy(e, 0, e, D, D)
                inputs = HashMap()
                for (k in 2 until lmOut.size) inputs[lmOut[k].replace("present", "past")] = r[k] as OnnxTensor
                embT = tensor(e, longArrayOf(2, 1, D.toLong())); inputs["inputs_embeds"] = embT
            }
        } finally {
            prev?.close()
        }
        val sp = generated.drop(1).filter { it < START_SPEECH }.map { it.toLong() }.toLongArray()
        return sp to forced
    }

    /** S3Gen: speech tokens → 24 kHz waveform. */
    fun decode(tokens: LongArray, voice: CbVoice): FloatArray {
        if (tokens.isEmpty()) return FloatArray(0)
        return busy(synBusy) { decodeImpl(tokens, voice) }
    }

    private fun decodeImpl(tokens: LongArray, voice: CbVoice): FloatArray {
        val all = voice.genTokens + tokens
        return OnnxTensor.createTensor(env, LongBuffer.wrap(all), longArrayOf(1, all.size.toLong())).use { st ->
            tensor(voice.xvec, longArrayOf(1, 192)).use { xv ->
                tensor(voice.genMel, longArrayOf(1, voice.genMelFrames.toLong(), 80)).use { mel ->
                    synchronized(this) { dec.run(mapOf("speech_tokens" to st, "speaker_embeddings" to xv, "speaker_features" to mel)) }
                        .use { r -> (r[0] as OnnxTensor).floatBuffer.toArr() }
                }
            }
        }
    }

    private fun run1(s: OrtSession, name: String, x: FloatArray): FloatArray =
        tensor(x, longArrayOf(1, x.size.toLong())).use { t -> s.run(mapOf(name to t)).use { r -> (r[0] as OnnxTensor).floatBuffer.toArr() } }

    private fun tensor(a: FloatArray, shape: LongArray) = OnnxTensor.createTensor(env, FloatBuffer.wrap(a), shape)

    override fun close() {
        synchronized(this) { closeRegistration(); closeSynthesis() }
    }

    /** Repetition penalty → temperature → min-p → top-p → multinomial (HF logits-processor semantics). */
    class Sampler(private val p: CbParams) {
        private val rng = java.util.SplittableRandom(p.seed)
        fun sample(logits: FloatArray, generated: List<Int>): Int {
            val seen = HashSet<Int>(generated)
            for (id in seen) { val s = logits[id]; logits[id] = if (s < 0) s * p.repetitionPenalty else s / p.repetitionPenalty }
            val t = p.temperature
            var mx = Float.NEGATIVE_INFINITY
            for (i in logits.indices) { logits[i] /= t; if (logits[i] > mx) mx = logits[i] }
            val probs = DoubleArray(logits.size); var sum = 0.0
            for (i in logits.indices) { probs[i] = exp((logits[i] - mx).toDouble()); sum += probs[i] }
            val top = 1.0 / sum                              // probability of the top token is exp(0)/sum
            var kept = 0.0
            for (i in probs.indices) { val pr = probs[i] / sum; if (pr < p.minP * top) probs[i] = 0.0 else kept += probs[i] }
            if (p.topP < 1f) {
                val order = probs.indices.sortedByDescending { probs[it] }
                var cum = 0.0; var cut = false
                for ((n, i) in order.withIndex()) {
                    if (cut) probs[i] = 0.0
                    cum += probs[i] / kept
                    if (cum > p.topP && n > 0) cut = true
                }
                kept = probs.sum()
            }
            var r = rng.nextDouble() * kept
            for (i in probs.indices) { r -= probs[i]; if (r <= 0 && probs[i] > 0) return i }
            return probs.indices.last { probs[it] > 0 }
        }
    }

    /** Port of chatterbox AlignmentStreamAnalyzer (heads (12,15), (13,11), (9,2) averaged). */
    class AlignmentAnalyzer(private val i0: Int, private val j0: Int) {
        class Verdict(val suppressEos: Boolean, val forceEos: Boolean)
        private val S = j0 - i0
        private val rows = ArrayList<FloatArray>()
        private var pos = 0; private var textPos = 0; private var started = false
        private var complete = false; private var completedAt = -1

        fun step(att: FloatArray, tq: Int, ttot: Int, first: Boolean, generated: List<Int>): Verdict {
            val r0 = if (first) j0 else 0
            for (q in r0 until tq) {
                val row = FloatArray(S)
                for (s in 0 until S) {
                    var a = 0f
                    for (h in 0 until 3) a += att[(h * tq + q) * ttot + i0 + s]
                    row[s] = if (s > pos) 0f else a / 3f
                }
                rows.add(row)
            }
            val last = rows.last()
            var cur = 0
            for (s in 1 until S) if (last[s] > last[cur]) cur = s
            if (cur - textPos in -3..6) textPos = cur
            val T = rows.size
            var a1 = 0f
            for (t in maxOf(0, T - 2) until T) for (s in maxOf(0, S - 2) until S) a1 = max(a1, rows[t][s])
            var a2 = 0f
            for (t in 0 until T) for (s in 0 until minOf(4, S)) a2 = max(a2, rows[t][s])
            val falseStart = !started && (a1 > 0.1f || a2 < 0.5f)
            started = !falseStart
            complete = complete || textPos >= S - 3
            if (complete && completedAt < 0) completedAt = T
            var longTail = false; var rep = false
            if (complete) {
                var mx = 0f
                for (s in maxOf(0, S - 3) until S) { var acc = 0f; for (t in completedAt until T) acc += rows[t][s]; mx = max(mx, acc) }
                longTail = mx >= 5f
                var acc = 0f
                for (t in completedAt until T) { var m = 0f; for (s in 0 until maxOf(0, S - 5)) m = max(m, rows[t][s]); acc += m }
                rep = acc > 5f
            }
            val tokRep = generated.size >= 3 && generated[generated.size - 1] == generated[generated.size - 2]
            pos++
            return Verdict(cur < S - 3 && S > 5, longTail || rep || tokRep)
        }
    }

    companion object {
        const val SR = 24000
        const val D = 1024
        const val LAYERS = 30; const val HEADS = 16; const val HEAD_DIM = 64
        const val VOCAB = 8194
        const val START_TEXT = 255; const val STOP_TEXT = 0
        const val START_SPEECH = 6561; const val STOP_SPEECH = 6562
        const val DECODER_FAST = "s3gen_turbo.onnx"
        const val DECODER_HQ = "s3gen_hq.onnx"
        val MODEL_FILES = listOf("t3_lm.onnx", "t3_cond.onnx", "t3_embed.bin", "t3_embed.json", "tokenizer.json",
            "s3_tokenizer.onnx", "campplus.onnx", "s3_mel.onnx", "ve_mel.onnx", "ve_lstm.onnx", "ecapa.onnx", DECODER_FAST)

        fun normalize(v: FloatArray): FloatArray { var n = 0.0; for (x in v) n += x * x; val s = (1 / sqrt(n)).toFloat(); return FloatArray(v.size) { v[it] * s } }

        /** librosa.effects.trim(top_db) with frame 2048 / hop 512. */
        fun trimDb(y: FloatArray, topDb: Double): FloatArray {
            val frame = 2048; val hop = 512
            val pad = frame / 2
            val n = 1 + y.size / hop
            val rms = DoubleArray(n)
            for (i in 0 until n) {
                var acc = 0.0
                for (k in 0 until frame) {
                    var idx = i * hop + k - pad
                    // librosa centers frames with constant (zero) padding for rms
                    val v = if (idx < 0 || idx >= y.size) 0.0 else y[idx].toDouble()
                    acc += v * v
                }
                rms[i] = sqrt(acc / frame)
            }
            val ref = rms.max().coerceAtLeast(1e-10)
            val db = rms.map { 20 * kotlin.math.log10(max(it, 1e-10)) - 20 * kotlin.math.log10(ref) }
            val first = db.indexOfFirst { it > -topDb }; val last = db.indexOfLast { it > -topDb }
            if (first < 0) return y
            val s = first * hop; val e = minOf(y.size, (last + 1) * hop)
            return y.copyOfRange(s, maxOf(s + 1, e))
        }

        private fun FloatBuffer.toArr(): FloatArray { val a = FloatArray(remaining()); get(a); return a }
        private fun LongBuffer.toArr(): LongArray { val a = LongArray(remaining()); get(a); return a }
    }
}

/** fp16 embedding tables memory-mapped from t3_embed.bin. */
class EmbeddingTables(file: File, header: String) : Closeable {
    private val raf = RandomAccessFile(file, "r")
    private val buf: ShortBuffer = raf.channel.map(FileChannel.MapMode.READ_ONLY, 0, file.length()).order(ByteOrder.LITTLE_ENDIAN).asShortBuffer()
    private val offsets = IntArray(4); private val rowsN = IntArray(4)

    init {
        val h = MiniJson.parse(header) as Map<*, *>
        listOf("text_emb", "text_pos_emb", "speech_emb", "speech_pos_emb").forEachIndexed { i, k ->
            val e = h[k] as Map<*, *>
            offsets[i] = ((e["offset"] as Number).toLong() / 2).toInt()
            rowsN[i] = ((e["shape"] as List<*>)[0] as Number).toInt()
        }
    }

    /** dst[o..o+1024) += table[row]. */
    fun add(table: Int, row: Int, dst: FloatArray, o: Int) {
        require(row in 0 until rowsN[table]) { "embedding row out of range" }
        val base = offsets[table] + row * CbEngine.D
        for (k in 0 until CbEngine.D) dst[o + k] += halfToFloat(buf.get(base + k).toInt())
    }

    override fun close() { raf.close() }

    companion object {
        const val TEXT = 0; const val TEXT_POS = 1; const val SPEECH = 2; const val SPEECH_POS = 3

        fun halfToFloat(h: Int): Float {
            val s = (h ushr 15) and 1; val e = (h ushr 10) and 0x1f; val m = h and 0x3ff
            val bits = when {
                e == 0 -> if (m == 0) s shl 31 else {
                    var mm = m; var ee = -14
                    while (mm and 0x400 == 0) { mm = mm shl 1; ee-- }
                    (s shl 31) or ((ee + 127) shl 23) or ((mm and 0x3ff) shl 13)
                }
                e == 31 -> (s shl 31) or (0xff shl 23) or (m shl 13)
                else -> (s shl 31) or ((e - 15 + 127) shl 23) or (m shl 13)
            }
            return java.lang.Float.intBitsToFloat(bits)
        }
    }
}
