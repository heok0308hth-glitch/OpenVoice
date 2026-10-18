package io.github.koreanvoice.engine

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import io.github.koreanvoice.audio.AudioTools
import io.github.koreanvoice.audio.Resampler
import io.github.koreanvoice.text.G2pKo
import io.github.koreanvoice.text.KoreanNormalizer
import io.github.koreanvoice.text.MeloFrontend
import java.io.Closeable
import java.io.File
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.FloatBuffer
import java.nio.LongBuffer

/** User-facing synthesis parameters. */
data class SynthParams(
    /** Speaking rate (MeloTTS length_scale = 1/speed). */
    val speed: Float = 1.0f,
    /** MeloTTS noise_scale: variation of timbre/energy within an utterance. */
    val noiseScale: Float = 0.6f,
    /** MeloTTS noise_scale_w: variation of phoneme durations (rhythm). */
    val noiseScaleW: Float = 0.8f,
    /** Mix of stochastic (1) vs deterministic (0) duration predictor. */
    val sdpRatio: Float = 0.2f,
    /** OpenVoice converter sampling temperature. */
    val tau: Float = 0.3f,
    /**
     * Timbre emphasis: target embedding = source + strength × (target − source).
     * 1.0 is plain OpenVoice; values slightly above 1 push the result further towards the sample.
     */
    val timbreStrength: Float = 1.2f,
    /**
     * Use the embedding of each generated utterance as the converter's source embedding instead of
     * the fixed MeloTTS-KR speaker average (kr.pth). Measured on 3 reference speakers × 8 sentences:
     * speaker similarity 0.815 → 0.844 (α=1.0) / 0.852 (α=1.25), CER unchanged (tools/similarity_eval.py).
     */
    val adaptiveSource: Boolean = true,
)

/** Thread split between the two pipeline stages (TTS stage / converter stage). */
data class ThreadConfig(val ttsThreads: Int, val convertThreads: Int) {
    companion object {
        fun auto(): ThreadConfig {
            val cores = Runtime.getRuntime().availableProcessors()
            val total = cores.coerceIn(2, 8)
            val tts = (total + 1) / 2
            return ThreadConfig(tts, maxOf(1, total - tts))
        }
    }
}

/**
 * On-device OpenVoice V2 (Korean) inference: MeloTTS-KR base speaker + BERT prosody features +
 * OpenVoice tone color converter, all as ONNX Runtime sessions.
 */
class VoiceEngine(
    modelDir: File,
    threads: ThreadConfig = ThreadConfig.auto(),
    /** Busy-wait in ORT worker threads; turn off when two pipeline stages share the CPU. */
    allowSpinning: Boolean = true,
) : Closeable {
    private val env: OrtEnvironment = OrtEnvironment.getEnvironment()
    private val bert: OrtSession
    private val melo: OrtSession
    private val extract: OrtSession
    private val convert: OrtSession
    val frontend = MeloFrontend()
    val normalizer = KoreanNormalizer(G2pKo())
    /** Embedding of the MeloTTS-KR base speaker (OpenVoiceV2 base_speakers/ses/kr.pth). */
    val baseSe: FloatArray = loadBaseSe()

    init {
        fun opts(n: Int) = OrtSession.SessionOptions().apply {
            setIntraOpNumThreads(n)
            setInterOpNumThreads(1)
            setOptimizationLevel(OrtSession.SessionOptions.OptLevel.ALL_OPT)
            setExecutionMode(OrtSession.SessionOptions.ExecutionMode.SEQUENTIAL)
            if (!allowSpinning) addConfigEntry("session.intra_op.allow_spinning", "0")
        }
        fun load(name: String, n: Int): OrtSession {
            val f = File(modelDir, name)
            require(f.exists()) { "모델 파일이 없습니다: ${f.path}" }
            return env.createSession(f.path, opts(n))
        }
        bert = load("bert_kr.onnx", threads.ttsThreads)
        melo = load("melo_kr.onnx", threads.ttsThreads)
        extract = load("tone_extract.onnx", threads.convertThreads)
        convert = load("tone_convert.onnx", threads.convertThreads)
    }

    /** Stage 1: normalized Korean text → base-speaker speech at 22.05 kHz. */
    fun baseSpeech(text: String, p: SynthParams): FloatArray {
        val fr = frontend.process(text)
        if (fr.phones.size <= 2) return FloatArray(0)
        val n = fr.ids.size
        // BERT (first 10 layers of kykim/bert-kor-base) → token features [T,768]
        val hidden: FloatArray
        val t = fr.bertIds.size
        OnnxTensor.createTensor(env, LongBuffer.wrap(fr.bertIds), longArrayOf(1, t.toLong())).use { ids ->
            bert.run(mapOf("input_ids" to ids)).use { r -> hidden = (r[0] as OnnxTensor).floatBuffer.toArray() }
        }
        // expand to phone level: ja_bert[1,768,N]
        val jaBert = FloatArray(768 * n)
        var col = 0
        for (tok in 0 until t) {
            repeat(fr.word2ph[tok]) {
                for (d in 0 until 768) jaBert[d * n + col] = hidden[tok * 768 + d]
                col++
            }
        }
        check(col == n) { "word2ph mismatch: $col != $n" }
        val shape = longArrayOf(1, n.toLong())
        val inputs = mapOf(
            "ids" to OnnxTensor.createTensor(env, LongBuffer.wrap(fr.ids), shape),
            "tones" to OnnxTensor.createTensor(env, LongBuffer.wrap(fr.tones), shape),
            "langs" to OnnxTensor.createTensor(env, LongBuffer.wrap(fr.langs), shape),
            "ja_bert" to OnnxTensor.createTensor(env, FloatBuffer.wrap(jaBert), longArrayOf(1, 768, n.toLong())),
            "noise_scale" to OnnxTensor.createTensor(env, p.noiseScale),
            "length_scale" to OnnxTensor.createTensor(env, 1f / p.speed.coerceIn(0.5f, 2.0f)),
            "noise_scale_w" to OnnxTensor.createTensor(env, p.noiseScaleW),
            "sdp_ratio" to OnnxTensor.createTensor(env, p.sdpRatio),
        )
        val audio44 = try {
            melo.run(inputs).use { r -> (r[0] as OnnxTensor).floatBuffer.toArray() }
        } finally { inputs.values.forEach { it.close() } }
        return Resampler.decimate2(audio44)
    }

    /** Stage 2: tone color conversion of 22.05 kHz speech towards [targetSe]. */
    fun convert(wav22: FloatArray, targetSe: FloatArray, p: SynthParams, sourceSeOverride: FloatArray? = null): FloatArray {
        if (wav22.size < 2048) return wav22
        val sourceSe = sourceSeOverride ?: if (p.adaptiveSource && wav22.size > SR / 2) embed(wav22) else baseSe
        val tgt = if (p.timbreStrength == 1f) targetSe else FloatArray(256) { sourceSe[it] + p.timbreStrength * (targetSe[it] - sourceSe[it]) }
        val inputs = mapOf(
            "wav" to OnnxTensor.createTensor(env, FloatBuffer.wrap(wav22), longArrayOf(1, wav22.size.toLong())),
            "src_se" to OnnxTensor.createTensor(env, FloatBuffer.wrap(sourceSe), longArrayOf(1, 256, 1)),
            "tgt_se" to OnnxTensor.createTensor(env, FloatBuffer.wrap(tgt), longArrayOf(1, 256, 1)),
            "tau" to OnnxTensor.createTensor(env, p.tau),
        )
        return try {
            convert.run(inputs).use { r -> (r[0] as OnnxTensor).floatBuffer.toArray() }
        } finally { inputs.values.forEach { it.close() } }
    }

    /** Speaker embedding of one 22.05 kHz clip. */
    fun embed(wav22: FloatArray): FloatArray =
        OnnxTensor.createTensor(env, FloatBuffer.wrap(wav22), longArrayOf(1, wav22.size.toLong())).use { w ->
            extract.run(mapOf("wav" to w)).use { r -> (r[0] as OnnxTensor).floatBuffer.toArray() }
        }

    /**
     * Voice embedding from a reference recording (any sample rate, mono): high-pass, VAD, split the
     * voiced audio into ≤10 s pieces and average their embeddings (as OpenVoice's se_extractor does).
     */
    fun extractVoice(pcm: FloatArray, sampleRate: Int): Pair<FloatArray, AudioTools.SampleReport> {
        var x = if (sampleRate == SR) pcm else Resampler.resample(pcm, sampleRate, SR)
        x = AudioTools.highPass(x, SR)
        val peak = x.maxOfOrNull { kotlin.math.abs(it) } ?: 0f
        if (peak > 1e-4f) { val g = 0.95f / peak; for (i in x.indices) x[i] *= g }
        val report = AudioTools.analyze(x, SR)
        val pieces = AudioTools.voicedPieces(x, SR)
        val acc = FloatArray(256)
        var total = 0.0
        for (piece in pieces) {
            val e = embed(piece)
            val w = piece.size.toDouble()
            for (i in 0 until 256) acc[i] += (e[i] * w).toFloat()
            total += w
        }
        for (i in 0 until 256) acc[i] = (acc[i] / total).toFloat()
        return acc to report
    }

    /** Full chunk: base speech (+ optional conversion). */
    fun synthesizeChunk(text: String, voiceSe: FloatArray?, p: SynthParams): FloatArray {
        val base = baseSpeech(text, p)
        return if (voiceSe == null || base.isEmpty()) base else convert(base, voiceSe, p)
    }

    override fun close() {
        listOf(bert, melo, extract, convert).forEach { runCatching { it.close() } }
    }

    companion object {
        const val SR = 22050
        val MODEL_FILES = listOf("bert_kr.onnx", "melo_kr.onnx", "tone_extract.onnx", "tone_convert.onnx")

        private fun loadBaseSe(): FloatArray {
            val b = (VoiceEngine::class.java.getResourceAsStream("/koreanvoice/kr_base_se.bin") ?: error("kr_base_se.bin missing")).use { it.readBytes() }
            val bb = ByteBuffer.wrap(b).order(ByteOrder.LITTLE_ENDIAN)
            return FloatArray(256) { bb.getFloat() }
        }

        private fun FloatBuffer.toArray(): FloatArray { val a = FloatArray(remaining()); get(a); return a }
    }
}
