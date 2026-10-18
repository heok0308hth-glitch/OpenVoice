package io.github.koreanvoice.engine

import io.github.koreanvoice.audio.AudioTools
import io.github.koreanvoice.audio.Mp3Encoder
import io.github.koreanvoice.text.TextChunker
import java.io.File
import java.io.OutputStream
import java.security.MessageDigest
import java.util.concurrent.ArrayBlockingQueue
import java.util.concurrent.TimeUnit
import java.util.concurrent.atomic.AtomicReference

/**
 * Long-text synthesis (up to 100,000 characters) with:
 *  - Korean normalization + chunking,
 *  - a two-stage pipeline: the TTS stage (BERT + MeloTTS) of chunk i+1 runs while the converter
 *    stage handles chunk i, each with its own ONNX Runtime threads,
 *  - per-chunk results cached on disk, so a cancelled / killed job resumes where it stopped and
 *    memory stays bounded regardless of text length,
 *  - final assembly: silence trimming, loudness levelling, punctuation-aware pauses, limiter and
 *    streaming MP3 encoding.
 */
class SynthesisJob(
    private val engine: VoiceEngine,
    cacheRoot: File,
    private val text: String,
    private val voiceSe: FloatArray?,
    private val params: SynthParams,
    private val chunker: TextChunker = TextChunker(),
    /** Run TTS and conversion stages concurrently (pays off on phones with many cores). */
    private val pipelined: Boolean = true,
) {
    data class Progress(val done: Int, val total: Int, val elapsedMs: Long, val etaMs: Long, val currentText: String)

    class CancelledException : RuntimeException("취소됨")

    val chunks: List<TextChunker.Chunk> by lazy { chunker.split(engine.normalizer.normalize(text)) }
    val jobDir: File = File(cacheRoot, "job_" + key()).apply { mkdirs() }

    private fun key(): String {
        val md = MessageDigest.getInstance("SHA-256")
        md.update(text.toByteArray())
        md.update(params.toString().toByteArray())
        voiceSe?.forEach { md.update(java.lang.Float.floatToIntBits(it).toString().toByteArray()) }
        md.update("v1".toByteArray())
        return md.digest().take(10).joinToString("") { "%02x".format(it) }
    }

    private fun chunkFile(i: Int) = File(jobDir, "chunk_%05d.pcm".format(i))

    fun completedChunks(): Int = chunks.indices.count { chunkFile(it).exists() }

    /** Synthesize all missing chunks. Blocks; call from a background thread. */
    fun render(isCancelled: () -> Boolean = { false }, onProgress: (Progress) -> Unit = {}) {
        val total = chunks.size
        val todo = chunks.indices.filter { !chunkFile(it).exists() }
        val start = System.currentTimeMillis()
        val doneBefore = total - todo.size
        val totalChars = chunks.sumOf { it.text.length }.coerceAtLeast(1)
        var charsDone = 0
        val charsBefore = chunks.indices.filter { chunkFile(it).exists() }.sumOf { chunks[it].text.length }
        onProgress(Progress(doneBefore, total, 0, -1, ""))
        if (todo.isEmpty()) return
        if (!pipelined) {
            var done = doneBefore
            for (i in todo) {
                if (isCancelled()) throw CancelledException()
                val base = retry { engine.baseSpeech(chunks[i].text, params) }
                val out = if (voiceSe == null || base.isEmpty()) base else retry { engine.convert(base, voiceSe, params) }
                AudioTools.writePcm16(chunkFile(i), out)
                done++; charsDone += chunks[i].text.length
                val elapsed = System.currentTimeMillis() - start
                onProgress(Progress(done, total, elapsed, (elapsed.toDouble() / charsDone * (totalChars - charsBefore - charsDone)).toLong(), chunks[i].text))
            }
            return
        }

        val queue = ArrayBlockingQueue<Pair<Int, FloatArray>>(2)
        val error = AtomicReference<Throwable?>(null)
        val poison = -1 to FloatArray(0)
        val producer = Thread({
            try {
                for (i in todo) {
                    if (isCancelled() || error.get() != null) break
                    val base = retry { engine.baseSpeech(chunks[i].text, params) }
                    while (!queue.offer(i to base, 200, TimeUnit.MILLISECONDS)) {
                        if (isCancelled() || error.get() != null) return@Thread
                    }
                }
            } catch (t: Throwable) {
                error.compareAndSet(null, t)
            } finally {
                while (!queue.offer(poison, 200, TimeUnit.MILLISECONDS)) { if (error.get() != null) break }
            }
        }, "tts-stage")
        producer.start()
        var done = doneBefore
        try {
            while (true) {
                val item = queue.poll(200, TimeUnit.MILLISECONDS)
                if (item == null) {
                    if (error.get() != null || (!producer.isAlive && queue.isEmpty())) break
                    if (isCancelled()) break
                    continue
                }
                if (item.first < 0) break
                val (i, base) = item
                val out = if (voiceSe == null || base.isEmpty()) base else retry { engine.convert(base, voiceSe, params) }
                AudioTools.writePcm16(chunkFile(i), out)
                done++
                charsDone += chunks[i].text.length
                val elapsed = System.currentTimeMillis() - start
                val eta = if (charsDone == 0) -1 else (elapsed.toDouble() / charsDone * (totalChars - charsBefore - charsDone)).toLong()
                onProgress(Progress(done, total, elapsed, eta, chunks[i].text))
                if (isCancelled()) break
            }
        } finally {
            producer.join(30_000)
        }
        error.get()?.let { throw it }
        if (isCancelled()) throw CancelledException()
    }

    private fun <T> retry(block: () -> T): T = try { block() } catch (e: OutOfMemoryError) { System.gc(); block() } catch (e: Exception) { block() }

    /** Assemble cached chunks into an MP3 stream (all chunks must be rendered). */
    fun assembleMp3(out: OutputStream, bitrateKbps: Int = 64, id3: Mp3Encoder.Id3Info? = null, pauseScale: Float = 1f) {
        Mp3Encoder(out, VoiceEngine.SR, bitrateKbps, id3 = id3).use { enc ->
            assemble(pauseScale) { enc.encode(it) }
        }
    }

    fun assembleWav(file: File, pauseScale: Float = 1f) {
        val parts = ArrayList<FloatArray>()
        assemble(pauseScale) { parts.add(it) }
        val all = FloatArray(parts.sumOf { it.size })
        var p = 0
        for (a in parts) { System.arraycopy(a, 0, all, p, a.size); p += a.size }
        AudioTools.writeWav(file, all, VoiceEngine.SR)
    }

    private fun assemble(pauseScale: Float, sink: (FloatArray) -> Unit) {
        val sr = VoiceEngine.SR
        val speed = params.speed.coerceIn(0.5f, 2f)
        sink(AudioTools.silence(sr, 250))
        var prevGain = -1f
        for ((i, c) in chunks.withIndex()) {
            val f = chunkFile(i)
            check(f.exists()) { "chunk $i not rendered" }
            var x = AudioTools.readPcm16(f)
            x = AudioTools.trimSilence(x, sr)
            if (x.isNotEmpty()) {
                var g = AudioTools.normalizeGain(x, sr)
                // smooth level changes between neighbouring sentences
                if (prevGain > 0) g = (0.6f * g + 0.4f * prevGain)
                prevGain = g
                for (k in x.indices) x[k] *= g
                AudioTools.fade(x, sr)
                AudioTools.limit(x)
                sink(x)
            }
            sink(AudioTools.silence(sr, (c.pauseMs * pauseScale / speed).toInt()))
        }
    }

    fun deleteCache() { jobDir.deleteRecursively() }
}
