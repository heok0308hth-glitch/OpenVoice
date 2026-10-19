package io.github.koreanvoice.app

import android.content.Context
import android.media.MediaCodec
import android.media.MediaExtractor
import android.media.MediaFormat
import android.net.Uri
import java.nio.ByteOrder

/**
 * Decodes any audio format Android supports (mp3, m4a/aac, wav, flac, ogg/opus, amr, 3gp, webm …)
 * and the audio track of videos (mp4, mkv, webm, 3gp, mov …) from a content Uri — local files,
 * Google Drive, Downloads, etc. — into mono float PCM.
 */
object AudioDecoder {

    class Decoded(val samples: FloatArray, val sampleRate: Int, val truncated: Boolean)

    /** MIME types offered in the file picker for learning a voice. */
    val PICK_TYPES = arrayOf("audio/*", "video/*")

    fun decode(context: Context, uri: Uri, maxSeconds: Int = 300): Decoded {
        val parts = ArrayList<FloatArray>(); var sr = 0; var trunc = false
        decodeWindows(context, uri, windowSeconds = maxSeconds, maxSeconds = maxSeconds) { w, rate, t -> parts.add(w); sr = rate; trunc = t }
        val all = FloatArray(parts.sumOf { it.size }); var p = 0
        for (a in parts) { System.arraycopy(a, 0, all, p, a.size); p += a.size }
        if (all.isEmpty()) error("오디오를 해석할 수 없습니다")
        return Decoded(all, sr, trunc)
    }

    /**
     * Streams the decoded audio in windows of [windowSeconds] (bounded memory for long videos);
     * [onWindow] gets (samples, sampleRate, truncated-at-limit).
     */
    fun decodeWindows(context: Context, uri: Uri, windowSeconds: Int = 60, maxSeconds: Int = 1200,
                      onWindow: (FloatArray, Int, Boolean) -> Unit) {
        val ex = MediaExtractor()
        ex.setDataSource(context, uri, null)
        val track = (0 until ex.trackCount).firstOrNull {
            ex.getTrackFormat(it).getString(MediaFormat.KEY_MIME)?.startsWith("audio/") == true
        } ?: run { ex.release(); error("오디오 트랙을 찾을 수 없습니다") }
        ex.selectTrack(track)
        val fmt = ex.getTrackFormat(track)
        val mime = fmt.getString(MediaFormat.KEY_MIME)!!
        var sampleRate = fmt.getInteger(MediaFormat.KEY_SAMPLE_RATE)
        var channels = fmt.getInteger(MediaFormat.KEY_CHANNEL_COUNT)
        val codec = MediaCodec.createDecoderByType(mime)
        codec.configure(fmt, null, null, 0)
        codec.start()
        val out = FloatList()
        var emitted = 0L
        val info = MediaCodec.BufferInfo()
        var inputDone = false
        var outputDone = false
        var pcmFloat = false
        var truncated = false
        try {
            while (!outputDone) {
                if (!inputDone) {
                    val inIdx = codec.dequeueInputBuffer(10_000)
                    if (inIdx >= 0) {
                        val buf = codec.getInputBuffer(inIdx)!!
                        val n = ex.readSampleData(buf, 0)
                        if (n < 0) {
                            codec.queueInputBuffer(inIdx, 0, 0, 0, MediaCodec.BUFFER_FLAG_END_OF_STREAM); inputDone = true
                        } else {
                            codec.queueInputBuffer(inIdx, 0, n, ex.sampleTime, 0); ex.advance()
                        }
                    }
                }
                val outIdx = codec.dequeueOutputBuffer(info, 10_000)
                when {
                    outIdx == MediaCodec.INFO_OUTPUT_FORMAT_CHANGED -> {
                        val of = codec.outputFormat
                        sampleRate = of.getInteger(MediaFormat.KEY_SAMPLE_RATE)
                        channels = of.getInteger(MediaFormat.KEY_CHANNEL_COUNT)
                        pcmFloat = of.containsKey(MediaFormat.KEY_PCM_ENCODING) &&
                            of.getInteger(MediaFormat.KEY_PCM_ENCODING) == android.media.AudioFormat.ENCODING_PCM_FLOAT
                    }
                    outIdx >= 0 -> {
                        val buf = codec.getOutputBuffer(outIdx)!!.order(ByteOrder.nativeOrder())
                        buf.position(info.offset); buf.limit(info.offset + info.size)
                        if (pcmFloat) {
                            val fb = buf.asFloatBuffer()
                            val frames = fb.remaining() / channels
                            for (f in 0 until frames) { var a = 0f; for (c in 0 until channels) a += fb.get(); out.add(a / channels) }
                        } else {
                            val sb = buf.asShortBuffer()
                            val frames = sb.remaining() / channels
                            for (f in 0 until frames) { var a = 0f; for (c in 0 until channels) a += sb.get() / 32768f; out.add(a / channels) }
                        }
                        codec.releaseOutputBuffer(outIdx, false)
                        if (info.flags and MediaCodec.BUFFER_FLAG_END_OF_STREAM != 0) outputDone = true
                        if (emitted + out.size > maxSeconds.toLong() * sampleRate) { truncated = true; outputDone = true }
                        if (out.size >= windowSeconds.toLong() * sampleRate || outputDone) {
                            if (out.size > 0) onWindow(out.toArray(), sampleRate, truncated)
                            emitted += out.size; out.clear()
                        }
                    }
                }
            }
        } finally {
            runCatching { codec.stop() }; codec.release(); ex.release()
        }
        if (out.size > 0) { onWindow(out.toArray(), sampleRate, truncated); emitted += out.size }
        if (emitted == 0L) error("오디오를 해석할 수 없습니다")
    }

    private class FloatList {
        private var a = FloatArray(1 shl 16)
        var size = 0; private set
        fun add(v: Float) { if (size == a.size) a = a.copyOf(a.size * 2); a[size++] = v }
        fun toArray() = a.copyOf(size)
        fun clear() { size = 0 }
    }
}
