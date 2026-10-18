package io.github.koreanvoice.audio

import de.sciss.jump3r.mp3.BitStream
import de.sciss.jump3r.mp3.GainAnalysis
import de.sciss.jump3r.mp3.GetAudio
import de.sciss.jump3r.mp3.ID3Tag
import de.sciss.jump3r.mp3.Lame
import de.sciss.jump3r.mp3.LameGlobalFlags
import de.sciss.jump3r.mp3.MPEGMode
import de.sciss.jump3r.mp3.Parse
import de.sciss.jump3r.mp3.Presets
import de.sciss.jump3r.mp3.Quantize
import de.sciss.jump3r.mp3.QuantizePVT
import de.sciss.jump3r.mp3.Reservoir
import de.sciss.jump3r.mp3.Takehiro
import de.sciss.jump3r.mp3.VBRTag
import de.sciss.jump3r.mp3.Version
import de.sciss.jump3r.mpg.Common
import de.sciss.jump3r.mpg.Interface
import de.sciss.jump3r.mpg.MPGLib
import java.io.OutputStream

/**
 * Streaming mono CBR MP3 encoder on top of jump3r (pure-Java LAME port, LGPL-2.1, used unmodified
 * as a library). jump3r's own `LameEncoder` wrapper depends on javax.sound, which Android lacks,
 * so the encoder modules are wired up here the same way that wrapper does.
 */
class Mp3Encoder(
    private val out: OutputStream,
    val sampleRate: Int,
    bitrateKbps: Int = 64,
    quality: Int = 3,
    id3: Id3Info? = null,
) : AutoCloseable {

    data class Id3Info(val title: String, val artist: String, val comment: String)

    private val lame = Lame()
    private val gfp: LameGlobalFlags
    private val mp3buf = ByteArray(1 shl 17)
    private var closed = false

    init {
        val gaud = GetAudio(); val ga = GainAnalysis(); val bs = BitStream(); val p = Presets()
        val qupvt = QuantizePVT(); val qu = Quantize(); val vbr = VBRTag(); val ver = Version()
        val id3t = ID3Tag(); val rv = Reservoir(); val tak = Takehiro(); val parse = Parse()
        val mpg = MPGLib(); val intf = Interface(); val common = Common()
        lame.setModules(ga, bs, p, qupvt, qu, vbr, ver, id3t, mpg)
        bs.setModules(ga, mpg, ver, vbr)
        id3t.setModules(bs, ver)
        p.setModules(lame)
        qu.setModules(bs, rv, qupvt, tak)
        qupvt.setModules(tak, rv, lame.enc.psy)
        rv.setModules(bs)
        tak.setModules(qupvt)
        vbr.setModules(lame, bs, ver)
        gaud.setModules(parse, mpg)
        parse.setModules(ver, id3t, p)
        mpg.setModules(intf, common)
        intf.setModules(vbr, common)

        gfp = lame.lame_init()
        gfp.num_channels = 1
        gfp.in_samplerate = sampleRate
        gfp.out_samplerate = sampleRate
        gfp.mode = MPEGMode.MONO
        gfp.brate = bitrateKbps
        gfp.quality = quality
        gfp.bWriteVbrTag = false
        id3t.id3tag_init(gfp)
        gfp.write_id3tag_automatic = false
        val rc = lame.lame_init_params(gfp)
        check(rc >= 0) { "LAME 초기화 실패 ($rc)" }
        if (id3 != null) out.write(id3v2(id3))
    }

    fun encode(samples: FloatArray, offset: Int = 0, length: Int = samples.size) {
        check(!closed)
        var pos = offset
        val end = offset + length
        val block = 8192
        val buf = IntArray(block)
        while (pos < end) {
            val n = minOf(block, end - pos)
            for (i in 0 until n) buf[i] = (samples[pos + i].coerceIn(-1f, 1f) * 32767f).toInt() shl 16
            val bytes = lame.lame_encode_buffer_int(gfp, buf, buf, n, mp3buf, 0, mp3buf.size)
            check(bytes >= 0) { "MP3 인코딩 오류 ($bytes)" }
            if (bytes > 0) out.write(mp3buf, 0, bytes)
            pos += n
        }
    }

    fun finish() {
        if (closed) return
        val bytes = lame.lame_encode_flush(gfp, mp3buf, 0, mp3buf.size)
        if (bytes > 0) out.write(mp3buf, 0, bytes)
        lame.lame_close(gfp)
        out.flush()
        closed = true
    }

    override fun close() {
        finish()
        out.close()
    }

    companion object {
        /** Minimal ID3v2.3 tag with UTF-16 text frames (TIT2, TPE1, COMM). */
        fun id3v2(info: Id3Info): ByteArray {
            fun textFrame(id: String, text: String): ByteArray {
                val body = byteArrayOf(1) + byteArrayOf(0xFF.toByte(), 0xFE.toByte()) + text.toByteArray(Charsets.UTF_16LE)
                return id.toByteArray() + int32(body.size) + byteArrayOf(0, 0) + body
            }
            fun commFrame(text: String): ByteArray {
                val bom = byteArrayOf(0xFF.toByte(), 0xFE.toByte())
                val body = byteArrayOf(1) + "kor".toByteArray() + bom + byteArrayOf(0, 0) + bom + text.toByteArray(Charsets.UTF_16LE)
                return "COMM".toByteArray() + int32(body.size) + byteArrayOf(0, 0) + body
            }
            val frames = textFrame("TIT2", info.title) + textFrame("TPE1", info.artist) + commFrame(info.comment)
            val size = frames.size
            val syncsafe = byteArrayOf(((size shr 21) and 0x7f).toByte(), ((size shr 14) and 0x7f).toByte(),
                ((size shr 7) and 0x7f).toByte(), (size and 0x7f).toByte())
            return "ID3".toByteArray() + byteArrayOf(3, 0, 0) + syncsafe + frames
        }

        private fun int32(v: Int) = byteArrayOf((v ushr 24).toByte(), (v ushr 16).toByte(), (v ushr 8).toByte(), v.toByte())
    }
}
