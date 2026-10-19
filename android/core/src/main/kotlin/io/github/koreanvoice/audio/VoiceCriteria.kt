package io.github.koreanvoice.audio

import kotlin.math.PI
import kotlin.math.abs
import kotlin.math.cos
import kotlin.math.exp
import kotlin.math.ln
import kotlin.math.log10
import kotlin.math.max
import kotlin.math.min
import kotlin.math.sqrt

/**
 * About 100 measurable voice characteristics in 8 groups — timbre, pitch, sentence endings (말끝), rhythm &
 * speaking style, emphasis/dynamics, voice quality, mood and recording environment (reverb/background).
 * A candidate is compared with the natural spread of the same criteria across the speaker's own real
 * recordings, so "similar" means "within how this person actually varies".
 */
object VoiceCriteria {
    enum class Group(val label: String, val weight: Double) {
        TIMBRE("음색", 0.24), PITCH("음높이", 0.15), ENDING("말끝", 0.10), RHYTHM("리듬·말투", 0.12),
        DYNAMICS("강세·에너지", 0.10), QUALITY("음질", 0.12), MOOD("분위기", 0.07), ENVIRONMENT("울림·녹음 환경", 0.10)
    }

    /** name, group, floor (smallest spread used for normalisation, in the criterion's unit). */
    class Def(val name: String, val group: Group, val floor: Double)

    val DEFS: List<Def> = buildList {
        val P = Group.PITCH; val E = Group.ENDING; val R = Group.RHYTHM; val D = Group.DYNAMICS
        val Q = Group.QUALITY; val M = Group.MOOD; val V = Group.ENVIRONMENT; val T = Group.TIMBRE
        listOf("평균 음높이(중앙값)" to 1.0, "평균 음높이(평균)" to 1.0, "억양 폭" to 0.5, "낮은 음 (5%)" to 1.2, "높은 음 (95%)" to 1.2,
            "음역" to 1.2, "음높이 사분범위" to 0.8, "문장 내 하강 기울기" to 2.0, "음높이 떨림" to 0.08, "음높이 변화 속도" to 0.1,
            "구절 내 억양 변화" to 0.4, "높은 음 비율" to 0.05).forEach { add(Def(it.first, P, it.second)) }
        listOf("말끝 음높이 변화" to 0.8, "말끝 상대 높이" to 1.0, "마지막 구절 길이" to 0.15, "말끝 늘이기" to 0.15,
            "말끝 음량 감쇠" to 15.0, "올림 말끝 비율" to 0.2, "내림 말끝 비율" to 0.2, "말끝 갈라짐(크리키)" to 0.08,
            "말끝 억양 변화" to 0.5, "구절 뒤 쉼 길이" to 0.08).forEach { add(Def(it.first, E, it.second)) }
        listOf("음절 속도" to 0.4, "유성음 비율" to 0.05, "평균 구절 길이" to 0.08, "구절 길이 편차" to 0.06, "쉼 빈도" to 0.15,
            "평균 쉼 길이" to 0.06, "쉼 길이 편차" to 0.05, "리듬 변동성(nPVI)" to 6.0, "말하는 시간 비율" to 0.05,
            "조음 속도" to 0.5).forEach { add(Def(it.first, R, it.second)) }
        listOf("음량 변화 폭" to 0.8, "음량 범위" to 1.5, "음절 리듬 변조" to 0.05, "강세 시작 기울기" to 1.0, "피크 대비 평균" to 1.0,
            "음량-음높이 연동" to 0.12, "큰 소리 여유" to 1.0, "강세 비율" to 0.05).forEach { add(Def(it.first, D, it.second)) }
        listOf("배음 선명도" to 0.04, "밝기(스펙트럼 중심)" to 0.12, "밝기 변화" to 0.06, "평탄도" to 0.02, "고역 한계(85%)" to 0.2,
            "숨소리(고역/저역)" to 1.5, "저역 비중" to 1.5, "영교차율" to 0.01, "스펙트럼 변화량" to 0.5, "주기성 편차" to 0.03)
            .forEach { add(Def(it.first, Q, it.second)) }
        listOf("활기" to 0.4, "따뜻함" to 2.0, "생동감" to 0.1, "차분함" to 0.05).forEach { add(Def(it.first, M, it.second)) }
        listOf("울림(쉼 채움)" to 0.05, "잔향 감쇠" to 20.0, "배경 잡음 수준(상대)" to 3.0, "배경 잡음 수준(절대)" to 3.0,
            "잡음 색깔" to 0.25, "잡음 평탄도" to 0.05, "스펙트럼 기울기" to 0.15, "포락선 번짐" to 0.05).forEach { add(Def(it.first, V, it.second)) }
        for (b in 0 until VoiceAnalysis.LTAS_BANDS) add(Def("주파수 대역 ${b + 1}", T, 2.0))
        for (c in 1..13) add(Def("음색 계수 $c", T, 1.5))
    }

    val SIZE get() = DEFS.size

    private fun pct(v: List<Double>, p: Double): Double { if (v.isEmpty()) return Double.NaN; val s = v.sorted(); return s[((s.size - 1) * p).toInt()] }
    private fun mean(v: List<Double>) = if (v.isEmpty()) Double.NaN else v.average()
    private fun std(v: List<Double>): Double { if (v.size < 2) return Double.NaN; val m = v.average(); return sqrt(v.sumOf { (it - m) * (it - m) } / v.size) }
    private fun slope(y: List<Double>, dt: Double): Double {
        if (y.size < 3) return Double.NaN
        val n = y.size; val mx = (n - 1) / 2.0; val my = y.average()
        var num = 0.0; var den = 0.0
        for (i in 0 until n) { num += (i - mx) * (y[i] - my); den += (i - mx) * (i - mx) }
        return num / den / dt
    }

    /** Measures all criteria (NaN where not measurable, e.g. no phrase end in the clip). */
    fun measure(x: FloatArray, sr: Int, x16In: FloatArray? = null): DoubleArray {
        val x16 = x16In ?: if (sr == 16000) x else Resampler.resample(x, sr, 16000)
        val aper = DoubleArray(x16.size / 160 + 1) { Double.NaN }
        val f0 = VoiceAnalysis.yin(x16, 16000, aperiodicity = { f, a -> if (f < aper.size) aper[f] = a })
        val db = AudioTools.frameDb(x16, 160)
        val nF = min(f0.size, max(0, db.size - 2))
        fun dbAt(f: Int) = db[min(db.size - 1, f + 2)]          // YIN frame f is centred 20 ms later
        val peak = db.maxOrNull() ?: -120.0
        val active = BooleanArray(db.size) { db[it] > peak - 35 }
        val st = DoubleArray(nF) { if (f0[it] > 0) 12 * ln(f0[it] / 100.0) / ln(2.0) else Double.NaN }
        val voicedSt = st.filter { !it.isNaN() }
        val med = pct(voicedSt, 0.5)
        val core = voicedSt.filter { abs(it - med) < 12 }
        val out = ArrayList<Double>()

        // voiced runs (gaps ≤ 2 frames bridged)
        data class Run(val s: Int, val e: Int)
        val runs = ArrayList<Run>()
        run {
            var i = 0
            while (i < nF) {
                if (st[i].isNaN()) { i++; continue }
                var last = i; var j = i + 1
                while (j < nF && j - last <= 3) { if (!st[j].isNaN()) last = j; j++ }
                runs.add(Run(i, last + 1)); i = last + 1
            }
        }
        val longRuns = runs.filter { it.e - it.s >= 8 }
        // phrase ends: runs followed by ≥ 150 ms unvoiced (or the clip end)
        val ends = longRuns.filterIndexed { k, r -> val next = runs.firstOrNull { it.s >= r.e }; next == null || next.s - r.e >= 15 }

        // ---------------- pitch (12)
        out += med; out += mean(core); out += std(core); out += pct(core, 0.05); out += pct(core, 0.95)
        out += pct(core, 0.95) - pct(core, 0.05); out += pct(core, 0.75) - pct(core, 0.25)
        out += mean(longRuns.filter { it.e - it.s >= 20 }.map { r -> slope((r.s until r.e).map { st[it] }.filter { !it.isNaN() }, 0.01) })
        val deltas = runs.flatMap { r -> (r.s + 1 until r.e).mapNotNull { i -> if (!st[i].isNaN() && !st[i - 1].isNaN()) st[i] - st[i - 1] else null } }.filter { abs(it) < 3 }
        out += mean(deltas.map { abs(it) }); out += std(deltas)
        out += mean(longRuns.map { r -> std((r.s until r.e).map { st[it] }.filter { !it.isNaN() }) }.filter { !it.isNaN() })
        out += if (core.isEmpty()) Double.NaN else core.count { it > med + 3 }.toDouble() / core.size

        // ---------------- endings (10)
        val moves = ArrayList<Double>(); val rels = ArrayList<Double>(); val lastDur = ArrayList<Double>(); val eSlopes = ArrayList<Double>()
        val creak = ArrayList<Double>(); val endStd = ArrayList<Double>(); val pauses = ArrayList<Double>()
        val meanRun = mean(longRuns.map { (it.e - it.s) * 0.01 })
        for (r in ends) {
            val w0 = max(r.s, r.e - 25)
            val win = (w0 until r.e).map { st[it] }.filter { !it.isNaN() }
            if (win.size < 6) continue
            moves += win.takeLast(5).average() - win.take(5).average()
            val rmed = pct((r.s until r.e).map { st[it] }.filter { !it.isNaN() }, 0.5)
            rels += win.takeLast(5).average() - rmed
            lastDur += (r.e - r.s) * 0.01
            eSlopes += slope((max(r.s, r.e - 20) until r.e).map { dbAt(it) }, 0.01)
            creak += (w0 until r.e).count { i -> !st[i].isNaN() && ((aper.getOrNull(i) ?: 0.0) > 0.3 || st[i] < med - 7) }.toDouble() / (r.e - w0)
            endStd += std(win)
            val next = runs.firstOrNull { it.s >= r.e }
            if (next != null) pauses += min(1.0, (next.s - r.e) * 0.01)
        }
        out += mean(moves); out += mean(rels); out += mean(lastDur); out += mean(lastDur) / meanRun; out += mean(eSlopes)
        out += if (moves.isEmpty()) Double.NaN else moves.count { it > 1 }.toDouble() / moves.size
        out += if (moves.isEmpty()) Double.NaN else moves.count { it < -1 }.toDouble() / moves.size
        out += mean(creak); out += mean(endStd); out += mean(pauses)

        // ---------------- rhythm (10)
        val activeSec = active.count { it } * 0.01
        var onsets = 0
        for (i in 2 until nF) if (!st[i].isNaN() && st[i - 1].isNaN() && st[i - 2].isNaN()) onsets++
        val voicedFrames = st.count { !it.isNaN() }
        // pauses = inactive stretches ≥ 150 ms between active speech
        val pauseLens = ArrayList<Double>()
        run {
            val first = active.indexOfFirst { it }; val last = active.indexOfLast { it }
            var i = first
            while (i in 0 until last) {
                if (!active[i]) { var j = i; while (j < last && !active[j]) j++; if (j - i >= 15) pauseLens += (j - i) * 0.01; i = j } else i++
            }
        }
        val runDur = longRuns.map { (it.e - it.s) * 0.01 }
        out += if (activeSec > 0.3) onsets / activeSec else Double.NaN
        out += if (activeSec > 0) voicedFrames * 0.01 / activeSec else Double.NaN
        out += mean(runDur); out += std(runDur)
        out += if (activeSec > 0.3) pauseLens.size / activeSec else Double.NaN
        out += if (pauseLens.isEmpty()) 0.0 else pauseLens.average()
        out += if (pauseLens.size < 2) 0.0 else std(pauseLens)
        out += if (runDur.size < 2) Double.NaN else 100.0 * runDur.zipWithNext { a, b -> abs(a - b) / ((a + b) / 2) }.average()
        out += if (db.isEmpty()) Double.NaN else active.count { it }.toDouble() / db.size
        out += if (voicedFrames > 30) onsets / (voicedFrames * 0.01) else Double.NaN

        // ---------------- dynamics (8)
        val actDb = db.filterIndexed { i, _ -> active[i] }
        out += std(actDb); out += pct(actDb, 0.95) - pct(actDb, 0.05)
        out += modulationRatio(db, active)
        val attacks = ArrayList<Double>()
        for (i in 2 until nF) if (!st[i].isNaN() && st[i - 1].isNaN() && st[i - 2].isNaN()) {
            var mx = 0.0; for (k in 0 until 5) if (i + k + 2 < db.size) mx = max(mx, db[i + k + 2] - db[i + k + 1]); attacks += mx
        }
        out += mean(attacks)
        val rms = AudioTools.rms(x16); val pk = x16.maxOfOrNull { abs(it) } ?: 0f
        out += 20 * log10((pk + 1e-9) / (rms + 1e-9))
        val pairs = (0 until nF).filter { !st[it].isNaN() }.map { st[it] to dbAt(it) }
        out += correlation(pairs)
        out += pct(actDb, 0.9) - pct(actDb, 0.5)
        val p50 = pct(actDb, 0.5)
        out += if (actDb.isEmpty()) Double.NaN else actDb.count { it > p50 + 6 }.toDouble() / actDb.size

        // ---------------- voice quality (10) + spectral data for mood/environment/timbre
        val spec = Spectral(x16, db, active, st)
        val vAper = (0 until nF).filter { !st[it].isNaN() }.mapNotNull { aper.getOrNull(it)?.takeIf { a -> !a.isNaN() } }
        out += mean(vAper.map { 1 - it })
        out += spec.centroidMean; out += spec.centroidStd; out += spec.flatness; out += spec.rolloff
        out += spec.highRatio; out += spec.lowRatio; out += spec.zcrVoiced; out += spec.flux; out += std(vAper)

        // ---------------- mood (4)
        val f0Std = out[2]; val eStd = out[32]; val onsetRate = out[22]
        out += 0.5 * f0Std + 0.25 * eStd + 0.6 * onsetRate
        out += spec.lowRatio - spec.highRatio
        out += out[9] * onsetRate
        out += (if (pauseLens.isEmpty()) 0.0 else pauseLens.average()) / max(0.3, 0.5 * f0Std + 0.25 * eStd + 0.6 * onsetRate)

        // ---------------- environment (8)
        val prof = VoiceAnalysis.analyze(x, sr, x16, hopMs = 20)
        out += prof.tailRatio
        val decays = ends.mapNotNull { r -> val seg = (r.e + 2 until min(db.size, r.e + 17)).map { db[it] }; if (seg.size >= 8) slope(seg, 0.01) else null }
        out += mean(decays)
        val p10 = pct(db.toList(), 0.10); val p90 = pct(db.toList(), 0.90)
        out += p10 - p90; out += p10
        out += spec.noiseCentroid; out += spec.noiseFlatness
        out += VoiceAnalysis.tilt(prof.ltas)
        out += envelopeSmear(db, active)

        // ---------------- timbre (37)
        prof.ltas.forEach { out += it }
        spec.mfcc.forEach { out += it }
        check(out.size == DEFS.size) { "criteria size ${out.size} != ${DEFS.size}" }
        return out.toDoubleArray()
    }

    private fun correlation(p: List<Pair<Double, Double>>): Double {
        if (p.size < 10) return Double.NaN
        val mx = p.map { it.first }.average(); val my = p.map { it.second }.average()
        var sxy = 0.0; var sxx = 0.0; var syy = 0.0
        for ((a, b) in p) { sxy += (a - mx) * (b - my); sxx += (a - mx) * (a - mx); syy += (b - my) * (b - my) }
        return if (sxx <= 0 || syy <= 0) Double.NaN else sxy / sqrt(sxx * syy)
    }

    /** Share of the loudness envelope's modulation energy at syllable rates (2–8 Hz) within 0.5–20 Hz. */
    private fun modulationRatio(db: DoubleArray, active: BooleanArray): Double {
        val first = active.indexOfFirst { it }; val last = active.indexOfLast { it }
        if (first < 0 || last - first < 64) return Double.NaN
        val env = (first..last).map { 10.0.pow(db[it] / 20) }
        val n = Fft.nextPow2(env.size); val m = env.average()
        val re = DoubleArray(n); val im = DoubleArray(n)
        for (i in env.indices) re[i] = env[i] - m
        Fft.transform(re, im)
        var band = 0.0; var all = 0.0
        for (k in 1 until n / 2) { val f = k * 100.0 / n; val pw = re[k] * re[k] + im[k] * im[k]; if (f in 0.5..20.0) all += pw; if (f in 2.0..8.0) band += pw }
        return if (all > 0) band / all else Double.NaN
    }

    /** Envelope autocorrelation at 60 ms: reverberation smears syllable boundaries. */
    private fun envelopeSmear(db: DoubleArray, active: BooleanArray): Double {
        val first = active.indexOfFirst { it }; val last = active.indexOfLast { it }
        if (first < 0 || last - first < 30) return Double.NaN
        val env = (first..last).map { 10.0.pow(db[it] / 20) }; val m = env.average()
        var num = 0.0; var den = 0.0
        for (i in env.indices) { val a = env[i] - m; den += a * a; if (i + 6 < env.size) num += a * (env[i + 6] - m) }
        return if (den > 0) num / den else Double.NaN
    }

    private fun Double.pow(e: Double) = Math.pow(this, e)

    /** Frame-wise spectral statistics at 16 kHz (512-point FFT, 10 ms hop). */
    private class Spectral(x: FloatArray, db: DoubleArray, active: BooleanArray, st: DoubleArray) {
        var centroidMean = Double.NaN; var centroidStd = Double.NaN; var flatness = Double.NaN; var rolloff = Double.NaN
        var highRatio = Double.NaN; var lowRatio = Double.NaN; var zcrVoiced = Double.NaN; var flux = Double.NaN
        var noiseCentroid = Double.NaN; var noiseFlatness = Double.NaN
        val mfcc = DoubleArray(13) { Double.NaN }

        init {
            val n = 512; val hop = 160
            val win = DoubleArray(n) { 0.5 - 0.5 * cos(2 * PI * it / n) }
            val mel = melBank(26, n, 16000)
            val cents = ArrayList<Double>(); val flats = ArrayList<Double>(); val rolls = ArrayList<Double>()
            val highs = ArrayList<Double>(); val lows = ArrayList<Double>(); val zcrs = ArrayList<Double>(); val fluxes = ArrayList<Double>()
            val nC = ArrayList<Double>(); val nFl = ArrayList<Double>()
            val ceps = Array(13) { ArrayList<Double>() }
            val p20 = db.sorted().let { if (it.isEmpty()) -120.0 else it[it.size / 5] }
            var prev: DoubleArray? = null
            val re = DoubleArray(n); val im = DoubleArray(n)
            var f = 0
            while (f * hop + n <= x.size) {
                val s = f * hop
                val dIdx = min(db.size - 1, f + 1)
                val isActive = active.getOrElse(dIdx) { false }
                val isQuiet = db.getOrElse(dIdx) { 0.0 } <= p20
                if (isActive || isQuiet) {
                    for (i in 0 until n) { re[i] = x[s + i] * win[i]; im[i] = 0.0 }
                    Fft.transform(re, im)
                    val pw = DoubleArray(n / 2 + 1) { re[it] * re[it] + im[it] * im[it] + 1e-12 }
                    val tot = pw.sum()
                    var c = 0.0; for (k in pw.indices) c += k * 16000.0 / n * pw[k]; c /= tot
                    val fl = exp(pw.sumOf { ln(it) } / pw.size) / (tot / pw.size)
                    if (isActive) {
                        cents += c / 1000; flats += fl
                        var acc = 0.0; var k85 = 0; while (k85 < pw.size && acc < 0.85 * tot) { acc += pw[k85]; k85++ }
                        rolls += k85 * 16000.0 / n / 1000
                        val k4 = 4000 * n / 16000; val k300 = 300 * n / 16000
                        highs += 10 * log10(pw.copyOfRange(k4, pw.size).sum() / pw.copyOfRange(0, k4).sum())
                        lows += 10 * log10(pw.copyOfRange(0, k300).sum() / tot)
                        val frameF0 = st.getOrElse(f - 1) { Double.NaN }
                        if (!frameF0.isNaN()) { var z = 0; for (i in s + 1 until s + n) if ((x[i] >= 0) != (x[i - 1] >= 0)) z++; zcrs += z.toDouble() / n }
                        val lp = DoubleArray(pw.size) { ln(pw[it]) }
                        prev?.let { pv -> var d = 0.0; for (k in lp.indices) d += (lp[k] - pv[k]) * (lp[k] - pv[k]); fluxes += sqrt(d / lp.size) }
                        prev = lp
                        val me = DoubleArray(mel.size) { b -> var e = 1e-10; for ((k, w) in mel[b]) e += w * pw[k]; ln(e) }
                        for (c1 in 1..13) { var sum = 0.0; for (b in me.indices) sum += me[b] * cos(PI * c1 * (b + 0.5) / me.size); ceps[c1 - 1] += sum }
                    } else { nC += c / 1000; nFl += fl; prev = null }
                }
                f++
            }
            centroidMean = mean(cents); centroidStd = std(cents); flatness = mean(flats); rolloff = mean(rolls)
            highRatio = mean(highs); lowRatio = mean(lows); zcrVoiced = mean(zcrs); flux = mean(fluxes)
            noiseCentroid = mean(nC); noiseFlatness = mean(nFl)
            for (i in 0 until 13) mfcc[i] = mean(ceps[i])
        }

        private fun melBank(nb: Int, n: Int, sr: Int): Array<List<Pair<Int, Double>>> {
            fun hz2mel(h: Double) = 2595 * log10(1 + h / 700)
            fun mel2hz(m: Double) = 700 * (Math.pow(10.0, m / 2595) - 1)
            val lo = hz2mel(60.0); val hi = hz2mel(7600.0)
            val pts = DoubleArray(nb + 2) { mel2hz(lo + (hi - lo) * it / (nb + 1)) * n / sr }
            return Array(nb) { b ->
                val l = pts[b]; val c = pts[b + 1]; val r = pts[b + 2]
                (l.toInt()..r.toInt() + 1).mapNotNull { k ->
                    val w = if (k < c) (k - l) / (c - l) else (r - k) / (r - c)
                    if (w > 0 && k <= n / 2) k to w else null
                }
            }
        }
    }

    // ------------------------------------------------------------------ comparison

    /** The speaker's own criteria distribution (from their real recordings). */
    class Target(val mean: DoubleArray, val spread: DoubleArray) {
        companion object {
            fun of(samples: List<DoubleArray>): Target {
                val m = DoubleArray(SIZE) { i -> samples.map { it[i] }.filter { !it.isNaN() }.let { if (it.isEmpty()) Double.NaN else it.average() } }
                val s = DoubleArray(SIZE) { i ->
                    val v = samples.map { it[i] }.filter { !it.isNaN() }
                    val sd = if (v.size < 2) 0.0 else sqrt(v.sumOf { (it - m[i]) * (it - m[i]) } / (v.size - 1))
                    max(sd, DEFS[i].floor)
                }
                return Target(m, s)
            }
        }
    }

    class Report(val overall: Double, val groups: Map<Group, Double>, val perCriterion: DoubleArray) {
        /** The criteria furthest from the original (name, similarity 0..1). */
        fun worst(n: Int = 5) = DEFS.indices.filter { !perCriterion[it].isNaN() }.sortedBy { perCriterion[it] }.take(n).map { DEFS[it].name to perCriterion[it] }
    }

    /** Similarity per criterion = exp(−z²/2) with z = distance / the speaker's own spread; groups are averaged. */
    fun compare(c: DoubleArray, t: Target): Report {
        val per = DoubleArray(SIZE) { i ->
            if (c[i].isNaN() || t.mean[i].isNaN()) Double.NaN else { val z = (c[i] - t.mean[i]) / t.spread[i]; exp(-0.5 * z * z) }
        }
        val groups = Group.entries.associateWith { g -> DEFS.indices.filter { DEFS[it].group == g && !per[it].isNaN() }.map { per[it] }.let { if (it.isEmpty()) Double.NaN else it.average() } }
        val valid = groups.filter { !it.value.isNaN() }
        val overall = valid.entries.sumOf { it.key.weight * it.value } / valid.keys.sumOf { it.weight }.coerceAtLeast(1e-9)
        return Report(overall, groups, per)
    }
}
