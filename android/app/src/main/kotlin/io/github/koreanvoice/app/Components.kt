package io.github.koreanvoice.app

import androidx.compose.foundation.Canvas
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.Share
import androidx.compose.material3.Button
import androidx.compose.material3.Card
import androidx.compose.material3.CardDefaults
import androidx.compose.material3.Icon
import androidx.compose.material3.LinearProgressIndicator
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedButton
import androidx.compose.material3.Slider
import androidx.compose.material3.Switch
import androidx.compose.material3.Text
import androidx.compose.material3.TextButton
import androidx.compose.runtime.Composable
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.PathEffect
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextOverflow
import androidx.compose.ui.unit.dp

@Composable
fun LabeledSlider(label: String, value: Float, range: ClosedFloatingPointRange<Float>, fmt: String, onChange: (Float) -> Unit) {
    Column {
        Row { Text(label, style = MaterialTheme.typography.labelLarge, modifier = Modifier.weight(1f)); Text(fmt.format(value), style = MaterialTheme.typography.labelLarge) }
        Slider(value = value, onValueChange = onChange, valueRange = range)
    }
}

@Composable
fun LabeledSwitch(label: String, detail: String, checked: Boolean, onChange: (Boolean) -> Unit) {
    Row(verticalAlignment = Alignment.CenterVertically) {
        Column(Modifier.weight(1f)) {
            Text(label, style = MaterialTheme.typography.labelLarge)
            if (detail.isNotEmpty()) Text(detail, style = MaterialTheme.typography.bodySmall)
        }
        Switch(checked, onChange)
    }
}

/** Score progression of auto-learning rounds (dashed line = starting point). */
@Composable
fun ScoreChart(scores: List<Double>, baseline: Double, modifier: Modifier = Modifier) {
    val pts = scores.filter { !it.isNaN() }
    if (pts.isEmpty()) return
    val color = MaterialTheme.colorScheme.primary
    val base = MaterialTheme.colorScheme.outline
    Canvas(modifier.fillMaxWidth().height(70.dp)) {
        val all = pts + listOf(baseline).filter { !it.isNaN() }
        val lo = all.min() - 0.01; val hi = all.max() + 0.01
        fun y(v: Double) = (size.height * (1 - (v - lo) / (hi - lo))).toFloat()
        if (!baseline.isNaN()) drawLine(base, Offset(0f, y(baseline)), Offset(size.width, y(baseline)), 2f, pathEffect = PathEffect.dashPathEffect(floatArrayOf(8f, 8f)))
        val step = if (pts.size > 1) size.width / (pts.size - 1) else 0f
        for (i in pts.indices) {
            val p = Offset(i * step, y(pts[i]))
            drawCircle(color, 6f, p)
            if (i > 0) drawLine(color, Offset((i - 1) * step, y(pts[i - 1])), p, 4f)
        }
    }
}

@Composable
fun JobCard(job: JobState, playing: String?, onGoLibrary: () -> Unit, kinds: Set<TaskKind> = TaskKind.entries.toSet()) {
    val ctx = LocalContext.current
    val kind = when (job) {
        is JobState.Preparing -> job.kind; is JobState.Running -> job.kind; is JobState.Done -> job.kind
        is JobState.Failed -> job.kind; is JobState.Cancelled -> job.kind; JobState.Idle -> null
    }
    if (kind == null || kind !in kinds) return
    when (job) {
        JobState.Idle -> {}
        is JobState.Preparing -> Card(Modifier.fillMaxWidth()) {
            Column(Modifier.padding(16.dp)) { Text("${job.kind.label}: ${job.message}"); LinearProgressIndicator(Modifier.fillMaxWidth().padding(top = 8.dp)) }
        }
        is JobState.Running -> Card(Modifier.fillMaxWidth()) {
            Column(Modifier.padding(16.dp), verticalArrangement = Arrangement.spacedBy(6.dp)) {
                Text("${job.kind.label} 중: ${job.title}", fontWeight = FontWeight.Bold)
                LinearProgressIndicator(progress = { job.fraction.coerceIn(0f, 1f) }, modifier = Modifier.fillMaxWidth())
                Text(job.primary + (if (job.etaMs > 0) " · 남은 시간 약 ${formatDuration(job.etaMs)}" else ""))
                if (job.secondary.isNotEmpty()) Text(job.secondary, maxLines = 2, overflow = TextOverflow.Ellipsis, style = MaterialTheme.typography.bodySmall)
                if (job.kind == TaskKind.AUTOLEARN && job.scores.isNotEmpty()) {
                    Text("원본과의 유사도 (점선 = 학습 전)", style = MaterialTheme.typography.labelMedium)
                    ScoreChart(job.scores, job.baseline)
                }
                Text("화면을 꺼도 계속됩니다.", style = MaterialTheme.typography.bodySmall)
                OutlinedButton(onClick = { SynthesisService.cancel(ctx) }) { Text("취소") }
            }
        }
        is JobState.Done -> Card(Modifier.fillMaxWidth(), colors = CardDefaults.cardColors(containerColor = MaterialTheme.colorScheme.secondaryContainer)) {
            Column(Modifier.padding(16.dp), verticalArrangement = Arrangement.spacedBy(6.dp)) {
                Text("완료: ${job.title}", fontWeight = FontWeight.Bold)
                Text(job.summary, style = MaterialTheme.typography.bodyMedium)
                Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                    job.file?.let { f ->
                        Button(onClick = { Player.toggle(f) }) { Text(if (playing == f.path) "정지" else "재생") }
                        OutlinedButton(onClick = { Sharing.share(ctx, f) }) { Icon(Icons.Filled.Share, null); Text("공유") }
                        TextButton(onClick = onGoLibrary) { Text("보관함") }
                    }
                    TextButton(onClick = { AppGraph.jobs.reset() }) { Text("닫기") }
                }
            }
        }
        is JobState.Failed -> Card(Modifier.fillMaxWidth(), colors = CardDefaults.cardColors(containerColor = MaterialTheme.colorScheme.errorContainer)) {
            Column(Modifier.padding(16.dp)) {
                Text("${job.kind.label} 실패: ${job.message}")
                TextButton(onClick = { AppGraph.jobs.reset() }) { Text("닫기") }
            }
        }
        is JobState.Cancelled -> Card(Modifier.fillMaxWidth()) {
            Column(Modifier.padding(16.dp)) {
                Text(if (job.kind == TaskKind.SYNTHESIS) "취소되었습니다. 같은 글로 다시 시작하면 완료된 문장은 건너뜁니다." else "${job.kind.label}이(가) 취소되었습니다.")
                TextButton(onClick = { AppGraph.jobs.reset() }) { Text("닫기") }
            }
        }
    }
}

val ChipGreen = Color(0xFF2E7D32)
