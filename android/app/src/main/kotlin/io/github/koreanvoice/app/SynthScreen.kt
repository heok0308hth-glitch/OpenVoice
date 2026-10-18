package io.github.koreanvoice.app

import android.widget.Toast
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.ExperimentalLayoutApi
import androidx.compose.foundation.layout.FlowRow
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.heightIn
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.verticalScroll
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.ContentPaste
import androidx.compose.material.icons.filled.Delete
import androidx.compose.material.icons.filled.ExpandLess
import androidx.compose.material.icons.filled.ExpandMore
import androidx.compose.material.icons.filled.FileOpen
import androidx.compose.material.icons.filled.PlayArrow
import androidx.compose.material.icons.filled.Share
import androidx.compose.material.icons.filled.Stop
import androidx.compose.material3.Button
import androidx.compose.material3.Card
import androidx.compose.material3.CardDefaults
import androidx.compose.material3.CircularProgressIndicator
import androidx.compose.material3.DropdownMenuItem
import androidx.compose.material3.ExperimentalMaterial3Api
import androidx.compose.material3.ExposedDropdownMenuBox
import androidx.compose.material3.ExposedDropdownMenuDefaults
import androidx.compose.material3.FilterChip
import androidx.compose.material3.Icon
import androidx.compose.material3.IconButton
import androidx.compose.material3.LinearProgressIndicator
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.MenuAnchorType
import androidx.compose.material3.OutlinedButton
import androidx.compose.material3.OutlinedTextField
import androidx.compose.material3.Slider
import androidx.compose.material3.Text
import androidx.compose.material3.TextButton
import androidx.compose.runtime.Composable
import androidx.compose.runtime.collectAsState
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableFloatStateOf
import androidx.compose.runtime.mutableIntStateOf
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.saveable.rememberSaveable
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.platform.LocalClipboardManager
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextOverflow
import androidx.compose.ui.unit.dp
import io.github.koreanvoice.engine.SynthParams

const val MAX_CHARS = 100_000

@OptIn(ExperimentalMaterial3Api::class, ExperimentalLayoutApi::class)
@Composable
fun SynthScreen(vm: MainViewModel, modifier: Modifier, onGoVoices: () -> Unit, onGoLibrary: () -> Unit) {
    val ctx = LocalContext.current
    val settings = AppGraph.settings
    val voices by AppGraph.voices.voices.collectAsState()
    val job by AppGraph.jobs.state.collectAsState()
    val preview by vm.preview.collectAsState()
    val playing by Player.playing.collectAsState()
    val clipboard = LocalClipboardManager.current

    var text by rememberSaveable { mutableStateOf(settings.draftText) }
    var title by rememberSaveable { mutableStateOf("") }
    var voiceId by rememberSaveable { mutableStateOf(settings.lastVoiceId) }
    if (voiceId != null && voices.none { it.id == voiceId }) voiceId = null
    var showSettings by rememberSaveable { mutableStateOf(false) }
    val p0 = remember { settings.params }
    var speed by remember { mutableFloatStateOf(p0.speed) }
    var noise by remember { mutableFloatStateOf(p0.noiseScale) }
    var noiseW by remember { mutableFloatStateOf(p0.noiseScaleW) }
    var timbre by remember { mutableFloatStateOf(p0.timbreStrength) }
    var pause by remember { mutableFloatStateOf(settings.pauseScale) }
    var bitrate by remember { mutableIntStateOf(settings.bitrate) }
    var perf by remember { mutableStateOf(settings.performanceMode) }

    fun params() = SynthParams(speed = speed, noiseScale = noise, noiseScaleW = noiseW, timbreStrength = timbre).also { settings.params = it }

    val openText = rememberLauncherForActivityResult(ActivityResultContracts.OpenDocument()) { uri ->
        if (uri != null) vm.loadTextFile(uri) { t ->
            text = if (t.length > MAX_CHARS) t.take(MAX_CHARS).also { Toast.makeText(ctx, "100,000자까지만 불러왔습니다.", Toast.LENGTH_LONG).show() } else t
            settings.draftText = text
        }
    }

    Column(modifier.fillMaxSize().verticalScroll(rememberScrollState()).padding(16.dp), verticalArrangement = Arrangement.spacedBy(12.dp)) {
        Text("텍스트를 목소리로", style = MaterialTheme.typography.headlineSmall, fontWeight = FontWeight.Bold)

        // voice picker
        var expanded by remember { mutableStateOf(false) }
        val selected = voices.find { it.id == voiceId }
        ExposedDropdownMenuBox(expanded = expanded, onExpandedChange = { expanded = it }) {
            OutlinedTextField(
                value = selected?.name ?: "기본 음성 (MeloTTS 한국어, 복제 없음)",
                onValueChange = {}, readOnly = true, label = { Text("목소리") },
                trailingIcon = { ExposedDropdownMenuDefaults.TrailingIcon(expanded) },
                modifier = Modifier.menuAnchor(MenuAnchorType.PrimaryNotEditable).fillMaxWidth(),
            )
            ExposedDropdownMenu(expanded = expanded, onDismissRequest = { expanded = false }) {
                DropdownMenuItem(text = { Text("기본 음성 (복제 없음)") }, onClick = { voiceId = null; settings.lastVoiceId = null; expanded = false })
                voices.forEach { v ->
                    DropdownMenuItem(text = { Text("${v.name}  · 품질 ${v.score}") }, onClick = { voiceId = v.id; settings.lastVoiceId = v.id; expanded = false })
                }
                DropdownMenuItem(text = { Text("+ 새 목소리 추가…") }, onClick = { expanded = false; onGoVoices() })
            }
        }

        OutlinedTextField(value = title, onValueChange = { title = it.take(60) }, label = { Text("제목 (파일 이름, 선택)") },
            singleLine = true, modifier = Modifier.fillMaxWidth())

        OutlinedTextField(
            value = text,
            onValueChange = { v ->
                text = if (v.length > MAX_CHARS) v.take(MAX_CHARS) else v
                settings.draftText = text
            },
            label = { Text("읽을 글 (한국어)") },
            placeholder = { Text("여기에 글을 붙여넣거나 TXT 파일을 불러오세요.\n숫자·날짜·영어·한자도 자연스럽게 읽어 드립니다.") },
            modifier = Modifier.fillMaxWidth().heightIn(min = 180.dp, max = 360.dp),
            supportingText = {
                val chunks = text.length
                Text("${"%,d".format(chunks)} / ${"%,d".format(MAX_CHARS)}자" + estimateLabel(text.length))
            },
        )
        Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
            OutlinedButton(onClick = { openText.launch(arrayOf("text/plain", "text/*")) }) { Icon(Icons.Filled.FileOpen, null); Spacer(Modifier.width(4.dp)); Text("TXT") }
            OutlinedButton(onClick = {
                clipboard.getText()?.text?.let { text = (text + it).take(MAX_CHARS); settings.draftText = text }
            }) { Icon(Icons.Filled.ContentPaste, null); Spacer(Modifier.width(4.dp)); Text("붙여넣기") }
            OutlinedButton(onClick = { text = ""; settings.draftText = "" }) { Icon(Icons.Filled.Delete, null); Spacer(Modifier.width(4.dp)); Text("지우기") }
        }

        // settings
        Card(Modifier.fillMaxWidth()) {
            Column(Modifier.padding(12.dp)) {
                Row(verticalAlignment = Alignment.CenterVertically) {
                    Text("음성 설정", style = MaterialTheme.typography.titleMedium, modifier = Modifier.weight(1f))
                    IconButton(onClick = { showSettings = !showSettings }) { Icon(if (showSettings) Icons.Filled.ExpandLess else Icons.Filled.ExpandMore, null) }
                }
                if (showSettings) {
                    LabeledSlider("말하기 속도", speed, 0.7f..1.5f, "%.2fx") { speed = it }
                    LabeledSlider("감정·억양 변화", noise, 0.2f..0.9f, "%.2f") { noise = it }
                    LabeledSlider("리듬 다양성", noiseW, 0.3f..1.2f, "%.2f") { noiseW = it }
                    LabeledSlider("음색 강조 (목소리 닮음 정도)", timbre, 1.0f..1.5f, "%.2f") { timbre = it }
                    LabeledSlider("문장 사이 쉼", pause, 0.5f..2.0f, "%.1fx") { pause = it; settings.pauseScale = it }
                    Text("MP3 음질", style = MaterialTheme.typography.labelLarge)
                    FlowRow(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                        listOf(48, 64, 96, 128).forEach { b ->
                            FilterChip(selected = bitrate == b, onClick = { bitrate = b; settings.bitrate = b }, label = { Text("${b}kbps") })
                        }
                    }
                    Text("성능 모드", style = MaterialTheme.typography.labelLarge)
                    FlowRow(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                        PerformanceMode.entries.forEach { m ->
                            FilterChip(selected = perf == m, onClick = { perf = m; settings.performanceMode = m }, label = { Text(m.label) })
                        }
                    }
                    Text(perf.description, style = MaterialTheme.typography.bodySmall)
                    TextButton(onClick = { speed = 1f; noise = 0.6f; noiseW = 0.8f; timbre = 1.2f; pause = 1f; settings.pauseScale = 1f }) { Text("기본값으로") }
                }
            }
        }

        val busy = AppGraph.jobs.isBusy
        Row(horizontalArrangement = Arrangement.spacedBy(8.dp), verticalAlignment = Alignment.CenterVertically) {
            OutlinedButton(enabled = text.isNotBlank() && preview !is MainViewModel.PreviewState.Working,
                onClick = { vm.preview(text, voiceId, params()) }) {
                if (preview is MainViewModel.PreviewState.Working) CircularProgressIndicator(Modifier.size(18.dp), strokeWidth = 2.dp)
                else Icon(Icons.Filled.PlayArrow, null)
                Spacer(Modifier.width(4.dp)); Text("미리듣기")
            }
            Button(enabled = text.isNotBlank() && !busy, onClick = {
                val t = title.ifBlank { text.trim().lineSequence().first().take(20) }
                SynthesisService.start(ctx, SynthesisRequest(text, t, voiceId, params(), bitrate, pause))
            }, modifier = Modifier.weight(1f)) { Text("MP3 만들기") }
        }
        when (val pv = preview) {
            is MainViewModel.PreviewState.Ready -> Row(verticalAlignment = Alignment.CenterVertically) {
                Text("미리듣기 준비됨 (처리 속도 ${"%.2f".format(pv.rtf)}× 실시간)", style = MaterialTheme.typography.bodySmall, modifier = Modifier.weight(1f))
                IconButton(onClick = { Player.toggle(pv.file) }) { Icon(if (playing == pv.file.path) Icons.Filled.Stop else Icons.Filled.PlayArrow, null) }
            }
            is MainViewModel.PreviewState.Error -> Text("미리듣기 오류: ${pv.message}", color = MaterialTheme.colorScheme.error)
            else -> {}
        }

        JobCard(job, playing, onGoLibrary)
        Text(
            "※ 본인 또는 동의를 받은 사람의 목소리만 사용하세요. 생성된 MP3에는 'AI 생성 음성' 정보가 기록됩니다.",
            style = MaterialTheme.typography.bodySmall,
        )
    }
}

private fun estimateLabel(chars: Int): String {
    if (chars < 200) return ""
    val audioMin = chars / 7.0 / 60.0 // Korean read speech ≈ 7 characters per second
    return " · 예상 음성 길이 약 ${if (audioMin >= 60) "%.1f시간".format(audioMin / 60) else "%.0f분".format(audioMin.coerceAtLeast(1.0))}"
}

@Composable
fun LabeledSlider(label: String, value: Float, range: ClosedFloatingPointRange<Float>, fmt: String, onChange: (Float) -> Unit) {
    Column {
        Row { Text(label, style = MaterialTheme.typography.labelLarge, modifier = Modifier.weight(1f)); Text(fmt.format(value), style = MaterialTheme.typography.labelLarge) }
        Slider(value = value, onValueChange = onChange, valueRange = range)
    }
}

@Composable
private fun JobCard(job: JobState, playing: String?, onGoLibrary: () -> Unit) {
    val ctx = LocalContext.current
    when (job) {
        JobState.Idle -> {}
        is JobState.Preparing -> Card(Modifier.fillMaxWidth()) {
            Column(Modifier.padding(16.dp)) { Text(job.message); LinearProgressIndicator(Modifier.fillMaxWidth().padding(top = 8.dp)) }
        }
        is JobState.Running -> Card(Modifier.fillMaxWidth()) {
            val p = job.progress
            Column(Modifier.padding(16.dp), verticalArrangement = Arrangement.spacedBy(6.dp)) {
                Text("생성 중: ${job.title}", fontWeight = FontWeight.Bold)
                LinearProgressIndicator(progress = { if (p.total == 0) 0f else p.done.toFloat() / p.total }, modifier = Modifier.fillMaxWidth())
                Text("${p.done} / ${p.total} 문장" + (if (p.etaMs > 0) " · 남은 시간 약 ${formatDuration(p.etaMs)}" else " · 남은 시간 계산 중…"))
                if (p.currentText.isNotEmpty()) Text("“${p.currentText}”", maxLines = 2, overflow = TextOverflow.Ellipsis, style = MaterialTheme.typography.bodySmall)
                Text("화면을 꺼도 계속됩니다. 중단해도 다음에 같은 글로 다시 시작하면 이어서 진행됩니다.", style = MaterialTheme.typography.bodySmall)
                OutlinedButton(onClick = { SynthesisService.cancel(ctx) }) { Text("취소") }
            }
        }
        is JobState.Finalizing -> Card(Modifier.fillMaxWidth()) {
            Column(Modifier.padding(16.dp)) { Text("MP3로 합치는 중…"); LinearProgressIndicator(Modifier.fillMaxWidth().padding(top = 8.dp)) }
        }
        is JobState.Done -> Card(Modifier.fillMaxWidth(), colors = CardDefaults.cardColors(containerColor = MaterialTheme.colorScheme.secondaryContainer)) {
            Column(Modifier.padding(16.dp), verticalArrangement = Arrangement.spacedBy(6.dp)) {
                Text("완료: ${job.title}", fontWeight = FontWeight.Bold)
                Text("길이 ${formatDuration((job.durationSec * 1000).toLong())} · 소요 ${formatDuration(job.elapsedMs)}" +
                    if (job.savedToMusic) "\n음악/KoreanVoice 폴더에도 저장되었습니다." else "")
                Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                    Button(onClick = { Player.toggle(job.file) }) { Text(if (playing == job.file.path) "정지" else "재생") }
                    OutlinedButton(onClick = { Sharing.share(ctx, job.file) }) { Icon(Icons.Filled.Share, null); Text("공유") }
                    TextButton(onClick = onGoLibrary) { Text("보관함") }
                }
            }
        }
        is JobState.Failed -> Card(Modifier.fillMaxWidth(), colors = CardDefaults.cardColors(containerColor = MaterialTheme.colorScheme.errorContainer)) {
            Column(Modifier.padding(16.dp)) {
                Text("실패: ${job.message}")
                TextButton(onClick = { AppGraph.jobs.reset() }) { Text("닫기") }
            }
        }
        JobState.Cancelled -> Card(Modifier.fillMaxWidth()) {
            Column(Modifier.padding(16.dp)) {
                Text("취소되었습니다. 같은 글로 다시 시작하면 완료된 문장은 건너뜁니다.")
                TextButton(onClick = { AppGraph.jobs.reset() }) { Text("닫기") }
            }
        }
    }
}
