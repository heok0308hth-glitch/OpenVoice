package io.github.koreanvoice.app

import android.net.Uri
import android.widget.Toast
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
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
import androidx.compose.material.icons.filled.Stop
import androidx.compose.material3.AlertDialog
import androidx.compose.material3.AssistChip
import androidx.compose.material3.Button
import androidx.compose.material3.Card
import androidx.compose.material3.CircularProgressIndicator
import androidx.compose.material3.DropdownMenu
import androidx.compose.material3.DropdownMenuItem
import androidx.compose.material3.ExperimentalMaterial3Api
import androidx.compose.material3.ExposedDropdownMenuBox
import androidx.compose.material3.ExposedDropdownMenuDefaults
import androidx.compose.material3.FilterChip
import androidx.compose.material3.Icon
import androidx.compose.material3.IconButton
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.MenuAnchorType
import androidx.compose.material3.OutlinedButton
import androidx.compose.material3.OutlinedTextField
import androidx.compose.material3.Text
import androidx.compose.material3.TextButton
import androidx.compose.runtime.Composable
import androidx.compose.runtime.collectAsState
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableIntStateOf
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.saveable.rememberSaveable
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.platform.LocalClipboardManager
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.text.TextRange
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.input.TextFieldValue
import androidx.compose.ui.unit.dp
import io.github.koreanvoice.audio.Dsp
import io.github.koreanvoice.expressive.Emotion
import io.github.koreanvoice.expressive.Markup

const val MAX_CHARS = 100_000

private val BACKGROUNDS = listOf("없음", "빗소리", "바람", "실내", "백색소음")

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

    var field by remember { mutableStateOf(TextFieldValue(settings.draftText)) }
    var title by rememberSaveable { mutableStateOf("") }
    var voiceId by rememberSaveable { mutableStateOf(settings.lastVoiceId) }
    val usable = voices.filter { it.segments > 0 }
    if (voiceId == null || usable.none { it.id == voiceId }) voiceId = usable.firstOrNull()?.id
    var showSettings by rememberSaveable { mutableStateOf(false) }
    var showHelp by rememberSaveable { mutableStateOf(false) }
    var showDict by remember { mutableStateOf(false) }
    var d by remember { mutableStateOf(settings.delivery) }
    var bitrate by remember { mutableIntStateOf(settings.bitrate) }
    var perf by remember { mutableStateOf(settings.performanceMode) }
    var hq by remember { mutableStateOf(settings.highQualityDecoder) }
    var bgUri by remember { mutableStateOf<Uri?>(null) }
    fun upd(n: DeliverySettings) { d = n; settings.delivery = n }
    fun setText(v: TextFieldValue) {
        field = if (v.text.length > MAX_CHARS) v.copy(text = v.text.take(MAX_CHARS)) else v
        settings.draftText = field.text
    }
    fun insert(tag: String) {
        val t = field.text; val s = field.selection.min; val e = field.selection.max
        setText(TextFieldValue(t.substring(0, s) + tag + t.substring(e), TextRange(s + tag.length)))
    }

    val openText = rememberLauncherForActivityResult(ActivityResultContracts.OpenDocument()) { uri ->
        if (uri != null) vm.loadTextFile(uri) { t ->
            if (t.length > MAX_CHARS) Toast.makeText(ctx, "100,000자까지만 불러왔습니다.", Toast.LENGTH_LONG).show()
            setText(TextFieldValue(t.take(MAX_CHARS)))
        }
    }
    val openBg = rememberLauncherForActivityResult(ActivityResultContracts.OpenDocument()) { uri -> if (uri != null) { bgUri = uri; upd(d.copy(background = "파일")) } }

    Column(modifier.fillMaxSize().verticalScroll(rememberScrollState()).padding(16.dp), verticalArrangement = Arrangement.spacedBy(12.dp)) {
        Text("텍스트를 목소리로", style = MaterialTheme.typography.headlineSmall, fontWeight = FontWeight.Bold)

        // voice picker
        var expanded by remember { mutableStateOf(false) }
        val selected = voices.find { it.id == voiceId }
        ExposedDropdownMenuBox(expanded = expanded, onExpandedChange = { expanded = it }) {
            OutlinedTextField(
                value = selected?.let { "${it.name} · ${"%.0f".format(it.seconds)}초 학습" + if (it.autoLearnRuns > 0) " · 자동학습 ${it.autoLearnRuns}회" else "" }
                    ?: "목소리를 먼저 만들어 주세요",
                onValueChange = {}, readOnly = true, label = { Text("목소리") },
                trailingIcon = { ExposedDropdownMenuDefaults.TrailingIcon(expanded) },
                modifier = Modifier.menuAnchor(MenuAnchorType.PrimaryNotEditable).fillMaxWidth(),
            )
            ExposedDropdownMenu(expanded = expanded, onDismissRequest = { expanded = false }) {
                usable.forEach { v ->
                    DropdownMenuItem(text = { Text("${v.name} · ${"%.0f".format(v.seconds)}초") }, onClick = { voiceId = v.id; settings.lastVoiceId = v.id; expanded = false })
                }
                DropdownMenuItem(text = { Text("+ 새 목소리 만들기…") }, onClick = { expanded = false; onGoVoices() })
            }
        }

        OutlinedTextField(value = title, onValueChange = { title = it.take(60) }, label = { Text("제목 (파일 이름, 선택)") },
            singleLine = true, modifier = Modifier.fillMaxWidth())

        OutlinedTextField(
            value = field, onValueChange = { setText(it) },
            label = { Text("읽을 글 (한국어)") },
            placeholder = { Text("글을 붙여넣거나 TXT 파일을 불러오세요.\n[기쁨], [쉼:1], [울림:홀], {NASA|나사} 같은 태그로 부분마다 읽는 법을 지정할 수 있습니다.") },
            modifier = Modifier.fillMaxWidth().heightIn(min = 180.dp, max = 380.dp),
            supportingText = { Text("${"%,d".format(field.text.length)} / ${"%,d".format(MAX_CHARS)}자" + estimateLabel(field.text.length)) },
        )
        FlowRow(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
            OutlinedButton(onClick = { openText.launch(arrayOf("text/plain", "text/*")) }) { Icon(Icons.Filled.FileOpen, null); Spacer(Modifier.width(4.dp)); Text("TXT") }
            OutlinedButton(onClick = { clipboard.getText()?.text?.let { insert(it) } }) { Icon(Icons.Filled.ContentPaste, null); Spacer(Modifier.width(4.dp)); Text("붙여넣기") }
            OutlinedButton(onClick = { setText(TextFieldValue("")) }) { Icon(Icons.Filled.Delete, null); Spacer(Modifier.width(4.dp)); Text("지우기") }
        }

        // professional tag toolbar
        Card(Modifier.fillMaxWidth()) {
            Column(Modifier.padding(12.dp), verticalArrangement = Arrangement.spacedBy(6.dp)) {
                Row(verticalAlignment = Alignment.CenterVertically) {
                    Text("부분 지정 태그 (커서 위치에 삽입)", style = MaterialTheme.typography.titleSmall, modifier = Modifier.weight(1f))
                    TextButton(onClick = { showHelp = !showHelp }) { Text(if (showHelp) "설명 닫기" else "설명") }
                }
                FlowRow(horizontalArrangement = Arrangement.spacedBy(6.dp)) {
                    TagMenu("감정", Emotion.entries.filter { it != Emotion.NEUTRAL }.flatMap { e -> listOf("[${e.label}]" to e.label, "[${e.label}:강]" to "${e.label} (강하게)", "[${e.label}:약]" to "${e.label} (살짝)") }, ::insert)
                    TagMenu("속도", listOf("[속도:빠르게]" to "빠르게", "[속도:느리게]" to "느리게", "[속도:1.2]" to "1.2배", "[속도:0.8]" to "0.8배", "[/속도]" to "원래대로"), ::insert)
                    TagMenu("높이", listOf("[높이:높게]" to "높게", "[높이:낮게]" to "낮게", "[높이:+1]" to "+1 반음", "[높이:-1]" to "-1 반음", "[/높이]" to "원래대로"), ::insert)
                    TagMenu("쉼", listOf("[쉼:0.5]" to "0.5초", "[쉼:1]" to "1초", "[쉼:2]" to "2초", "[쉼:3]" to "3초"), ::insert)
                    TagMenu("울림", Dsp.Room.entries.filter { it != Dsp.Room.NONE }.map { "[울림:${it.label}]" to it.label } + listOf("[울림:없음]" to "없음"), ::insert)
                    TagMenu("상태", Dsp.State.entries.filter { it != Dsp.State.NORMAL }.map { "[${it.label}]" to it.label } + listOf("[/상태]" to "원래대로"), ::insert)
                    TagMenu("톤", Dsp.Tone.entries.map { "[톤:${it.label}]" to it.label }, ::insert)
                    TagMenu("소리(실험)", Markup.NONVERBAL.keys.map { "[$it]" to it }, ::insert)
                    AssistChip(onClick = { insert("{표기|읽는법}") }, label = { Text("읽는법") })
                    AssistChip(onClick = { insert("[기본]") }, label = { Text("초기화") })
                }
                if (showHelp) Text(MARKUP_HELP, style = MaterialTheme.typography.bodySmall)
            }
        }

        // delivery settings
        Card(Modifier.fillMaxWidth()) {
            Column(Modifier.padding(12.dp), verticalArrangement = Arrangement.spacedBy(4.dp)) {
                Row(verticalAlignment = Alignment.CenterVertically) {
                    Text("감정·연출 설정", style = MaterialTheme.typography.titleMedium, modifier = Modifier.weight(1f))
                    IconButton(onClick = { showSettings = !showSettings }) { Icon(if (showSettings) Icons.Filled.ExpandLess else Icons.Filled.ExpandMore, null) }
                }
                Text("기본 감정: ${d.emotion.label}${if (d.autoEmotion) " + 문장별 자동" else ""} · 울림 ${d.room.label} · 상태 ${d.state.label} · 배경 ${d.background}",
                    style = MaterialTheme.typography.bodySmall)
                if (showSettings) {
                    Text("기본 감정", style = MaterialTheme.typography.labelLarge)
                    FlowRow(horizontalArrangement = Arrangement.spacedBy(6.dp)) {
                        Emotion.entries.forEach { e -> FilterChip(selected = d.emotion == e, onClick = { upd(d.copy(emotion = e)) }, label = { Text(e.label) }) }
                    }
                    LabeledSlider("감정 강도", d.intensity, 0.5f..1.6f, "%.1f") { upd(d.copy(intensity = it)) }
                    LabeledSwitch("문장별 자동 감정", "대사(따옴표)·문장 부호·감정 단어를 보고 문장마다 감정을 자동으로 정합니다. 태그로 지정한 부분은 그대로 둡니다.", d.autoEmotion) { upd(d.copy(autoEmotion = it)) }
                    LabeledSwitch("감정에 맞는 원본 구간 사용", "학습된 음성 중 감정에 어울리는 구간(활기찬/차분한)을 골라 참조합니다.", d.emotionRouting) { upd(d.copy(emotionRouting = it)) }
                    LabeledSwitch("자동 학습 결과 적용", "자동 학습으로 찾은 음높이·음색·울림 보정을 적용합니다.", d.useTuning) { upd(d.copy(useTuning = it)) }
                    LabeledSlider("말하기 속도", d.tempo, 0.6f..1.6f, "%.2fx") { upd(d.copy(tempo = it)) }
                    LabeledSlider("음높이 (반음)", d.pitch, -6f..6f, "%+.1f") { upd(d.copy(pitch = it)) }
                    Text("울림 (공간)", style = MaterialTheme.typography.labelLarge)
                    FlowRow(horizontalArrangement = Arrangement.spacedBy(6.dp)) {
                        Dsp.Room.entries.forEach { r -> FilterChip(selected = d.room == r, onClick = { upd(d.copy(room = r, wet = -1f)) }, label = { Text(r.label) }) }
                    }
                    if (d.room != Dsp.Room.NONE) LabeledSlider("울림 양", if (d.wet < 0) d.room.defaultWet.toFloat() else d.wet, 0.02f..0.8f, "%.2f") { upd(d.copy(wet = it)) }
                    Text("상태", style = MaterialTheme.typography.labelLarge)
                    FlowRow(horizontalArrangement = Arrangement.spacedBy(6.dp)) {
                        Dsp.State.entries.forEach { s -> FilterChip(selected = d.state == s, onClick = { upd(d.copy(state = s)) }, label = { Text(s.label) }) }
                    }
                    Text("음색 톤", style = MaterialTheme.typography.labelLarge)
                    FlowRow(horizontalArrangement = Arrangement.spacedBy(6.dp)) {
                        Dsp.Tone.entries.forEach { t -> FilterChip(selected = d.tone == t, onClick = { upd(d.copy(tone = t)) }, label = { Text(t.label) }) }
                    }
                    Text("배경", style = MaterialTheme.typography.labelLarge)
                    FlowRow(horizontalArrangement = Arrangement.spacedBy(6.dp)) {
                        BACKGROUNDS.forEach { b -> FilterChip(selected = d.background == b, onClick = { bgUri = null; upd(d.copy(background = b)) }, label = { Text(b) }) }
                        FilterChip(selected = d.background == "파일" && bgUri != null, onClick = { openBg.launch(arrayOf("audio/*", "video/*")) }, label = { Text("음악/파일…") })
                    }
                    if (d.background != "없음") {
                        LabeledSlider("배경 음량 (dB)", d.backgroundDb, -40f..-8f, "%.0f") { upd(d.copy(backgroundDb = it)) }
                        Text("말할 때는 배경이 자동으로 작아집니다(더킹).", style = MaterialTheme.typography.bodySmall)
                    }
                    LabeledSwitch("방송용 마스터링", "음량을 고르게 압축해 또렷하게 들리게 합니다.", d.mastering) { upd(d.copy(mastering = it)) }
                    LabeledSlider("문장 사이 쉼", d.pauseScale, 0.5f..2.0f, "%.1fx") { upd(d.copy(pauseScale = it)) }
                    Text("MP3 음질", style = MaterialTheme.typography.labelLarge)
                    FlowRow(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                        listOf(64, 96, 128, 192).forEach { b -> FilterChip(selected = bitrate == b, onClick = { bitrate = b; settings.bitrate = b }, label = { Text("${b}kbps") }) }
                    }
                    if (AppGraph.models.hasHqDecoder()) LabeledSwitch("고품질 디코더", "음질이 약간 더 좋지만 디코딩이 약 5배 느립니다.", hq) { hq = it; settings.highQualityDecoder = it }
                    Text("성능 모드", style = MaterialTheme.typography.labelLarge)
                    FlowRow(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                        PerformanceMode.entries.forEach { m -> FilterChip(selected = perf == m, onClick = { perf = m; settings.performanceMode = m }, label = { Text(m.label) }) }
                    }
                    Row {
                        TextButton(onClick = { showDict = true }) { Text("발음 사전…") }
                        TextButton(onClick = { upd(DeliverySettings()); bgUri = null }) { Text("기본값으로") }
                    }
                }
            }
        }

        val busy = AppGraph.jobs.isBusy
        Row(horizontalArrangement = Arrangement.spacedBy(8.dp), verticalAlignment = Alignment.CenterVertically) {
            OutlinedButton(enabled = field.text.isNotBlank() && preview !is MainViewModel.PreviewState.Working && !busy,
                onClick = { vm.preview(field.text, voiceId, d) }) {
                if (preview is MainViewModel.PreviewState.Working) CircularProgressIndicator(Modifier.size(18.dp), strokeWidth = 2.dp)
                else Icon(Icons.Filled.PlayArrow, null)
                Spacer(Modifier.width(4.dp)); Text("미리듣기")
            }
            Button(enabled = field.text.isNotBlank() && !busy && voiceId != null, onClick = {
                val t = title.ifBlank { Markup.applyReadings(field.text).replace(Regex("\\[[^\\]]*\\]"), "").trim().lineSequence().first().take(20) }
                SynthesisService.start(ctx, TaskRequest.Synthesis(field.text, t, voiceId!!, d, bitrate, if (d.background == "파일") bgUri else null, settings.dictionary))
            }, modifier = Modifier.weight(1f)) { Text("MP3 만들기") }
        }
        when (val pv = preview) {
            is MainViewModel.PreviewState.Working -> Text(pv.message, style = MaterialTheme.typography.bodySmall)
            is MainViewModel.PreviewState.Ready -> Row(verticalAlignment = Alignment.CenterVertically) {
                Text("미리듣기 준비됨 (처리 시간 = 음성 길이의 ${"%.1f".format(pv.rtf)}배)", style = MaterialTheme.typography.bodySmall, modifier = Modifier.weight(1f))
                IconButton(onClick = { Player.toggle(pv.file) }) { Icon(if (playing == pv.file.path) Icons.Filled.Stop else Icons.Filled.PlayArrow, null) }
            }
            is MainViewModel.PreviewState.Error -> Text("미리듣기 오류: ${pv.message}", color = MaterialTheme.colorScheme.error)
            else -> {}
        }
        JobCard(job, playing, onGoLibrary, setOf(TaskKind.SYNTHESIS))
        Text("※ 본인 또는 동의를 받은 사람의 목소리만 사용하세요. 생성된 MP3에는 'AI 생성 음성' 정보가 기록됩니다.", style = MaterialTheme.typography.bodySmall)
    }

    if (showDict) {
        var txt by remember { mutableStateOf(settings.dictionaryText) }
        AlertDialog(onDismissRequest = { showDict = false }, title = { Text("발음 사전") },
            text = {
                Column {
                    Text("한 줄에 하나씩 ‘표기=읽는법’ 형식으로 적으면 글 전체에 적용됩니다.\n예) GPT=지피티\n예) 1등=일등", style = MaterialTheme.typography.bodySmall)
                    OutlinedTextField(txt, { txt = it }, modifier = Modifier.fillMaxWidth().heightIn(min = 160.dp))
                }
            },
            confirmButton = { TextButton(onClick = { settings.dictionaryText = txt; showDict = false }) { Text("저장") } },
            dismissButton = { TextButton(onClick = { showDict = false }) { Text("취소") } })
    }
}

@Composable
private fun TagMenu(label: String, items: List<Pair<String, String>>, onPick: (String) -> Unit) {
    var open by remember { mutableStateOf(false) }
    Box {
        AssistChip(onClick = { open = true }, label = { Text(label) })
        DropdownMenu(expanded = open, onDismissRequest = { open = false }) {
            items.forEach { (tag, name) -> DropdownMenuItem(text = { Text("$name   $tag") }, onClick = { onPick(tag); open = false }) }
        }
    }
}

private const val MARKUP_HELP = """• 감정: [기쁨] [슬픔] [화남] [차분] [신남] [진지] [다정] [놀람] [긴장] [두려움] — 다음 태그가 나올 때까지 유지. [기쁨:강] [기쁨:약] 처럼 강도 지정
• 속도·높이: [속도:1.2] [속도:느리게] [높이:+2] [높이:낮게]
• 쉼: [쉼:1.5] (초)
• 울림: [울림:방|스튜디오|욕실|무대|홀|성당|동굴|없음], 양 지정 [울림:홀:0.4]
• 상태: [전화] [라디오] [확성기] [멀리] [가까이] [속삭임] [물속] [로봇]
• 톤: [톤:밝게|따뜻하게|선명하게|어둡게]
• 읽는 법: {NASA|나사}, {3.1절|삼일절}
• 되돌리기: [/감정] [/속도] [/상태] … 또는 [기본]
• 소리(실험적): [웃음] [한숨] [헛기침] [숨] [흐느낌] — 모델이 항상 소리를 내지는 않습니다."""

private fun estimateLabel(chars: Int): String {
    if (chars < 200) return ""
    val audioMin = chars / 7.0 / 60.0 // Korean read speech ≈ 7 characters per second
    return " · 예상 음성 길이 약 ${if (audioMin >= 60) "%.1f시간".format(audioMin / 60) else "%.0f분".format(audioMin.coerceAtLeast(1.0))}"
}
