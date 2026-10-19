package io.github.koreanvoice.app

import android.Manifest
import android.content.pm.PackageManager
import android.net.Uri
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
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.verticalScroll
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.Add
import androidx.compose.material.icons.filled.AudioFile
import androidx.compose.material.icons.filled.AutoAwesome
import androidx.compose.material.icons.filled.Delete
import androidx.compose.material.icons.filled.Edit
import androidx.compose.material.icons.filled.Mic
import androidx.compose.material.icons.filled.PlayArrow
import androidx.compose.material.icons.filled.RecordVoiceOver
import androidx.compose.material.icons.filled.Stop
import androidx.compose.material.icons.filled.VideoFile
import androidx.compose.material3.AlertDialog
import androidx.compose.material3.Button
import androidx.compose.material3.Card
import androidx.compose.material3.Checkbox
import androidx.compose.material3.ExtendedFloatingActionButton
import androidx.compose.material3.FilterChip
import androidx.compose.material3.HorizontalDivider
import androidx.compose.material3.Icon
import androidx.compose.material3.IconButton
import androidx.compose.material3.LinearProgressIndicator
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedButton
import androidx.compose.material3.OutlinedTextField
import androidx.compose.material3.Text
import androidx.compose.material3.TextButton
import androidx.compose.runtime.Composable
import androidx.compose.runtime.collectAsState
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.rememberCoroutineScope
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.core.content.ContextCompat
import io.github.koreanvoice.expressive.AutoTuner
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext

@Composable
fun VoicesScreen(vm: MainViewModel, modifier: Modifier, onGoLibrary: () -> Unit) {
    val voices by AppGraph.voices.voices.collectAsState()
    val job by AppGraph.jobs.state.collectAsState()
    val playing by Player.playing.collectAsState()
    val preview by vm.preview.collectAsState()
    var adding by remember { mutableStateOf<String?>(null) }      // "" = new voice, id = add material
    var detail by remember { mutableStateOf<String?>(null) }
    var autoFor by remember { mutableStateOf<VoiceInfo?>(null) }
    var renaming by remember { mutableStateOf<VoiceInfo?>(null) }
    var deleting by remember { mutableStateOf<VoiceInfo?>(null) }
    val busy = AppGraph.jobs.isBusy

    Box(modifier.fillMaxSize()) {
        LazyColumn(Modifier.fillMaxSize().padding(horizontal = 16.dp), verticalArrangement = Arrangement.spacedBy(10.dp)) {
            item {
                Text("목소리", style = MaterialTheme.typography.headlineSmall, fontWeight = FontWeight.Bold, modifier = Modifier.padding(top = 16.dp))
                Text("음성 파일·영상·녹음으로 목소리를 학습합니다. 자료를 더 넣을수록(추가 학습), 자동 학습을 돌릴수록 원본에 가까워집니다.",
                    style = MaterialTheme.typography.bodyMedium)
            }
            item { JobCard(job, playing, onGoLibrary, setOf(TaskKind.LEARN, TaskKind.AUTOLEARN)) }
            if (voices.isEmpty()) item {
                Card(Modifier.fillMaxWidth()) {
                    Column(Modifier.padding(16.dp)) {
                        Icon(Icons.Filled.RecordVoiceOver, null)
                        Text("아직 등록된 목소리가 없습니다. ‘목소리 만들기’를 눌러 음성/영상 파일(내 파일, Google 드라이브 등)을 고르거나 직접 녹음하세요.")
                    }
                }
            }
            items(voices, key = { it.id }) { v ->
                Card(Modifier.fillMaxWidth()) {
                    Column(Modifier.padding(16.dp), verticalArrangement = Arrangement.spacedBy(4.dp)) {
                        Text(v.name, style = MaterialTheme.typography.titleMedium)
                        Text("학습 음성 ${"%.0f".format(v.seconds)}초 · 구간 ${v.segments}개 · 학습 ${v.generation}회" +
                            if (v.autoLearnRuns > 0) " · 자동학습 ${v.autoLearnRuns}회" else "", style = MaterialTheme.typography.bodySmall)
                        if (v.scoreHistory.size >= 2) {
                            Text("최근 자동 학습: ${"%.3f".format(v.scoreHistory.first())} → ${"%.3f".format(v.scoreHistory.maxOrNull() ?: 0.0)}", style = MaterialTheme.typography.bodySmall)
                            ScoreChart(v.scoreHistory.drop(1), v.scoreHistory.first())
                        }
                        if (v.generations.isNotEmpty()) GenerationList(vm, v, playing)
                        if (v.seconds in 0.1..15.0) Text("• 음성이 짧습니다. 20초~2분 분량을 더 넣으면 더 닮아집니다.", style = MaterialTheme.typography.bodySmall, color = MaterialTheme.colorScheme.error)
                        FlowRow(horizontalArrangement = Arrangement.spacedBy(4.dp)) {
                            TextButton(enabled = !busy && preview !is MainViewModel.PreviewState.Working && v.segments > 0, onClick = { vm.testVoice(v.id) }) {
                                Icon(Icons.Filled.PlayArrow, null); Text("들어보기")
                            }
                            TextButton(enabled = !busy, onClick = { adding = v.id }) { Icon(Icons.Filled.Add, null); Text("추가 학습") }
                            TextButton(enabled = !busy && v.segments > 0, onClick = { autoFor = v }) { Icon(Icons.Filled.AutoAwesome, null); Text("자동 학습") }
                            TextButton(onClick = { detail = v.id }) { Text("구간 보기") }
                            IconButton(onClick = { renaming = v }) { Icon(Icons.Filled.Edit, "이름 변경") }
                            IconButton(enabled = !busy, onClick = { deleting = v }) { Icon(Icons.Filled.Delete, "삭제") }
                        }
                    }
                }
            }
            item {
                when (val pv = preview) {
                    is MainViewModel.PreviewState.Working -> Text(pv.message, style = MaterialTheme.typography.bodySmall)
                    is MainViewModel.PreviewState.Error -> Text("오류: ${pv.message}", color = MaterialTheme.colorScheme.error)
                    else -> {}
                }
                Spacer(Modifier.padding(40.dp))
            }
        }
        ExtendedFloatingActionButton(
            onClick = { if (!busy) adding = "" },
            icon = { Icon(Icons.Filled.Add, null) }, text = { Text("목소리 만들기") },
            modifier = Modifier.align(Alignment.BottomEnd).padding(16.dp),
        )
    }

    adding?.let { target -> LearnDialog(vm, target.ifEmpty { null }) { adding = null } }
    autoFor?.let { v -> AutoLearnDialog(v) { autoFor = null } }
    detail?.let { id -> SegmentsDialog(vm, id, playing) { detail = null } }
    renaming?.let { v ->
        var name by remember(v.id) { mutableStateOf(v.name) }
        AlertDialog(onDismissRequest = { renaming = null }, title = { Text("이름 변경") },
            text = { OutlinedTextField(name, { name = it.take(30) }, singleLine = true) },
            confirmButton = { TextButton(onClick = { AppGraph.voices.rename(v.id, name); renaming = null }) { Text("저장") } },
            dismissButton = { TextButton(onClick = { renaming = null }) { Text("취소") } })
    }
    deleting?.let { v ->
        AlertDialog(onDismissRequest = { deleting = null }, title = { Text("목소리 삭제") }, text = { Text("‘${v.name}’과(와) 학습된 자료를 모두 삭제할까요?") },
            confirmButton = { TextButton(onClick = { AppGraph.voices.delete(v.id); deleting = null }) { Text("삭제") } },
            dismissButton = { TextButton(onClick = { deleting = null }) { Text("취소") } })
    }
}

/** New voice ([voiceId] = null) or additional material for an existing one. */
@OptIn(ExperimentalLayoutApi::class)
@Composable
private fun LearnDialog(vm: MainViewModel, voiceId: String?, onClose: () -> Unit) {
    val ctx = LocalContext.current
    var consent by remember { mutableStateOf(AppGraph.settings.consentAccepted) }
    var name by remember { mutableStateOf("") }
    var uris by remember { mutableStateOf(listOf<Uri>()) }
    var recorded by remember { mutableStateOf<Pair<FloatArray, Int>?>(null) }
    var recording by remember { mutableStateOf(false) }
    val recorder = remember { VoiceRecorder() }
    val recSec by recorder.seconds.collectAsState()
    val recLevel by recorder.level.collectAsState()
    val scope = rememberCoroutineScope()
    val existing = voiceId?.let { AppGraph.voices.get(it) }

    val pick = rememberLauncherForActivityResult(ActivityResultContracts.OpenMultipleDocuments()) { picked -> uris = (uris + picked).distinct() }
    fun startRecording() {
        recording = true
        scope.launch {
            val pcm = withContext(Dispatchers.IO) { recorder.record(maxSec = 180) }
            recording = false
            if (pcm.size > recorder.sampleRate) recorded = pcm to recorder.sampleRate
        }
    }
    val micPermission = rememberLauncherForActivityResult(ActivityResultContracts.RequestPermission()) { ok -> if (ok) startRecording() }

    AlertDialog(
        onDismissRequest = { if (!recording) onClose() },
        title = { Text(if (existing == null) "새 목소리" else "‘${existing.name}’ 추가 학습") },
        text = {
            Column(Modifier.verticalScroll(rememberScrollState()), verticalArrangement = Arrangement.spacedBy(8.dp)) {
                Row(verticalAlignment = Alignment.CenterVertically) {
                    Checkbox(consent, { consent = it; AppGraph.settings.consentAccepted = it })
                    Text("본인의 목소리이거나, 목소리 주인의 동의를 받았습니다. 타인을 사칭하거나 속이는 데 사용하지 않겠습니다.",
                        style = MaterialTheme.typography.bodySmall)
                }
                if (existing == null) OutlinedTextField(name, { name = it.take(30) }, label = { Text("목소리 이름") }, singleLine = true)
                Text("음성 파일(mp3·m4a·wav·flac…)이나 영상(mp4·mkv·webm…)을 고르세요. 영상은 소리만 뽑아 말하는 구간을 자동으로 찾고, " +
                    "가장 많이 말하는 사람의 목소리만 골라 학습합니다. 잡음·음악이 적을수록 좋습니다.", style = MaterialTheme.typography.bodySmall)
                uris.forEach { u -> Text("• ${vm.displayName(u)}", style = MaterialTheme.typography.bodyMedium) }
                recorded?.let { Text("• 직접 녹음 ${"%.0f".format(it.first.size.toDouble() / it.second)}초", style = MaterialTheme.typography.bodyMedium) }
                if (recording) {
                    Card { Column(Modifier.padding(12.dp)) {
                        Text("녹음 중 ${"%.0f".format(recSec)}초 — 아래 문장을 감정을 살려 자연스럽게 읽어 주세요", fontWeight = FontWeight.Bold)
                        LinearProgressIndicator(progress = { recLevel.coerceIn(0f, 1f) }, modifier = Modifier.fillMaxWidth())
                        Text(RECORDING_SCRIPT.trim(), style = MaterialTheme.typography.bodyMedium)
                        Button(onClick = { recorder.stop() }) { Icon(Icons.Filled.Stop, null); Text("녹음 끝") }
                    } }
                } else FlowRow(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                    OutlinedButton(enabled = consent, onClick = { pick.launch(arrayOf("audio/*")) }) { Icon(Icons.Filled.AudioFile, null); Spacer(Modifier.width(4.dp)); Text("음성 파일") }
                    OutlinedButton(enabled = consent, onClick = { pick.launch(arrayOf("video/*")) }) { Icon(Icons.Filled.VideoFile, null); Spacer(Modifier.width(4.dp)); Text("영상") }
                    OutlinedButton(enabled = consent, onClick = {
                        if (ContextCompat.checkSelfPermission(ctx, Manifest.permission.RECORD_AUDIO) == PackageManager.PERMISSION_GRANTED) startRecording()
                        else micPermission.launch(Manifest.permission.RECORD_AUDIO)
                    }) { Icon(Icons.Filled.Mic, null); Spacer(Modifier.width(4.dp)); Text("녹음") }
                }
            }
        },
        confirmButton = {
            TextButton(enabled = consent && !recording && (uris.isNotEmpty() || recorded != null), onClick = {
                vm.learn(voiceId, name, uris, recorded); onClose()
            }) { Text("학습 시작") }
        },
        dismissButton = { if (!recording) TextButton(onClick = onClose) { Text("취소") } },
    )
}

@OptIn(ExperimentalLayoutApi::class)
@Composable
private fun AutoLearnDialog(v: VoiceInfo, onClose: () -> Unit) {
    val ctx = LocalContext.current
    val modes = listOf(
        Triple("빠르게", AutoTuner.Config.QUICK, "3회차 × 30가지 (새로 말하기 5번 × 디코더·음색 변형 6가지)"),
        Triple("표준", AutoTuner.Config.STANDARD, "5회차 × 96가지 (새로 말하기 8번 × 디코더·음색 변형 12가지)"),
        Triple("정밀", AutoTuner.Config.PRECISE, "5회차 × 84가지 (새로 말하기 14번 × 디코더·음색 변형 6가지)"),
    )
    var mode by remember { mutableStateOf(1) }
    AlertDialog(onDismissRequest = onClose, title = { Text("자동 학습: ${v.name}") },
        text = {
            Column(verticalArrangement = Arrangement.spacedBy(8.dp)) {
                Text("학습된 목소리로 감정 세기·말 속도감·억양 변화·참조 구간(말투)·음색 혼합·음색 보정·음색 기울기·말끝 높낮이·말끝 길이를 매번 다르게 바꿔 가며 말해 보고, " +
                    "화자 인식 모델 3개(합성에 쓰이지 않는 독립 판정 모델 ECAPA 중심)와 원본의 특징 약 100가지(음색·음높이·말끝·리듬과 말투·강세·음질·분위기·울림과 배경)를 " +
                    "원본이 평소 보이는 변화 폭과 비교해 가장 닮은 방식을 찾습니다. " +
                    "회차마다 가장 닮은 목소리를 새 세대로 저장하고, 원본과의 차이를 되먹여(재학습) 다음 세대는 그 주변을 더 촘촘히 탐색합니다. " +
                    "끝으로 원본의 녹음 환경(울림·마이크 색깔·배경 잡음)도 분석해 둡니다. " +
                    "마지막에 탐색에 쓰지 않은 새 문장으로 다시 검증해, 분명히 나아진 경우에만 적용합니다(아니면 기본값 유지).", style = MaterialTheme.typography.bodySmall)
                FlowRow(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                    modes.forEachIndexed { i, m -> FilterChip(selected = mode == i, onClick = { mode = i }, label = { Text(m.first) }) }
                }
                val m = modes[mode]
                Text(m.third, style = MaterialTheme.typography.bodyMedium)
                // measured on a 4-core server: ≈25 s per generated delivery incl. its variants; phones ≈2–3× slower (estimate)
                val gens = m.second.rounds * m.second.generativePerRound + 3
                Text("휴대폰에서 대략 ${gens * 25 * 2 / 60}~${gens * 25 * 3 / 60}분 걸릴 것으로 예상합니다(4코어 서버 실측 ${gens * 25 / 60}분 기준). 충전 중에 실행하는 것을 권장합니다.", style = MaterialTheme.typography.bodySmall)
                HorizontalDivider()
                Text("※ 점수는 기기 안의 측정치이며 사람이 듣는 느낌과 완전히 같지는 않습니다.", style = MaterialTheme.typography.bodySmall)
            }
        },
        confirmButton = {
            TextButton(onClick = {
                val m = modes[mode]
                SynthesisService.start(ctx, TaskRequest.AutoLearn(v.id, "${v.name} 자동 학습", m.second, m.first)); onClose()
            }) { Text("시작") }
        },
        dismissButton = { TextButton(onClick = onClose) { Text("취소") } })
}

/** Auto-learning generations: listen to each and choose which one the voice uses. */
@Composable
private fun GenerationList(vm: MainViewModel, v: VoiceInfo, playing: String?) {
    Text("자동 학습 세대 (각 세대는 이전 세대에서 가장 닮은 목소리를 다시 학습한 결과)", style = MaterialTheme.typography.labelMedium)
    Row(verticalAlignment = Alignment.CenterVertically) {
        FilterChip(selected = v.selectedGeneration == -1, onClick = { vm.useGeneration(v.id, -1) }, label = { Text("기본") })
        Spacer(Modifier.width(4.dp))
        Text(if (v.selectedGeneration == -1) "사용 중" else "", style = MaterialTheme.typography.bodySmall)
    }
    v.generations.forEachIndexed { i, (round, score, crit) ->
        Row(verticalAlignment = Alignment.CenterVertically) {
            FilterChip(selected = v.selectedGeneration == i, onClick = { vm.useGeneration(v.id, i) }, label = { Text("${round}세대") })
            Text("  점수 ${"%.3f".format(score)} · 특징 유사도 ${"%.0f".format(crit * 100)}%" + if (v.selectedGeneration == i) " · 사용 중" else "",
                style = MaterialTheme.typography.bodySmall, modifier = Modifier.weight(1f))
            IconButton(onClick = { vm.playGeneration(v.id, i) }) { Icon(Icons.Filled.PlayArrow, "${round}세대 듣기") }
        }
    }
    if (v.hasEnvironment) Text("원본 녹음 환경(울림·마이크 색깔${if (v.roomToneSec > 1) "·배경" else ""}) 분석됨 — 생성 화면에서 '원본 녹음 환경 재현'을 켤 수 있습니다.",
        style = MaterialTheme.typography.bodySmall)
}

@Composable
private fun SegmentsDialog(vm: MainViewModel, voiceId: String, playing: String?, onClose: () -> Unit) {
    val bank = AppGraph.voices.bank(voiceId)
    var version by remember { mutableStateOf(0) }
    AlertDialog(onDismissRequest = onClose, title = { Text("학습된 구간") },
        text = {
            if (bank == null) Text("목소리를 찾을 수 없습니다.") else {
                val prompt = bank.tuning.promptSegment
                version.let { }
                LazyColumn(verticalArrangement = Arrangement.spacedBy(4.dp)) {
                    items(bank.segments.toList(), key = { it.id }) { s ->
                        Row(verticalAlignment = Alignment.CenterVertically) {
                            Column(Modifier.weight(1f)) {
                                Text("${"%.1f".format(s.seconds)}초 · 품질 ${s.quality}" + (if (s.id == prompt) " · 자동학습 기준" else ""), style = MaterialTheme.typography.bodyMedium)
                                Text("${s.source.take(30)} · 활기 ${"%.1f".format(s.profile.arousal)}", style = MaterialTheme.typography.bodySmall)
                            }
                            IconButton(onClick = { vm.playSegment(voiceId, s.id) }) { Icon(Icons.Filled.PlayArrow, "재생") }
                            IconButton(enabled = bank.segments.size > 1 && !AppGraph.jobs.isBusy, onClick = { bank.removeSegment(s.id); AppGraph.voices.refresh(); version++ }) {
                                Icon(Icons.Filled.Delete, "삭제")
                            }
                        }
                    }
                }
            }
        },
        confirmButton = { TextButton(onClick = onClose) { Text("닫기") } })
}
