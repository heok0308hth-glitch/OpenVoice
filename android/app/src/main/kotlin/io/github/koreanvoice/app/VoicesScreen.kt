package io.github.koreanvoice.app

import android.Manifest
import android.content.pm.PackageManager
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
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
import androidx.compose.material.icons.filled.Delete
import androidx.compose.material.icons.filled.Edit
import androidx.compose.material.icons.filled.Mic
import androidx.compose.material.icons.filled.PlayArrow
import androidx.compose.material.icons.filled.RecordVoiceOver
import androidx.compose.material.icons.filled.Stop
import androidx.compose.material3.AlertDialog
import androidx.compose.material3.AssistChip
import androidx.compose.material3.Button
import androidx.compose.material3.Card
import androidx.compose.material3.Checkbox
import androidx.compose.material3.ExtendedFloatingActionButton
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
import androidx.compose.foundation.layout.Box
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext

@Composable
fun VoicesScreen(vm: MainViewModel, modifier: Modifier) {
    val voices by AppGraph.voices.voices.collectAsState()
    val playing by Player.playing.collectAsState()
    val preview by vm.preview.collectAsState()
    var adding by remember { mutableStateOf(false) }
    var renaming by remember { mutableStateOf<VoiceProfile?>(null) }
    var deleting by remember { mutableStateOf<VoiceProfile?>(null) }

    Box(modifier.fillMaxSize()) {
        LazyColumn(Modifier.fillMaxSize().padding(horizontal = 16.dp), verticalArrangement = Arrangement.spacedBy(10.dp)) {
            item {
                Text("목소리", style = MaterialTheme.typography.headlineSmall, fontWeight = FontWeight.Bold, modifier = Modifier.padding(top = 16.dp))
                Text("10~30초 분량의 깨끗한 음성 샘플로 목소리를 복제합니다. 여러 파일을 함께 넣으면 더 정확해집니다.",
                    style = MaterialTheme.typography.bodyMedium)
            }
            if (voices.isEmpty()) item {
                Card(Modifier.fillMaxWidth()) {
                    Column(Modifier.padding(16.dp)) {
                        Icon(Icons.Filled.RecordVoiceOver, null)
                        Text("아직 등록된 목소리가 없습니다. 아래의 ‘목소리 추가’를 눌러 파일(내 파일, Google 드라이브 등)을 고르거나 직접 녹음하세요.")
                    }
                }
            }
            items(voices, key = { it.id }) { v ->
                Card(Modifier.fillMaxWidth()) {
                    Column(Modifier.padding(16.dp), verticalArrangement = Arrangement.spacedBy(4.dp)) {
                        Row(verticalAlignment = Alignment.CenterVertically) {
                            Text(v.name, style = MaterialTheme.typography.titleMedium, modifier = Modifier.weight(1f))
                            AssistChip(onClick = {}, label = { Text("품질 ${v.score}") })
                        }
                        Text("음성 ${"%.1f".format(v.voicedSec)}초 · 샘플 ${v.sources.size}개", style = MaterialTheme.typography.bodySmall)
                        v.warnings.forEach { Text("• $it", style = MaterialTheme.typography.bodySmall, color = MaterialTheme.colorScheme.error) }
                        Row {
                            val sample = AppGraph.voices.sampleFile(v.id)
                            if (sample.exists()) TextButton(onClick = { Player.toggle(sample) }) {
                                Icon(if (playing == sample.path) Icons.Filled.Stop else Icons.Filled.AudioFile, null); Text("원본")
                            }
                            TextButton(enabled = preview !is MainViewModel.PreviewState.Working, onClick = { vm.testVoice(v) }) {
                                Icon(Icons.Filled.PlayArrow, null); Text("복제 듣기")
                            }
                            IconButton(onClick = { renaming = v }) { Icon(Icons.Filled.Edit, "이름 변경") }
                            IconButton(onClick = { deleting = v }) { Icon(Icons.Filled.Delete, "삭제") }
                        }
                    }
                }
            }
            item { Spacer(Modifier.padding(40.dp)) }
        }
        ExtendedFloatingActionButton(
            onClick = { vm.resetVoiceBuild(); adding = true },
            icon = { Icon(Icons.Filled.Add, null) }, text = { Text("목소리 추가") },
            modifier = Modifier.align(Alignment.BottomEnd).padding(16.dp),
        )
    }

    if (adding) AddVoiceDialog(vm) { adding = false }
    renaming?.let { v ->
        var name by remember(v.id) { mutableStateOf(v.name) }
        AlertDialog(onDismissRequest = { renaming = null }, title = { Text("이름 변경") },
            text = { OutlinedTextField(name, { name = it.take(30) }, singleLine = true) },
            confirmButton = { TextButton(onClick = { AppGraph.voices.rename(v.id, name); renaming = null }) { Text("저장") } },
            dismissButton = { TextButton(onClick = { renaming = null }) { Text("취소") } })
    }
    deleting?.let { v ->
        AlertDialog(onDismissRequest = { deleting = null }, title = { Text("목소리 삭제") }, text = { Text("‘${v.name}’을(를) 삭제할까요?") },
            confirmButton = { TextButton(onClick = { AppGraph.voices.delete(v.id); deleting = null }) { Text("삭제") } },
            dismissButton = { TextButton(onClick = { deleting = null }) { Text("취소") } })
    }
}

@Composable
private fun AddVoiceDialog(vm: MainViewModel, onClose: () -> Unit) {
    val ctx = LocalContext.current
    val state by vm.voiceBuild.collectAsState()
    var consent by remember { mutableStateOf(AppGraph.settings.consentAccepted) }
    var name by remember { mutableStateOf("") }
    var recording by remember { mutableStateOf(false) }
    val recorder = remember { VoiceRecorder() }
    val recSec by recorder.seconds.collectAsState()
    val recLevel by recorder.level.collectAsState()
    val scope = rememberCoroutineScope()

    val pick = rememberLauncherForActivityResult(ActivityResultContracts.OpenMultipleDocuments()) { uris ->
        if (uris.isNotEmpty()) vm.analyzeUris(uris)
    }
    fun startRecording() {
        recording = true
        scope.launch {
            val pcm = withContext(Dispatchers.IO) { recorder.record() }
            recording = false
            if (pcm.size > recorder.sampleRate) vm.analyzeRecording(pcm, recorder.sampleRate)
        }
    }
    val micPermission = rememberLauncherForActivityResult(ActivityResultContracts.RequestPermission()) { ok -> if (ok) startRecording() }

    AlertDialog(
        onDismissRequest = { if (!recording) onClose() },
        title = { Text("새 목소리") },
        text = {
            Column(Modifier.verticalScroll(rememberScrollState()), verticalArrangement = Arrangement.spacedBy(8.dp)) {
                Row(verticalAlignment = Alignment.CenterVertically) {
                    Checkbox(consent, { consent = it; AppGraph.settings.consentAccepted = it })
                    Text("본인의 목소리이거나, 목소리 주인의 동의를 받았습니다. 타인을 사칭하거나 속이는 데 사용하지 않겠습니다.",
                        style = MaterialTheme.typography.bodySmall)
                }
                when (val s = state) {
                    is MainViewModel.VoiceBuildState.Working -> { Text(s.message); LinearProgressIndicator(Modifier.fillMaxWidth()) }
                    is MainViewModel.VoiceBuildState.Error -> Text(s.message, color = MaterialTheme.colorScheme.error)
                    is MainViewModel.VoiceBuildState.Analyzed -> {
                        s.items.forEach { it ->
                            Text("✓ ${it.label}: 음성 ${"%.1f".format(it.report.voicedSec)}초, 품질 ${it.report.score}점", style = MaterialTheme.typography.bodyMedium)
                            it.report.warnings.forEach { w -> Text("  • $w", style = MaterialTheme.typography.bodySmall, color = MaterialTheme.colorScheme.error) }
                        }
                        OutlinedTextField(name, { name = it.take(30) }, label = { Text("목소리 이름") }, singleLine = true)
                    }
                    is MainViewModel.VoiceBuildState.Saved -> Text("‘${s.profile.name}’ 저장 완료! 생성 화면에서 선택하세요.")
                    MainViewModel.VoiceBuildState.Idle -> Text("파일(내 파일·Google 드라이브·다운로드 등)을 고르거나 직접 녹음하세요. 여러 개를 추가할 수 있습니다.",
                        style = MaterialTheme.typography.bodySmall)
                }
                if (recording) {
                    Card { Column(Modifier.padding(12.dp)) {
                        Text("녹음 중 ${"%.0f".format(recSec)}초 — 아래 문장을 자연스럽게 읽어 주세요", fontWeight = FontWeight.Bold)
                        LinearProgressIndicator(progress = { recLevel.coerceIn(0f, 1f) }, modifier = Modifier.fillMaxWidth())
                        Text(RECORDING_SCRIPT.trim(), style = MaterialTheme.typography.bodyMedium)
                        Button(onClick = { recorder.stop() }) { Icon(Icons.Filled.Stop, null); Text("녹음 끝") }
                    } }
                } else if (state !is MainViewModel.VoiceBuildState.Working && state !is MainViewModel.VoiceBuildState.Saved) {
                    Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                        OutlinedButton(enabled = consent, onClick = { pick.launch(arrayOf("audio/*", "video/mp4")) }) {
                            Icon(Icons.Filled.AudioFile, null); Spacer(Modifier.width(4.dp)); Text("파일 선택")
                        }
                        OutlinedButton(enabled = consent, onClick = {
                            if (ContextCompat.checkSelfPermission(ctx, Manifest.permission.RECORD_AUDIO) == PackageManager.PERMISSION_GRANTED) startRecording()
                            else micPermission.launch(Manifest.permission.RECORD_AUDIO)
                        }) { Icon(Icons.Filled.Mic, null); Spacer(Modifier.width(4.dp)); Text("녹음") }
                    }
                }
            }
        },
        confirmButton = {
            when (state) {
                is MainViewModel.VoiceBuildState.Analyzed -> TextButton(enabled = consent, onClick = { vm.saveVoice(name) }) { Text("저장") }
                is MainViewModel.VoiceBuildState.Saved -> TextButton(onClick = onClose) { Text("닫기") }
                else -> {}
            }
        },
        dismissButton = { if (!recording) TextButton(onClick = onClose) { Text("취소") } },
    )
}
