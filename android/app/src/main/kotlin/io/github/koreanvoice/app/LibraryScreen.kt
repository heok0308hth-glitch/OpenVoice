package io.github.koreanvoice.app

import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.Delete
import androidx.compose.material.icons.filled.PlayArrow
import androidx.compose.material.icons.filled.SaveAlt
import androidx.compose.material.icons.filled.Share
import androidx.compose.material.icons.filled.Stop
import androidx.compose.material3.Card
import androidx.compose.material3.Icon
import androidx.compose.material3.IconButton
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.collectAsState
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableIntStateOf
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import java.io.File
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale

@Composable
fun LibraryScreen(modifier: Modifier) {
    val ctx = LocalContext.current
    val playing by Player.playing.collectAsState()
    val job by AppGraph.jobs.state.collectAsState()
    var refresh by remember { mutableIntStateOf(0) }
    var files by remember { mutableStateOf(emptyList<File>()) }
    LaunchedEffect(refresh, job) {
        files = AppGraph.outputDir.listFiles { f -> f.extension == "mp3" }.orEmpty().sortedByDescending { it.lastModified() }
    }
    var exporting by remember { mutableStateOf<File?>(null) }
    val export = rememberLauncherForActivityResult(ActivityResultContracts.CreateDocument("audio/mpeg")) { uri ->
        val f = exporting
        if (uri != null && f != null) Sharing.exportTo(ctx, f, uri)
        exporting = null
    }
    val fmt = remember { SimpleDateFormat("yyyy.MM.dd HH:mm", Locale.KOREA) }

    LazyColumn(modifier.fillMaxSize().padding(horizontal = 16.dp), verticalArrangement = Arrangement.spacedBy(8.dp)) {
        item {
            Text("보관함", style = MaterialTheme.typography.headlineSmall, fontWeight = FontWeight.Bold, modifier = Modifier.padding(top = 16.dp))
            Text("‘저장’으로 원하는 위치(내 파일, Google 드라이브 등)에 MP3를 내보낼 수 있습니다.", style = MaterialTheme.typography.bodySmall)
        }
        if (files.isEmpty()) item { Text("아직 생성된 파일이 없습니다.") }
        items(files, key = { it.path }) { f ->
            Card(Modifier.fillMaxWidth()) {
                Row(Modifier.padding(start = 16.dp, end = 4.dp, top = 8.dp, bottom = 8.dp), verticalAlignment = Alignment.CenterVertically) {
                    Column(Modifier.weight(1f)) {
                        Text(f.nameWithoutExtension, style = MaterialTheme.typography.titleSmall)
                        Text("${fmt.format(Date(f.lastModified()))} · ${"%.1f".format(f.length() / 1e6)} MB", style = MaterialTheme.typography.bodySmall)
                    }
                    IconButton(onClick = { Player.toggle(f) }) { Icon(if (playing == f.path) Icons.Filled.Stop else Icons.Filled.PlayArrow, "재생") }
                    IconButton(onClick = { exporting = f; export.launch(f.name) }) { Icon(Icons.Filled.SaveAlt, "저장") }
                    IconButton(onClick = { Sharing.share(ctx, f) }) { Icon(Icons.Filled.Share, "공유") }
                    IconButton(onClick = { if (playing == f.path) Player.stop(); f.delete(); refresh++ }) { Icon(Icons.Filled.Delete, "삭제") }
                }
            }
        }
    }
}
