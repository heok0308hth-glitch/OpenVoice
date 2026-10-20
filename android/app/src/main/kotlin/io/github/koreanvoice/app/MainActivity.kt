package io.github.koreanvoice.app

import android.Manifest
import android.os.Build
import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.compose.setContent
import androidx.activity.enableEdgeToEdge
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.foundation.isSystemInDarkTheme
import android.widget.Toast
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.ExperimentalLayoutApi
import androidx.compose.foundation.layout.WindowInsets
import androidx.compose.foundation.layout.consumeWindowInsets
import androidx.compose.foundation.layout.imePadding
import androidx.compose.foundation.layout.isImeVisible
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.AlertDialog
import androidx.compose.material3.TextButton
import androidx.compose.ui.platform.LocalClipboardManager
import androidx.compose.ui.text.AnnotatedString
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.text.input.TextFieldValue
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.Info
import androidx.compose.material.icons.filled.LibraryMusic
import androidx.compose.material.icons.filled.RecordVoiceOver
import androidx.compose.material.icons.filled.TextFields
import androidx.compose.material3.Button
import androidx.compose.material3.Icon
import androidx.compose.material3.LinearProgressIndicator
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.NavigationBar
import androidx.compose.material3.NavigationBarItem
import androidx.compose.material3.Scaffold
import androidx.compose.material3.Surface
import androidx.compose.material3.Text
import androidx.compose.material3.darkColorScheme
import androidx.compose.material3.lightColorScheme
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.collectAsState
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableIntStateOf
import androidx.compose.runtime.rememberCoroutineScope
import androidx.compose.runtime.saveable.rememberSaveable
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.unit.dp
import androidx.lifecycle.viewmodel.compose.viewModel
import kotlinx.coroutines.launch

class MainActivity : ComponentActivity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        enableEdgeToEdge()
        // debug builds only: open the UI without the model pack (for UI testing on emulators)
        if (BuildConfig.DEBUG && intent?.getBooleanExtra("ui_test", false) == true) uiTestMode = true
        setContent { AppTheme { Root() } }
    }
}

private val Brand = Color(0xFF3D5AFE)
private var uiTestMode = false

@Composable
fun AppTheme(content: @Composable () -> Unit) {
    val dark = isSystemInDarkTheme()
    val scheme = if (dark) darkColorScheme(primary = Color(0xFF9FA8FF), secondary = Color(0xFF80CBC4))
    else lightColorScheme(primary = Brand, secondary = Color(0xFF00897B))
    MaterialTheme(colorScheme = scheme, content = content)
}

@Composable
private fun Root() {
    val modelState by AppGraph.models.state.collectAsState()
    LaunchedEffect(Unit) { AppGraph.models.check() }
    var crash by remember { mutableStateOf(CrashReport.pending()) }
    Surface(Modifier.fillMaxSize(), color = MaterialTheme.colorScheme.background) {
        when (val s = modelState) {
            ModelManager.State.Ready -> MainTabs()
            else -> if (uiTestMode) MainTabs() else ModelSetupScreen(s)
        }
    }
    crash?.let { r -> CrashDialog(r) { CrashReport.dismiss(); crash = null } }
}

@Composable
private fun CrashDialog(r: CrashReport.Report, onClose: () -> Unit) {
    val clipboard = LocalClipboardManager.current
    val context = LocalContext.current
    AlertDialog(
        onDismissRequest = onClose,
        title = { Text(r.title) },
        text = {
            Column(Modifier.verticalScroll(rememberScrollState())) {
                Text(r.advice, style = MaterialTheme.typography.bodyMedium)
                Spacer(Modifier.height(12.dp))
                Text(r.details, style = MaterialTheme.typography.bodySmall, color = MaterialTheme.colorScheme.onSurfaceVariant)
            }
        },
        confirmButton = { TextButton(onClick = onClose) { Text("확인") } },
        dismissButton = {
            TextButton(onClick = {
                clipboard.setText(AnnotatedString(r.title + "\n" + r.details))
                Toast.makeText(context, "복사했습니다", Toast.LENGTH_SHORT).show()
            }) { Text("내용 복사") }
        },
    )
}

@Composable
private fun ModelSetupScreen(state: ModelManager.State) {
    val scope = rememberCoroutineScope()
    Column(
        Modifier.fillMaxSize().padding(32.dp),
        verticalArrangement = Arrangement.Center,
        horizontalAlignment = Alignment.CenterHorizontally,
    ) {
        Text("한국어 보이스클론", style = MaterialTheme.typography.headlineMedium, fontWeight = FontWeight.Bold)
        Text("감정 표현 음성 합성 · 완전 온디바이스 · 무료", style = MaterialTheme.typography.bodyMedium)
        Spacer(Modifier.height(32.dp))
        when (state) {
            ModelManager.State.Checking -> { LinearProgressIndicator(Modifier.fillMaxWidth()); Text("모델 확인 중…") }
            is ModelManager.State.Working -> {
                LinearProgressIndicator(progress = { state.progress }, modifier = Modifier.fillMaxWidth())
                Spacer(Modifier.height(8.dp))
                Text("${state.label} ${(state.progress * 100).toInt()}%")
            }
            is ModelManager.State.NeedsDownload -> {
                Text(
                    "음성 합성 모델(약 ${state.totalBytes / 1_000_000} MB)을 한 번 내려받아야 합니다.\n" +
                        "다운로드 후에는 인터넷 없이 기기 안에서만 동작합니다. Wi-Fi 사용을 권장합니다.",
                    textAlign = TextAlign.Center,
                )
                Spacer(Modifier.height(16.dp))
                Button(onClick = { scope.launch { AppGraph.models.install() } }) { Text("모델 다운로드") }
            }
            is ModelManager.State.Failed -> {
                Text("모델 준비 실패: ${state.message}", color = MaterialTheme.colorScheme.error, textAlign = TextAlign.Center)
                Spacer(Modifier.height(16.dp))
                Button(onClick = { scope.launch { AppGraph.models.install() } }) { Text("다시 시도") }
            }
            ModelManager.State.Ready -> {}
        }
    }
}

@OptIn(ExperimentalLayoutApi::class)
@Composable
private fun MainTabs() {
    var tab by rememberSaveable { mutableIntStateOf(0) }
    var editing by rememberSaveable { mutableStateOf(false) }
    val vm: MainViewModel = viewModel()
    val ctx = LocalContext.current
    val openTxt = rememberLauncherForActivityResult(ActivityResultContracts.OpenDocument()) { uri ->
        if (uri != null) vm.loadTextFile(uri) { t ->
            if (t.length > MAX_CHARS) Toast.makeText(ctx, "100,000자까지만 불러왔습니다.", Toast.LENGTH_LONG).show()
            vm.updateScript(TextFieldValue(t.take(MAX_CHARS)))
        }
    }
    val launchTxt = { openTxt.launch(arrayOf("text/plain", "text/*")) }
    val notifPermission = rememberLauncherForActivityResult(ActivityResultContracts.RequestPermission()) {}
    LaunchedEffect(Unit) {
        if (Build.VERSION.SDK_INT >= 33) notifPermission.launch(Manifest.permission.POST_NOTIFICATIONS)
    }
    if (editing) {
        ScriptEditor(vm, onClose = { editing = false; vm.saveScriptNow() }, onOpenTxt = launchTxt)
        return
    }
    // the bottom navigation hides while the keyboard is up, and every tab is padded for the keyboard
    val imeVisible = WindowInsets.isImeVisible
    Scaffold(
        bottomBar = {
            if (!imeVisible) NavigationBar {
                listOf(
                    Triple("생성", Icons.Filled.TextFields, 0),
                    Triple("목소리", Icons.Filled.RecordVoiceOver, 1),
                    Triple("보관함", Icons.Filled.LibraryMusic, 2),
                    Triple("정보", Icons.Filled.Info, 3),
                ).forEach { (label, icon, i) ->
                    NavigationBarItem(selected = tab == i, onClick = { tab = i }, icon = { Icon(icon, null) }, label = { Text(label) })
                }
            }
        },
    ) { pad ->
        val m = Modifier.padding(pad).consumeWindowInsets(pad).imePadding()
        when (tab) {
            0 -> SynthScreen(vm, m, onGoVoices = { tab = 1 }, onGoLibrary = { tab = 2 }, onEdit = { editing = true }, onOpenTxt = launchTxt)
            1 -> VoicesScreen(vm, m, onGoLibrary = { tab = 2 })
            2 -> LibraryScreen(m)
            else -> AboutScreen(m)
        }
    }
}
