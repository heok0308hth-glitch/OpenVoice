package io.github.koreanvoice.app

import android.app.ActivityManager
import android.content.Context
import android.os.Build
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.Card
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedButton
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.collectAsState
import androidx.compose.runtime.getValue
import androidx.compose.runtime.rememberCoroutineScope
import androidx.compose.material3.LinearProgressIndicator
import kotlinx.coroutines.launch
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Modifier
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.text.font.FontFamily
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp

@Composable
fun AboutScreen(modifier: Modifier) {
    val ctx = LocalContext.current
    val licenses = remember { runCatching { ctx.assets.open("THIRD_PARTY_NOTICES.txt").bufferedReader().readText() }.getOrDefault("") }
    var showLicenses by remember { mutableStateOf(false) }
    var cacheMsg by remember { mutableStateOf("") }
    val am = ctx.getSystemService(Context.ACTIVITY_SERVICE) as ActivityManager
    val mem = ActivityManager.MemoryInfo().also { am.getMemoryInfo(it) }
    val ramGb = mem.totalMem / 1e9

    Column(modifier.fillMaxSize().verticalScroll(rememberScrollState()).padding(16.dp), verticalArrangement = Arrangement.spacedBy(12.dp)) {
        Text("정보", style = MaterialTheme.typography.headlineSmall, fontWeight = FontWeight.Bold)
        Card(Modifier.fillMaxWidth()) {
            Column(Modifier.padding(16.dp), verticalArrangement = Arrangement.spacedBy(4.dp)) {
                Text("한국어 보이스클론 ${BuildConfig.VERSION_NAME}", fontWeight = FontWeight.Bold)
                Text("Resemble AI의 Chatterbox Multilingual(MIT, 한국어 지원·감정 조절) 모델을 휴대폰용으로 변환·압축해 기기 안에서 실행하는 비공식 앱입니다. " +
                    "유료 API나 서버를 사용하지 않으며, 텍스트·음성·영상 자료는 기기 밖으로 전송되지 않습니다.")
            }
        }
        Card(Modifier.fillMaxWidth()) {
            Column(Modifier.padding(16.dp), verticalArrangement = Arrangement.spacedBy(4.dp)) {
                Text("이 기기", fontWeight = FontWeight.Bold)
                Text("${Build.MANUFACTURER} ${Build.MODEL} · Android ${Build.VERSION.RELEASE} · CPU ${Runtime.getRuntime().availableProcessors()}코어 · RAM ${"%.1f".format(ramGb)}GB")
                Text(if (ramGb < 5.5) "⚠ RAM 6GB 미만 기기는 메모리 부족·느린 속도가 생길 수 있습니다. 다른 앱을 닫고 사용하세요."
                    else "권장 사양(RAM 6GB 이상, 64비트 ARM)을 충족합니다.", style = MaterialTheme.typography.bodySmall)
                Text("처리 속도는 기기에 따라 음성 길이의 약 2~6배 시간이 걸립니다(긴 글은 백그라운드에서 진행).", style = MaterialTheme.typography.bodySmall)
                Text("성능 모드: ${AppGraph.settings.performanceMode.label}", style = MaterialTheme.typography.bodySmall)
            }
        }
        Card(Modifier.fillMaxWidth()) {
            Column(Modifier.padding(16.dp), verticalArrangement = Arrangement.spacedBy(4.dp)) {
                Text("기능", fontWeight = FontWeight.Bold)
                Text("• 목소리 학습: 음성 파일·영상·녹음 → 말하는 구간 자동 분리, 잡음 구간·다른 화자 자동 제외, 자료를 추가할수록 누적 학습\n" +
                    "• 자동 학습: 회차마다 최대 100가지 방식으로 말해 보고 원본과 가장 닮은 방식을 찾아 목소리에 되먹임(최대 5회차)\n" +
                    "• 감정 10종 + 강도, 문장별 자동 감정(대사/서술 구분), 감정에 맞는 원본 구간 선택\n" +
                    "• 부분 지정 태그: 감정·속도·음높이·쉼·울림·상태(전화/라디오/속삭임 등)·톤·읽는 법, 발음 사전\n" +
                    "• 울림 8종(방~성당), 배경(빗소리/바람/음악 파일 + 자동 더킹), 방송용 마스터링\n" +
                    "• 한국어 정규화: 숫자·날짜·시간·단위·영어·한자 읽기, 최대 10만 자 자동 분할·순서대로 합치기, 중단 후 이어서 생성",
                    style = MaterialTheme.typography.bodySmall)
            }
        }
        val hq by AppGraph.models.hqState.collectAsState()
        val scope = rememberCoroutineScope()
        Card(Modifier.fillMaxWidth()) {
            Column(Modifier.padding(16.dp), verticalArrangement = Arrangement.spacedBy(4.dp)) {
                Text("고품질 디코더 (선택)", fontWeight = FontWeight.Bold)
                Text("기본 디코더(1단계)보다 음질이 약간 좋은 10단계 디코더입니다. 대신 디코딩이 약 5배 느립니다. 약 ${AppGraph.models.hqDecoderBytes() / 1_000_000}MB.",
                    style = MaterialTheme.typography.bodySmall)
                when (val s = hq) {
                    is ModelManager.State.Working -> { LinearProgressIndicator(progress = { s.progress }, modifier = Modifier.fillMaxWidth()); Text("${(s.progress * 100).toInt()}%") }
                    is ModelManager.State.Failed -> Text("실패: ${s.message}", color = MaterialTheme.colorScheme.error)
                    else -> {}
                }
                if (AppGraph.models.hasHqDecoder()) Text("설치됨 — 생성 화면의 ‘감정·연출 설정’에서 켤 수 있습니다.", style = MaterialTheme.typography.bodySmall)
                else if (hq !is ModelManager.State.Working) OutlinedButton(onClick = { scope.launch { AppGraph.models.installHq() } }) { Text("내려받기") }
            }
        }
        OutlinedButton(onClick = {
            val n = AppGraph.jobCacheDir.listFiles()?.size ?: 0
            AppGraph.jobCacheDir.deleteRecursively(); ctx.cacheDir.listFiles()?.forEach { it.deleteRecursively() }
            cacheMsg = "임시 파일 정리 완료 (작업 캐시 ${n}개)"
        }) { Text("임시 파일 정리") }
        if (cacheMsg.isNotEmpty()) Text(cacheMsg, style = MaterialTheme.typography.bodySmall)
        OutlinedButton(onClick = { showLicenses = !showLicenses }) { Text(if (showLicenses) "라이선스 닫기" else "오픈소스 라이선스") }
        if (showLicenses) Text(licenses, fontFamily = FontFamily.Monospace, fontSize = 11.sp)
    }
}
