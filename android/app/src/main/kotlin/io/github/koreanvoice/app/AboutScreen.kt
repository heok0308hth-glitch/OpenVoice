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
import androidx.compose.runtime.getValue
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
                Text("OpenVoice V2(MyShell, MIT)의 음색 변환기와 MeloTTS-Korean(MIT)을 기기 안에서 실행하는 비공식 앱입니다. " +
                    "유료 API나 서버를 사용하지 않으며, 텍스트·음성 샘플은 기기 밖으로 전송되지 않습니다.")
            }
        }
        Card(Modifier.fillMaxWidth()) {
            Column(Modifier.padding(16.dp), verticalArrangement = Arrangement.spacedBy(4.dp)) {
                Text("이 기기", fontWeight = FontWeight.Bold)
                Text("${Build.MANUFACTURER} ${Build.MODEL} · Android ${Build.VERSION.RELEASE} · CPU ${Runtime.getRuntime().availableProcessors()}코어 · RAM ${"%.1f".format(ramGb)}GB")
                Text(if (ramGb < 3.5) "⚠ RAM이 4GB 미만이면 메모리 부족이 생길 수 있습니다. 다른 앱을 닫고 ‘절전’ 모드를 사용하세요."
                    else "권장 사양(RAM 4GB 이상, 64비트 ARM)을 충족합니다.", style = MaterialTheme.typography.bodySmall)
                Text("성능 모드: ${AppGraph.settings.performanceMode.label}", style = MaterialTheme.typography.bodySmall)
            }
        }
        Card(Modifier.fillMaxWidth()) {
            Column(Modifier.padding(16.dp), verticalArrangement = Arrangement.spacedBy(4.dp)) {
                Text("한국어 특화 처리", fontWeight = FontWeight.Bold)
                Text("• 숫자 읽기: 세 개/삼 개월, 세 시/십삼 시, 날짜·시간·전화번호·단위·통화·분수\n" +
                    "• 한자→한글 (두음법칙 적용), 영어 외래어 사전 + 발음 사전(CMUdict)\n" +
                    "• MeloTTS 학습 시 사용된 g2pkk 발음 규칙을 그대로 이식 (원본과 3,020문장 중 3,018문장 일치)\n" +
                    "• mecab 형태소 분석기 대신, mecab의 판단을 증류한 경량 규칙표(14KB)로 용언 경음화 처리\n" +
                    "• 문장 단위 자동 분할·쉼 조절, 문장별 음량 평준화, 중단 후 이어서 생성",
                    style = MaterialTheme.typography.bodySmall)
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
