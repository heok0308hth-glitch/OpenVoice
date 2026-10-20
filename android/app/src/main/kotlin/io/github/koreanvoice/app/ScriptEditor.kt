package io.github.koreanvoice.app

import androidx.activity.compose.BackHandler
import androidx.compose.foundation.background
import androidx.compose.foundation.horizontalScroll
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.WindowInsets
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.safeDrawing
import androidx.compose.foundation.layout.windowInsetsPadding
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.text.BasicTextField
import androidx.compose.foundation.layout.ExperimentalLayoutApi
import androidx.compose.foundation.layout.isImeVisible
import androidx.compose.runtime.rememberUpdatedState
import androidx.compose.ui.platform.LocalDensity
import androidx.compose.runtime.mutableFloatStateOf
import androidx.compose.foundation.verticalScroll
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.automirrored.filled.ArrowBack
import androidx.compose.material.icons.automirrored.filled.Redo
import androidx.compose.material.icons.automirrored.filled.Undo
import androidx.compose.material.icons.filled.ContentPaste
import androidx.compose.material.icons.filled.FileOpen
import androidx.compose.material.icons.filled.KeyboardHide
import androidx.compose.material.icons.filled.MoreVert
import androidx.compose.material.icons.filled.PlayArrow
import androidx.compose.material3.AlertDialog
import androidx.compose.material3.AssistChip
import androidx.compose.material3.DropdownMenu
import androidx.compose.material3.DropdownMenuItem
import androidx.compose.material3.HorizontalDivider
import androidx.compose.material3.Icon
import androidx.compose.material3.IconButton
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Surface
import androidx.compose.material3.Text
import androidx.compose.material3.TextButton
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.collectAsState
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.focus.FocusRequester
import androidx.compose.ui.focus.focusRequester
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.SolidColor
import androidx.compose.ui.platform.LocalClipboardManager
import androidx.compose.ui.platform.LocalSoftwareKeyboardController
import androidx.compose.ui.text.AnnotatedString
import androidx.compose.ui.text.SpanStyle
import androidx.compose.ui.text.TextRange
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.input.OffsetMapping
import androidx.compose.ui.text.input.TextFieldValue
import androidx.compose.ui.text.input.TransformedText
import androidx.compose.ui.text.input.VisualTransformation
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import io.github.koreanvoice.audio.Dsp
import io.github.koreanvoice.expressive.Emotion
import io.github.koreanvoice.expressive.Markup

/** A tag group of the toolbar: items (tag, label) and the tag that ends the setting (for wrapping a selection). */
class TagGroup(val label: String, val items: List<Pair<String, String>>, val close: String?)

val TAG_GROUPS: List<TagGroup> = listOf(
    TagGroup("감정", Emotion.entries.filter { it != Emotion.NEUTRAL }.flatMap { e ->
        listOf("[${e.label}]" to e.label, "[${e.label}:강]" to "${e.label} (강하게)", "[${e.label}:약]" to "${e.label} (살짝)") }, "[/감정]"),
    TagGroup("쉼", listOf("[쉼:0.5]" to "0.5초 쉬기", "[쉼:1]" to "1초 쉬기", "[쉼:2]" to "2초 쉬기", "[쉼:3]" to "3초 쉬기"), null),
    TagGroup("속도", listOf("[속도:빠르게]" to "빠르게", "[속도:느리게]" to "느리게", "[속도:1.2]" to "1.2배", "[속도:0.8]" to "0.8배"), "[/속도]"),
    TagGroup("높이", listOf("[높이:높게]" to "높게", "[높이:낮게]" to "낮게", "[높이:+1]" to "+1 반음", "[높이:-1]" to "-1 반음"), "[/높이]"),
    TagGroup("말끝", listOf("[말끝:올림]" to "끝을 올려서", "[말끝:살짝올림]" to "살짝 올려서", "[말끝:내림]" to "끝을 내려서",
        "[말끝:살짝내림]" to "살짝 내려서", "[말끝:길게]" to "끝을 길게", "[말끝:짧게]" to "끝을 짧게"), "[/말끝]"),
    TagGroup("울림", Dsp.Room.entries.filter { it != Dsp.Room.NONE }.map { "[울림:${it.label}]" to it.label }, "[울림:없음]"),
    TagGroup("상태", Dsp.State.entries.filter { it != Dsp.State.NORMAL }.map { "[${it.label}]" to it.label }, "[/상태]"),
    TagGroup("톤", Dsp.Tone.entries.filter { it != Dsp.Tone.NORMAL }.map { "[톤:${it.label}]" to it.label }, "[톤:기본]"),
    TagGroup("소리(실험)", Markup.NONVERBAL.keys.map { "[$it]" to it }, null),
)

/** Colours markup tags ([...]) and readings ({표기|읽는법}) so the script stays readable. */
class TagHighlight(private val tag: Color, private val reading: Color) : VisualTransformation {
    private val re = Regex("""\[[^\[\]\n]{1,40}\]|\{[^\{\}\n]*\|[^\{\}\n]*\}""")
    override fun filter(text: AnnotatedString): TransformedText {
        if (text.isEmpty() || (!text.contains('[') && !text.contains('{'))) return TransformedText(text, OffsetMapping.Identity)
        val b = AnnotatedString.Builder(text)
        for (m in re.findAll(text.text)) {
            val c = if (m.value.startsWith("{")) reading else tag
            b.addStyle(SpanStyle(color = c, fontWeight = FontWeight.SemiBold, background = c.copy(alpha = 0.10f)), m.range.first, m.range.last + 1)
        }
        return TransformedText(b.toAnnotatedString(), OffsetMapping.Identity)
    }
}

/**
 * Full-screen script editor: the text fills the space above the on-screen keyboard and the tag toolbar sits
 * directly above it. Tags insert at the cursor; with text selected they wrap the selection
 * ("[기쁨]선택한 부분[/감정]").
 */
@OptIn(ExperimentalLayoutApi::class)
@Composable
fun ScriptEditor(vm: MainViewModel, onClose: () -> Unit, onOpenTxt: () -> Unit) {
    val keyboard = LocalSoftwareKeyboardController.current
    val clipboard = LocalClipboardManager.current
    val focus = remember { FocusRequester() }
    var menu by remember { mutableStateOf(false) }
    var help by remember { mutableStateOf(false) }
    var confirmClear by remember { mutableStateOf(false) }
    val field = vm.script
    val tagColor = MaterialTheme.colorScheme.primary
    val readingColor = MaterialTheme.colorScheme.tertiary
    val highlight = remember(tagColor, readingColor) { TagHighlight(tagColor, readingColor) }
    val preview by vm.preview.collectAsState()
    val playing by Player.playing.collectAsState()
    BackHandler { onClose() }
    var textBottom by remember { mutableFloatStateOf(0f) }
    val tapMargin = with(LocalDensity.current) { 12.dp.toPx() }
    val imeVisible = rememberUpdatedState(WindowInsets.isImeVisible)

    /** The selection, or the paragraph at the cursor, prefixed by the style tags in effect at that point. */
    fun previewPart() {
        val t = field.text
        val (s0, e0) = if (!field.selection.collapsed) field.selection.min to field.selection.max else {
            val c = field.selection.start.coerceIn(0, t.length)
            val a = t.lastIndexOf('\n', (c - 1).coerceAtLeast(0)).let { if (it < 0 || it >= c) 0 else it + 1 }
            val b = t.indexOf('\n', c).let { if (it < 0) t.length else it }
            a to b
        }
        val part = t.substring(s0, e0).trim()
        if (part.isEmpty()) return
        val active = Regex("""\[[^\[\]\n]{1,40}\]""").findAll(t.substring(0, s0)).map { it.value }
            .filter { tag -> !tag.startsWith("[쉼") && Markup.NONVERBAL.keys.none { tag == "[$it]" } }.joinToString("")
        keyboard?.hide()
        vm.preview(active + part, AppGraph.settings.lastVoiceId, AppGraph.settings.delivery, maxChars = 220)
    }
    LaunchedEffect(Unit) { runCatching { focus.requestFocus() } }

    fun apply(tag: String, close: String?) {
        val t = field.text; val s = field.selection.min; val e = field.selection.max
        val new = if (e > s && close != null) {
            val wrapped = tag + t.substring(s, e) + close
            TextFieldValue(t.substring(0, s) + wrapped + t.substring(e), TextRange(s + wrapped.length))
        } else TextFieldValue(t.substring(0, s) + tag + t.substring(e), TextRange(s + tag.length))
        vm.updateScript(new)
    }

    Surface(Modifier.fillMaxSize(), color = MaterialTheme.colorScheme.background) {
        Column(Modifier.fillMaxSize().windowInsetsPadding(WindowInsets.safeDrawing).hideKeyboardOnTap(clearFocus = false)) {
            // top bar
            Row(Modifier.fillMaxWidth().padding(horizontal = 4.dp), verticalAlignment = Alignment.CenterVertically) {
                IconButton(onClick = onClose) { Icon(Icons.AutoMirrored.Filled.ArrowBack, "완료") }
                Column(Modifier.weight(1f)) {
                    Text("대본 편집", style = MaterialTheme.typography.titleMedium)
                    Text("${"%,d".format(field.text.length)} / ${"%,d".format(MAX_CHARS)}자" + estimateLabel(field.text.length), style = MaterialTheme.typography.bodySmall)
                }
                IconButton(enabled = vm.canUndo, onClick = { vm.undo() }) { Icon(Icons.AutoMirrored.Filled.Undo, "실행 취소") }
                IconButton(enabled = vm.canRedo, onClick = { vm.redo() }) { Icon(Icons.AutoMirrored.Filled.Redo, "다시 실행") }
                Box {
                    IconButton(onClick = { menu = true }) { Icon(Icons.Filled.MoreVert, "더보기") }
                    DropdownMenu(expanded = menu, onDismissRequest = { menu = false }) {
                        DropdownMenuItem(text = { Text("붙여넣기") }, leadingIcon = { Icon(Icons.Filled.ContentPaste, null) },
                            onClick = { menu = false; clipboard.getText()?.text?.let { apply(it, null) } })
                        DropdownMenuItem(text = { Text("TXT 파일 불러오기") }, leadingIcon = { Icon(Icons.Filled.FileOpen, null) }, onClick = { menu = false; onOpenTxt() })
                        DropdownMenuItem(text = { Text("태그 설명") }, onClick = { menu = false; help = true })
                        DropdownMenuItem(text = { Text("전체 지우기") }, onClick = { menu = false; confirmClear = true })
                    }
                }
                TextButton(onClick = onClose) { Text("완료") }
            }
            when (val pv = preview) {
                is MainViewModel.PreviewState.Working -> Text("  ${pv.message}", style = MaterialTheme.typography.bodySmall, color = MaterialTheme.colorScheme.primary)
                is MainViewModel.PreviewState.Error -> Text("  미리듣기 오류: ${pv.message}", style = MaterialTheme.typography.bodySmall, color = MaterialTheme.colorScheme.error)
                is MainViewModel.PreviewState.Ready -> Row(verticalAlignment = Alignment.CenterVertically) {
                    Text("  미리듣기 준비됨", style = MaterialTheme.typography.bodySmall, modifier = Modifier.weight(1f))
                    TextButton(onClick = { Player.toggle(pv.file) }) { Text(if (playing == pv.file.path) "정지" else "다시 듣기") }
                }
                else -> {}
            }
            HorizontalDivider()
            // the text: takes all space left above the keyboard and scrolls internally, so the cursor stays visible
            BasicTextField(
                value = field,
                onValueChange = { vm.updateScript(it) },
                // colouring is skipped for very long scripts to keep typing light
                visualTransformation = if (field.text.length <= 30_000) highlight else VisualTransformation.None,
                textStyle = MaterialTheme.typography.bodyLarge.copy(color = MaterialTheme.colorScheme.onBackground, fontSize = 17.sp, lineHeight = 26.sp),
                cursorBrush = SolidColor(MaterialTheme.colorScheme.primary),
                onTextLayout = { textBottom = it.size.height.toFloat() },
                // tapping the blank space under the text hides the keyboard (tap again to bring it back)
                modifier = Modifier.weight(1f).fillMaxWidth().padding(horizontal = 16.dp, vertical = 8.dp)
                    .hideKeyboardOnTapBelow({ textBottom + tapMargin }, { imeVisible.value && field.text.isNotEmpty() })
                    .focusRequester(focus),
                decorationBox = { inner ->
                    Box {
                        if (field.text.isEmpty()) Text("여기에 읽을 글을 쓰거나 붙여넣으세요.\n아래 태그 버튼으로 감정·쉼·말끝·울림 등을 부분마다 지정할 수 있습니다.\n글자를 선택한 뒤 태그를 누르면 그 부분에만 적용됩니다.",
                            style = MaterialTheme.typography.bodyLarge, color = MaterialTheme.colorScheme.onSurfaceVariant)
                        inner()
                    }
                },
            )
            // tag toolbar right above the keyboard
            HorizontalDivider()
            Row(Modifier.fillMaxWidth().background(MaterialTheme.colorScheme.surfaceContainer).horizontalScroll(rememberScrollState())
                .padding(horizontal = 8.dp, vertical = 4.dp), horizontalArrangement = Arrangement.spacedBy(6.dp), verticalAlignment = Alignment.CenterVertically) {
                IconButton(onClick = { keyboard?.hide() }) { Icon(Icons.Filled.KeyboardHide, "키보드 숨기기") }
                AssistChip(onClick = { previewPart() }, enabled = preview !is MainViewModel.PreviewState.Working,
                    leadingIcon = { Icon(Icons.Filled.PlayArrow, null) }, label = { Text(if (field.selection.collapsed) "이 문단 듣기" else "선택 듣기") })
                TAG_GROUPS.forEach { g -> ToolbarMenu(g, field.selection.collapsed.not()) { tag -> apply(tag, g.close) } }
                AssistChip(onClick = {
                    // {표기|읽는법}: with a selection, the selection becomes the written form and the cursor waits for the reading
                    val t = field.text; val s0 = field.selection.min; val e0 = field.selection.max
                    val written = if (e0 > s0) t.substring(s0, e0) else "표기"
                    val ins = "{$written|"
                    vm.updateScript(TextFieldValue(t.substring(0, s0) + ins + "}" + t.substring(e0), TextRange(s0 + ins.length)))
                }, label = { Text("읽는법") })
                AssistChip(onClick = { apply("[기본]", null) }, label = { Text("초기화") })
            }
        }
    }

    if (help) AlertDialog(onDismissRequest = { help = false }, title = { Text("태그 설명") },
        text = { Column(Modifier.verticalScroll(rememberScrollState())) { Text(MARKUP_HELP, style = MaterialTheme.typography.bodySmall) } },
        confirmButton = { TextButton(onClick = { help = false }) { Text("닫기") } })
    if (confirmClear) AlertDialog(onDismissRequest = { confirmClear = false }, title = { Text("전체 지우기") }, text = { Text("대본을 모두 지울까요? (실행 취소로 되돌릴 수 있습니다)") },
        confirmButton = { TextButton(onClick = { vm.updateScript(TextFieldValue("")); confirmClear = false }) { Text("지우기") } },
        dismissButton = { TextButton(onClick = { confirmClear = false }) { Text("취소") } })
}

@Composable
private fun ToolbarMenu(g: TagGroup, hasSelection: Boolean, onPick: (String) -> Unit) {
    var open by remember { mutableStateOf(false) }
    Box {
        AssistChip(onClick = { open = true }, label = { Text(g.label) })
        DropdownMenu(expanded = open, onDismissRequest = { open = false }) {
            if (hasSelection && g.close != null) Text("  선택한 부분에만 적용", style = MaterialTheme.typography.labelSmall, color = MaterialTheme.colorScheme.primary)
            g.items.forEach { (tag, name) -> DropdownMenuItem(text = { Text("$name   $tag") }, onClick = { onPick(tag); open = false }) }
        }
    }
}
