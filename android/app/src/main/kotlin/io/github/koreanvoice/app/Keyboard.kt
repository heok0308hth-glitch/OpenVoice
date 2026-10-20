package io.github.koreanvoice.app

import androidx.compose.foundation.gestures.awaitEachGesture
import androidx.compose.foundation.gestures.awaitFirstDown
import androidx.compose.foundation.gestures.detectTapGestures
import androidx.compose.ui.Modifier
import androidx.compose.ui.composed
import androidx.compose.ui.input.pointer.PointerEventPass
import androidx.compose.ui.input.pointer.pointerInput
import androidx.compose.ui.platform.LocalFocusManager
import androidx.compose.ui.platform.LocalSoftwareKeyboardController

/**
 * Tapping an empty spot (one no button, field or list consumed) hides the on-screen keyboard;
 * with [clearFocus] the text cursor is dropped as well.
 */
fun Modifier.hideKeyboardOnTap(clearFocus: Boolean = true): Modifier = composed {
    val keyboard = LocalSoftwareKeyboardController.current
    val focus = LocalFocusManager.current
    pointerInput(clearFocus) {
        detectTapGestures(onTap = { if (clearFocus) focus.clearFocus(); keyboard?.hide() })
    }
}

/**
 * For a text field that fills its space: a tap below the last line of text ([contentBottom], in the
 * field's local pixels) hides the keyboard instead of moving the cursor there, while [active] is true.
 * Drags are swallowed only when they start there (nothing to scroll: the text is shorter than the field).
 */
fun Modifier.hideKeyboardOnTapBelow(contentBottom: () -> Float, active: () -> Boolean): Modifier = composed {
    val keyboard = LocalSoftwareKeyboardController.current
    pointerInput(Unit) {
        awaitEachGesture {
            val down = awaitFirstDown(requireUnconsumed = false, pass = PointerEventPass.Initial)
            if (!active() || down.position.y <= contentBottom()) return@awaitEachGesture
            down.consume()
            var moved = false
            while (true) {
                val c = awaitPointerEvent(PointerEventPass.Initial).changes.firstOrNull { it.id == down.id } ?: break
                if ((c.position - down.position).getDistance() > viewConfiguration.touchSlop) moved = true
                c.consume()
                if (!c.pressed) { if (!moved) keyboard?.hide(); break }
            }
        }
    }
}
