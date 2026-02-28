"""
Flicker-free subtitle display via MPV's osd-overlay API.

Segments are stored in memory as (abs_start, abs_end, text) triples.
A background thread polls time-pos at ~20 Hz and pushes the matching
text to the overlay, hiding it when no segment covers the current position.

This avoids the sub-reload flash that the SRT-file approach requires, and
guarantees only one segment is ever shown at a time (no stacking).
"""
import bisect
import logging
import threading
from typing import Callable, List, Optional, Tuple

log = logging.getLogger("overlay")

_OVERLAY_ID = 1
_OCR_OVERLAY_ID = 2
_POLL_INTERVAL = 0.05  # seconds (20 Hz)

# Virtual coordinate space used for the osd-overlay res_x/res_y parameters.
# MPV scales this to the actual display size, so positions are resolution-independent.
_OSD_RES_X = 1280
_OSD_RES_Y = 720


def _redistribute_lines(text: str, max_lines: int) -> str:
    """If *text* has more than *max_lines* newline-separated lines, merge
    adjacent lines onto the same row (joined by ``  |  ``) so the total
    number of rows does not exceed *max_lines*.

    When *max_lines* is 0 or the text already fits, returns *text* unchanged.
    """
    if max_lines <= 0:
        return text
    lines = text.split("\n")
    if len(lines) <= max_lines:
        return text
    # Distribute lines as evenly as possible across max_lines rows.
    rows: list[list[str]] = [[] for _ in range(max_lines)]
    for i, line in enumerate(lines):
        rows[i % max_lines].append(line)
    return "\n".join("  |  ".join(parts) for parts in rows)


def _ass_top_event(text: str, margin_top: int = 30, font_size: int = 0) -> str:
    """Format text for top-centre osd-overlay placement.
    \\an8 = top-centre; \\pos(x, y) places it margin_top virtual pixels below the top edge.
    """
    x = _OSD_RES_X // 2
    y = margin_top
    fs = "\\fs%d" % font_size if font_size > 0 else ""
    return "{\\an8\\pos(%d,%d)%s}" % (x, y, fs) + text.replace("\n", "\\N")


def _ass_event(text: str, margin_bottom: int = 50, font_size: int = 0) -> str:
    """Format text for osd-overlay ass-events.
    \\an2 = bottom-center; \\pos(x, y) lifts the subtitle off the edge by margin_bottom
    virtual pixels (in the 1280×720 OSD space).
    """
    x = _OSD_RES_X // 2
    y = _OSD_RES_Y - margin_bottom
    fs = "\\fs%d" % font_size if font_size > 0 else ""
    return "{\\an2\\pos(%d,%d)%s}" % (x, y, fs) + text.replace("\n", "\\N")


class OverlayManager:
    """Drives a single MPV osd-overlay for subtitle display."""

    def __init__(self, command: Callable, margin_bottom: int = 50, font_size: int = 0,
                 max_display_seconds: float = 0.0):
        """
        Args:
            command:             MPVMonitor.command — forwards calls to MPV JSON IPC.
            margin_bottom:       Pixels from the bottom edge in the 1280×720 virtual OSD space.
            font_size:           Font size in the 1280×720 virtual OSD space (0 = MPV default).
            max_display_seconds: Hide subtitle after this many seconds on screen (0 = disabled).
        """
        self._cmd = command
        self._margin_bottom = margin_bottom
        self._font_size = font_size
        self._max_display = max_display_seconds
        self._segments: List[Tuple[float, float, str]] = []
        self._starts: List[float] = []
        self._lock = threading.Lock()
        self._current_text: Optional[str] = None
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._loop, daemon=True, name="overlay")
        self._thread.start()

    # ── public API ───────────────────────────────────────────────────────────

    def clear(self):
        """Discard all segments and hide the overlay immediately."""
        with self._lock:
            self._segments.clear()
            self._starts.clear()
        self._push("")

    def add_segments(self, segments: List[Tuple[float, float, str]]):
        """Append (abs_start, abs_end, text) triples to the display list."""
        with self._lock:
            self._segments.extend(segments)
            self._starts.extend(s[0] for s in segments)

    def shutdown(self):
        """Stop the background thread and blank the overlay."""
        self._stop.set()
        self._push("")

    # ── internals ────────────────────────────────────────────────────────────

    def _text_at(self, pos: float) -> str:
        with self._lock:
            if not self._starts:
                return ""
            idx = bisect.bisect_right(self._starts, pos) - 1
            if idx < 0:
                return ""
            start, end, text = self._segments[idx]
            if pos < end:
                if self._max_display > 0 and pos - start >= self._max_display:
                    return ""
                return text
        return ""

    def _push(self, text: str):
        """Send an osd-overlay command only when the displayed text changes."""
        if text == self._current_text:
            return
        self._current_text = text
        data = _ass_event(text, self._margin_bottom, self._font_size) if text else ""
        try:
            self._cmd("osd-overlay", _OVERLAY_ID, "ass-events", data, _OSD_RES_X, _OSD_RES_Y, 0, False, False)
        except Exception:
            log.debug("osd-overlay update failed", exc_info=True)

    def _loop(self):
        while not self._stop.wait(_POLL_INTERVAL):
            try:
                pos = float(self._cmd("get_property", "time-pos") or 0.0)
                self._push(self._text_at(pos))
            except Exception:
                pass


_OCR_REFRESH_INTERVAL = 1.0  # seconds between keep-alive pushes for the OCR overlay


class PushOverlay:
    """On-demand overlay for OCR results shown at the top of the screen.

    A background thread re-sends the current text at _OCR_REFRESH_INTERVAL so
    that MPV keeps the overlay visible even if it drops the one-shot command
    (observed with osd-overlay ID != 1 on some MPV builds).
    """

    def __init__(self, command: Callable, margin_top: int = 30, max_lines: int = 0, font_size: int = 0):
        self._cmd = command
        self._margin_top = margin_top
        self._max_lines = max_lines
        self._font_size = font_size
        self._current: Optional[str] = None
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._loop, daemon=True, name="ocr-overlay")
        self._thread.start()

    def show(self, text: str):
        """Display *text* at the top of the screen."""
        with self._lock:
            if text == self._current:
                return
            self._current = text
        log.debug("ocr overlay show: %r", text[:60])
        self._push(text)

    def hide(self):
        """Remove the OCR overlay."""
        with self._lock:
            if self._current == "":
                return
            self._current = ""
        self._push("")

    def shutdown(self):
        self._stop.set()
        self.hide()

    # ── internals ────────────────────────────────────────────────────────────

    def _push(self, text: str):
        if text and self._max_lines > 0:
            text = _redistribute_lines(text, self._max_lines)
        data = _ass_top_event(text, self._margin_top, self._font_size) if text else ""
        try:
            self._cmd(
                "osd-overlay", _OCR_OVERLAY_ID, "ass-events",
                data, _OSD_RES_X, _OSD_RES_Y, 0, False, False,
            )
        except Exception:
            log.warning("osd-overlay (ocr) update failed", exc_info=True)

    def _loop(self):
        """Periodically re-push the current text so MPV keeps it rendered."""
        while not self._stop.wait(_OCR_REFRESH_INTERVAL):
            with self._lock:
                if self._current:
                    self._push(self._current)
