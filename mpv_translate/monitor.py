"""
MPV monitor.

Listens for MPV events over the JSON IPC socket and drives the core loop:

  start-file  → cancel any running job, start fresh from position 0
  seek        → cancel running job immediately, restart from current
                position (with look-back applied in core_loop)
  end-file    → cancel
  toggle key  → enable/disable translation

The thread pool has max_workers=1 so only one translation job runs at a time.
Cancellation is cooperative via threading.Event; a running Whisper call
finishes its current chunk but then stops immediately.
"""
import logging
import pathlib
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from threading import Event
from typing import Any, Optional

from python_mpv_jsonipc import MPV, MPVError

from .config import Config, get_config
from .coreloop import core_loop
from .ocr_loop import OcrLoop
from .overlay import OverlayManager, PushOverlay
from .scheduler import GpuScheduler
from .subtitle import SRTFile
from .translate import get_model

log = logging.getLogger("monitor")


def _unstructure(arg: Any) -> Any:
    if isinstance(arg, pathlib.Path):
        return str(arg)
    return arg


class MPVMonitor:
    def __init__(self, config: Optional[Config] = None):
        self.config = config or get_config()
        self.shutdown = Event()
        self.mpv = MPV(
            mpv_location=self.config.mpv.executable,
            start_mpv=self.config.mpv.start_mpv,
            ipc_socket=self.config.mpv.ipc_socket,
            quit_callback=self.shutdown.set,
            **self.config.mpv.start_args,
        )
        self.enabled = True
        self._cancel = Event()
        self._cancel.set()  # nothing running yet
        self._paused_by_seek = False  # True only when we paused MPV waiting for first subtitle
        self._ocr_seek_in_progress = False  # True while we manage OCR across a seek/load pause
        self._overlay = OverlayManager(self.command, margin_bottom=self.config.subtitle.margin_bottom)

        # GPU scheduler: coordinates GPU access between audio and OCR pipelines.
        self._gpu_scheduler: Optional[GpuScheduler] = (
            GpuScheduler() if self.config.ocr.enabled else None
        )

        # OCR overlay + loop (enabled via config.ocr.enabled)
        self._ocr_overlay = PushOverlay(
            self.command,
            margin_top=self.config.ocr.margin_top,
            max_lines=self.config.ocr.max_lines,
        )
        self._ocr_loop: Optional[OcrLoop] = (
            OcrLoop(self.command, self.config, self._ocr_overlay,
                    gpu_scheduler=self._gpu_scheduler)
            if self.config.ocr.enabled
            else None
        )

        # Single-worker pool: at most one translation job at a time.
        self._pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="translate")

        # Seek debounce state
        self._seek_timer: Optional[threading.Timer] = None
        self._seek_lock = threading.Lock()
        self._seek_debounce = 0.075  # seconds to wait after last seek before launching
        self._translated_up_to: float = 0.0  # furthest translated position (seconds)

        self._register()
        self._launch_if_already_playing()

    # ── MPV interface ────────────────────────────────────────────────────────

    def command(self, name: str, *args: Any) -> Any:
        return self.mpv.command(name, *map(_unstructure, args))  # type: ignore

    def block(self):
        """Block until MPV quits."""
        self.shutdown.wait()
        self._overlay.shutdown()
        if self._ocr_loop:
            self._ocr_loop.shutdown()

    # ── event registration ───────────────────────────────────────────────────

    def _register(self):
        cfg = self.config.mpv
        self.mpv.bind_key_press(cfg.toggle_binding, self._on_toggle)
        # start-file fires before the file is open (path not yet available) — use it
        # only to cancel the current job and reset state.  file-loaded fires once the
        # file is fully open and path/duration are guaranteed to be available.
        self.mpv.bind_event("start-file", self._on_start_file)
        self.mpv.bind_event("file-loaded", self._on_file_loaded)
        self.mpv.bind_event("end-file", self._on_cancel)
        self.mpv.bind_event("seek", self._on_seek)
        if self._ocr_loop:
            self.mpv.bind_property_observer("pause", self._on_pause_change)

    def _launch_if_already_playing(self):
        """If MPV already has a file loaded when we connect, start translating it."""
        try:
            path = self.command("get_property", "path")
        except MPVError:
            return
        if not path:
            return
        try:
            position = float(self.command("get_property", "time-pos") or 0.0)
        except (MPVError, TypeError):
            position = 0.0
        log.info("file already loaded on connect (%s @ %.2fs) — starting translation", path, position)
        self._launch(position=position)
        if self._ocr_loop:
            self._ocr_loop.start()

    # ── event handlers ───────────────────────────────────────────────────────

    def _on_toggle(self):
        self.enabled = not self.enabled
        if self.enabled:
            self.command("show-text", "mpv-translate: ON")
            self._launch()
        else:
            self.command("show-text", "mpv-translate: OFF")
            self._overlay.clear()
            self._translated_up_to = 0.0
            self._stop()

    def _on_start_file(self, _event: Any = None):
        """Cancel any running job and clear the overlay; the file is about to change."""
        self._stop()
        self._overlay.clear()
        self._paused_by_seek = False
        self._ocr_seek_in_progress = False
        self._translated_up_to = 0.0

        if self._ocr_loop:
            self._ocr_loop.stop()

    def _on_file_loaded(self, _event: Any = None):
        """File is fully open — path and duration are now available; start translating.

        Pauses MPV briefly (up to max_wait) so that both the audio translation job
        and the OCR loop can produce their first results before playback begins.
        """
        first_ready: Optional[Event] = None
        ocr_first_ready: Optional[Event] = None

        try:
            was_paused = bool(self.command("get_property", "pause"))
        except (MPVError, TypeError):
            was_paused = False

        if self.enabled:
            first_ready = Event()
        if self._ocr_loop:
            ocr_first_ready = Event()

        # Pause MPV so we can wait for the first translations before playback starts.
        max_wait = self.config.translate.max_wait
        if max_wait > 0 and not was_paused and (first_ready is not None or ocr_first_ready is not None):
            self._ocr_seek_in_progress = True  # prevent _on_pause_change from stopping OCR
            try:
                self.command("set_property", "pause", True)
                self._paused_by_seek = True
            except MPVError:
                log.warning("could not pause MPV on file load")
                first_ready = ocr_first_ready = None
                self._paused_by_seek = False
                self._ocr_seek_in_progress = False

        cancel = None
        if self.enabled:
            cancel = self._launch(position=0.0, first_ready=first_ready)

        if self._ocr_loop:
            self._ocr_loop.start(first_ready=ocr_first_ready)

        # Start the waiter only if we actually paused MPV.
        if self._paused_by_seek and (first_ready is not None or ocr_first_ready is not None):
            sentinel = cancel if cancel is not None else Event()
            max_wait = self.config.translate.max_wait
            threading.Thread(
                target=self._wait_and_resume,
                args=(first_ready or Event(), ocr_first_ready, sentinel, max_wait),
                daemon=True,
                name="load-waiter",
            ).start()

    def _on_seek(self, _event: Any = None):
        """Seek: cancel immediately, debounce rapid seeks, then restart.

        MPV fires many seek events in quick succession when scrubbing or holding
        arrow keys.  We cancel the running job immediately on the *first* event
        but delay the (expensive) re-launch until seeks have settled for
        ``_seek_debounce`` seconds.
        """
        if not self.enabled:
            return

        # Cancel the running job immediately (cheap, idempotent).
        self._stop()

        with self._seek_lock:
            # Cancel any pending debounce timer — we'll schedule a fresh one.
            if self._seek_timer is not None:
                self._seek_timer.cancel()
            self._seek_timer = threading.Timer(self._seek_debounce, self._do_seek)
            self._seek_timer.daemon = True
            self._seek_timer.start()

    def _do_seek(self):
        """Actually restart translation after the debounce window closes.

        Two modes:
          • **backward seek** (into already-translated region): the overlay already
            has segments for this position, so we skip pausing MPV and resume
            translation from the frontier (_translated_up_to).
          • **forward seek** (past the frontier): pause MPV, wait for the first
            subtitle from the new position, then resume.
        """

        try:
            position = float(self.command("get_property", "time-pos") or 0.0)
        except (MPVError, TypeError):
            position = 0.0

        # ── backward seek into already-translated region ────────────────────
        # The overlay already has segments covering this position — no need to
        # re-translate or pause MPV.  Resume from the frontier.
        if self._translated_up_to > 0 and position < self._translated_up_to:
            log.info(
                "seek to %.2fs (translated up to %.2fs) — resuming from frontier",
                position, self._translated_up_to,
            )
            # If a previous seek paused MPV for us, resume now — we have content.
            if self._paused_by_seek:
                try:
                    self.command("set_property", "pause", False)
                except MPVError:
                    pass
                self._paused_by_seek = False

            if self._ocr_loop:
                self._ocr_loop.stop()
                self._ocr_loop.start()

            self._launch(
                position=self._translated_up_to,
                translated_up_to=self._translated_up_to,
            )
            return

        # ── forward seek (past the translated frontier) ─────────────────────
        try:
            was_paused = bool(self.command("get_property", "pause"))
        except (MPVError, TypeError):
            was_paused = False

        log.info("seek to %.2fs — launching new job", position)

        # Create a resume mechanism if MPV is playing, OR if a previous seek already
        # paused it for us (_paused_by_seek).  Skip only when the user paused manually
        # or when max_wait is 0 (user opted out of pause-on-seek).
        max_wait = self.config.translate.max_wait
        first_ready: Optional[Event] = None
        ocr_first_ready: Optional[Event] = None
        if max_wait > 0 and (not was_paused or self._paused_by_seek):
            first_ready = Event()
            if self._ocr_loop:
                ocr_first_ready = Event()
            if not was_paused:
                # Set flag before pausing so _on_pause_change(True) doesn't stop OCR.
                self._ocr_seek_in_progress = True
                try:
                    self.command("set_property", "pause", True)
                    self._paused_by_seek = True
                except MPVError:
                    log.warning("could not pause MPV after seek")
                    first_ready = None
                    ocr_first_ready = None
                    self._paused_by_seek = False
                    self._ocr_seek_in_progress = False

        # Restart the OCR loop at the new position immediately.  It uses capture_frame_av
        # (file-based) so it works correctly even while MPV is paused.
        if self._ocr_loop and ocr_first_ready is not None:
            self._ocr_loop.stop()
            self._ocr_loop.start(first_ready=ocr_first_ready)

        cancel = self._launch(
            position=position,
            first_ready=first_ready,
            translated_up_to=self._translated_up_to,
        )

        if first_ready is not None and cancel is not None:
            max_wait = self.config.translate.max_wait
            threading.Thread(
                target=self._wait_and_resume,
                args=(first_ready, ocr_first_ready, cancel, max_wait),
                daemon=True,
                name="seek-waiter",
            ).start()

    def _on_cancel(self, _event: Any = None):
        self._stop()
        if self._ocr_loop:
            self._ocr_loop.stop()

    def _on_pause_change(self, _name: Any, paused: Any):
        if not self._ocr_loop:
            return
        if paused:
            # During a seek or file-load, we manage OCR ourselves — don't let the
            # pause callback stop the loop we just (re)started.
            if self._ocr_seek_in_progress:
                return
            self._ocr_loop.stop()
        else:
            self._ocr_seek_in_progress = False
            self._ocr_loop.start()

    # ── job management ───────────────────────────────────────────────────────

    def _stop(self):
        """Signal the running job to stop and wait briefly for the pool slot."""
        self._cancel.set()

    def _launch(
        self,
        position: Optional[float] = None,
        first_ready: Optional[Event] = None,
        translated_up_to: float = 0.0,
    ) -> Optional[Event]:
        """Start a new translation job and return its cancel Event, or None on failure."""
        if not self.enabled:
            return None

        path: Optional[str] = None
        try:
            path = self.command("get_property", "path")
        except MPVError:
            pass
        if not path:
            return None

        if position is None:
            try:
                position = float(self.command("get_property", "time-pos") or 0.0)
            except (MPVError, TypeError):
                position = 0.0

        # Create a fresh cancel event for the new job
        cancel = Event()
        self._cancel = cancel

        log.info("launching translation job: path=%s position=%.2f", path, position)
        self._pool.submit(
            self._run_job, path=path, position=position, cancel=cancel,
            first_ready=first_ready, translated_up_to=translated_up_to,
        )
        return cancel

    # ── seek waiter ──────────────────────────────────────────────────────────

    def _wait_and_resume(
        self,
        first_ready: Event,
        ocr_first_ready: Optional[Event],
        cancel: Event,
        max_wait: float,
    ):
        """Block until first subtitle AND first OCR result are ready (or max_wait expires).

        Checks `cancel` before resuming: if a new seek already cancelled this job
        we leave MPV in whatever state the new seek set up.
        """
        t_start = time.monotonic()
        triggered = first_ready.wait(timeout=max_wait)
        if not triggered:
            log.info("seek-waiter: max_wait %.1fs elapsed waiting for audio, resuming anyway", max_wait)
        if ocr_first_ready is not None and not ocr_first_ready.is_set():
            elapsed = time.monotonic() - t_start
            remaining = max(0.0, max_wait - elapsed)
            if remaining > 0:
                ocr_first_ready.wait(timeout=remaining)
        if cancel.is_set():
            log.debug("seek-waiter: job was cancelled, skipping resume")
            return
        try:
            self.command("set_property", "pause", False)
            self._paused_by_seek = False
        except MPVError:
            log.warning("seek-waiter: could not resume MPV")

    # ── worker ───────────────────────────────────────────────────────────────

    def _get_playback_pos(self) -> float:
        return float(self.command("get_property", "time-pos") or 0.0)

    def _run_job(self, *, path: str, position: float, cancel: Event,
                 first_ready: Optional[Event] = None, translated_up_to: float = 0.0):
        try:
            for item in core_loop(
                path=path, position=position, cancel=cancel,
                first_ready=first_ready, get_playback_pos=self._get_playback_pos,
                gpu_scheduler=self._gpu_scheduler,
                translated_up_to=translated_up_to,
            ):
                if cancel.is_set():
                    break

                if isinstance(item, SRTFile):
                    pass  # overlay is cleared on file-change, not per-job
                elif item is True:
                    self.command("show-text", "mpv-translate: done")
                elif isinstance(item, list):
                    # Chunk of (abs_start, abs_end, text) triples — hand to overlay.
                    self._overlay.add_segments(item)
                    # Track the furthest point we've translated.
                    if item:
                        self._translated_up_to = max(
                            self._translated_up_to,
                            max(end for _, end, _ in item),
                        )

        except Exception:
            log.exception("translation job failed for %s", path)
