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
from concurrent.futures import Future, ThreadPoolExecutor
from threading import Event
from typing import Any, Optional

from python_mpv_jsonipc import MPV, MPVError

from .config import Config, get_config
from .coreloop import core_loop
from .ocr_loop import OcrLoop
from .overlay import OverlayManager, PushOverlay
from .scheduler import GpuScheduler
from .subtitle import SRTFile, hms
from .translate import cleanup as _cleanup_models, get_model

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
        self._overlay = OverlayManager(self.command, margin_bottom=self.config.subtitle.margin_bottom, font_size=self.config.subtitle.font_size, max_display_seconds=self.config.subtitle.max_display_seconds)

        # GPU scheduler: coordinates GPU access between audio and OCR pipelines.
        # Only used in "interleave" mode; "simultaneous" lets both run freely.
        self._gpu_scheduler: Optional[GpuScheduler] = (
            GpuScheduler()
            if self.config.ocr.enabled and self.config.ocr.gpu_mode == "interleave"
            else None
        )

        # OCR overlay + loop (enabled via config.ocr.enabled)
        self._ocr_overlay = PushOverlay(
            self.command,
            margin_top=self.config.ocr.margin_top,
            max_lines=self.config.ocr.max_lines,
            font_size=self.config.ocr.font_size,
        )
        self._ocr_loop: Optional[OcrLoop] = (
            OcrLoop(self.command, self.config, self._ocr_overlay,
                    gpu_scheduler=self._gpu_scheduler)
            if self.config.ocr.enabled
            else None
        )

        # Single-worker pool: at most one translation job at a time.
        self._pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="translate")
        self._current_future: Optional[Future] = None
        self._launch_lock = threading.Lock()  # serializes _launch calls
        self._generation = 0  # incremented on every launch/file-change; stale jobs check this

        # Seek debounce state
        self._seek_timer: Optional[threading.Timer] = None
        self._seek_lock = threading.Lock()
        self._seek_debounce = 0.30  # seconds to wait after last seek before launching
        self._seek_generation = 0   # incremented per seek; stale _do_seek calls bail out
        self._translated_up_to: float = 0.0  # furthest translated position (seconds)

        self._register()
        self._launch_if_already_playing()

    # ── MPV interface ────────────────────────────────────────────────────────

    def command(self, name: str, *args: Any) -> Any:
        return self.mpv.command(name, *map(_unstructure, args))  # type: ignore

    def block(self):
        """Block until MPV quits."""
        self.shutdown.wait()
        self._stop()
        self._overlay.shutdown()
        if self._ocr_loop:
            self._ocr_loop.shutdown()
        with self._seek_lock:
            if self._seek_timer is not None:
                self._seek_timer.cancel()
                self._seek_timer = None
        self._pool.shutdown(wait=True, cancel_futures=True)
        _cleanup_models()

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
        log.info("file already loaded on connect (%s @ %s) — starting translation", path, hms(position))
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
        self._generation += 1  # invalidate any in-flight job immediately
        self._stop()
        self._overlay.clear()
        self._paused_by_seek = False
        self._ocr_seek_in_progress = False
        self._translated_up_to = 0.0

        if self._gpu_scheduler:
            self._gpu_scheduler.reset()
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

        # Give OCR first GPU access so on-screen text appears before audio
        # translation starts its (expensive) first Whisper chunk.
        if self._ocr_loop and self._gpu_scheduler:
            self._gpu_scheduler.set_ocr_priority()

        cancel = None
        if self.enabled:
            cancel = self._launch(position=0.0, first_ready=first_ready)
            # If _launch was skipped (lock busy from a concurrent stop/start),
            # schedule a retry so the file still gets translated.
            if cancel is None:
                log.debug("file-loaded launch skipped — scheduling retry")
                t = threading.Timer(
                    0.5, self._retry_file_launch,
                    args=(first_ready,),
                )
                t.daemon = True
                t.start()

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

    def _retry_file_launch(self, first_ready: Optional[Event] = None):
        """Retry a file-loaded launch that was skipped due to lock contention."""
        if not self.enabled:
            return
        # If a job is already running (another launch succeeded), skip.
        if self._current_future is not None and not self._current_future.done():
            return
        cancel = self._launch(position=0.0, first_ready=first_ready)
        if cancel is None:
            log.debug("file-loaded retry still blocked — giving up")

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
            # Bump generation so any already-running _do_seek bails out.
            self._seek_generation += 1
            gen = self._seek_generation
            # Cancel any pending debounce timer — we'll schedule a fresh one.
            if self._seek_timer is not None:
                self._seek_timer.cancel()
            self._seek_timer = threading.Timer(
                self._seek_debounce, self._do_seek, args=(gen,),
            )
            self._seek_timer.daemon = True
            self._seek_timer.start()

    def _do_seek(self, seek_gen: int = 0):
        """Actually restart translation after the debounce window closes.

        Two modes:
          • **backward seek** (into already-translated region): the overlay already
            has segments for this position, so we skip pausing MPV and resume
            translation from the frontier (_translated_up_to).
          • **forward seek** (past the frontier): pause MPV, wait for the first
            subtitle from the new position, then resume.
        """
        if self.shutdown.is_set():
            return
        # Stale seek — a newer one has been scheduled; bail out.
        if seek_gen != self._seek_generation:
            return

        # Clear stale state and give audio first GPU access after a seek.
        if self._gpu_scheduler:
            self._gpu_scheduler.reset()
            self._gpu_scheduler.set_audio_priority()

        try:
            position = float(self.command("get_property", "time-pos") or 0.0)
        except (MPVError, TypeError):
            position = 0.0

        # ── backward seek into already-translated region ────────────────────
        # The overlay already has segments covering this position — no need to
        # re-translate or pause MPV.  Resume from the frontier.
        if self._translated_up_to > 0 and position < self._translated_up_to:
            log.info(
                "seek to %s (translated up to %s) — resuming from frontier",
                hms(position), hms(self._translated_up_to),
            )
            # If a previous seek paused MPV for us, resume now — we have content.
            if self._paused_by_seek:
                try:
                    self.command("set_property", "pause", False)
                except MPVError:
                    pass
                self._paused_by_seek = False

            # Skip re-launch if a non-cancelled job is already running from
            # the frontier.  If cancel is set the job is dying — launch a
            # replacement so the pool queues it behind the exiting one.
            if self._current_future is not None and not self._current_future.done():
                if not self._cancel.is_set():
                    return

            # OCR reads from the current playback position — no restart needed
            # for backward seeks.

            self._launch(
                position=self._translated_up_to,
                translated_up_to=self._translated_up_to,
            )
            return

        # ── forward seek (past the translated frontier) ─────────────────────
        # Check generation again — another seek may have arrived while we
        # were doing the work above.
        if seek_gen != self._seek_generation:
            return

        try:
            was_paused = bool(self.command("get_property", "pause"))
        except (MPVError, TypeError):
            was_paused = False

        log.info("seek to %s — launching new job", hms(position))

        # Create a resume mechanism if MPV is playing, OR if a previous seek already
        # paused it for us (_paused_by_seek).  Skip only when the user paused manually
        # or when max_wait is 0 (user opted out of pause-on-seek).
        max_wait = self.config.translate.max_wait
        first_ready: Optional[Event] = None
        if max_wait > 0 and (not was_paused or self._paused_by_seek):
            first_ready = Event()
            if not was_paused:
                # Set flag before pausing so _on_pause_change(True) doesn't stop OCR.
                self._ocr_seek_in_progress = True
                try:
                    self.command("set_property", "pause", True)
                    self._paused_by_seek = True
                except MPVError:
                    log.warning("could not pause MPV after seek")
                    first_ready = None
                    self._paused_by_seek = False
                    self._ocr_seek_in_progress = False

        # OCR reads from the current playback position — no need to restart it
        # on within-file seeks.  Just reset the scheduler so timing adapts to
        # the new position.  (File changes already restart OCR via
        # _on_start_file / _on_file_loaded.)

        cancel = self._launch(
            position=position,
            first_ready=first_ready,
            translated_up_to=self._translated_up_to,
        )

        # If _launch was skipped (lock busy), retry after a short delay.
        # The generation check at the top of _do_seek will bail out if a
        # newer seek has arrived in the meantime.
        if cancel is None:
            log.debug("seek launch skipped — scheduling retry")
            with self._seek_lock:
                self._seek_timer = threading.Timer(
                    0.5, self._do_seek, args=(seek_gen,),
                )
                self._seek_timer.daemon = True
                self._seek_timer.start()
            return

        if first_ready is not None:
            max_wait = self.config.translate.max_wait
            threading.Thread(
                target=self._wait_and_resume,
                args=(first_ready, None, cancel, max_wait),
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
        """Start a new translation job and return its cancel Event, or None on failure.

        Thread-safe: uses a non-blocking lock so concurrent callers (e.g.
        multiple debounced seeks) don't pile up waiting.
        """
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

        # Non-blocking acquire: if another _launch is in progress, skip
        # entirely and let the caller retry.  This prevents thread pile-ups
        # during rapid file switching.
        if not self._launch_lock.acquire(blocking=False):
            log.debug("launch lock busy — skipping")
            return None
        try:
            # Cancel the old job so it exits at the next check point.
            self._cancel.set()

            # Try to cancel a queued-but-not-started future outright.
            if self._current_future is not None:
                self._current_future.cancel()
                # Don't wait — the pool serialises jobs and the stale one will
                # exit via generation check.  Waiting here causes lock pile-ups
                # during rapid file switching.

            # Create a fresh cancel event for the new job.
            self._generation += 1
            gen = self._generation
            cancel = Event()
            self._cancel = cancel

            log.info("launching translation job: path=%s position=%s", path, hms(position))
            self._current_future = self._pool.submit(
                self._run_job, path=path, position=position, cancel=cancel,
                first_ready=first_ready, translated_up_to=translated_up_to,
                generation=gen,
            )
            return cancel
        finally:
            self._launch_lock.release()

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
                 first_ready: Optional[Event] = None, translated_up_to: float = 0.0,
                 generation: int = 0):
        try:
            for item in core_loop(
                path=path, position=position, cancel=cancel,
                first_ready=first_ready, get_playback_pos=self._get_playback_pos,
                gpu_scheduler=self._gpu_scheduler,
                translated_up_to=translated_up_to,
            ):
                if cancel.is_set() or self._generation != generation:
                    break

                if isinstance(item, SRTFile):
                    pass  # overlay is cleared on file-change, not per-job
                elif item is True:
                    self.command("show-text", "mpv-translate: done")
                elif isinstance(item, list):
                    # Double-check generation before touching shared state.
                    if self._generation != generation:
                        break
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
