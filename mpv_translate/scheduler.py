"""GPU scheduler for fair rotation between audio translation and OCR."""
import logging
import threading
import time
from contextlib import contextmanager
from typing import Optional

log = logging.getLogger("scheduler")


class GpuScheduler:
    """Mutual-exclusion scheduler for GPU access with fair rotation.

    When both audio translation and OCR are active, prevents them from
    competing for GPU simultaneously.  When only one pipeline runs,
    the lock is uncontested and adds negligible overhead.

    Priority modes
    ──────────────
    • **OCR priority** — audio callers (``defer_to_ocr=True``) wait until
      OCR finishes.  Used at file load (OCR scans first) and after each
      audio chunk (OCR catches up to the audio frontier).
    • **Audio priority** — OCR callers (``defer_to_audio=True``) wait until
      audio finishes.  Used after seeks so the first subtitle appears fast.
    """

    def __init__(self, yield_ms: float = 5.0):
        self._lock = threading.Lock()
        self._waiters = 0
        self._waiter_lock = threading.Lock()
        self._yield_sec = yield_ms / 1000.0
        self._ocr_priority = threading.Event()
        self._audio_priority = threading.Event()
        self._priority_set_at: float = 0.0
        self._audio_frontier: float = 0.0

    # ── priority control ──────────────────────────────────────────────────

    def set_ocr_priority(self):
        """Give OCR first access to the GPU.

        Audio callers (``defer_to_ocr=True``) will wait until
        :meth:`clear_ocr_priority` is called.
        """
        self._priority_set_at = time.monotonic()
        self._ocr_priority.set()

    def clear_ocr_priority(self):
        """Resume fair scheduling between audio and OCR."""
        self._ocr_priority.clear()

    def set_audio_priority(self):
        """Give audio first access to the GPU.

        OCR callers (``defer_to_audio=True``) will wait until
        :meth:`clear_audio_priority` is called.
        """
        self._priority_set_at = time.monotonic()
        self._audio_priority.set()

    def clear_audio_priority(self):
        self._audio_priority.clear()

    def reset(self):
        """Clear all priority/frontier state (e.g. on seek or file change)."""
        self._ocr_priority.clear()
        self._audio_priority.clear()
        self._audio_frontier = 0.0

    def yield_to_ocr(self, frontier: float):
        """Called by audio after each chunk to hand GPU time to OCR.

        Clears audio priority (audio's turn is over), sets the audio frontier,
        and activates OCR priority so the next audio ``gpu(defer_to_ocr=True)``
        blocks until OCR catches up and calls :meth:`clear_ocr_priority`.
        """
        self._audio_priority.clear()
        self._audio_frontier = frontier
        self.set_ocr_priority()

    @property
    def audio_frontier(self) -> float:
        """Timestamp (seconds) that audio has translated up to."""
        return self._audio_frontier

    # ── GPU acquisition ───────────────────────────────────────────────────

    @contextmanager
    def gpu(
        self,
        cancel: Optional[threading.Event] = None,
        defer_to_ocr: bool = False,
        defer_to_audio: bool = False,
    ):
        """Acquire exclusive GPU access.

        Polls the lock every 100 ms, checking *cancel* between attempts.
        Yields ``True`` if acquired, ``False`` if cancelled first.
        On release, sleeps briefly when other callers are waiting (fairness).
        """
        # Defer to the other pipeline when its priority is active.
        for flag in (
            (self._ocr_priority if defer_to_ocr else None),
            (self._audio_priority if defer_to_audio else None),
        ):
            if flag is None:
                continue
            while flag.is_set():
                if cancel is not None and cancel.is_set():
                    yield False
                    return
                # Safety timeout: auto-clear after 60 s.
                if time.monotonic() - self._priority_set_at > 60.0:
                    log.debug("priority safety timeout — clearing")
                    flag.clear()
                    break
                time.sleep(0.1)

        acquired = False
        try:
            with self._waiter_lock:
                self._waiters += 1

            while True:
                acquired = self._lock.acquire(timeout=0.1)
                if acquired:
                    break
                if cancel is not None and cancel.is_set():
                    yield False
                    return

            yield True
        finally:
            with self._waiter_lock:
                self._waiters -= 1
                has_waiters = self._waiters > 0

            if acquired:
                self._lock.release()
                if has_waiters:
                    time.sleep(self._yield_sec)
