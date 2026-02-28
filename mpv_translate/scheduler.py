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

    Supports an OCR-priority mode: while active, callers that pass
    ``defer_to_ocr=True`` (audio translation) wait for OCR to finish
    its first pass before competing for the lock.
    """

    def __init__(self, yield_ms: float = 5.0):
        self._lock = threading.Lock()
        self._waiters = 0
        self._waiter_lock = threading.Lock()
        self._yield_sec = yield_ms / 1000.0
        self._ocr_priority = threading.Event()
        self._priority_set_at: float = 0.0

    def set_ocr_priority(self):
        """Give OCR first access to the GPU.

        Audio callers (``defer_to_ocr=True``) will wait until
        :meth:`clear_ocr_priority` is called.  Auto-clears after 10 s
        as a safety net.
        """
        self._priority_set_at = time.monotonic()
        self._ocr_priority.set()

    def clear_ocr_priority(self):
        """Resume fair scheduling between audio and OCR."""
        self._ocr_priority.clear()

    @contextmanager
    def gpu(self, cancel: Optional[threading.Event] = None, defer_to_ocr: bool = False):
        """Acquire exclusive GPU access.

        Polls the lock every 100 ms, checking *cancel* between attempts.
        Yields ``True`` if acquired, ``False`` if cancelled first.
        On release, sleeps briefly when other callers are waiting (fairness).

        When *defer_to_ocr* is True and OCR priority is active, waits for
        the priority to clear (or 10 s safety timeout) before competing.
        """
        # Defer to OCR when priority is active.
        if defer_to_ocr:
            while self._ocr_priority.is_set():
                if cancel is not None and cancel.is_set():
                    yield False
                    return
                # Safety timeout: auto-clear after 10 s.
                if time.monotonic() - self._priority_set_at > 10.0:
                    log.debug("OCR priority safety timeout — clearing")
                    self._ocr_priority.clear()
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
