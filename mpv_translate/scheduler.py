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
    """

    def __init__(self, yield_ms: float = 5.0):
        self._lock = threading.Lock()
        self._waiters = 0
        self._waiter_lock = threading.Lock()
        self._yield_sec = yield_ms / 1000.0

    @contextmanager
    def gpu(self, cancel: Optional[threading.Event] = None):
        """Acquire exclusive GPU access.

        Polls the lock every 100 ms, checking *cancel* between attempts.
        Yields ``True`` if acquired, ``False`` if cancelled first.
        On release, sleeps briefly when other callers are waiting (fairness).
        """
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
