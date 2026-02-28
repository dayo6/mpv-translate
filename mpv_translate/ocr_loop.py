"""
OCR polling loop.

Extracts on-screen text with easyocr, translates it offline via
OPUS-MT (MarianMT), and pushes the result to a PushOverlay at the top of the screen.

Two operating modes
───────────────────
**Scan mode** (``lookahead_seconds > 0``)
    Binary-search lookahead: samples thumbnails across the upcoming window,
    pinpoints text transitions via binary search (comparing text-region crops
    with movement tolerance), runs full OCR only at transition points, batch-
    translates, then displays each result at the correct playback time.  This
    catches every title card, no matter how fast they change.

**Tick mode** (``lookahead_seconds == 0``)
    Periodic single-frame polling with Bayesian-adaptive interval, frame-diff
    gating, and per-block stability tracking.  Falls back to this when no
    lookahead is configured.

Shared behaviour
────────────────
• Length guard — blocks / combined result exceeding `max_chars` are dropped.
• Watermark filtering — persistent text is pre-scanned and spatially masked.
• Visibility  — the overlay is hidden as soon as the OCR result is empty.
• Lazy imports — heavy models are imported only when the loop first runs.
"""
import logging
import math
import os
import tempfile
import threading
import time
from contextlib import nullcontext
from threading import Event
from typing import TYPE_CHECKING, Callable, Optional

if TYPE_CHECKING:
    from .config import Config
    from .overlay import PushOverlay

log = logging.getLogger("ocr_loop")


class _BayesianScheduler:
    """Adaptive OCR timing using Bayesian estimates of text transition times.

    Tracks how long text stays visible (*on*-durations) and hidden
    (*off*-durations), maintains conjugate-prior posteriors for each, and
    uses the resulting hazard function to choose:

    • **optimal_offset** — where within ``[0, lookahead]`` to capture the
      next frame.  When a transition is expected soon the offset shrinks
      toward 0 (precise timing); when the state is stable it grows toward
      ``lookahead`` (early detection).

    • **next_interval** — how long to wait before the next tick.  Polls
      faster (Gaussian peak) near the expected transition and slower during
      stable stretches.
    """

    __slots__ = (
        "base_interval", "lookahead",
        "_on_sum", "_on_n", "_off_sum", "_off_n",
        "_state_start", "_text_on",
    )

    def __init__(self, base_interval: float, lookahead: float):
        self.base_interval = base_interval
        self.lookahead = lookahead
        # Conjugate Normal prior (pseudo-observations).
        self._on_sum = 8.0    # prior mean 4 s visible (2 × 4)
        self._on_n = 2.0
        self._off_sum = 6.0   # prior mean 3 s hidden  (2 × 3)
        self._off_n = 2.0
        self._state_start: float = 0.0
        self._text_on: bool = False

    # ── posterior updates ──────────────────────────────────────────────────

    def notify(self, text_showing: bool, now: float) -> None:
        """Record a state transition (text appeared / disappeared)."""
        if self._state_start > 0:
            dur = now - self._state_start
            if dur > 0.3:  # ignore sub-second jitter
                if self._text_on:
                    self._on_sum += dur
                    self._on_n += 1
                else:
                    self._off_sum += dur
                    self._off_n += 1
        self._text_on = text_showing
        self._state_start = now

    def reset(self) -> None:
        """Clear state tracking (e.g. on seek) while keeping learned priors."""
        self._state_start = 0.0
        self._text_on = False

    # ── queries ────────────────────────────────────────────────────────────

    @property
    def _expected(self) -> float:
        """Posterior mean of the current state's duration."""
        if self._text_on:
            return self._on_sum / self._on_n
        return self._off_sum / self._off_n

    def _hazard(self) -> float:
        """Sigmoid hazard — rises toward 1 as elapsed approaches the posterior mean."""
        if self._state_start <= 0:
            return 0.0  # no state yet → full lookahead for earliest detection
        elapsed = time.monotonic() - self._state_start
        ratio = elapsed / max(self._expected, 0.5)
        return 1.0 / (1.0 + math.exp(-4.0 * (ratio - 0.8)))

    def optimal_offset(self) -> float:
        """Lookahead offset in seconds — shrinks as a transition nears."""
        if self.lookahead <= 0:
            return 0.0
        h = self._hazard()
        return self.lookahead * max(0.1, 1.0 - h)

    def next_interval(self) -> float:
        """Seconds to wait before the next OCR capture."""
        if self._state_start <= 0:
            return self.base_interval
        elapsed = time.monotonic() - self._state_start
        ratio = elapsed / max(self._expected, 0.5)
        # Gaussian peak at ratio=1.0 → fastest polling near expected transition.
        proximity = math.exp(-2.0 * (ratio - 1.0) ** 2)
        interval = self.base_interval * (1.0 - 0.5 * proximity)
        return max(0.3, min(interval, self.base_interval * 1.5))


class OcrLoop:
    """Background thread that polls MPV for video frames and shows OCR translations."""

    def __init__(
        self,
        command: Callable,
        config: "Config",
        overlay: "PushOverlay",
        gpu_scheduler=None,
    ):
        self._cmd = command
        self._cfg = config.ocr
        self._overlay = overlay
        self._gpu = gpu_scheduler
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None

        # Per-block age counters: consecutive frames each block has appeared.
        # Used for both stability gating (age >= stability_frames to show)
        # and watermark suppression (age >= watermark_frames to hide).
        self._block_ages: dict = {}
        self._last_shown_raw: str = ""   # raw text that produced the current overlay
        # Display duration limiting
        self._shown_at: Optional[float] = None
        self._expired: set = set()       # combined texts suppressed after expiry
        # Cooldown: recently shown text → monotonic time first shown.
        # Prevents title cards that fade in/out from being shown twice.
        self._recently_shown: dict[str, float] = {}
        # Pre-detected watermarks (text present across multiple video samples).
        # Cached per video path so re-scans are skipped on seek.
        self._watermarks: set = set()
        self._watermark_bboxes: list[tuple[int, int, int, int]] = []
        self._watermark_path: str = ""
        # Per-tick bbox tracking for runtime watermark promotion.
        self._block_bboxes: dict[str, tuple[int, int, int, int]] = {}
        # Fires once after the first definitive OCR decision (show or hide)
        self._first_ready: Optional[Event] = None
        self._sched = _BayesianScheduler(self._cfg.interval, self._cfg.lookahead_seconds)
        self._prev_thumb = None  # 64×64 grayscale for frame-diff gate
        self._diff_skip = 0     # consecutive ticks skipped by frame-diff gate
        # Crop hash cache: quantised bbox key → (recognised text, crop pixel hash).
        # Skips easyocr recognition when the pixel content of a detected region hasn't changed.
        self._crop_cache: dict[tuple, tuple[str, bytes]] = {}
        # Last detected bboxes for region-aware frame diff.
        self._last_regions: list[tuple[int, int, int, int]] = []

    # ── public API ───────────────────────────────────────────────────────────

    def start(self, first_ready: Optional[Event] = None):
        """Start the OCR polling thread."""
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()
        self._block_ages = {}
        self._block_bboxes = {}
        self._last_shown_raw = ""
        self._shown_at = None
        self._expired.clear()
        self._recently_shown.clear()
        self._first_ready = first_ready
        self._sched.reset()
        self._prev_thumb = None
        self._diff_skip = 0
        self._crop_cache = {}
        self._last_regions = []
        self._thread = threading.Thread(
            target=self._loop, daemon=True, name="ocr-loop"
        )
        self._thread.start()

    def stop(self):
        """Stop polling and blank the OCR overlay."""
        self._stop.set()
        self._overlay.hide()
        self._block_ages = {}
        self._block_bboxes = {}
        self._last_shown_raw = ""
        self._shown_at = None
        self._expired.clear()
        self._recently_shown.clear()
        self._first_ready = None
        self._sched.reset()
        self._prev_thumb = None
        self._diff_skip = 0
        self._crop_cache = {}
        self._last_regions = []

    def shutdown(self):
        self.stop()

    # ── internals ────────────────────────────────────────────────────────────

    def _notify_ready(self):
        """Fire the first_ready event once after a definitive show/hide decision."""
        if self._first_ready is not None:
            self._first_ready.set()
            self._first_ready = None
            # Release GPU priority so audio translation can proceed.
            if self._gpu is not None:
                self._gpu.clear_ocr_priority()

    @staticmethod
    def _bboxes_overlap(a: tuple, b: tuple, tolerance: int = 50) -> bool:
        """Check if two bounding boxes overlap by centre proximity."""
        ax = (a[0] + a[1]) / 2
        ay = (a[2] + a[3]) / 2
        bx = (b[0] + b[1]) / 2
        by = (b[2] + b[3]) / 2
        return abs(ax - bx) < tolerance and abs(ay - by) < tolerance

    def _scan_watermarks(self, tmpfile: str):
        """Sample frames from across the video to pre-detect persistent watermarks.

        Text blocks whose bounding boxes overlap across >= 2 sampled frames are
        treated as permanent watermarks (studio logos, channel names).  Matching
        by spatial position (not exact text) catches watermarks that OCR reads
        slightly differently each time.

        Stops scanning early once watermarks are confirmed (2 frames agree).
        Results are cached per video path so seeks don't re-scan.
        """
        from .ocr import capture_frame_av, extract_blocks  # noqa: PLC0415

        try:
            path = str(self._cmd("get_property", "path") or "")
            duration = float(self._cmd("get_property", "duration") or 0)
        except Exception:
            return

        if not path or duration <= 0:
            return

        # Already scanned this video — reuse cached watermarks.
        if path == self._watermark_path:
            if self._watermarks:
                log.debug("reusing %d cached watermark(s)", len(self._watermarks))
            return

        timestamps = [duration * f for f in (0.1, 0.3, 0.5, 0.7, 0.9)]
        # Collect (text, bbox) from every frame for spatial matching.
        all_frame_blocks: list[list[tuple[str, tuple]]] = []
        n_captured = 0

        for ts in timestamps:
            if self._stop.is_set():
                return
            if not capture_frame_av(path, ts, tmpfile):
                continue
            blocks = extract_blocks(tmpfile, self._cfg)
            frame_blocks = [(t, b) for t, b in blocks if len(t) <= self._cfg.max_chars]
            all_frame_blocks.append(frame_blocks)
            n_captured += 1
            # Early exit: check if any bbox from frame 0 overlaps with frame 1+.
            if n_captured >= 2:
                found = False
                for _, bbox_a in all_frame_blocks[0]:
                    for fi in range(1, len(all_frame_blocks)):
                        for _, bbox_b in all_frame_blocks[fi]:
                            if self._bboxes_overlap(bbox_a, bbox_b):
                                found = True
                                break
                        if found:
                            break
                    if found:
                        break
                if found:
                    break

        self._watermark_path = path
        if n_captured < 2:
            self._watermarks = set()
            self._watermark_bboxes = []
            return

        # A bbox is a watermark if it overlaps with a bbox in at least one other frame.
        watermark_bboxes: list[tuple] = []
        watermark_texts: set[str] = set()
        for i, blocks_i in enumerate(all_frame_blocks):
            for text_i, bbox_i in blocks_i:
                for j, blocks_j in enumerate(all_frame_blocks):
                    if i == j:
                        continue
                    if any(self._bboxes_overlap(bbox_i, bbox_j) for _, bbox_j in blocks_j):
                        watermark_bboxes.append(bbox_i)
                        watermark_texts.add(text_i)
                        break

        self._watermarks = watermark_texts
        self._watermark_bboxes = watermark_bboxes
        if self._watermarks:
            log.info(
                "pre-detected %d watermark(s) from %d frames: %s",
                len(self._watermarks), n_captured,
                ", ".join(repr(w) for w in self._watermarks),
            )

    @staticmethod
    def _crop_hash(img, bbox: tuple[int, int, int, int]) -> bytes:
        """Fast perceptual hash of a cropped region (~0.5 ms).

        Downscales the crop to a tiny 16x16 grayscale thumbnail and returns
        the raw bytes.  Two crops with the same hash contain visually identical
        text — easyocr recognition can be skipped entirely.
        """
        x_min, x_max, y_min, y_max = bbox
        crop = img.crop((x_min, y_min, x_max, y_max))
        return crop.convert("L").resize((16, 16)).tobytes()

    def _regions_changed(self, tmpfile: str) -> bool:
        """Check if the previously detected text regions have changed pixels.

        Compares 16x16 grayscale thumbnails of each region from the last tick
        against the new frame.  If all text regions look the same, returns
        False — the caller can skip easyocr detection entirely.

        Returns True (assume changed) when there are no cached regions, or
        when any region's content has shifted.
        """
        if not self._last_regions:
            return True
        try:
            from PIL import Image  # noqa: PLC0415
            img = Image.open(tmpfile)
        except Exception:
            return True
        for bbox in self._last_regions:
            key = self._bbox_key_static(bbox)
            cached = self._crop_cache.get(key)
            if cached is None:
                return True
            new_hash = self._crop_hash(img, bbox)
            if new_hash != cached[1]:
                return True
        return False

    @staticmethod
    def _bbox_key_static(bbox: tuple[int, int, int, int]) -> tuple[int, int]:
        """Quantise a bbox centre to a 50px grid for spatial grouping."""
        cx = (bbox[0] + bbox[1]) // 2
        cy = (bbox[2] + bbox[3]) // 2
        return (cx // 50 * 50, cy // 50 * 50)

    @staticmethod
    def _has_text_activity(thumb) -> bool:
        """Fast edge-density check for text-like content (~0.2 ms).

        Operates on the 64x64 grayscale thumbnail already computed by
        ``_frame_changed``.  Text overlays produce clusters of sharp edges
        that raise the local variance; near-uniform frames (solid colours,
        fades, dark scenes) score far below the threshold and can skip
        easyocr detection entirely.

        Conservative: returns True (= "might have text") whenever in doubt
        so we never miss actual text.
        """
        import numpy as np  # noqa: PLC0415
        # Horizontal gradient approximation — text has sharp horizontal strokes.
        grad = np.abs(np.diff(thumb, axis=1))
        strong = np.count_nonzero(grad > 15)
        # ~0.8 % of 64×63 = ~32 strong-edge pixels.  A single short word
        # easily produces 30-50; a text-free flat scene produces < 10.
        return strong > 32

    def _frame_changed(self, tmpfile: str) -> bool:
        """Quick pixel-diff gate: skip full OCR if the frame is unchanged.

        Compares a small grayscale thumbnail (64x64) against the previous
        capture.  Aggressive downscaling makes this robust to small movements
        like scrolling/animated text or camera shake, while still catching
        new text, scene cuts, and text disappearances (~1-2 ms per check).
        """
        try:
            from PIL import Image   # noqa: PLC0415
            import numpy as np      # noqa: PLC0415
            thumb = np.asarray(
                Image.open(tmpfile).convert("L").resize((64, 64)),
                dtype=np.float32,
            )
        except Exception:
            self._prev_thumb = None
            return True  # can't load → assume changed

        prev = self._prev_thumb
        self._prev_thumb = thumb

        if prev is None:
            return True

        diff = float(np.mean(np.abs(thumb - prev)))
        # Higher threshold when text is already showing — tolerate small
        # movements (scrolling text, pans) without re-triggering full OCR.
        threshold = 8.0 if self._last_shown_raw else 3.0
        return diff > threshold

    # ── binary-search scan mode ───────────────────────────────────────────

    @staticmethod
    def _load_thumb_from(tmpfile: str):
        """Load a 64×64 grayscale thumbnail from an image file."""
        try:
            from PIL import Image  # noqa: PLC0415
            import numpy as np     # noqa: PLC0415
            return np.asarray(
                Image.open(tmpfile).convert("L").resize((64, 64)),
                dtype=np.float32,
            )
        except Exception:
            return None

    @staticmethod
    def _thumbs_match(a, b, threshold: float = 5.0) -> bool:
        """Return True if two 64×64 thumbnails are visually similar."""
        if a is None or b is None:
            return False
        import numpy as np  # noqa: PLC0415
        return float(np.mean(np.abs(a - b))) < threshold

    @staticmethod
    def _region_crops(img, regions: list) -> list:
        """Extract padded 32×32 grayscale crop arrays for each text region.

        Padding (30 px) tolerates small text movement between frames.
        """
        import numpy as np  # noqa: PLC0415
        crops = []
        w, h = img.size
        for bbox in regions:
            x_min, x_max, y_min, y_max = bbox
            pad = 30
            crop = img.crop((
                max(0, x_min - pad), max(0, y_min - pad),
                min(w, x_max + pad), min(h, y_max + pad),
            ))
            crops.append(np.asarray(
                crop.convert("L").resize((32, 32)), dtype=np.float32,
            ))
        return crops

    def _regions_match_file(self, tmpfile: str, ref_regions: list,
                            ref_crops: list, threshold: float = 8.0) -> bool:
        """Compare text region crops from *tmpfile* against reference crops.

        Returns True when every region crop is visually close to its reference
        (mean-absolute-difference < *threshold*).  Padding in the crop handles
        small text movements.
        """
        import numpy as np  # noqa: PLC0415
        try:
            from PIL import Image  # noqa: PLC0415
            img = Image.open(tmpfile)
        except Exception:
            return False
        w, h = img.size
        for bbox, ref in zip(ref_regions, ref_crops):
            x_min, x_max, y_min, y_max = bbox
            pad = 30
            crop = np.asarray(
                img.crop((
                    max(0, x_min - pad), max(0, y_min - pad),
                    min(w, x_max + pad), min(h, y_max + pad),
                )).convert("L").resize((32, 32)),
                dtype=np.float32,
            )
            if float(np.mean(np.abs(crop - ref))) > threshold:
                return False
        return True

    def _check_changed_at(self, path: str, ts: float,
                          ref_regions: list, ref_crops: list, ref_thumb,
                          tmpfile: str) -> bool:
        """Capture frame at *ts* and check whether text content differs.

        Uses region crops when regions are known (focused on text areas,
        ignores background motion); falls back to full-frame thumbnail
        when no text regions exist.
        """
        from .ocr import capture_frame_av  # noqa: PLC0415
        if not capture_frame_av(path, ts, tmpfile):
            return True  # can't capture → assume changed
        if ref_regions and ref_crops:
            return not self._regions_match_file(tmpfile, ref_regions, ref_crops)
        thumb = self._load_thumb_from(tmpfile)
        return not self._thumbs_match(ref_thumb, thumb)

    def _bisect(self, path: str, lo: float, hi: float,
                ref_regions: list, ref_crops: list, ref_thumb,
                tmpfile: str) -> float:
        """Binary search for the approximate timestamp where text changes.

        Narrows the interval to ≤ 0.5 s precision.  Compares text-region
        crops when known, full-frame thumbnails otherwise.
        """
        while hi - lo > 0.5 and not self._stop.is_set():
            mid = (lo + hi) / 2
            if self._check_changed_at(path, mid, ref_regions, ref_crops,
                                      ref_thumb, tmpfile):
                hi = mid
            else:
                lo = mid
        return hi

    def _ocr_at(self, path: str, ts: float, tmpfile: str):
        """Full OCR (detect + recognise + filter) at a video timestamp.

        Returns ``(combined_text, regions, region_crops)``.
        """
        from .ocr import capture_frame_av, detect_regions, recognize_region  # noqa: PLC0415

        if not capture_frame_av(path, ts, tmpfile):
            return ("", [], [])

        result = detect_regions(
            tmpfile, self._cfg,
            exclude_bboxes=self._watermark_bboxes or None,
        )
        if result is None:
            return ("", [], [])

        img, bboxes = result
        blocks: list[tuple[str, tuple]] = []
        for bbox in bboxes:
            text = recognize_region(img, bbox, self._cfg)
            if text and len(text) <= self._cfg.max_chars and text not in self._watermarks:
                blocks.append((text, bbox))

        combined = "\n".join(t for t, _ in blocks)
        if len(combined) > self._cfg.max_chars:
            combined = ""
            blocks = []

        regions = [bbox for _, bbox in blocks]
        crops = self._region_crops(img, regions) if regions else []
        return (combined, regions, crops)

    def _scan_window(self, path: str, pos: float, tmpfile: str):
        """Scan the lookahead window using binary search for text transitions.

        Algorithm
        ---------
        1. Sample thumbnails at ~1 s intervals across ``[pos, pos+lookahead]``
           (~50 ms per sample — cheap).
        2. Between adjacent samples whose thumbnails differ, binary-search
           for the exact transition time.  The search compares text-region
           crops (focused on text, ignores background), falling back to
           full-frame thumbnails when no regions are known.
        3. Full OCR (detect + recognise) only at the start and at each
           transition point found.
        4. Batch-translate all unique texts.

        Returns ``[(video_ts, raw_text, translation), ...]``.
        """
        from .ocr import capture_frame_av  # noqa: PLC0415
        from .ocr_translate import translate_text  # noqa: PLC0415

        lookahead = self._cfg.lookahead_seconds
        scan_step = min(1.0, lookahead / 2)

        # Phase 1: quick thumbnail sampling to locate change intervals.
        samples: list[tuple[float, object]] = []
        t = pos
        while t <= pos + lookahead + 0.01:
            if self._stop.is_set():
                return []
            if capture_frame_av(path, t, tmpfile):
                thumb = self._load_thumb_from(tmpfile)
                if thumb is not None:
                    samples.append((t, thumb))
            t += scan_step

        if not samples:
            return []

        # Phase 2: full OCR at window start.
        with (self._gpu.gpu(self._stop, defer_to_audio=True) if self._gpu else nullcontext(True)) as acquired:
            if not acquired:
                return []
            start_text, start_regions, start_crops = self._ocr_at(path, pos, tmpfile)
        start_thumb = samples[0][1]

        # Phase 3: find change intervals and binary-search each one.
        transitions: list[tuple[float, str]] = []
        prev_text = start_text
        prev_regions = start_regions
        prev_crops = start_crops
        prev_thumb = start_thumb

        for i in range(len(samples) - 1):
            if self._stop.is_set():
                return []
            _ts_a, thumb_a = samples[i]
            ts_b, _thumb_b = samples[i + 1]

            if self._thumbs_match(thumb_a, _thumb_b):
                continue  # no visible change in this interval

            # Binary search for transition point.
            ts_a = samples[i][0]
            transition_ts = self._bisect(
                path, ts_a, ts_b,
                prev_regions, prev_crops, prev_thumb, tmpfile,
            )

            # Full OCR at the transition.
            with (self._gpu.gpu(self._stop, defer_to_audio=True) if self._gpu else nullcontext(True)) as acquired:
                if not acquired:
                    return []
                new_text, new_regions, new_crops = self._ocr_at(
                    path, transition_ts, tmpfile,
                )
            if new_text != prev_text:
                transitions.append((transition_ts, new_text))
                prev_text = new_text
            if new_regions:
                prev_regions = new_regions
                prev_crops = new_crops
            prev_thumb = self._load_thumb_from(tmpfile)

        # Build result list.
        result_raw: list[tuple[float, str]] = []
        if start_text and start_text != self._last_shown_raw:
            result_raw.append((pos, start_text))
        elif not start_text and self._last_shown_raw:
            result_raw.append((pos, ""))
        result_raw.extend(transitions)

        if not result_raw:
            return []

        # Phase 4: batch translate all unique texts.
        unique = {raw for _, raw in result_raw if raw}
        trans_map: dict[str, str] = {}
        for text in unique:
            if self._stop.is_set():
                return []
            with (self._gpu.gpu(self._stop, defer_to_audio=True) if self._gpu else nullcontext(True)) as acquired:
                if not acquired:
                    return []
                trans_map[text] = translate_text(
                    text, self._cfg.source_lang, self._cfg.target_lang,
                )

        return [(ts, raw, trans_map.get(raw, "")) for ts, raw in result_raw]

    def _wait_for_pos(self, target: float):
        """Block until playback reaches *target* seconds (or stop is set)."""
        while not self._stop.is_set():
            try:
                current = float(self._cmd("get_property", "time-pos") or 0.0)
            except Exception:
                return
            if current >= target - 0.2:
                return
            wait = min(target - current, 0.3)
            if self._stop.wait(wait):
                return

    def _show_entry(self, raw: str, translation: str) -> None:
        """Display or hide an OCR entry, handling cooldown/expiry/min-duration."""
        if raw:
            if self._cfg.cooldown_seconds > 0:
                st = self._recently_shown.get(raw)
                if st and time.monotonic() - st < self._cfg.cooldown_seconds:
                    return
            if raw in self._expired:
                return
            if raw == self._last_shown_raw:
                return

            log.info("OCR translation: %r -> %r", raw, translation)
            self._overlay.show(translation)
            self._last_shown_raw = raw
            self._shown_at = time.monotonic()
            self._recently_shown[raw] = time.monotonic()
        else:
            # Text disappeared.
            if self._last_shown_raw:
                min_d = self._cfg.min_display_seconds
                if (min_d > 0 and self._shown_at is not None
                        and time.monotonic() - self._shown_at < min_d):
                    return  # keep showing until minimum duration
                self._expired.discard(self._last_shown_raw)
            self._overlay.hide()
            self._last_shown_raw = ""
            self._shown_at = None

    def _check_display_expiry(self) -> None:
        """Hide the overlay if the current text has been shown too long."""
        if (self._last_shown_raw
                and self._cfg.max_display_seconds > 0
                and self._shown_at is not None
                and time.monotonic() - self._shown_at
                >= self._cfg.max_display_seconds):
            self._overlay.hide()
            self._expired.add(self._last_shown_raw)
            self._last_shown_raw = ""
            self._shown_at = None

    def _loop_scan(self, tmpfile: str):
        """Main loop for lookahead mode: binary-search scan → display → repeat.

        When interleaving with audio translation (gpu scheduler present),
        operates in catch-up mode: after audio finishes a chunk, OCR batch-
        scans consecutive windows up to the audio frontier, queues results,
        releases GPU priority, then displays at playback time.
        """
        scan_pos: float = 0.0
        pending: list[tuple[float, str, str]] = []

        while not self._stop.is_set():
            # ── Phase A: Catch-up scan to audio frontier ─────────────────
            if (self._gpu is not None
                    and self._gpu._ocr_priority.is_set()
                    and self._gpu.audio_frontier > scan_pos):

                frontier = self._gpu.audio_frontier
                try:
                    path = str(self._cmd("get_property", "path") or "")
                except Exception:
                    path = ""

                if path:
                    log.debug(
                        "OCR catch-up: scanning %.1f → %.1f",
                        scan_pos, frontier,
                    )
                    while scan_pos < frontier and not self._stop.is_set():
                        try:
                            entries = self._scan_window(path, scan_pos, tmpfile)
                        except Exception:
                            log.warning("scan_window error in catch-up", exc_info=True)
                            entries = []
                        pending.extend(entries)
                        scan_pos += self._cfg.lookahead_seconds

                # Done catching up — let audio proceed with its next chunk.
                self._gpu.clear_ocr_priority()
                log.debug("OCR catch-up done, released priority")

            # ── Phase B: Display queued entries from catch-up ─────────────
            while pending and not self._stop.is_set():
                # Interrupt display if audio yielded again (new catch-up needed).
                if (self._gpu is not None
                        and self._gpu._ocr_priority.is_set()
                        and self._gpu.audio_frontier > scan_pos):
                    break

                video_ts, raw, translation = pending.pop(0)
                self._wait_for_pos(video_ts)
                if self._stop.is_set():
                    break
                self._show_entry(raw, translation)
                self._notify_ready()

            # If a new catch-up is needed, loop back to Phase A immediately.
            if (self._gpu is not None
                    and self._gpu._ocr_priority.is_set()
                    and self._gpu.audio_frontier > scan_pos):
                continue

            # ── Phase C: Normal mode — single window at playback position ─
            try:
                pos = float(self._cmd("get_property", "time-pos") or 0.0)
                path = str(self._cmd("get_property", "path") or "")
            except Exception:
                if self._stop.wait(1.0):
                    break
                continue

            if not path:
                if self._stop.wait(1.0):
                    break
                continue

            scan_pos = pos
            self._check_display_expiry()

            try:
                entries = self._scan_window(path, pos, tmpfile)
            except Exception:
                log.warning("scan_window error", exc_info=True)
                entries = []

            scan_pos = pos + self._cfg.lookahead_seconds

            if not entries:
                self._notify_ready()
                if self._stop.wait(self._cfg.interval):
                    break
                continue

            for video_ts, raw, translation in entries:
                if self._stop.is_set():
                    break

                self._wait_for_pos(video_ts)
                if self._stop.is_set():
                    break

                self._show_entry(raw, translation)
                self._notify_ready()

    # ── tick-based polling mode (no lookahead) ────────────────────────────

    def _loop(self):
        fd, tmpfile = tempfile.mkstemp(suffix=".png", prefix="mpv-ocr-")
        os.close(fd)
        try:
            # Pre-scan for watermarks before the first tick.
            if self._cfg.watermark_frames > 0:
                self._scan_watermarks(tmpfile)
            if self._cfg.lookahead_seconds > 0:
                self._loop_scan(tmpfile)
            elif self._cfg.binary_refine:
                self._loop_poll_refine(tmpfile)
            else:
                while not self._stop.wait(self._sched.next_interval()):
                    try:
                        self._tick(tmpfile)
                    except Exception:
                        log.warning("OCR tick error", exc_info=True)
        finally:
            try:
                os.unlink(tmpfile)
            except OSError:
                pass

    # ── poll-and-refine mode (binary search on transitions) ──────────────

    def _loop_poll_refine(self, tmpfile: str):
        """Poll at regular intervals; binary-search text boundaries on transitions.

        Every ``interval`` seconds, capture a frame and do lightweight detection.
        When the text state changes (appears / disappears / changes), binary-
        search between the previous and current check timestamps (via
        ``capture_frame_av``) to find the precise transition, then run full OCR
        and translate only at that point.
        """
        from .ocr import capture_frame_av, capture_screenshot  # noqa: PLC0415
        from .ocr import detect_regions, recognize_region       # noqa: PLC0415
        from .ocr_translate import translate_text                # noqa: PLC0415

        prev_ts: float = 0.0       # video timestamp of previous poll
        prev_had_text: bool = False
        prev_text: str = ""
        prev_regions: list = []
        prev_crops: list = []
        prev_thumb = None

        while not self._stop.wait(self._cfg.interval):
            # 1. Current position and path.
            try:
                pos = float(self._cmd("get_property", "time-pos") or 0.0)
                path = str(self._cmd("get_property", "path") or "")
            except Exception:
                continue
            if not path:
                continue

            # 2. Capture frame at current position.
            if not capture_frame_av(path, pos, tmpfile):
                if not capture_screenshot(self._cmd, tmpfile):
                    continue

            # 3. Thumbnail diff gate — skip when the frame is unchanged.
            if not self._frame_changed(tmpfile):
                self._check_display_expiry()
                continue

            # 4. Detect text regions (CRAFT).
            with (self._gpu.gpu(self._stop, defer_to_audio=True)
                  if self._gpu else nullcontext(True)) as acquired:
                if not acquired:
                    continue
                result = detect_regions(
                    tmpfile, self._cfg,
                    exclude_bboxes=self._watermark_bboxes or None,
                )
            if result is None:
                has_text = False
                combined = ""
                cur_regions: list = []
                cur_crops: list = []
            else:
                img, bboxes = result
                # Quick recognition pass.
                blocks: list[tuple[str, tuple]] = []
                with (self._gpu.gpu(self._stop, defer_to_audio=True)
                      if self._gpu else nullcontext(True)) as acquired:
                    if not acquired:
                        continue
                    for bbox in bboxes:
                        text = recognize_region(img, bbox, self._cfg)
                        if (text
                                and len(text) <= self._cfg.max_chars
                                and text not in self._watermarks):
                            blocks.append((text, bbox))
                combined = "\n".join(t for t, _ in blocks)
                if len(combined) > self._cfg.max_chars:
                    combined = ""
                    blocks = []
                has_text = bool(combined)
                cur_regions = [bbox for _, bbox in blocks]
                cur_crops = (self._region_crops(img, cur_regions)
                             if cur_regions else [])

            cur_thumb = self._prev_thumb  # set by _frame_changed

            # 5. Compare against previous state and act on transitions.
            if has_text and not prev_had_text:
                # ── Text appeared: binary search for start ────────────────
                if prev_ts > 0 and pos - prev_ts > 0.5:
                    start_ts = self._bisect(
                        path, prev_ts, pos,
                        [], [], prev_thumb, tmpfile,
                    )
                else:
                    start_ts = pos

                # Full OCR at transition point.
                with (self._gpu.gpu(self._stop, defer_to_audio=True)
                      if self._gpu else nullcontext(True)) as acquired:
                    if not acquired:
                        continue
                    refined_text, refined_regions, refined_crops = \
                        self._ocr_at(path, start_ts, tmpfile)

                if not refined_text:
                    refined_text = combined

                with (self._gpu.gpu(self._stop, defer_to_audio=True)
                      if self._gpu else nullcontext(True)) as acquired:
                    if not acquired:
                        continue
                    translation = translate_text(
                        refined_text,
                        self._cfg.source_lang,
                        self._cfg.target_lang,
                    )

                log.info(
                    "poll-refine: text appeared at %.1fs (polled %.1f→%.1f), "
                    "%r -> %r",
                    start_ts, prev_ts, pos, refined_text, translation,
                )
                self._show_entry(refined_text, translation)
                self._notify_ready()

            elif not has_text and prev_had_text:
                # ── Text disappeared: binary search for end ───────────────
                if prev_ts > 0 and pos - prev_ts > 0.5:
                    end_ts = self._bisect(
                        path, prev_ts, pos,
                        prev_regions, prev_crops, prev_thumb, tmpfile,
                    )
                else:
                    end_ts = pos

                log.info(
                    "poll-refine: text disappeared at %.1fs (polled %.1f→%.1f)",
                    end_ts, prev_ts, pos,
                )
                self._show_entry("", "")
                self._notify_ready()

            elif has_text and prev_had_text and combined != prev_text:
                # ── Text changed: binary search for transition ────────────
                if prev_ts > 0 and pos - prev_ts > 0.5:
                    change_ts = self._bisect(
                        path, prev_ts, pos,
                        prev_regions, prev_crops, prev_thumb, tmpfile,
                    )
                else:
                    change_ts = pos

                with (self._gpu.gpu(self._stop, defer_to_audio=True)
                      if self._gpu else nullcontext(True)) as acquired:
                    if not acquired:
                        continue
                    refined_text, refined_regions, refined_crops = \
                        self._ocr_at(path, change_ts, tmpfile)

                if not refined_text:
                    refined_text = combined

                with (self._gpu.gpu(self._stop, defer_to_audio=True)
                      if self._gpu else nullcontext(True)) as acquired:
                    if not acquired:
                        continue
                    translation = translate_text(
                        refined_text,
                        self._cfg.source_lang,
                        self._cfg.target_lang,
                    )

                log.info(
                    "poll-refine: text changed at %.1fs (polled %.1f→%.1f), "
                    "%r -> %r",
                    change_ts, prev_ts, pos, refined_text, translation,
                )
                self._show_entry(refined_text, translation)
                self._notify_ready()

            else:
                # ── No transition — enforce display limits ────────────────
                self._check_display_expiry()
                self._notify_ready()

            # 6. Update state for next poll.
            prev_ts = pos
            prev_had_text = has_text
            prev_text = combined
            prev_regions = cur_regions
            prev_crops = cur_crops
            prev_thumb = cur_thumb

    # ── tick-based polling mode (no lookahead) ────────────────────────────

    def _tick(self, tmpfile: str):
        from .ocr import capture_frame_av, capture_screenshot  # noqa: PLC0415
        from .ocr import detect_regions, recognize_region       # noqa: PLC0415
        from .ocr_translate import translate_text                # noqa: PLC0415

        wf = self._cfg.watermark_frames
        t_start = time.monotonic()
        offset = self._sched.optimal_offset()

        # 1. Capture video frame.
        # Proactive lookahead: always capture ahead when configured, using
        # a Bayesian-tuned offset that shrinks near expected transitions.
        try:
            pos = float(self._cmd("get_property", "time-pos") or 0.0)
            path = str(self._cmd("get_property", "path") or "")
        except Exception:
            pos, path = 0.0, ""
        target = pos + offset
        if path and capture_frame_av(path, target, tmpfile):
            pass
        elif not capture_screenshot(self._cmd, tmpfile):
            return

        # 1b. Quick frame-diff gate — skip full OCR when the frame is unchanged.
        #     When nothing is showing, allow at most 1 skip so new text is
        #     caught within 2 ticks.  When text is showing, allow up to 5.
        if not self._frame_changed(tmpfile):
            self._diff_skip += 1
            max_skip = 5 if self._last_shown_raw else 1
            if self._diff_skip < max_skip:
                # Still enforce display duration limit even when skipping OCR.
                if (self._last_shown_raw
                        and self._cfg.max_display_seconds > 0
                        and self._shown_at is not None
                        and time.monotonic() - self._shown_at >= self._cfg.max_display_seconds):
                    log.debug("ocr overlay expired after %.0fs (frame unchanged)", self._cfg.max_display_seconds)
                    self._sched.notify(False, time.monotonic())
                    self._overlay.hide()
                    self._expired.add(self._last_shown_raw)
                    self._last_shown_raw = ""
                    self._shown_at = None
                return
            # max skips reached — fall through to full OCR
        self._diff_skip = 0

        # 1c. Bayesian-guided processing level.
        #     Low hazard  = deep in a stable period → aggressive lightweight gates.
        #     High hazard = transition expected soon → full OCR to catch it.
        hazard = self._sched._hazard()
        transition_zone = hazard > 0.3

        # 1d. Edge density pre-filter — skip easyocr on clearly text-free frames.
        #     Only applied when nothing is showing and the Bayesian model says
        #     we're in a stable off-period (low hazard).  Near expected text-on
        #     transitions the pre-filter is bypassed for maximum sensitivity.
        if not self._last_shown_raw and not transition_zone:
            if self._prev_thumb is not None and not self._has_text_activity(self._prev_thumb):
                log.debug("edge density below threshold, hazard=%.2f — skipping OCR", hazard)
                self._notify_ready()
                return

        # 1e. Region-aware diff gate — if the frame changed overall but the
        #     text regions specifically haven't, skip the expensive easyocr
        #     detection and reuse the previous bboxes.  Catches background
        #     motion with static text overlays (~2 ms vs ~200 ms for CRAFT).
        #     Disabled near expected transitions so we don't miss text appearing
        #     or disappearing.
        reuse_regions = False
        if self._last_regions and not transition_zone and not self._regions_changed(tmpfile):
            reuse_regions = True
            log.debug("text regions unchanged, hazard=%.2f — skipping easyocr", hazard)

        # 2. Detect text regions (easyocr CRAFT) or reuse previous bboxes.
        #    GPU scheduler: hold lock for detect + recognise, release before filtering.
        with (self._gpu.gpu(self._stop, defer_to_audio=True) if self._gpu else nullcontext(True)) as acquired:
            if not acquired:
                return
            if reuse_regions:
                try:
                    from PIL import Image  # noqa: PLC0415
                    img = Image.open(tmpfile)
                except Exception:
                    return
                bboxes = list(self._last_regions)
            else:
                result = detect_regions(
                    tmpfile, self._cfg,
                    exclude_bboxes=self._watermark_bboxes or None,
                )
                if result is None:
                    return
                img, bboxes = result

            # 2b. Recognise each region — crop hash cache skips easyocr for
            #     unchanged crops (~0.5 ms hash vs ~80 ms recognition per region).
            blocks: list[tuple[str, tuple[int, int, int, int]]] = []
            new_crop_cache: dict[tuple, tuple[str, bytes]] = {}
            cache_hits = 0
            for bbox in bboxes:
                key = self._bbox_key_static(bbox)
                chash = self._crop_hash(img, bbox)
                cached = self._crop_cache.get(key)
                if cached is not None and cached[1] == chash:
                    # Crop pixels identical — reuse recognised text.
                    text = cached[0]
                    cache_hits += 1
                else:
                    text = recognize_region(img, bbox, self._cfg)
                if text is not None:
                    new_crop_cache[key] = (text, chash)
                    blocks.append((text, bbox))
            self._crop_cache = new_crop_cache
            self._last_regions = [bbox for _, bbox in blocks] if blocks else []
            if cache_hits:
                log.debug("crop cache: %d/%d hits, hazard=%.2f", cache_hits, len(bboxes), hazard)

        raw_blocks = [text for text, _bbox in blocks]
        self._block_bboxes = {text: bbox for text, bbox in blocks}
        raw_blocks_detected = raw_blocks  # unfiltered copy for scene-cut detection

        # 3. Drop individual blocks that are too long (paragraph guard).
        raw_blocks = [b for b in raw_blocks if len(b) <= self._cfg.max_chars]

        # 3a. Remove pre-detected watermarks by text (safety net — regions are
        #     already spatially masked, but text matching catches edge cases).
        if self._watermarks:
            raw_blocks = [b for b in raw_blocks if b not in self._watermarks]

        # 3b. Per-block age tracking — count consecutive frames each block appears.
        # Uses bbox centre (rounded to 50px grid) as key so OCR text variation
        # in the same spatial region still accumulates age correctly.
        new_ages: dict = {}
        block_bbox_key: dict[str, tuple] = {}
        for b in raw_blocks:
            bbox = self._block_bboxes.get(b)
            if bbox:
                bk = self._bbox_key_static(bbox)
                block_bbox_key[b] = bk
                # Carry forward age from any previous block at the same position.
                prev_age = self._block_ages.get(bk, 0)
                new_ages[bk] = prev_age + 1
            else:
                new_ages[b] = self._block_ages.get(b, 0) + 1
        self._block_ages = new_ages

        # 3c. Per-block stability filter — only keep blocks seen in >=
        #     stability_frames consecutive ticks.  Individual blocks stabilise
        #     independently, so OCR noise in one region does not reset
        #     stability for other regions.
        #     First detection (nothing showing yet) skips the filter so text
        #     appears immediately.  When watermarks have been pre-scanned,
        #     non-watermark text is already trusted — skip stability entirely.
        sf = self._cfg.stability_frames
        if sf > 1 and self._last_shown_raw and not self._watermark_path:
            raw_blocks = [
                b for b in raw_blocks
                if self._block_ages.get(block_bbox_key.get(b, b), 0) >= sf
            ]

        # 3d. Per-block watermark filter — drop blocks persisting >=
        #     watermark_frames consecutive ticks and promote them to spatial
        #     exclusion so easyocr skips those regions entirely in future frames.
        if wf > 0:
            for b in raw_blocks:
                key = block_bbox_key.get(b, b)
                if self._block_ages.get(key, 0) >= wf and b in self._block_bboxes:
                    self._watermark_bboxes.append(self._block_bboxes[b])
                    self._watermarks.add(b)
                    log.info("runtime watermark %r promoted to spatial exclusion", b)
            raw_blocks = [
                b for b in raw_blocks
                if self._block_ages.get(block_bbox_key.get(b, b), 0) < wf
            ]

        # 4. Combine into one string (newline-separated).
        combined = "\n".join(raw_blocks)

        # 5. Drop if the combined result is also too long.
        if len(combined) > self._cfg.max_chars:
            combined = ""

        # 6. Nothing on screen → apply lookahead delay, enforce min display, hide.
        if not combined:
            # When capturing ahead, delay so the overlay clears when the text
            # actually disappears from the screen.
            if offset > 0:
                elapsed = time.monotonic() - t_start
                wait = offset - elapsed
                if wait > 0 and self._stop.wait(wait):
                    return  # cancelled during lookahead wait
            # Minimum display duration — keep the overlay visible for at least
            # min_display_seconds even after the source text leaves the screen.
            min_d = self._cfg.min_display_seconds
            if (min_d > 0 and self._last_shown_raw
                    and self._shown_at is not None
                    and time.monotonic() - self._shown_at < min_d):
                return  # keep showing until minimum duration is met
            # When text naturally leaves the screen, remove it from the expiry set
            # so it can be shown again if it reappears later (different scene / time).
            if self._last_shown_raw:
                self._expired.discard(self._last_shown_raw)
                self._sched.notify(False, time.monotonic())
            self._overlay.hide()
            self._last_shown_raw = ""
            self._shown_at = None
            # True scene cut (no blocks detected at all) resets watermark age counters
            # so the next scene starts fresh.
            if wf > 0 and not raw_blocks_detected:
                self._block_ages = {}
            self._notify_ready()
            return

        # 7. Already showing this exact text — enforce display duration limit.
        if combined == self._last_shown_raw:
            if (self._cfg.max_display_seconds > 0
                    and self._shown_at is not None
                    and time.monotonic() - self._shown_at >= self._cfg.max_display_seconds):
                log.debug("ocr overlay expired after %.0fs", self._cfg.max_display_seconds)
                self._sched.notify(False, time.monotonic())
                self._overlay.hide()
                self._last_shown_raw = ""
                self._shown_at = None
                self._expired.add(combined)
            return

        # 7b. Skip text that was already shown and expired this session.
        if combined in self._expired:
            return

        # 7c. Cooldown — suppress re-showing text that was recently displayed.
        cooldown = self._cfg.cooldown_seconds
        if cooldown > 0:
            shown_time = self._recently_shown.get(combined)
            if shown_time is not None and time.monotonic() - shown_time < cooldown:
                return

        # 8. Translate proactively — do the work during the lookahead window
        #    so the result is ready the instant the text reaches the screen.
        log.debug("stable text detected: %r", combined)
        with (self._gpu.gpu(self._stop, defer_to_audio=True) if self._gpu else nullcontext(True)) as acquired:
            if not acquired:
                return
            translation = translate_text(combined, self._cfg.source_lang, self._cfg.target_lang)

        # 9. Lookahead timing — sleep for remaining lead time after translation
        #    so the overlay appears when the text actually reaches the screen.
        if offset > 0:
            elapsed = time.monotonic() - t_start
            wait = offset - elapsed
            if wait > 0 and self._stop.wait(wait):
                return  # stop was requested during the wait

        # 10. Display.
        if translation:
            log.info("OCR translation: %r -> %r", combined, translation)
            now = time.monotonic()
            if self._last_shown_raw:
                self._sched.notify(False, now)
            self._sched.notify(True, now)
            self._overlay.show(translation)
            self._last_shown_raw = combined
            self._shown_at = now
            self._recently_shown[combined] = now
        else:
            if self._last_shown_raw:
                self._sched.notify(False, time.monotonic())
            self._overlay.hide()
            self._last_shown_raw = ""
            self._shown_at = None
        self._notify_ready()
