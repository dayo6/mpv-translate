"""
Core processing loop.

On every start (including after seek) we back up by `lookback_seconds` so
that Whisper has context for any sentence that was already in progress at the
seek point.  The look-back audio produces subtitles that are still valid (they
cover real speech) so MPV will display them if the playback position falls
inside their window.

Cancellation is checked:
  • before starting each new audio chunk
  • after each translated segment is written

This means a running Whisper call (which blocks while it processes one chunk)
will complete, but the results are discarded as soon as the cancel flag is set,
and no further chunks are started.
"""
import logging
import time
from collections import deque
from concurrent.futures import Future, ThreadPoolExecutor
from contextlib import nullcontext
from threading import Event
from typing import Generator, Optional, Union

from .audio import AudioReader
from .config import get_config
from .subtitle import SRTFile
from .translate import Segment, translate_chunk, has_words


def _merge_overlapping(segments: list[Segment]) -> list[Segment]:
    """Split overlapping segments into non-overlapping sub-interval entries.

    Instead of collapsing two overlapping segments into one long combined
    subtitle (which stays on screen too long), we split at every boundary:
        speaker A alone → A + B (actual overlap) → speaker B alone.
    Non-overlapping segments are passed through unchanged.
    """
    if not segments:
        return []

    # Collect all start/end boundary points and sort them.
    events = sorted({t for seg in segments for t in (seg.start, seg.end)})

    result: list[Segment] = []
    for i in range(len(events) - 1):
        t_start, t_end = events[i], events[i + 1]
        # Segments that span this interval (started at or before, end at or after).
        active = [seg.text for seg in segments if seg.start <= t_start and seg.end >= t_end]
        if not active:
            continue
        text = "\n".join(active)
        # Merge with the previous entry when text is identical and times are adjacent.
        if result and result[-1].text == text and result[-1].end == t_start:
            result[-1] = Segment(result[-1].start, t_end, text)
        else:
            result.append(Segment(t_start, t_end, text))

    return result

log = logging.getLogger("coreloop")


def _prefetch_and_probe(
    reader: AudioReader,
    pos: float,
    duration: float,
    lang: Optional[str],
):
    """Read next audio chunk and probe for speech in one background call.

    Returns ``(read_result, has_speech)`` where *read_result* is the same
    ``(chunk, chunk_start)`` tuple from :meth:`AudioReader.read_chunk` (or
    ``None`` at EOF) and *has_speech* is the ``has_words`` gate result so
    the caller can skip the redundant gate check.
    """
    result = reader.read_chunk(pos, duration)
    if result is None:
        return None, False
    chunk, _ = result
    speech = has_words(chunk, lang)
    return result, speech


def core_loop(
    path: str,
    position: float,
    language: Optional[str] = None,
    cancel: Optional[Event] = None,
    first_ready: Optional[Event] = None,
    get_playback_pos: Optional[callable] = None,
    gpu_scheduler=None,
    translated_up_to: float = 0.0,
) -> Generator[Union[SRTFile, list[tuple[float, float, str]], bool], None, None]:
    cancel = cancel or Event()
    config = get_config()

    # ── subtitle file ────────────────────────────────────────────────────────
    subtitle = SRTFile(config.subtitle.get_subtitle(path))
    yield subtitle  # caller (monitor) loads this file into MPV

    # ── seek look-back ───────────────────────────────────────────────────────
    # Back up from the seek/start position so we catch the sentence that was
    # already in progress.  max(0) avoids seeking before the file start.
    lookback = config.translate.lookback_seconds
    current_pos = max(0.0, position - lookback)
    log.debug(
        "starting at %.2fs (position=%.2fs, lookback=%.2fs)", current_pos, position, lookback
    )

    # After a seek we use a shorter first chunk so the initial subtitle appears quickly.
    # `first_ready` being set means this is a seek start; after that chunk we revert.
    _seek_start = first_ready is not None
    max_duration = config.translate.seek_chunk_duration if _seek_start else config.translate.chunk_duration
    recent_durations: deque[float] = deque(maxlen=config.translate.adaptive_history)
    lang = language or config.translate.language or None
    last_abs_end: float = translated_up_to  # suppress already-translated & chunk-boundary duplicates

    reader = AudioReader(path)
    prefetch_pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="prefetch")
    prefetch_future: Optional[Future] = None

    try:
        while not cancel.is_set():
            # ── read next audio chunk (use prefetched result when available) ──
            if prefetch_future is not None:
                result, probed_speech = prefetch_future.result()
                prefetch_future = None
            else:
                result = reader.read_chunk(current_pos, max_duration)
                probed_speech = None  # no probe — must run gate check below

            if result is None:
                log.info("no more audio at %.2fs — finished", current_pos)
                yield True
                break

            chunk, chunk_start = result

            if cancel.is_set():
                break

            # ── word-detection gate: skip chunks with no intelligible speech ──
            # When the prefetch already probed, reuse its result.
            has_speech = probed_speech if probed_speech is not None else has_words(chunk, lang)

            if not has_speech:
                log.debug("no words at %.2fs — skipping chunk", chunk_start)
                current_pos = chunk_start + max_duration
                continue

            # ── throttle: don't race too far ahead of playback ────────────────
            max_lead = config.translate.max_lead
            if get_playback_pos is not None and max_lead > 0:
                while not cancel.is_set():
                    try:
                        playback = get_playback_pos()
                    except Exception:
                        break
                    lead = chunk_start - playback
                    if lead <= max_lead:
                        break
                    log.debug(
                        "audio %.0fs ahead (chunk=%.1f, playback=%.1f), waiting",
                        lead, chunk_start, playback,
                    )
                    cancel.wait(1.0)

            log.debug("translating chunk at %.2fs", chunk_start)
            t0 = time.time()

            # ── translate (offline via Whisper) ──────────────────────────────
            with (gpu_scheduler.gpu(cancel, defer_to_ocr=True) if gpu_scheduler else nullcontext(True)) as acquired:
                if not acquired:
                    break
                segments, lang = translate_chunk(chunk, lang)

            elapsed = time.time() - t0
            log.debug("chunk %.2fs → %d segments in %.2fs", chunk_start, len(segments), elapsed)

            # word_end_padding can extend a segment's end past the next segment's start,
            # causing both to show simultaneously. Cap each segment at the next one's start
            # so that subtitles replace each other cleanly instead of stacking.
            for i in range(len(segments) - 1):
                if segments[i].end > segments[i + 1].start:
                    segments[i].end = segments[i + 1].start

            segments = _merge_overlapping(segments)

            if not segments:
                # No speech — skip forward by max_duration to avoid getting stuck
                current_pos = chunk_start + max_duration
                continue

            # ── write segments and collect (abs_start, abs_end, text) triples ─
            #
            # End-time extension to prevent subtitle gaps:
            #   • intra-chunk: extend each segment's end to the next segment's start
            #     when the gap is small (< 2 s) — fills natural Whisper boundaries.
            #   • last segment: extend by a short hold so the subtitle stays visible
            #     while Whisper processes the next chunk.
            last_end: Optional[float] = None
            written: list[tuple[float, float, str]] = []
            with subtitle.open("a"):
                for i, seg in enumerate(segments):
                    if cancel.is_set():
                        break
                    if i + 1 < len(segments):
                        next_start = segments[i + 1].start
                        gap = next_start - seg.end
                        seg_end = next_start if 0 < gap < config.translate.gap_fill_threshold else seg.end
                    else:
                        seg_end = seg.end + config.translate.last_segment_hold
                    abs_start = chunk_start + seg.start
                    abs_end = chunk_start + seg_end
                    speech_end = chunk_start + seg.end  # unextended, used for overlap detection
                    # Skip segments whose speech is already covered by the previous chunk.
                    if speech_end <= last_abs_end:
                        last_end = seg.end
                        continue
                    # Trim display start to avoid time-overlap with the previous subtitle.
                    abs_start = max(abs_start, last_abs_end)
                    subtitle.write(abs_start, abs_end, seg.text)
                    written.append((abs_start, abs_end, seg.text))
                    last_end = seg.end  # position tracking uses original (unextended) end
                    last_abs_end = speech_end
                    d = seg.end - seg.start
                    if d > 0:
                        recent_durations.append(d)

            if last_end is None or cancel.is_set():
                break

            # After the first seek chunk the monitor has reloaded subs — signal
            # the waiter thread so MPV can resume playback, then switch to
            # normal sizing.
            if _seek_start:
                if first_ready is not None:
                    first_ready.set()
                _seek_start = False
                max_duration = config.translate.chunk_duration

            # Advance past the last spoken sentence
            current_pos = chunk_start + last_end

            # ── adaptive chunk sizing ─────────────────────────────────────────
            # Keep the last N durations and target ~N sentences per chunk.
            if recent_durations:
                avg = sum(recent_durations) / len(recent_durations)
                max_duration = max(
                    config.translate.min_chunk_duration,
                    min(config.translate.chunk_duration, avg * config.translate.adaptive_target_sentences),
                )
                log.debug("chunk_duration → %.1fs (avg sentence %.1fs)", max_duration, avg)

            # ── prefetch next chunk + probe for speech while caller processes ──
            if not cancel.is_set():
                prefetch_future = prefetch_pool.submit(
                    _prefetch_and_probe, reader, current_pos, max_duration, lang
                )

            yield written  # list of (abs_start, abs_end, text) as written to the SRT

    finally:
        if prefetch_future is not None:
            prefetch_future.cancel()
        prefetch_pool.shutdown(wait=True, cancel_futures=True)
        reader.close()

    log.info("core_loop exiting (cancel=%s)", cancel.is_set())
