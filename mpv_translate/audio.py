import itertools
from typing import Any, BinaryIO, Iterator, Union

import av
import numpy as np


def _safe_av_path(path):
    """Prefix local paths with file: so ffmpeg won't parse @ # ? as URL syntax."""
    if isinstance(path, str) and "://" not in path:
        return "file:" + path.replace("\\", "/")
    return path


def _ignore_invalid_frames(frames):
    iterator = iter(frames)
    while True:
        try:
            yield next(iterator)
        except StopIteration:
            break
        except (av.InvalidDataError, IndexError):
            continue


def _chunk_frames(frames, start: float, duration: float):
    for _t, chunk in itertools.groupby(
        frames, key=lambda frame: (frame.time - start) // duration
    ):
        if _t < 0:
            continue
        yield chunk


def _group_frames(frames, num_samples: int | None = None):
    fifo = av.AudioFifo()
    for frame in frames:
        frame.pts = None
        fifo.write(frame)
        if num_samples is not None and fifo.samples >= num_samples:
            yield fifo.read()
    if fifo.samples > 0:
        yield fifo.read()


def _resample_frames(frames, resampler: av.AudioResampler):
    for frame in itertools.chain(frames, [None]):
        yield from resampler.resample(frame)


class AudioReader:
    """Persistent audio container — avoids re-opening the file on every chunk.

    For sequential playback the container is opened once and re-used across
    reads.  Only if a read fails (stale container, network hiccup) is the
    container re-opened and the read retried.
    """

    def __init__(self, input_file: Union[str, BinaryIO], *, sampling_rate: int = 16000):
        self._input_file = input_file
        self._sampling_rate = sampling_rate
        self._container: "av.InputContainer | None" = None

    # -- internal helpers -----------------------------------------------------

    def _open(self) -> None:
        if self._container is not None:
            try:
                self._container.close()
            except Exception:
                pass
        self._container = av.open(_safe_av_path(self._input_file), mode="r", metadata_errors="ignore")

    def _read(self, start: float, duration: float) -> "tuple[Any, float] | None":
        container = self._container
        if container is None or not container.streams.audio:
            return None
        container.seek(int(start * 1_000_000))
        frames = container.decode(audio=0)
        frames = _ignore_invalid_frames(frames)

        first = next(frames, None)
        if first is None:
            return None
        frames = itertools.chain([first], frames)
        chunks = _chunk_frames(frames, start, duration)
        try:
            chunk = next(chunks)
        except StopIteration:
            return None

        resampler = av.AudioResampler(format="s16", layout="mono", rate=self._sampling_rate)
        chunk_frames = list(_resample_frames(_group_frames(chunk, 500000), resampler))
        del resampler

        if not chunk_frames:
            return None

        audio = np.concatenate(
            [frame.to_ndarray().reshape(-1) for frame in chunk_frames], axis=0
        )
        audio = audio.astype(np.float32)
        audio *= 1.0 / 32768.0
        max_val = np.abs(audio).max()
        if max_val > 1e-4:
            audio *= 0.95 / max_val
        return audio, start

    # -- public API -----------------------------------------------------------

    def read_chunk(self, start: float, duration: float) -> "tuple[Any, float] | None":
        """Read a single audio chunk starting at *start* for up to *duration* seconds.

        Returns ``(audio_array, actual_start)`` or ``None`` if no audio at that
        position.  The underlying container is kept open between calls; if a read
        fails the container is re-opened and the read retried once.
        """
        if self._container is None:
            self._open()
        try:
            return self._read(start, duration)
        except Exception:
            # Container may be stale (network timeout, format quirk) — re-open once.
            self._open()
            return self._read(start, duration)

    def close(self) -> None:
        if self._container is not None:
            try:
                self._container.close()
            except Exception:
                pass
            self._container = None

    def __enter__(self) -> "AudioReader":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


def read_chunk(
    input_file: Union[str, BinaryIO],
    start: float,
    duration: float,
    *,
    sampling_rate: int = 16000,
) -> "tuple[Any, float] | None":
    """One-shot convenience wrapper around :class:`AudioReader`."""
    with AudioReader(input_file, sampling_rate=sampling_rate) as reader:
        return reader.read_chunk(start, duration)
