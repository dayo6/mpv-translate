import logging
import os.path
import pathlib
from functools import cache
from typing import Any, List, Optional

import cattrs
from attr import dataclass, field

try:
    import tomllib as toml
except ImportError:
    import toml


def expand_path(path: str) -> pathlib.Path:
    return pathlib.Path(os.path.expandvars(os.path.expanduser(path)))


def ensure_path(path: str) -> pathlib.Path:
    p = expand_path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


@dataclass(kw_only=True)
class ModelConfig:
    model: str = "large-v3"
    args: dict[str, Any] = field(factory=dict)
    task_args: dict[str, Any] = field(factory=lambda: {"beam_size": 5, "vad_filter": True})


@dataclass(kw_only=True)
class TranslateConfig:
    # Source language for Whisper (e.g. "ja", "zh", None = auto-detect)
    language: Optional[str] = None
    # How many seconds to look back from seek position to catch the start of a sentence
    lookback_seconds: float = 8.0
    # Max audio chunk duration fed to Whisper per iteration
    chunk_duration: float = 15.0
    # Show original language text above the English translation
    show_original: bool = True
    # Show the English translation (set to False with show_original=True for original-only)
    show_translation: bool = True
    # Confidence threshold for language auto-detection
    confidence_threshold: float = 0.95
    # Adaptive chunk sizing: lower bound in seconds
    min_chunk_duration: float = 3.0
    # Adaptive chunk sizing: number of recent segment durations to average
    adaptive_history: int = 5
    # Adaptive chunk sizing: target N sentences per chunk (multiplier on avg segment duration)
    adaptive_target_sentences: float = 2.0
    # Extend a segment's end to the next segment's start when the gap is smaller than this (seconds)
    gap_fill_threshold: float = 2.0
    # How long the last subtitle of a chunk stays visible while the next chunk loads (seconds)
    last_segment_hold: float = 2.0
    # Seconds to keep the subtitle visible after the last spoken word (word_timestamps mode).
    # Prevents subtitles from vanishing the instant speech ends.
    word_end_padding: float = 0.4
    # Audio duration for the FIRST chunk after a seek (seconds).
    # Smaller = faster first subtitle; must be >= lookback_seconds to cover the seek point.
    seek_chunk_duration: float = 12.0
    # After a seek, MPV is paused and waits up to this many seconds for the first subtitle
    # before resuming playback automatically (seconds).
    max_wait: float = 6.0
    # Filter out common Whisper hallucination phrases (e.g. "Thanks for watching!").
    suppress_hallucinations: bool = True
    # Small Whisper model for word-level pre-filtering (empty string = disabled).
    # Chunks where this model finds no words are skipped before the main model runs.
    gate_model: str = "tiny"
    # Maximum seconds audio translation may lead playback (0 = unlimited).
    # Audio pauses when it gets this far ahead, freeing GPU for OCR.
    max_lead: float = 30.0


@dataclass(kw_only=True)
class MpvConfig:
    executable: Optional[str] = None
    start_mpv: bool = False
    start_args: dict[str, Any] = field(factory=dict)
    ipc_socket: Optional[str] = "mpvsocket"
    toggle_binding: str = "ctrl+."


@dataclass(kw_only=True)
class OcrConfig:
    # Set to true to enable on-screen text OCR and translation
    enabled: bool = False
    # Seconds between frame captures
    interval: float = 1.0
    # Skip any text block (or combined text) longer than this many characters
    max_chars: int = 120
    # Minimum easyocr detection confidence (0.0–1.0)
    min_confidence: float = 0.5
    # Drop any detected text block shorter than this many characters.
    # Filters single-character OCR noise (stray digits, punctuation, etc.)
    min_length: int = 2
    # easyocr language list for detection and recognition (e.g. ["ja"], ["ko"], ["en"])
    language: List[str] = field(factory=lambda: ["ja"])
    # Top margin for the overlay in 1280×720 virtual space
    margin_top: int = 30
    # Fraction of frame width/height defining corner watermark exclusion zones (0.0–0.5)
    corner_fraction: float = 0.1
    # How many consecutive frames with the same text before showing it
    stability_frames: int = 2
    # OPUS-MT translation language codes (Helsinki-NLP/opus-mt-{src}-{tgt})
    source_lang: str = "ja"
    target_lang: str = "en"
    # Use GPU for easyocr (set to false if OCR and Whisper compete for VRAM)
    gpu: bool = False
    # Seconds ahead of current playback to read frames from the video file.
    # 0.0 = capture the current MPV frame (no lookahead).
    # >0  = proactive lookahead with Bayesian-adaptive offset and interval.
    lookahead_seconds: float = 0.0
    # Hide the overlay after displaying the same text for this many seconds (0 = disabled).
    max_display_seconds: float = 0.0
    # Suppress blocks present in every consecutive frame for N+ frames (0 = disabled).
    # Blocks that persist this long are treated as permanent watermarks and excluded.
    watermark_frames: int = 0
    # Minimum seconds to keep the overlay visible after showing it.
    # Prevents brief text cards from vanishing before the user can read them.
    min_display_seconds: float = 0.0
    # Suppress re-showing the same text within this many seconds.
    # Prevents title cards that fade in/out from being shown twice.
    cooldown_seconds: float = 0.0


@dataclass(kw_only=True)
class SubtitleConfig:
    path: pathlib.Path = field(
        default=pathlib.Path("~/.config/mpv-translate/subs"),
        converter=ensure_path,
    )
    only_network: bool = False
    # Pixels from the bottom edge in a 1280×720 virtual space (proportionally scaled).
    # 0 = flush with the bottom; increase to raise subtitles higher.
    margin_bottom: int = 50

    def get_subtitle(self, fname: str) -> pathlib.Path:
        if "://" in fname:
            stem = pathlib.Path(fname.split("/")[-1].split("?")[0]).stem
            return (self.path / stem).with_suffix(".translate.srt")
        else:
            if self.only_network:
                stem = pathlib.Path(fname).stem
                return (self.path / stem).with_suffix(".translate.srt")
            return pathlib.Path(fname).with_suffix(".translate.srt")


@dataclass(kw_only=True)
class Config:
    model: ModelConfig = field(factory=ModelConfig)
    translate: TranslateConfig = field(factory=TranslateConfig)
    mpv: MpvConfig = field(factory=MpvConfig)
    subtitle: SubtitleConfig = field(factory=SubtitleConfig)
    ocr: OcrConfig = field(factory=OcrConfig)


def load_config_paths(*paths: pathlib.Path) -> "Config":
    for p in paths:
        if not p.exists():
            continue
        logging.getLogger("config").info("using configuration from %s", p)
        raw: Any = toml.loads(p.read_text(encoding="utf8"))
        conv = cattrs.GenConverter(forbid_extra_keys=True)
        return conv.structure(raw, Config)
    raise RuntimeError("could not find configuration file")


@cache
def get_config() -> Config:
    return load_config_paths(
        expand_path(__file__).parent / "config.toml",
        expand_path(".") / "mpv-translate.toml",
        expand_path("~/.config/mpv-translate") / "config.toml",
    )
