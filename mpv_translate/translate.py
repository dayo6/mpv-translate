"""
Offline translation via faster-whisper's built-in translate task.

Whisper translates from any language → English with no network required.
When show_original=True (config default) we run transcribe+translate and
merge the resulting segments so each subtitle entry shows:
    <original line>
    <English translation>

GPU / CPU fallback
------------------
If the config requests CUDA but cuBLAS DLLs are missing (a common setup
issue), the first transcribe call raises RuntimeError.  We catch that,
destroy the GPU model, recreate it on CPU with int8, and continue — no
manual config change needed.  A warning is logged so the user knows.
"""
import glob
import logging
import os
import re
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any, Optional

from faster_whisper import WhisperModel

from .config import get_config

log = logging.getLogger("translate")

_model: Optional[WhisperModel] = None
_model_device: str = "unknown"
_cuda_dirs_registered: bool = False
_gate: Optional[WhisperModel] = None
_whisper_pool = ThreadPoolExecutor(max_workers=2, thread_name_prefix="whisper")


def _find_cublas_dir() -> Optional[str]:
    """Return a directory that contains cublas64_12.dll, or None."""
    target = "cublas64_12.dll"
    home = os.path.expanduser("~")

    # ── tier 1: CUDA Toolkit standard install ─────────────────────────────
    for pat in [
        r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12*\bin",
        r"C:\Program Files\NVIDIA\CUDA\v12*\bin",
    ]:
        for path in sorted(glob.glob(pat), reverse=True):
            if os.path.isfile(os.path.join(path, target)):
                return path

    # ── tier 2: PyTorch in the current venv ───────────────────────────────
    try:
        import torch as _torch  # noqa: PLC0415
        lib = os.path.join(os.path.dirname(_torch.__file__), "lib")
        if os.path.isfile(os.path.join(lib, target)):
            return lib
    except ImportError:
        pass

    # ── tier 3: torch/lib inside venvs in common user directories ─────────
    # glob("*") on Windows skips dot-dirs (.venv etc.) so we use scandir
    # which returns ALL entries including hidden ones.
    def _check_torch_lib(base: str) -> Optional[str]:
        """Return torch/lib path under base if it contains cublas64_12.dll."""
        tail = os.path.join("site-packages", "torch", "lib")
        for lib_sub in ("Lib", "lib"):
            p = os.path.join(base, lib_sub, tail)
            if os.path.isfile(os.path.join(p, target)):
                return p
        return None

    search_roots = [
        home,
        os.path.join(home, "Documents"),
        os.path.join(home, "Desktop"),
        os.path.join(home, "AppData", "Local", "Programs"),
    ]
    for root in search_roots:
        try:
            for d1 in os.scandir(root):
                if not d1.is_dir():
                    continue
                # depth 1 — e.g. ~/Documents/ComfyUI
                found = _check_torch_lib(d1.path)
                if found:
                    return found
                # depth 2 — e.g. ~/Documents/ComfyUI/.venv
                try:
                    for d2 in os.scandir(d1.path):
                        if not d2.is_dir():
                            continue
                        found = _check_torch_lib(d2.path)
                        if found:
                            return found
                except PermissionError:
                    pass
        except PermissionError:
            pass

    return None


def _register_cuda_dll_dirs() -> None:
    """On Windows, make CUDA 12 DLLs available to ctranslate2.

    ctranslate2 bundles cudnn but not cuBLAS.  It loads cuBLAS via a plain
    LoadLibraryW() call which only searches PATH and the process module list —
    AddDllDirectory() / os.add_dll_directory() are NOT consulted for that path.

    The reliable fix is to pre-load the DLLs with ctypes using their full path
    BEFORE the model is created.  Once a DLL is in the process module cache,
    any subsequent LoadLibraryW("cublas64_12.dll") call (with no path) will
    return the already-loaded handle.
    """
    global _cuda_dirs_registered
    if _cuda_dirs_registered or os.name != "nt":
        return
    _cuda_dirs_registered = True

    path = _find_cublas_dir()
    if not path:
        log.warning(
            "cublas64_12.dll not found — GPU acceleration unavailable.\n"
            "  To enable GPU:\n"
            "    Option A) Install CUDA 12 Toolkit: https://developer.nvidia.com/cuda-downloads\n"
            "    Option B) pip install torch --index-url https://download.pytorch.org/whl/cu124\n"
            "  Falling back to CPU."
        )
        return

    import ctypes  # noqa: PLC0415

    # Load in dependency order: runtime → LT → cuBLAS
    for dll_name in ("cudart64_12.dll", "cublasLt64_12.dll", "cublas64_12.dll"):
        dll_path = os.path.join(path, dll_name)
        if not os.path.isfile(dll_path):
            continue
        try:
            ctypes.CDLL(dll_path)
            log.debug("pre-loaded %s", dll_name)
        except OSError as e:
            log.warning("could not pre-load %s: %s", dll_name, e)

    log.info("CUDA DLLs pre-loaded from: %s", path)


def _cuda_error(exc: Exception) -> bool:
    msg = str(exc).lower()
    return any(k in msg for k in ("cublas", "cudnn", "cuda", ".dll", ".so", "cannot be loaded"))


def _load_model(device: str, compute_type: str, extra_args: dict) -> WhisperModel:
    config = get_config()
    args = {**extra_args, "device": device, "compute_type": compute_type}
    log.info("loading whisper model [%s] args=%s", config.model.model, args)
    # local_files_only avoids an HF API network call on every startup
    return WhisperModel(config.model.model, local_files_only=True, **args)


def get_model() -> WhisperModel:
    global _model, _model_device
    if _model is not None:
        return _model

    _register_cuda_dll_dirs()

    config = get_config()
    raw_args = dict(config.model.args)
    device = raw_args.pop("device", "cpu")
    compute_type = raw_args.pop("compute_type", "int8")

    try:
        _model = _load_model(device, compute_type, raw_args)
        _model_device = device
    except Exception as e:
        if device != "cpu" and _cuda_error(e):
            log.warning("GPU model load failed (%s) — falling back to CPU int8", e)
            _model = _load_model("cpu", "int8", raw_args)
            _model_device = "cpu"
        else:
            raise

    log.info("whisper model ready on %s", _model_device)
    return _model


def get_gate_model() -> WhisperModel:
    """Load a small Whisper model on CPU for word-level pre-filtering."""
    global _gate
    if _gate is not None:
        return _gate
    config = get_config()
    name = config.translate.gate_model
    log.info("loading gate model [%s] on cpu/int8", name)
    _gate = WhisperModel(name, device="cpu", compute_type="int8")
    return _gate


def has_words(audio: Any, language: Optional[str] = None) -> bool:
    """Fast check: does this audio contain actual spoken words?

    Uses a small Whisper model (e.g. tiny) to run a quick transcribe.
    If no segments are found, the chunk has no intelligible speech.
    Returns True immediately when the gate model is disabled (empty string).
    """
    config = get_config()
    if not config.translate.gate_model:
        return True
    model = get_gate_model()
    segments, _info = model.transcribe(
        audio,
        task="transcribe",
        language=language or None,
        beam_size=1,
        vad_filter=True,
    )
    return next(segments, None) is not None


def _cpu_fallback(exc: Exception) -> WhisperModel:
    """Recreate model on CPU after a runtime CUDA failure."""
    global _model, _model_device
    log.warning(
        "CUDA runtime error during encode (%s)\n"
        "  → cuBLAS 12 DLLs are missing (install CUDA 12 Toolkit or use cpu in config.toml)\n"
        "  → falling back to CPU int8 for this session",
        exc,
    )
    config = get_config()
    raw_args = {k: v for k, v in config.model.args.items() if k not in ("device", "compute_type")}
    _model = _load_model("cpu", "int8", raw_args)
    _model_device = "cpu"
    return _model


@dataclass
class Segment:
    start: float
    end: float
    text: str


def _run(
    audio: Any,
    task: str,
    language: Optional[str],
) -> tuple[list[Any], str]:
    """Run one whisper task; returns (segments_list, detected_language).
    Retries once on CPU if a CUDA runtime error occurs."""
    config = get_config()

    def _transcribe(model: WhisperModel) -> tuple[Any, Any]:
        return model.transcribe(
            audio,
            task=task,
            language=language or None,
            **config.model.task_args,
        )

    model = get_model()
    try:
        segments_gen, info = _transcribe(model)
        segs = list(segments_gen)
    except RuntimeError as e:
        if _cuda_error(e):
            model = _cpu_fallback(e)
            segments_gen, info = _transcribe(model)
            segs = list(segments_gen)
        else:
            raise

    lang = language or info.language
    return segs, lang


def _seg_end(s: Any, padding: float = 0.0) -> float:
    """Return the end of the last spoken word if word timestamps are available.

    Whisper's segment .end includes trailing non-speech audio (groans, breath
    sounds, etc.).  When word_timestamps=True is in task_args, .words[-1].end
    gives the timestamp right after the final spoken word instead.
    `padding` is added so the subtitle doesn't vanish the instant speech ends.
    """
    words = getattr(s, "words", None)
    return (words[-1].end if words else s.end) + padding


# Phrases that Whisper commonly hallucinates when audio is silent or music-only.
# Matched case-insensitively against the last line of a subtitle (the English translation).
_HALLUCINATION_RE = re.compile(
    r"^\s*(?:"
    # YouTube-style calls to action
    r"thank(?:s| you) for (?:watching|viewing|listening)"
    r"|please (?:like|subscribe|share|comment)"
    r"|(?:like|subscribe)(?:,? and|,) (?:subscribe|like|share|comment|the channel)"
    r"|don'?t forget to (?:like|subscribe|share)"
    r"|see you (?:in the next|next time|next video|soon)"
    r"|(?:check out|watch) (?:my|the|our) (?:other )?(?:videos?|channel|content)"
    r"|(?:hit|smash|click) the (?:like|subscribe|notification|bell) button"
    r"|subscribe(?: for more)?"
    r"|follow (?:us|me) on (?:twitter|x|instagram|facebook|tiktok|youtube)"
    r"|support (?:us|me) on patreon"
    r"|(?:this (?:video|episode|content) (?:is |was )?)?(?:sponsored|brought to you) by"
    # Subtitle/transcript credits (Amara.org and similar — very common Whisper hallucination)
    r"|amara\.?org"
    r"|(?:sub(?:title|caption)s?|captions?) (?:(?:created |made |edited )?by|from)(?: the)? amara"
    r"|(?:transcript(?:ion)?|transcribed|translated|captioned|subtitled) by"
    r")\s*[.!?♪❤]*\s*$",
    re.IGNORECASE,
)


def _is_hallucination(text: str) -> bool:
    """Return True if *text* looks like a Whisper hallucination phrase.

    In show_original mode the text is "original\\ntranslation" — we check
    only the last line (the English translation) to avoid false positives on
    legitimate original-language content.
    """
    lines = [ln.strip() for ln in text.strip().splitlines() if ln.strip()]
    if not lines:
        return True
    return bool(_HALLUCINATION_RE.match(lines[-1]))


def translate_chunk(
    audio: Any,
    language: Optional[str] = None,
) -> tuple[list[Segment], Optional[str]]:
    """
    Translate an audio chunk.

    Returns (segments, detected_language) where each Segment has
    .start/.end (seconds within the chunk) and .text (subtitle text).

    If config.translate.show_original is True, each segment's text is:
        "<original text>\\n<English translation>"  (when show_translation is also True)
        "<original text>"                          (when show_translation is False)
    Otherwise only the English translation is shown.
    """
    config = get_config()
    show_original = config.translate.show_original
    show_translation = config.translate.show_translation
    padding = config.translate.word_end_padding

    if show_original:
        if not show_translation:
            # Original language only — single transcribe pass
            orig_segs, lang = _run(audio, "transcribe", language)
            if not orig_segs:
                return [], lang
            return [Segment(s.start, _seg_end(s, padding), s.text.strip()) for s in orig_segs], lang

        # Run transcribe and translate concurrently — both tasks are
        # independent and ctranslate2 is thread-safe for inference.
        fut_orig = _whisper_pool.submit(_run, audio, "transcribe", language)
        fut_trans = _whisper_pool.submit(_run, audio, "translate", language)
        orig_segs, lang = fut_orig.result()
        trans_segs, _ = fut_trans.result()

        if not orig_segs:
            return [], lang

        merged: list[Segment] = []
        for o, t in zip(orig_segs, trans_segs):
            text = o.text.strip()
            translation = t.text.strip()
            if translation and translation.lower() != text.lower():
                text = f"{text}\n{translation}"
            merged.append(Segment(o.start, _seg_end(o, padding), text))

        for seg in orig_segs[len(trans_segs):]:
            merged.append(Segment(seg.start, _seg_end(seg, padding), seg.text.strip()))
        for seg in trans_segs[len(orig_segs):]:
            merged.append(Segment(seg.start, _seg_end(seg, padding), seg.text.strip()))

        if config.translate.suppress_hallucinations:
            merged = [s for s in merged if not _is_hallucination(s.text)]
        return merged, lang

    else:
        trans_segs, lang = _run(audio, "translate", language)
        segs = [Segment(s.start, _seg_end(s, padding), s.text.strip()) for s in trans_segs]
        if config.translate.suppress_hallucinations:
            segs = [s for s in segs if not _is_hallucination(s.text)]
        return segs, lang
