"""
Offline text translation via Helsinki-NLP OPUS-MT (MarianMT).

Uses the transformers library with torch for GPU-accelerated seq2seq
translation.  The model is downloaded from Hugging Face Hub on first use
and cached locally (~/.cache/huggingface/).  Subsequent sessions are fully
offline.

The loaded model is cached so it is only initialised once per session.
"""
import logging
from typing import Optional

log = logging.getLogger("ocr_translate")

_model = None
_tokenizer = None
_device: str = "cpu"
_model_langs: Optional[tuple[str, str]] = None


def _model_name(source_lang: str, target_lang: str) -> str:
    return f"Helsinki-NLP/opus-mt-{source_lang}-{target_lang}"


def _ensure_model(source_lang: str, target_lang: str):
    global _model, _tokenizer, _model_langs
    if _model_langs == (source_lang, target_lang) and _model is not None:
        return _model, _tokenizer

    import torch  # noqa: PLC0415
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer  # noqa: PLC0415

    name = _model_name(source_lang, target_lang)
    log.info("loading translation model: %s", name)

    try:
        _tokenizer = AutoTokenizer.from_pretrained(name, local_files_only=True)
        _model = AutoModelForSeq2SeqLM.from_pretrained(name, local_files_only=True)
    except (OSError, ValueError):
        log.info("opus-mt: model not cached — downloading %s …", name)
        _tokenizer = AutoTokenizer.from_pretrained(name)
        _model = AutoModelForSeq2SeqLM.from_pretrained(name)

    if _device == "cuda" and torch.cuda.is_available():
        _model = _model.half().cuda()
        log.info("opus-mt: %s ready on CUDA (float16)", name)
    else:
        log.info("opus-mt: %s ready on CPU", name)

    _model.eval()
    _model_langs = (source_lang, target_lang)
    return _model, _tokenizer


def warm_up(source_lang: str, target_lang: str, gpu: bool = False) -> None:
    """Pre-load the translation model so the first real call is fast."""
    global _device
    _device = "cuda" if gpu else "cpu"
    try:
        _ensure_model(source_lang, target_lang)
        # Trigger JIT / CUDA warm-up with a throwaway translation.
        translate_text("test", source_lang, target_lang)
        log.info("opus-mt: %s→%s translator ready", source_lang, target_lang)
    except Exception:
        log.warning("opus-mt warm-up failed", exc_info=True)


def translate_text(text: str, source_lang: str, target_lang: str) -> str:
    """Translate *text* offline using OPUS-MT.

    Multi-line input is split into individual lines and batch-translated,
    then re-joined with newlines.  Returns *text* unchanged on failure.
    """
    lines = [line.strip() for line in text.split("\n") if line.strip()]
    if not lines:
        return ""
    try:
        translated = _translate_batch(lines, source_lang, target_lang)
        result = "\n".join(t for t in translated if t.strip())
        return result if result else text
    except Exception:
        log.debug("translation failed", exc_info=True)
        return text


def _translate_batch(
    texts: list[str], source_lang: str, target_lang: str,
) -> list[str]:
    """Translate a list of strings in one batched model call.

    Uses ``max_new_tokens`` capped to 3x the source length (OCR text is short)
    and ``repetition_penalty`` to prevent degenerate "a... a... a..." loops
    that MarianMT produces on out-of-distribution input.
    """
    import torch  # noqa: PLC0415

    model, tokenizer = _ensure_model(source_lang, target_lang)
    device = next(model.parameters()).device
    inputs = tokenizer(
        texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=512,
    ).to(device)
    # Cap output length relative to input — OCR text is short so we never
    # need 512 output tokens.  The 3x multiplier handles CJK→English
    # expansion.  repetition_penalty kills degenerate repeat loops.
    max_src = int(inputs["input_ids"].shape[1])
    with torch.no_grad():
        generated = model.generate(
            **inputs,
            num_beams=4,
            max_new_tokens=max(20, max_src * 3),
            repetition_penalty=1.5,
        )
    return tokenizer.batch_decode(generated, skip_special_tokens=True)
