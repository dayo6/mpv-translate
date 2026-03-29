"""
Offline translation via Meta NLLB-200 (No Language Left Behind).

Significantly more accurate than OPUS-MT, especially for CJK → English.
The 3.3B model (~6.5 GB VRAM in float16) is downloaded from Hugging Face
on first use and cached locally.  Subsequent sessions are fully offline.
"""
import logging
from typing import Optional

log = logging.getLogger("nllb")

_model = None
_tokenizer = None
_device: str = "cpu"

# NLLB uses BCP-47-style codes with script suffix.
# Map common short codes (ja, en, zh, ko, …) → NLLB flores-200 codes.
_LANG_MAP: dict[str, str] = {
    "af": "afr_Latn", "am": "amh_Ethi", "ar": "arb_Arab",
    "az": "azj_Latn", "be": "bel_Cyrl", "bg": "bul_Cyrl",
    "bn": "ben_Beng", "bs": "bos_Latn", "ca": "cat_Latn",
    "cs": "ces_Latn", "cy": "cym_Latn", "da": "dan_Latn",
    "de": "deu_Latn", "el": "ell_Grek", "en": "eng_Latn",
    "es": "spa_Latn", "et": "est_Latn", "fa": "pes_Arab",
    "fi": "fin_Latn", "fr": "fra_Latn", "ga": "gle_Latn",
    "gl": "glg_Latn", "gu": "guj_Gujr", "ha": "hau_Latn",
    "he": "heb_Hebr", "hi": "hin_Deva", "hr": "hrv_Latn",
    "hu": "hun_Latn", "hy": "hye_Armn", "id": "ind_Latn",
    "is": "isl_Latn", "it": "ita_Latn", "ja": "jpn_Jpan",
    "ka": "kat_Geor", "kk": "kaz_Cyrl", "km": "khm_Khmr",
    "kn": "kan_Knda", "ko": "kor_Hang", "lo": "lao_Laoo",
    "lt": "lit_Latn", "lv": "lvs_Latn", "mk": "mkd_Cyrl",
    "ml": "mal_Mlym", "mn": "khk_Cyrl", "mr": "mar_Deva",
    "ms": "zsm_Latn", "my": "mya_Mymr", "ne": "npi_Deva",
    "nl": "nld_Latn", "no": "nob_Latn", "pa": "pan_Guru",
    "pl": "pol_Latn", "pt": "por_Latn", "ro": "ron_Latn",
    "ru": "rus_Cyrl", "si": "sin_Sinh", "sk": "slk_Latn",
    "sl": "slv_Latn", "sq": "als_Latn", "sr": "srp_Cyrl",
    "sv": "swe_Latn", "sw": "swh_Latn", "ta": "tam_Taml",
    "te": "tel_Telu", "th": "tha_Thai", "tl": "tgl_Latn",
    "tr": "tur_Latn", "uk": "ukr_Cyrl", "ur": "urd_Arab",
    "uz": "uzn_Latn", "vi": "vie_Latn", "zh": "zho_Hans",
}

MODEL_NAME = "facebook/nllb-200-3.3B"


def _nllb_code(lang: str) -> str:
    """Convert a short language code to an NLLB flores-200 code."""
    code = _LANG_MAP.get(lang)
    if code is None:
        raise ValueError(
            f"unsupported NLLB language code: {lang!r}  "
            f"(known: {', '.join(sorted(_LANG_MAP))})"
        )
    return code


def warm_up(source_lang: str, target_lang: str, gpu: bool = True) -> None:
    """Pre-load the NLLB model so the first real call is fast."""
    global _device
    _device = "cuda" if gpu else "cpu"
    try:
        _ensure_model(source_lang, target_lang)
        translate_text("test", source_lang, target_lang)
        log.info("nllb: %s→%s translator ready on %s", source_lang, target_lang, _device)
    except Exception:
        log.warning("nllb warm-up failed", exc_info=True)


def _ensure_model(source_lang: str, target_lang: str):
    global _model, _tokenizer

    if _model is not None:
        return _model, _tokenizer

    import torch  # noqa: PLC0415
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer  # noqa: PLC0415

    log.info("loading NLLB model: %s", MODEL_NAME)

    try:
        _tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, local_files_only=True)
        _model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME, local_files_only=True)
    except (OSError, ValueError):
        log.info("nllb: model not cached — downloading %s (this may take a while)…", MODEL_NAME)
        _tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        _model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)

    if _device == "cuda" and torch.cuda.is_available():
        _model = _model.half().cuda()
        log.info("nllb: %s ready on CUDA (float16)", MODEL_NAME)
    else:
        log.info("nllb: %s ready on CPU", MODEL_NAME)

    _model.eval()
    return _model, _tokenizer


def translate_text(text: str, source_lang: str, target_lang: str) -> str:
    """Translate *text* offline using NLLB-200.

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
        log.debug("nllb translation failed", exc_info=True)
        return text


def _translate_batch(
    texts: list[str], source_lang: str, target_lang: str,
) -> list[str]:
    """Translate a list of strings in one batched model call."""
    import torch  # noqa: PLC0415

    src_code = _nllb_code(source_lang)
    tgt_code = _nllb_code(target_lang)

    model, tokenizer = _ensure_model(source_lang, target_lang)
    device = next(model.parameters()).device

    tokenizer.src_lang = src_code
    inputs = tokenizer(
        texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=512,
    ).to(device)

    tgt_token_id = tokenizer.convert_tokens_to_ids(tgt_code)

    max_src = int(inputs["input_ids"].shape[1])
    with torch.no_grad():
        generated = model.generate(
            **inputs,
            forced_bos_token_id=tgt_token_id,
            num_beams=4,
            max_new_tokens=max(40, max_src * 4),
            repetition_penalty=2.0,
            no_repeat_ngram_size=3,
            length_penalty=1.0,
        )
    return tokenizer.batch_decode(generated, skip_special_tokens=True)
