"""
On-screen text extraction using easyocr (detection + recognition).

Captures a video frame from MPV, detects text regions with easyocr's CRAFT
detector, then recognises each region with easyocr's recognition model.
Supports all languages that easyocr provides (ja, ko, zh_sim, en, de, fr, …).

Filters applied after recognition:
  • box centre inside a corner region → dropped (watermark guard)
  • text shorter than min_length → dropped (single-char noise)
  • purely numeric/symbolic text → dropped (timecodes, scores)
"""
import logging
from typing import TYPE_CHECKING, Callable, Optional

if TYPE_CHECKING:
    from PIL.Image import Image

    from .config import OcrConfig

log = logging.getLogger("ocr")

_detector = None
_detector_langs: Optional[tuple] = None
_detector_gpu: Optional[bool] = None

def _get_detector(languages: list, gpu: bool):
    global _detector, _detector_langs, _detector_gpu
    key = (tuple(languages), gpu)
    if _detector_langs == key and _detector is not None:
        return _detector
    import easyocr  # noqa: PLC0415
    log.info("initialising easyocr (languages=%s, gpu=%s)", languages, gpu)
    _detector = easyocr.Reader(languages, gpu=gpu)
    _detector_langs = key
    _detector_gpu = gpu
    return _detector


def _in_corner(x_min, x_max, y_min, y_max, img_w: int, img_h: int, fraction: float) -> bool:
    cx = (x_min + x_max) / 2.0
    cy = (y_min + y_max) / 2.0
    mx = img_w * fraction
    my = img_h * fraction
    near_x = cx < mx or cx > img_w - mx
    near_y = cy < my or cy > img_h - my
    return near_x and near_y


def capture_frame_av(video_path: str, timestamp: float, output_path: str) -> bool:
    """Decode a single video frame at *timestamp* seconds directly from the file.

    Uses PyAV (libavformat) so it works independently of MPV's current position.
    Falls back gracefully on network streams or seek errors.
    """
    try:
        import av as _av  # noqa: PLC0415
        safe_path = video_path if "://" in video_path else "file:" + video_path.replace("\\", "/")
        with _av.open(safe_path) as container:
            stream = next((s for s in container.streams if s.type == "video"), None)
            if stream is None:
                return False
            # Seek to just before target using the stream's time_base.
            target_ts = int(timestamp / stream.time_base)
            container.seek(target_ts, stream=stream)
            for frame in container.decode(stream):
                img = frame.to_image()
                img.save(output_path, format="PNG")
                return True
    except Exception:
        log.debug("av frame capture failed at %.2fs", timestamp, exc_info=True)
    return False


def warm_up(config: "OcrConfig") -> None:
    """Pre-load the easyocr detector and recognition models into GPU memory."""
    try:
        _get_detector(list(config.language), config.gpu)
    except Exception:
        log.warning("OCR warm-up failed", exc_info=True)


def capture_screenshot(command: Callable, path: str) -> bool:
    """Ask MPV to write the current video frame to *path* (PNG).

    Uses the "video" flag so overlays (subtitles, OSD) are excluded from
    the capture — we only want to OCR the raw video content.
    """
    try:
        command("screenshot-to-file", path, "video")
        return True
    except Exception:
        log.warning("screenshot-to-file failed", exc_info=True)
        return False


def detect_regions(
    image_path: str,
    config: "OcrConfig",
    exclude_bboxes: Optional[list] = None,
) -> Optional[tuple["Image", list[tuple[int, int, int, int]]]]:
    """Run easyocr CRAFT detection only — no recognition.

    Returns ``(pil_image, [(x_min, x_max, y_min, y_max), ...])`` or *None*
    on failure.  Corner regions are already filtered out.
    """
    try:
        from PIL import Image  # noqa: PLC0415
        img = Image.open(image_path)
        img_w, img_h = img.size
    except Exception:
        log.debug("failed to open screenshot %s", image_path, exc_info=True)
        return None

    import numpy as np  # noqa: PLC0415

    # Mask known watermark regions before detection.
    if exclude_bboxes:
        img_arr = np.array(img)
        _PAD = 10
        for ex_xmin, ex_xmax, ex_ymin, ex_ymax in exclude_bboxes:
            img_arr[
                max(0, ex_ymin - _PAD):min(img_h, ex_ymax + _PAD),
                max(0, ex_xmin - _PAD):min(img_w, ex_xmax + _PAD),
            ] = 255
        detect_input = img_arr
    else:
        detect_input = image_path

    # Invert dark frames so CRAFT can detect white-on-black text.
    img_arr = np.array(img) if not isinstance(detect_input, np.ndarray) else detect_input
    if float(np.mean(img_arr)) < 80:
        detect_input = 255 - img_arr
        log.debug("inverted dark frame for detection (mean=%.0f)", float(np.mean(img_arr)))

    try:
        detector = _get_detector(list(config.language), config.gpu)
        horizontal_list, _free_list = detector.detect(detect_input)
        regions = horizontal_list[0] if horizontal_list else []
    except Exception:
        log.debug("easyocr detection failed", exc_info=True)
        return None

    # Filter corners and convert to int tuples.
    bboxes: list[tuple[int, int, int, int]] = []
    for bbox in regions:
        x_min, x_max, y_min, y_max = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
        if _in_corner(x_min, x_max, y_min, y_max, img_w, img_h, config.corner_fraction):
            log.debug("dropping corner region at (%.0f, %.0f)", (x_min + x_max) / 2, (y_min + y_max) / 2)
            continue
        bboxes.append((x_min, x_max, y_min, y_max))

    return img, bboxes


def recognize_region(
    img: "Image",
    bbox: tuple[int, int, int, int],
    config: "OcrConfig",
) -> Optional[str]:
    """Run easyocr recognition on a single detected region.

    Returns the recognised text or *None* if it fails filters
    (too short, numeric-only, recognition error).
    """
    x_min, x_max, y_min, y_max = bbox
    try:
        import numpy as np  # noqa: PLC0415

        reader = _get_detector(list(config.language), config.gpu)
        img_arr = np.array(img)
        results = reader.recognize(
            img_arr,
            horizontal_list=[[x_min, x_max, y_min, y_max]],
            free_list=[],
            detail=0,
        )
        text = " ".join(results).strip() if results else ""
    except Exception:
        log.debug("easyocr recognition failed for region", exc_info=True)
        return None

    if len(text) < config.min_length:
        return None

    stripped = text.replace(" ", "").replace(".", "").replace("-", "").replace("/", "").replace(":", "")
    if stripped.isdigit() or not stripped:
        log.debug("dropping numeric/symbol-only text: %r", text)
        return None

    log.debug("keeping text: %r", text)
    return text


def extract_blocks(
    image_path: str,
    config: "OcrConfig",
    exclude_bboxes: Optional[list] = None,
) -> list[tuple[str, tuple[int, int, int, int]]]:
    """Detect text regions with easyocr, recognise each with easyocr.

    Returns a list of ``(text, (x_min, x_max, y_min, y_max))`` tuples.

    If *exclude_bboxes* is provided, those regions are painted white in the
    image before detection so easyocr skips them entirely (saves both
    detection and recognition time for known watermarks).

    Filters applied (in order):
    1. Box centre inside a corner region → dropped (watermark guard).
    2. Text shorter than config.min_length → dropped.
    3. Purely numeric/symbolic text → dropped.
    """
    result = detect_regions(image_path, config, exclude_bboxes)
    if result is None:
        return []
    img, bboxes = result
    if not bboxes:
        return []

    blocks: list[tuple[str, tuple[int, int, int, int]]] = []
    for bbox in bboxes:
        text = recognize_region(img, bbox, config)
        if text is not None:
            blocks.append((text, bbox))
    return blocks


def extract_text(image_path: str, config: "OcrConfig", exclude_bboxes: Optional[list] = None) -> list[str]:
    """Convenience wrapper around :func:`extract_blocks` returning only text strings."""
    return [text for text, _bbox in extract_blocks(image_path, config, exclude_bboxes)]
