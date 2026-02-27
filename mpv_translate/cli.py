import logging
import time
from typing import Optional

import click
from python_mpv_jsonipc import MPVError

from .coreloop import core_loop
from .monitor import MPVMonitor
from .subtitle import SRTFile
from .translate import get_model


@click.command()
@click.argument("path", required=False, default=None)
@click.option("--loglevel", default="INFO", show_default=True)
def cli(path: Optional[str], loglevel: str):
    """
    mpv-translate: live offline translation overlay for MPV.

    Attaches to a running MPV instance (via IPC socket configured in
    mpv.conf / config.toml) and streams translated subtitles as you watch.

    Optionally pass a PATH to load that file into MPV on startup.
    """
    logging.basicConfig(
        level=loglevel,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    for _noisy in ("faster_whisper", "httpcore", "httpx", "filelock",
                   "huggingface_hub", "PIL.PngImagePlugin"):
        logging.getLogger(_noisy).setLevel(logging.WARNING)

    # Pre-load the Whisper model so the first seek/play is fast
    get_model()

    # Pre-load OCR models so they are ready before the first file plays
    from .config import get_config      # noqa: PLC0415
    from .ocr import warm_up            # noqa: PLC0415
    _cfg = get_config()
    if _cfg.ocr.enabled:
        warm_up(_cfg.ocr)
        from .ocr_translate import warm_up as translate_warm_up  # noqa: PLC0415
        translate_warm_up(_cfg.ocr.source_lang, _cfg.ocr.target_lang, gpu=_cfg.ocr.gpu)

    # Retry connection — MPV's IPC pipe may not be ready yet.
    log = logging.getLogger("cli")
    timeout, interval = 30.0, 0.5
    deadline = time.monotonic() + timeout
    while True:
        try:
            monitor = MPVMonitor()
            break
        except MPVError:
            if time.monotonic() >= deadline:
                raise click.ClickException(
                    f"Could not connect to MPV IPC pipe within {timeout:.0f}s. "
                    "Is MPV running with input-ipc-server enabled?"
                )
            log.debug("MPV IPC pipe not ready, retrying…")
            time.sleep(interval)

    if path:
        monitor.command("loadfile", path)

    monitor.block()


@click.command()
@click.argument("path")
@click.option("--position", default=0.0, show_default=True)
@click.option("--language", default=None)
@click.option("--loglevel", default="INFO", show_default=True)
@click.option("--echo/--no-echo", default=False)
def direct(
    path: str,
    position: float,
    language: Optional[str],
    loglevel: str,
    echo: bool,
):
    """Translate a single file without MPV (for testing)."""
    logging.basicConfig(
        level=loglevel,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    logging.getLogger("faster_whisper").disabled = True
    get_model()

    for item in core_loop(path=path, position=position, language=language):
        if item is True:
            logging.info("done")
        elif isinstance(item, SRTFile):
            logging.info("writing to %s", item.path)
        elif echo and isinstance(item, tuple):
            segments, _ = item
            for seg in segments:
                print(seg.text)
