import logging
import pathlib
import re
import shutil
import time
from typing import Optional

import click
from python_mpv_jsonipc import MPVError

from .config import Config, get_config_path
from .coreloop import core_loop
from .monitor import MPVMonitor
from .subtitle import SRTFile
from .translate import get_model


def _resolve_executable(cfg: Config) -> None:
    """Check if the configured MPV executable exists; prompt if not found."""
    if not cfg.mpv.start_mpv:
        return

    exe = cfg.mpv.executable
    if exe and (pathlib.Path(exe).is_file() or shutil.which(exe)):
        return

    which_mpv = shutil.which("mpv")

    if exe:
        click.echo(f"MPV executable not found: {exe}")
    else:
        click.echo("MPV executable path not configured.")

    while True:
        prompt_kwargs: dict = {"text": "Enter path to mpv executable"}
        if which_mpv:
            prompt_kwargs["default"] = which_mpv
        answer = click.prompt(**prompt_kwargs)
        answer = answer.strip().strip('"').strip("'")
        resolved = pathlib.Path(answer)
        if resolved.is_file():
            cfg.mpv.executable = str(resolved)
            _save_executable(str(resolved))
            click.echo(f"Saved mpv executable: {resolved}")
            return
        found = shutil.which(answer)
        if found:
            cfg.mpv.executable = found
            _save_executable(found)
            click.echo(f"Saved mpv executable: {found}")
            return
        click.echo(f"Not found: {answer}")


def _save_executable(exe_path: str) -> None:
    """Persist the executable path to the loaded config file."""
    config_path = get_config_path()
    if not config_path:
        return
    # Normalise to forward slashes for TOML compatibility.
    exe_path = exe_path.replace("\\", "/")
    text = config_path.read_text(encoding="utf8")
    new_line = f'executable = "{exe_path}"'
    new_text, count = re.subn(
        r'^[#\s]*executable\s*=\s*"[^"]*"',
        new_line,
        text,
        count=1,
        flags=re.MULTILINE,
    )
    if count:
        config_path.write_text(new_text, encoding="utf8")


def _discover_mpv_pipe(base: str = "mpvsocket", pid: Optional[int] = None) -> Optional[str]:
    """Find an mpv IPC pipe on Windows.

    If *pid* is given, return ``mpvsocket-<pid>`` directly.
    Otherwise, scan ``\\\\.\\pipe\\`` for pipes matching ``mpvsocket-*``
    and return the first one found (falls back to the bare *base* name).
    """
    import os  # noqa: PLC0415

    if pid:
        return f"{base}-{pid}"

    # Enumerate Windows named pipes
    pipe_dir = r"\\.\pipe"
    try:
        pipes = os.listdir(pipe_dir)
    except OSError:
        return base

    # Prefer PID-specific pipes (created by playlist-manager.lua)
    candidates = sorted(p for p in pipes if p.startswith(f"{base}-"))
    if candidates:
        return candidates[0]

    # Fall back to the base name if it exists
    if base in pipes:
        return base
    return None


@click.command()
@click.argument("path", required=False, default=None)
@click.option("--pid", default=None, type=int, help="PID of the mpv instance to attach to")
@click.option("--loglevel", default="INFO", show_default=True)
def cli(path: Optional[str], pid: Optional[int], loglevel: str):
    """
    mpv-translate: live offline translation overlay for MPV.

    Attaches to a running MPV instance (via IPC socket configured in
    mpv.conf / config.toml) and streams translated subtitles as you watch.

    Optionally pass a PATH to load that file into MPV on startup.
    Use --pid to target a specific mpv instance.
    """
    logging.basicConfig(
        level=loglevel,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    for _noisy in ("faster_whisper", "httpcore", "httpx", "filelock",
                   "huggingface_hub", "PIL.PngImagePlugin"):
        logging.getLogger(_noisy).setLevel(logging.WARNING)

    # Validate MPV executable before loading heavy models.
    from .config import get_config      # noqa: PLC0415
    _cfg = get_config()
    _resolve_executable(_cfg)

    # Discover the right pipe for this mpv instance
    log = logging.getLogger("cli")
    base_socket = _cfg.mpv.ipc_socket or "mpvsocket"
    discovered = _discover_mpv_pipe(base_socket, pid)
    if discovered:
        log.info("using IPC pipe: %s", discovered)
        _cfg.mpv.ipc_socket = discovered
    else:
        log.warning("no mpv pipe found, using configured: %s", base_socket)

    # Pre-load the Whisper model so the first seek/play is fast
    get_model()

    # Pre-load text translation model for audio subtitles
    if _cfg.translate.translator in ("nllb", "opus") and _cfg.translate.language and _cfg.translate.language != _cfg.translate.target_lang:
        if _cfg.translate.translator == "nllb":
            from .nllb import warm_up as audio_translate_warm_up  # noqa: PLC0415
        else:
            from .ocr_translate import warm_up as audio_translate_warm_up  # noqa: PLC0415
        audio_translate_warm_up(_cfg.translate.language, _cfg.translate.target_lang, gpu=True)

    # Pre-load OCR models so they are ready before the first file plays
    from .ocr import warm_up            # noqa: PLC0415
    if _cfg.ocr.enabled:
        warm_up(_cfg.ocr)
        from .ocr_translate import warm_up as ocr_translate_warm_up  # noqa: PLC0415
        ocr_translate_warm_up(_cfg.ocr.source_lang, _cfg.ocr.target_lang, gpu=_cfg.ocr.gpu)

    # Retry connection — MPV's IPC pipe may not be ready yet.
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
                    "Is MPV running with input-ipc-server enabled?\n"
                    "  Fix: add this line to your mpv.conf:\n"
                    "    input-ipc-server=\\\\.\\pipe\\mpvsocket"
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
