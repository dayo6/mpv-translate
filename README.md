# mpv-translate

Fork of [alexkoay/mpv-whisper](https://github.com/alexkoay/mpv-whisper), rebuilt around **translation** instead of transcription.

Live offline subtitle translation overlay for [MPV](https://mpv.io/) using [faster-whisper](https://github.com/SYSTRAN/faster-whisper). Translates audio to English in real time — no internet required. Optionally shows the original language text alongside the translation, and can OCR on-screen text (signs, titles) for translation too.

## Features

- **Offline translation** — Whisper's `translate` task converts speech directly to English on your GPU or CPU
- **Bilingual subtitles** — show original text above the English translation (`show_original = true`)
- **Flicker-free overlay** — uses MPV's `osd-overlay` API with 20 Hz polling; subtitles update smoothly
- **Seek-aware** — cancels the current job and restarts from the new position with look-back
- **Adaptive chunk sizing** — targets ~2 sentences per Whisper call, balancing latency and context
- **Word-detection gate** — a tiny Whisper model skips silent/music-only chunks before running the main model
- **Hallucination suppression** — filters common Whisper phantom phrases ("Thanks for watching!", etc.)
- **OCR translation** (optional) — detects on-screen text with EasyOCR and translates it with OPUS-MT
- **GPU scheduling** — audio translation and OCR share a single GPU without collisions

## Requirements

- Python 3.9+
- [MPV](https://mpv.io/) with JSON IPC enabled
- CUDA-capable GPU recommended (CPU fallback available)

## Installation

Run the launcher script — it auto-installs on first run (creates `.venv`, installs dependencies):

```bash
# Windows
launch.bat

# Linux / macOS
chmod +x launch.sh
./launch.sh
```

You can also drag a video file onto `launch.bat` to open it directly.

For OCR support, activate the venv and install the extra:

```bash
# Windows
.venv\Scripts\activate && pip install -e ".[ocr]"

# Linux / macOS
source .venv/bin/activate && pip install -e ".[ocr]"
```

### MPV configuration

Add to your `mpv.conf`:

```ini
# Windows
input-ipc-server=\\.\pipe\mpvsocket

# Linux/macOS
# input-ipc-server=/tmp/mpvsocket
```

The socket name must match `ipc_socket` in the mpv-translate config (default: `"mpvsocket"`).

> The launcher scripts handle this automatically — they pass `--input-ipc-server` when starting MPV.
> The `mpv.conf` line is only needed if you start MPV separately.

## Usage

### Launcher (recommended)

The launcher finds MPV, starts it with IPC enabled, and attaches mpv-translate:

```bash
# Windows — double-click or run from terminal
launch.bat [optional-video.mkv]

# Linux / macOS
./launch.sh [optional-video.mkv]
```

### Attach to a running MPV instance

If MPV is already running with IPC enabled:

```bash
mpv-translate
```

Optionally load a file into MPV at the same time:

```bash
mpv-translate path/to/video.mkv
```

### Direct mode (no MPV)

Translate a file and write an SRT, useful for testing:

```bash
mpv-translate-direct path/to/video.mkv --echo
```

Options:
| Flag | Description |
|------|-------------|
| `--position FLOAT` | Start at this timestamp (seconds) |
| `--language CODE` | Force source language (skip auto-detect) |
| `--echo / --no-echo` | Print segment text to stdout |
| `--loglevel LEVEL` | Logging verbosity (default: `INFO`) |

### Keybinding

Press **Ctrl+.** (configurable) to toggle translation on/off while MPV is playing.

## Configuration

Settings are loaded in order (later files override earlier ones):

1. `mpv_translate/config.toml` — bundled defaults
2. `./mpv-translate.toml` — project-local override
3. `~/.config/mpv-translate/config.toml` — user override

### `[model]` — Whisper model

| Key | Default | Description |
|-----|---------|-------------|
| `model` | `"large-v3"` | Whisper model name |
| `args` | `{ device = "cuda", compute_type = "float16" }` | `WhisperModel()` init args |
| `task_args` | `{ beam_size = 3, vad_filter = true, ... }` | Per-call transcribe/translate args |

### `[translate]` — Translation engine

| Key | Default | Description |
|-----|---------|-------------|
| `language` | `"ja"` | Source language code (or `""` for auto-detect) |
| `show_original` | `false` | Show original text above translation |
| `show_translation` | `true` | Show English translation |
| `lookback_seconds` | `3.0` | Seconds to rewind after seek to catch sentence starts |
| `chunk_duration` | `30.0` | Max audio per Whisper call (seconds) |
| `seek_chunk_duration` | `9.0` | Shorter first chunk after seek for faster response |
| `min_chunk_duration` | `3.0` | Floor for adaptive chunk sizing |
| `adaptive_history` | `5` | Recent segments used for duration averaging |
| `adaptive_target_sentences` | `2.0` | Target sentences per chunk |
| `max_wait` | `6.0` | Seconds to pause MPV after seek awaiting first subtitle |
| `confidence_threshold` | `0.95` | Minimum confidence for auto-detected language |
| `word_end_padding` | `1.0` | Extra seconds after last word before subtitle hides |
| `gap_fill_threshold` | `0.0` | Fill gaps shorter than this between segments |
| `last_segment_hold` | `0.0` | Hold last subtitle while next chunk processes |
| `suppress_hallucinations` | `true` | Filter phantom Whisper phrases |
| `gate_model` | `"tiny"` | Small model for word-detection pre-filter (`""` to disable) |
| `max_lead` | `30.0` | Max seconds audio can race ahead of playback |

### `[mpv]` — MPV integration

| Key | Default | Description |
|-----|---------|-------------|
| `executable` | *(auto-detect)* | Path to `mpv.exe` |
| `start_mpv` | `false` | Launch a new MPV instance on startup |
| `ipc_socket` | `"mpvsocket"` | Named pipe / socket name (must match `mpv.conf`) |
| `toggle_binding` | `"ctrl+."` | Keybind to toggle translation |

### `[subtitle]` — Subtitle output

| Key | Default | Description |
|-----|---------|-------------|
| `path` | `"~/.config/mpv-translate/subs"` | Storage directory for translated SRT files |
| `only_network` | `false` | `true` = always use `path`; `false` = local files get `.translate.srt` alongside the video |
| `margin_bottom` | `50` | Pixels from bottom edge (in 1280x720 virtual space) |

### `[ocr]` — On-screen text OCR (optional)

| Key | Default | Description |
|-----|---------|-------------|
| `enabled` | `true` | Enable OCR detection and translation |
| `interval` | `1.0` | Frame capture rate (seconds) |
| `language` | `["ja"]` | EasyOCR language codes |
| `source_lang` | `"ja"` | OPUS-MT source language |
| `target_lang` | `"en"` | OPUS-MT target language |
| `gpu` | `true` | Use GPU for EasyOCR |
| `min_confidence` | `0.5` | Minimum OCR confidence |
| `min_length` | `2` | Skip text blocks shorter than this |
| `max_chars` | `120` | Skip text blocks longer than this |
| `corner_fraction` | `0.1` | Corner exclusion zone for watermarks |
| `stability_frames` | `2` | Consecutive frames before showing text |
| `lookahead_seconds` | `3.0` | Proactive frame capture ahead of playback |
| `margin_top` | `30` | Pixels from top edge (in 1280x720 virtual space) |
| `max_display_seconds` | `15.0` | Auto-hide OCR overlay after this duration |
| `min_display_seconds` | `3.0` | Minimum visibility before auto-hide |
| `watermark_frames` | `10` | Suppress text persistent for this many frames |
| `cooldown_seconds` | `30.0` | Suppress duplicate text within this window |

## Supported languages

Any language Whisper supports can be used as a source. Common codes:

| Code | Language | | Code | Language | | Code | Language |
|------|----------|-|------|----------|-|------|----------|
| `ja` | Japanese | | `zh` | Chinese | | `ko` | Korean |
| `de` | German | | `fr` | French | | `es` | Spanish |
| `it` | Italian | | `pt` | Portuguese | | `ru` | Russian |
| `ar` | Arabic | | `hi` | Hindi | | `th` | Thai |
| `vi` | Vietnamese | | `id` | Indonesian | | `tr` | Turkish |

Set `language = ""` in config to auto-detect.

## How it works

1. **MPV monitor** listens for file-load and seek events over the IPC socket
2. **Audio reader** (PyAV) extracts 16 kHz mono float32 audio in chunks
3. **Word-detection gate** runs a tiny Whisper model to skip silent chunks
4. **Whisper translate** converts speech to English (or runs bilingual transcribe + translate concurrently)
5. **Overlay manager** displays subtitles via `osd-overlay`, updating at 20 Hz with no flicker
6. **OCR loop** (optional) captures frames ahead of playback, detects text with EasyOCR, and translates with OPUS-MT

## License

MIT
