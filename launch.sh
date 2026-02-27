#!/usr/bin/env bash
# mpv-translate launcher (Linux / macOS)
# Drag a video file onto a terminal running this script, or pass it as $1.
# Auto-installs to .venv on first run.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV="$SCRIPT_DIR/.venv"
VENV_TRANSLATE="$VENV/bin/mpv-translate"
SOCKET="/tmp/mpvsocket"

# ── auto-install if needed ────────────────────────────────────────────────────
if [[ ! -x "$VENV_TRANSLATE" ]]; then
    echo "First run — installing mpv-translate into .venv ..."

    PY=""
    for cmd in python3 python; do
        if command -v "$cmd" &>/dev/null; then
            PY="$cmd"
            break
        fi
    done
    if [[ -z "$PY" ]]; then
        echo "Python not found. Install Python 3.9+ and ensure it is on PATH."
        exit 1
    fi
    echo "Using: $PY ($($PY --version 2>&1))"

    if [[ ! -f "$VENV/bin/python" ]]; then
        "$PY" -m venv "$VENV"
    fi
    "$VENV/bin/python" -m pip install --upgrade pip -q
    "$VENV/bin/pip" install -e "$SCRIPT_DIR" -q

    echo "Install complete."
    echo
fi

# ── locate mpv ────────────────────────────────────────────────────────────────
MPV=""

if command -v mpv &>/dev/null; then
    MPV=mpv
else
    for CANDIDATE in /usr/bin/mpv /usr/local/bin/mpv /opt/homebrew/bin/mpv "$SCRIPT_DIR/mpv"; do
        if [[ -x "$CANDIDATE" ]]; then
            MPV="$CANDIDATE"
            break
        fi
    done
fi

if [[ -z "$MPV" ]]; then
    echo "mpv not found on PATH or common locations."
    read -rp "Enter path to mpv binary: " MPV
    if [[ -z "$MPV" ]]; then
        echo "Cancelled."
        exit 1
    fi
    if [[ ! -x "$MPV" ]]; then
        echo "Not executable: $MPV"
        exit 1
    fi
fi

echo "Using MPV: $MPV"

# ── update config ipc_socket for Linux if still set to Windows default ─────────
# (only prints a warning; edit mpv_translate/config.toml to change it)
CFG="$SCRIPT_DIR/mpv_translate/config.toml"
if grep -q 'ipc_socket.*=.*"mpvsocket"' "$CFG" 2>/dev/null; then
    echo ""
    echo "  NOTE: config.toml has ipc_socket = \"mpvsocket\" (Windows default)."
    echo "  For Linux, consider changing it to: ipc_socket = \"$SOCKET\""
    echo "  File: $CFG"
    echo ""
fi

# ── start MPV with IPC socket ─────────────────────────────────────────────────
if [[ -n "${1:-}" ]]; then
    "$MPV" --input-ipc-server="$SOCKET" "$1" &
else
    "$MPV" --input-ipc-server="$SOCKET" &
fi
MPV_PID=$!

# Wait for MPV to create the socket
for i in 1 2 3 4 5; do
    [[ -S "$SOCKET" ]] && break
    sleep 0.5
done

if [[ ! -S "$SOCKET" ]]; then
    echo "Timed out waiting for MPV socket at $SOCKET"
    kill "$MPV_PID" 2>/dev/null || true
    exit 1
fi

# ── start mpv-translate ───────────────────────────────────────────────────────
"$VENV_TRANSLATE"

# Clean up MPV if it's still running when mpv-translate exits
kill "$MPV_PID" 2>/dev/null || true
