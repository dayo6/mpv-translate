"""Helper for launch.bat: print the configured mpv executable path."""
from mpv_translate.config import get_config

exe = get_config().mpv.executable
if exe:
    print(exe)
