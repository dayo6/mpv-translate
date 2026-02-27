@echo off
setlocal EnableDelayedExpansion

set "SCRIPT_DIR=%~dp0"
set "VENV_BIN=%SCRIPT_DIR%.venv\Scripts"
set "SOCKET=\\.\pipe\mpvsocket"

:: ── auto-install if needed ───────────────────────────────────────────────────
if not exist "%VENV_BIN%\mpv-translate.exe" (
    echo First run — installing mpv-translate into .venv ...

    set "PY="
    where py >nul 2>&1
    if !errorlevel! equ 0 (
        set "PY=py -3"
    ) else (
        where python >nul 2>&1
        if !errorlevel! equ 0 (
            set "PY=python"
        )
    )
    if not defined PY (
        echo Python not found. Install Python 3.9+ and ensure it is on PATH.
        pause
        exit /b 1
    )
    echo Using: !PY!

    if not exist "%VENV_BIN%\python.exe" (
        !PY! -m venv "%SCRIPT_DIR%.venv"
        if !errorlevel! neq 0 (
            echo Failed to create venv.
            pause
            exit /b 1
        )
    )
    "%VENV_BIN%\python.exe" -m pip install --upgrade pip -q
    "%VENV_BIN%\pip.exe" install -e "%SCRIPT_DIR%." -q
    if !errorlevel! neq 0 (
        echo Install failed.
        pause
        exit /b 1
    )

    echo Install complete.
    echo.
)

:: ── activate the venv ─────────────────────────────────────────────────────────
call "%VENV_BIN%\activate.bat"

:: ── CUDA 12 cuBLAS DLLs ───────────────────────────────────────────────────────
:: ctranslate2 needs cublas64_12.dll but doesn't bundle it on Windows.
:: Search in order: CUDA Toolkit, then PyTorch torch\lib in common venv locations.
:: NOTE: for /d with * skips dot-dirs (.venv etc.) so we check venv names explicitly.
set "CUDA_BIN="

:: 1. Standard CUDA Toolkit install
for /d %%V in ("C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.*") do (
    if exist "%%V\bin\cublas64_12.dll" set "CUDA_BIN=%%V\bin"
)
for /d %%V in ("C:\Program Files\NVIDIA\CUDA\v12.*") do (
    if exist "%%V\bin\cublas64_12.dll" set "CUDA_BIN=%%V\bin"
)

:: 2. PyTorch torch\lib inside venvs — check venv names explicitly (for /d skips dotdirs)
if not defined CUDA_BIN (
    for %%R in ("%USERPROFILE%" "%USERPROFILE%\Documents" "%USERPROFILE%\Desktop" "%USERPROFILE%\AppData\Local\Programs") do (
        for /d %%D in ("%%~R\*") do (
            :: depth-1 venv (e.g. ~/myproject/Lib/...)
            for %%V in (".venv" "venv" "env" ".env") do (
                if exist "%%D\%%V\Lib\site-packages\torch\lib\cublas64_12.dll" (
                    if not defined CUDA_BIN set "CUDA_BIN=%%D\%%V\Lib\site-packages\torch\lib"
                )
                if exist "%%D\%%V\lib\site-packages\torch\lib\cublas64_12.dll" (
                    if not defined CUDA_BIN set "CUDA_BIN=%%D\%%V\lib\site-packages\torch\lib"
                )
            )
            :: depth-2 venv (e.g. ~/projects/myproject/venv/Lib/...)
            for /d %%E in ("%%D\*") do (
                for %%V in (".venv" "venv" "env" ".env") do (
                    if exist "%%E\%%V\Lib\site-packages\torch\lib\cublas64_12.dll" (
                        if not defined CUDA_BIN set "CUDA_BIN=%%E\%%V\Lib\site-packages\torch\lib"
                    )
                    if exist "%%E\%%V\lib\site-packages\torch\lib\cublas64_12.dll" (
                        if not defined CUDA_BIN set "CUDA_BIN=%%E\%%V\lib\site-packages\torch\lib"
                    )
                )
            )
        )
    )
)

if defined CUDA_BIN (
    echo CUDA DLLs: !CUDA_BIN!
    set "PATH=!CUDA_BIN!;%PATH%"
) else (
    echo CUDA 12 DLLs not found - GPU unavailable, using CPU fallback.
    echo   To enable GPU, install CUDA 12: https://developer.nvidia.com/cuda-downloads
    echo.
)

:: ── locate mpv ────────────────────────────────────────────────────────────────
set "MPV="

:: 0. config.toml mpv.executable (highest priority)
"%VENV_BIN%\python.exe" "%SCRIPT_DIR%mpv_translate\_get_mpv_exe.py" > "%TEMP%\mpv_translate_exe.tmp" 2>nul
for /f "usebackq tokens=*" %%P in ("%TEMP%\mpv_translate_exe.tmp") do (
    if not "%%P"=="" (
        set "MPV=%%P"
        goto :mpv_found
    )
)

where mpv.exe >nul 2>&1
if %errorlevel% equ 0 (
    set "MPV=mpv"
    goto :mpv_found
)

for %%P in (
    "%LOCALAPPDATA%\Programs\mpv\mpv.exe"
    "C:\Program Files\mpv\mpv.exe"
    "C:\Program Files (x86)\mpv\mpv.exe"
    "%~dp0mpv.exe"
) do (
    if exist %%P (
        set "MPV=%%~P"
        goto :mpv_found
    )
)

echo mpv.exe not found on PATH or common locations.
set /p "MPV=Enter full path to mpv.exe: "
if "!MPV!"=="" (
    echo Cancelled.
    pause
    exit /b 1
)
if not exist "!MPV!" (
    echo File not found: !MPV!
    pause
    exit /b 1
)

:mpv_found
echo Using MPV: !MPV!

:: ── start MPV with IPC socket ─────────────────────────────────────────────────
:: %1 = optional video file (drag a file onto this .bat to open it directly)
if "%~1"=="" (
    start "" "!MPV!" --input-ipc-server=%SOCKET%
) else (
    start "" "!MPV!" --input-ipc-server=%SOCKET% "%~1"
)

:: Wait for MPV to open the named pipe
timeout /t 2 /nobreak >nul

:: ── start mpv-translate ───────────────────────────────────────────────────────
"%VENV_BIN%\mpv-translate.exe" --loglevel DEBUG

endlocal
