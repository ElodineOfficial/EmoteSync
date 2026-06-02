@echo off
setlocal enabledelayedexpansion
title EmoteSync
cd /d "%~dp0"

echo ==================================================
echo                    E M O T E S Y N C
echo ==================================================
echo.

REM ---------------------------------------------------------------
REM 1. Locate Python (prefer the py launcher, then python on PATH)
REM ---------------------------------------------------------------
set "PYEXE="
where py >nul 2>&1 && set "PYEXE=py -3"
if not defined PYEXE (
    where python >nul 2>&1 && set "PYEXE=python"
)
if not defined PYEXE (
    echo [!] Python was not found on this PC.
    where winget >nul 2>&1
    if !errorlevel! == 0 (
        echo     Installing Python 3.11 via winget...
        winget install -e --id Python.Python.3.11 --accept-source-agreements --accept-package-agreements
        echo.
        echo     Python installed. Please CLOSE this window and run EmoteSync.bat again.
    ) else (
        echo     Please install Python 3.10 - 3.12 from:
        echo         https://www.python.org/downloads/
        echo     IMPORTANT: tick "Add Python to PATH" in the installer.
    )
    echo.
    pause
    exit /b 1
)
echo [*] Using Python: %PYEXE%

REM ---------------------------------------------------------------
REM 2. Create a local virtual environment (.venv) once
REM ---------------------------------------------------------------
if not exist ".venv\Scripts\python.exe" (
    echo [*] Creating virtual environment ^(.venv^)...
    %PYEXE% -m venv .venv
    if errorlevel 1 (
        echo [!] Failed to create the virtual environment.
        pause
        exit /b 1
    )
)
set "VPY=.venv\Scripts\python.exe"

REM ---------------------------------------------------------------
REM 3. Install dependencies (only on first run / after a reset)
REM ---------------------------------------------------------------
if not exist ".venv\.installed" (
    echo.
    echo [*] Installing dependencies. This happens ONCE and can take several
    echo     minutes ^(it downloads PyTorch, Whisper, Transformers, etc^).
    echo.
    "%VPY%" -m pip install --upgrade pip
    "%VPY%" -m pip install -r requirements.txt
    if errorlevel 1 (
        echo.
        echo [!] Dependency installation failed. Scroll up for details.
        echo     If it mentions a network/proxy error, check your connection
        echo     and run EmoteSync.bat again.
        pause
        exit /b 1
    )
    echo installed> ".venv\.installed"
    echo [*] Dependencies installed.
)

REM ---------------------------------------------------------------
REM 4. Make sure ffmpeg is available (Whisper needs it to read audio)
REM ---------------------------------------------------------------
where ffmpeg >nul 2>&1
if errorlevel 1 (
    if not exist "ffmpeg\ffmpeg.exe" (
        echo [*] Setting up bundled ffmpeg...
        "%VPY%" -c "import imageio_ffmpeg,shutil,os; os.makedirs('ffmpeg',exist_ok=True); shutil.copy(imageio_ffmpeg.get_ffmpeg_exe(), os.path.join('ffmpeg','ffmpeg.exe'))"
        if errorlevel 1 (
            echo [!] Could not provision ffmpeg automatically.
            echo     Install it from https://ffmpeg.org and ensure it is on PATH.
            pause
            exit /b 1
        )
    )
    set "PATH=%CD%\ffmpeg;%PATH%"
)

REM ---------------------------------------------------------------
REM 5. Launch the app
REM ---------------------------------------------------------------
echo [*] Launching EmoteSync...
echo.
"%VPY%" emotesync.py
if errorlevel 1 (
    echo.
    echo [!] EmoteSync exited with an error. See the messages above.
    pause
)
endlocal
