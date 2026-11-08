@echo off
setlocal
cd /d "%~dp0"
set "PUZZLE_PYTHON=python"
if exist ".venv\Scripts\python.exe" set "PUZZLE_PYTHON=.venv\Scripts\python.exe"
curl.exe --noproxy "*" -fsS http://127.0.0.1:8765/mobile.html >nul 2>&1
if not errorlevel 1 (
    start "" "http://localhost:8765/"
    exit /b 0
)
start "" "http://localhost:8765/"
"%PUZZLE_PYTHON%" serve.py
pause
