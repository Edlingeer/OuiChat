@echo off
title OuiChat Server (mobile access)
cd /d "%~dp0"

REM ═══ OuiChat mobile-access server ═══════════════════════════════════════════
REM Starts ONLY the backend server (no browser) and exposes it over Tailscale
REM HTTPS so a phone can use it from anywhere:
REM   1. keep this window open (closing it stops the server)
REM   2. on the phone: Tailscale ON, then open the https://...ts.net URL printed below
REM Phone speech modes (button in the app header):
REM   device (Android) / hybrid Whisper-STT + phone TTS (iPhone default) / full server
REM ════════════════════════════════════════════════════════════════════════════

REM ── Backends ────────────────────────────────────────────────────────────────
set "LLM_BACKEND=ollama"
set "OLLAMA_MODEL=gemma4:31b-cloud"   REM free Ollama Cloud; comment out for local qwen3.6:27b
set "STT_BACKEND=whisper"             REM local Whisper — used by hybrid + server modes
set "TTS_BACKEND=piper"               REM local Piper — used only by full-server mode

REM ── Free port 8000 from any stale server ────────────────────────────────────
for /f "tokens=5" %%p in ('netstat -ano ^| findstr "LISTENING" ^| findstr ":8000"') do taskkill /F /PID %%p >nul 2>&1

REM ── Make sure the Tailscale HTTPS proxy points at this server ───────────────
set "TS=C:\Program Files\Tailscale\tailscale.exe"
if exist "%TS%" (
  "%TS%" serve --bg 8000 >nul 2>&1
  echo  Phone URL ^(Tailscale HTTPS^):
  "%TS%" serve status
) else (
  echo  [!] Tailscale CLI not found - phone access needs Tailscale installed.
)
echo  PC URL: http://localhost:8000
echo.

call "%USERPROFILE%\anaconda3\Scripts\activate.bat" french_teacher
uvicorn api:app --host 127.0.0.1 --port 8000
