@echo off
title OuiChat Launcher
cd /d "%~dp0"

REM ── LLM backend: "ollama" (local, fast on your GPU) or "nvidia" (hosted API) ─
set "LLM_BACKEND=ollama"
REM To use NVIDIA's hosted API instead, change the line above to:
REM     set "LLM_BACKEND=nvidia"
REM and set your key once, in a terminal:  setx NVIDIA_API_KEY "nvapi-xxxxxxxx"
REM (then reopen the terminal). NVIDIA model defaults to meta/llama-3.3-70b-instruct
REM (slow, ~20-80s/turn; google/gemma-4-31b-it is currently down). Override with:
REM     set "NVIDIA_MODEL=google/gemma-2-2b-it"
REM
REM Ollama model: set below (needs LLM_BACKEND=ollama). Comment this line out for
REM the local default (qwen3.6:27b). For a free Ollama CLOUD model run `ollama signin`
REM once, then use a "-cloud" tag. Verified free & working: gemma4:31b-cloud.
set "OLLAMA_MODEL=gemma4:31b-cloud"
REM ───────────────────────────────────────────────────────────────────────────

REM ── Optional ElevenLabs speech backends (default local Whisper/Piper) ────────
REM Set your key once, in a terminal:  setx ELEVENLABS_API_KEY "sk_xxxxxxxx"
REM (then reopen the terminal). Values: "whisper"/"piper" (local) or "elevenlabs".
set "STT_BACKEND=elevenlabs"    REM speech-to-text via ElevenLabs Scribe (verified working)
set "TTS_BACKEND=elevenlabs"    REM text-to-speech via ElevenLabs (verified working)
REM ───────────────────────────────────────────────────────────────────────────

REM Free port 8000 from any previous server still running (prevents a stale
REM server without the latest routes from lingering).
for /f "tokens=5" %%p in ('netstat -ano ^| findstr "LISTENING" ^| findstr ":8000"') do taskkill /F /PID %%p >nul 2>&1

echo  Starting OuiChat server (LLM: %LLM_BACKEND% %OLLAMA_MODEL%  STT: %STT_BACKEND%  TTS: %TTS_BACKEND%)...
start "OuiChat Server" cmd /k "call "%USERPROFILE%\anaconda3\Scripts\activate.bat" french_teacher && cd /d "%~dp0" && set "LLM_BACKEND=%LLM_BACKEND%" && set "OLLAMA_MODEL=%OLLAMA_MODEL%" && set "STT_BACKEND=%STT_BACKEND%" && set "TTS_BACKEND=%TTS_BACKEND%" && uvicorn api:app --port 8000"

echo  Waiting for server to be ready...
timeout /t 6 /nobreak >nul

echo  Opening Firefox...
start "" firefox --new-window http://localhost:8000

exit
