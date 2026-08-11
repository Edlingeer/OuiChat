# OuiChat — French A1 Conversation Bot

A French language learning bot for Brazilian Portuguese speakers at beginner levels
(CEFR A1–B1). It listens to you speak French, transcribes it, replies in French, displays
the reply, and speaks it aloud. Speech and voice run locally on your GPU; the LLM runs
locally via Ollama (default) or, optionally, on NVIDIA's hosted API. Grammar corrections
and vocabulary translations are given in Brazilian Portuguese.

Two ways to use it: a **terminal bot** (classic) or a **web interface** (browser-based).

---

## Prerequisites

| Tool | Install |
|---|---|
| Ollama | https://ollama.com/download |
| CUDA Toolkit 12.4+ | https://developer.nvidia.com/cuda-downloads |
| Python 3.11+ | https://python.org |

### The LLM

Two backends are supported (see [Choosing the LLM backend](#choosing-the-llm-backend)):

- **Local (default)** — pull the model with Ollama:
  ```bash
  ollama pull qwen3.6:27b
  ```
- **NVIDIA hosted API** — no local model needed; grab a free API key at
  [build.nvidia.com](https://build.nvidia.com). Any chat model in their catalog works
  (default is `meta/llama-3.3-70b-instruct`).

### Download the Piper French voice

Download both files and place them in the `OuiChat/` folder (next to `main.py`):

```
https://huggingface.co/rhasspy/piper-voices/resolve/main/fr/fr_FR/upmc/medium/fr_FR-upmc-medium.onnx
https://huggingface.co/rhasspy/piper-voices/resolve/main/fr/fr_FR/upmc/medium/fr_FR-upmc-medium.onnx.json
```

---

## Installation

```bash
conda env create -f environment.yml
conda activate french_teacher
```

---

## Running

### Quick start (Windows)

Double-click **`launch.bat`** — it activates the conda env, starts the web server, and
opens the browser for you. The LLM backend is selected by the `set "LLM_BACKEND=..."`
line near the top of that file (`ollama` by default; `nvidia` for the hosted API).

### Terminal bot (microphone + speakers)

```bash
python main.py
```

Press **Ctrl+C** to exit.

### Web interface (browser-based)

```bash
uvicorn api:app --port 8000
```

Then open `http://localhost:8000` in your browser. Tap **▶ Commencer la conversation**
(one tap unlocks audio + microphone — required on iOS) and the bot opens with a greeting.
Hold the microphone button to speak, release to send. The top-right buttons open live
panels for vocabulary, learner profile, and correction history.

> **First reply is slow.** The first LLM call of a session loads the model into VRAM
> (~30–60s for `qwen3.6:27b`); every turn after is fast. On the web app this happens during
> the opening greeting, so the model is already warm by the time you speak. (Cloud LLMs
> skip this entirely.)

### Speech modes (button in the app header)

The web app has three speech modes, cycled by the voice button in the header:

| Mode | STT (your speech) | TTS (bot voice) | Notes |
|---|---|---|---|
| **📱 Voix** | phone/browser (Web Speech) | phone/browser | No audio crosses the wire — only text. Default on Android/desktop Chrome. Phone STT may auto-correct your mistakes, hiding them from the tutor. |
| **🎙️ Mixte** | **server Whisper** | phone/browser | Best of both: faithful transcription (catches your errors) + free device voice. **Default on iPhone/iPad** (iOS speech recognition is unreliable). |
| **☁️ Voix** | server (`STT_BACKEND`) | server (`TTS_BACKEND`) | Full server pipeline — uses whatever backends the launcher sets (Whisper/Piper local, or ElevenLabs). |

The button only shows the modes your browser supports. If device speech recognition
fails (e.g. iOS `service-not-allowed`), the app automatically drops to **🎙️ Mixte**.

**Device voices (⚙ button):** the Web Speech API doesn't expose voice gender, and phone
voices often have opaque names ("Google français"), so automatic male/female matching is
best-effort. Use the ⚙ menu to assign which of your device's French voices speaks the
female and male personas (saved on the device); "Tester les deux voix" previews them.
Note: some engines expose only ONE French voice to browsers — the variant selected in the
system's TTS settings (Samsung TTS, and Google TTS's voice I–IV picker). To change the
voice in that case, pick a different variant in the phone's text-to-speech settings and
fully restart the browser.

### Phone access (Tailscale)

Run **`run_server.bat`** — it starts only the backend (no browser window) and exposes it
over Tailscale HTTPS. On the phone: Tailscale ON, then open the `https://….ts.net` URL the
script prints. HTTPS is mandatory — phone browsers block the microphone and speech APIs
on plain `http://`. Keep the server window open; closing it stops the server.

---

## Configuration

Both `main.py` and `api.py` share the same constants at the top of each file:

| Constant | Default | Description |
|---|---|---|
| `WHISPER_MODEL` | `medium` | Whisper model size |
| `WHISPER_DEVICE` | `cuda` | `cuda` or `cpu` |
| `SAMPLE_RATE` | `16000` | Microphone sample rate (Hz) |
| `VAD_THRESHOLD` | `0.5` | Silero-VAD speech confidence cutoff (terminal only) |
| `SILENCE_DURATION_S` | `1.8` | Seconds of silence that end a recording (terminal only) |
| `PIPER_VOICE_PATH` | `fr_FR-upmc-medium.onnx` | Path to Piper voice model |
| `VOCAB_FILE` | `vocabulary.md` | Vocabulary tracking file |
| `PROFILE_FILE` | `profile.json` | Learner knowledge graph |
| `CORRECTIONS_FILE` | `corrections.md` | Grammar correction log |
| `NEW_WORDS_PER_TURN` | `1` | Max new French words per bot reply |
| `DEFAULT_PERSONA` | `marion_cotillard` | Persona id from the `personas/` folder |
| `DEFAULT_LEVEL` | `a1` | Starting CEFR level: `a1`, `a2`, or `b1` |
| `AUDIO_DEVICE` | `None` | Mic input device index (terminal only; None = system default) |
| `AUDIO_OUTPUT_DEVICE` | `2` | Speaker output device index (terminal only) |
| `ENABLE_THINKING` | `False` | Enable model chain-of-thought (slower) |

### Choosing the LLM backend

The LLM backend and model live in `providers.py` and are controlled by environment
variables (no code edit needed):

| Variable | Default | Description |
|---|---|---|
| `LLM_BACKEND` | `ollama` | `ollama` (local) or `nvidia` (hosted API) |
| `OLLAMA_MODEL` | `qwen3.6:27b` | Ollama model tag (when `LLM_BACKEND=ollama`) |
| `NVIDIA_MODEL` | `meta/llama-3.3-70b-instruct` | NVIDIA catalog id (when `LLM_BACKEND=nvidia`) |
| `NVIDIA_API_KEY` | — | Required for the NVIDIA backend |
| `NVIDIA_BASE_URL` | `https://integrate.api.nvidia.com/v1` | OpenAI-compatible endpoint |

**Ollama Cloud models** (free tier): keep `LLM_BACKEND=ollama`, run `ollama signin` once,
then point `OLLAMA_MODEL` at a `-cloud` tag — no code change, nothing loads on your GPU.
Verified free & working: `gemma4:31b-cloud` and `gpt-oss:120b-cloud` (some tags such as
`qwen3.5:cloud` require a paid subscription). Free tier has session/weekly rate limits.

To use NVIDIA's hosted API instead of local Ollama, set the backend and your API key.

**Windows (this project's default platform):** the simplest path is `launch.bat` — set
`LLM_BACKEND=nvidia` in it, and store the key once (persists for future terminals):

```bat
setx NVIDIA_API_KEY "nvapi-xxxxxxxx"
```

Reopen the terminal afterwards so the new value is picked up. To set the variables just
for one session instead of persistently:

```bat
REM Command Prompt
set LLM_BACKEND=nvidia
set NVIDIA_API_KEY=nvapi-xxxxxxxx
uvicorn api:app --port 8000
```

```powershell
# PowerShell
$env:LLM_BACKEND   = "nvidia"
$env:NVIDIA_API_KEY = "nvapi-xxxxxxxx"
uvicorn api:app --port 8000
```

**macOS / Linux:**

```bash
export LLM_BACKEND=nvidia
export NVIDIA_API_KEY=nvapi-xxxxxxxx
python main.py        # or: uvicorn api:app --port 8000
```

> Do **not** include the word `Bearer` in `NVIDIA_API_KEY` — just the `nvapi-...` value.
> Note that NVIDIA's free tier can be slow (tens of seconds per reply) and not every model
> in the catalog is callable on every account.

Both backends share the same system prompt and `ENABLE_THINKING` toggle.

### Speech backends (STT / TTS) — optional ElevenLabs

Speech-to-text and text-to-speech default to the local **faster-whisper** and **Piper**
models. You can optionally route either (or both) through the **ElevenLabs** API instead —
handy for testing higher-quality voices. Config lives in `elevenlabs_api.py`; everything is
env-driven:

| Variable | Default | Description |
|---|---|---|
| `STT_BACKEND` | `whisper` | `whisper` (local) or `elevenlabs` |
| `TTS_BACKEND` | `piper` | `piper` (local) or `elevenlabs` |
| `ELEVENLABS_API_KEY` | — | **Your API key goes here** (required when either backend is `elevenlabs`) |
| `ELEVENLABS_VOICE_FEMALE` | Sarah | Voice id used for female personas (free-tier voice) |
| `ELEVENLABS_VOICE_MALE` | Adam | Voice id used for male personas (free-tier voice) |
| `ELEVENLABS_TTS_MODEL` | `eleven_multilingual_v2` | TTS model |
| `ELEVENLABS_STT_MODEL` | `scribe_v1` | STT model |

**Where to put your key (Windows):**

```bat
setx ELEVENLABS_API_KEY "sk_your_key_here"
```

Reopen the terminal, then enable the backend(s) — either flip `STT_BACKEND` / `TTS_BACKEND`
in `launch.bat`, or set them for one session:

```bat
set STT_BACKEND=elevenlabs
set TTS_BACKEND=elevenlabs
uvicorn api:app --port 8000
```

The default voice ids are generic multilingual voices; for the best French, replace them
with native-French voice ids from your ElevenLabs library via `ELEVENLABS_VOICE_MALE` /
`ELEVENLABS_VOICE_FEMALE`. Male/female selection still follows each persona's `genre`.

If ElevenLabs TTS fails mid-session (out of credits, network error, …), synthesis
automatically **falls back to local Piper** so the conversation never breaks — provided
Piper is installed (see lite mode below).

### Lightweight / all-cloud mode

`faster-whisper`, `piper`, and `sounddevice` are imported **lazily** — they load only when
a *local* backend is selected. So if you run everything in the cloud
(`STT_BACKEND=elevenlabs`, `TTS_BACKEND=elevenlabs`, and `LLM_BACKEND=nvidia` or an Ollama
`-cloud` model), none of the heavy libraries or the GPU are needed. Install just the thin
deps and run the web interface on any machine:

```bash
pip install -r requirements-lite.txt
uvicorn api:app --port 8000
```

Access the UI from any browser, including a phone (the server runs elsewhere; the phone is
just the client). Note: the automatic Piper TTS fallback only works if Piper is installed,
so a pure lite install trades that safety net for a smaller footprint.

---

## How it works

```
Microphone → silero-VAD → faster-whisper → Ollama/Qwen3.6 → Piper TTS → Speaker
  (terminal)                                      ↕
                               vocabulary.md · profile.json · corrections.md

Browser mic → POST /turn → faster-whisper → Ollama/Qwen3.6 → Piper TTS → WAV → Browser
  (web)                                              ↕
                               vocabulary.md · profile.json · corrections.md
```

0. **Opening** — the bot greets the learner and asks the first question to start the
   conversation (on launch, and after any persona/level switch or reset).
1. **VAD** — Silero records in 512-sample chunks, requiring several consecutive frames above
   threshold before committing to speech, to filter background noise (terminal only).
2. **STT** — faster-whisper transcribes French audio; low-confidence and short clips are dropped.
3. **LLM** — the model replies at A1 level, tags new vocabulary, grammar corrections, and
   personal facts shared by the learner.
4. **Memory** — The learner profile is stored as a typed knowledge graph (`profile.json`),
   e.g. `["apprenant", "habite_à", "Rio de Janeiro"]`. The bot extracts new facts from each
   turn and injects a natural-language summary into the system prompt.
5. **Vocabulary** — New words are logged to `vocabulary.md` in dictionary format, grouped by
   session date.
6. **TTS** — Piper synthesises only the French reply; corrections are display-only. The
   `fr_FR-upmc-medium` voice is multi-speaker, so the bot uses a male voice (*pierre*) for
   male personas and a female voice (*jessica*) for female ones, matching each persona's
   `genre` field.
