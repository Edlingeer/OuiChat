# OuiChat — French Conversation Bot

A French language learning bot for Brazilian Portuguese speakers at beginner levels
(CEFR A1–B1). You speak French, it transcribes you, replies in French as a famous
persona, shows the reply, and speaks it aloud. Grammar corrections and vocabulary
translations are given in Brazilian Portuguese.

Every heavy component is pluggable:

- **LLM** — local Ollama (default), free Ollama Cloud models, or NVIDIA's hosted API.
- **Speech-to-text** — local faster-whisper (default), ElevenLabs Scribe, or the
  browser's own speech recognition.
- **Text-to-speech** — local Piper (default), ElevenLabs, or the browser's own voices.

Two ways to use it: a **terminal bot** (`main.py`, microphone + speakers) or a
**web interface** (`api.py`, browser-based — also works from a phone).

---

## Prerequisites

| Tool | Needed for | Install |
|---|---|---|
| Python 3.11+ (conda recommended) | everything | https://python.org |
| Ollama | `LLM_BACKEND=ollama` (local or `-cloud` models) | https://ollama.com/download |
| CUDA Toolkit 12.4+ | local Whisper / Piper / local LLM on the GPU | https://developer.nvidia.com/cuda-downloads |
| Tailscale | optional — phone access via `run_server.bat` | https://tailscale.com |

For an all-cloud setup with no GPU, see [Lightweight / all-cloud mode](#lightweight--all-cloud-mode).

### The LLM

See [Choosing the LLM backend](#choosing-the-llm-backend) for details.

- **Local Ollama** — pull the model:
  ```bash
  ollama pull qwen3.6:27b
  ```
- **Ollama Cloud (free tier)** — run `ollama signin` once and use a `-cloud` tag such as
  `gemma4:31b-cloud`. **The bundled `launch.bat` / `run_server.bat` use this by default.**
- **NVIDIA hosted API** — no local model needed; grab a free API key at
  [build.nvidia.com](https://build.nvidia.com) (default model `meta/llama-3.3-70b-instruct`).

### Download the Piper French voice

Needed for local TTS (the default, and the fallback if ElevenLabs fails). Download both
files and place them in the project folder (next to `main.py`):

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

`environment.lock.yml` pins the exact versions of a known-working install
(`conda env create -f environment.lock.yml`). Without conda, `pip install -r requirements.txt`
installs the same packages (install a CUDA build of PyTorch yourself if you want GPU).

Always run the app from the **activated** env (`conda activate french_teacher`, or via
the `.bat` launchers). Calling the env's `python.exe` directly skips activation, and
local Whisper then fails on the GPU with `cublas64_12.dll is not found`.

The `.bat` launchers expect the env to be called `french_teacher` and Anaconda to be
installed at `%USERPROFILE%\anaconda3` — edit the `activate.bat` path in them if yours differs.

---

## Running

### Quick start (Windows)

Double-click **`launch.bat`** — it frees port 8000 (killing any stale server), starts
the web server in its own window, and opens Firefox at `http://localhost:8000`.

The backends are chosen by the `set "..."` lines near the top of the file:

| Line | Shipped value | Meaning |
|---|---|---|
| `LLM_BACKEND` | `ollama` | `ollama` or `nvidia` |
| `OLLAMA_MODEL` | `gemma4:31b-cloud` | free Ollama Cloud model — comment the line out to use the local default `qwen3.6:27b` |
| `STT_BACKEND` | `whisper` | `whisper` (local) or `elevenlabs` |
| `TTS_BACKEND` | `piper` | `piper` (local) or `elevenlabs` |

### Terminal bot (microphone + speakers)

```bash
python main.py
```

The bot greets you first, then listens; speak, pause, and it replies. Press **Ctrl+C**
to exit. If the mic or speakers aren't picked up, run `python check_audio.py` — it
records from every input device and plays a tone on every output device, so you can set
`AUDIO_DEVICE` / `AUDIO_OUTPUT_DEVICE` in `main.py` to the right indices.

The terminal bot uses the persona and level set by `DEFAULT_PERSONA` / `DEFAULT_LEVEL`
in `main.py`.

### Web interface (browser-based)

```bash
uvicorn api:app --port 8000
```

Then open `http://localhost:8000` in your browser:

- Pick a **persona** and a **level** (A1 / A2 / B1) from the dropdowns — switching
  either starts a fresh conversation.
- Tap **▶ Commencer la conversation** (one tap unlocks audio + microphone — required on
  iOS) and the bot opens with a greeting.
- **Hold** the microphone button to speak, release to send.
- **Nouvelle session** clears the conversation and gets a new greeting.
- Hover a word in a bot reply and click **+ Vocabulaire** to have it translated and
  added to your vocabulary list.
- The header buttons open live panels for **Vocabulaire**, **Profil** (what the bot has
  learned about you) and **Corrections** (history of your mistakes).

> **First reply is slow with a local model.** The first LLM call loads the model into
> VRAM (~30–60s for `qwen3.6:27b`); every turn after is fast. On the web app this
> happens during the opening greeting, so the model is warm by the time you speak.
> Cloud LLMs skip this entirely.

### Speech modes (button in the app header)

The web app has three speech modes, cycled by the voice button in the header:

| Mode | STT (your speech) | TTS (bot voice) | Notes |
|---|---|---|---|
| **📱 Voix** | device (Web Speech API) | device | Only text crosses the wire. Default on Android/desktop Chrome. Device STT may auto-correct your mistakes, hiding them from the tutor. |
| **🎙️ Mixte** | server (`STT_BACKEND`, Whisper by default) | device | Faithful transcription (catches your errors) + free device voice. **Default on iPhone/iPad** (iOS speech recognition is unreliable). |
| **☁️ Voix** | server (`STT_BACKEND`) | server (`TTS_BACKEND`) | Full server pipeline — Whisper/Piper locally, or ElevenLabs. |

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

Run **`run_server.bat`** — it frees port 8000, starts only the backend (no browser
window, listening on `127.0.0.1`) and runs `tailscale serve --bg 8000` to expose it over
Tailscale HTTPS. On the phone: Tailscale ON, then open the `https://….ts.net` URL the
script prints. HTTPS is mandatory — phone browsers block the microphone and speech APIs
on plain `http://`. Keep the server window open; closing it stops the server. Its
backends are set at the top of the file, same as `launch.bat`.

---

## Configuration

### Constants in `main.py` / `api.py`

Set at the top of each file (`api.py` only has the ones that apply to the web app):

| Constant | Default | Description |
|---|---|---|
| `WHISPER_MODEL` | `medium` | Whisper model size |
| `WHISPER_DEVICE` | `cuda` | `cuda` or `cpu` (use `cpu` if VRAM is tight alongside a local LLM) |
| `PIPER_VOICE_PATH` | `fr_FR-upmc-medium.onnx` | Path to the Piper voice model |
| `PIPER_SPEAKER_FEMALE` / `PIPER_SPEAKER_MALE` | `0` / `1` | Piper speaker ids (*jessica* / *pierre*) |
| `VOCAB_FILE` | `vocabulary.md` | Vocabulary tracking file |
| `PROFILE_FILE` | `profile.json` | Learner knowledge graph |
| `CORRECTIONS_FILE` | `corrections.md` | Grammar correction log |
| `NEW_WORDS_PER_TURN` | `1` | Max new French words per bot reply |
| `ENABLE_THINKING` | `False` | Enable model chain-of-thought (slower) |
| `DEFAULT_PERSONA` | `marion` | Persona id from the `personas/` folder (web: starting persona) |
| `DEFAULT_LEVEL` | `a1` | CEFR level: `a1`, `a2`, or `b1` (web: starting level) |
| `SAMPLE_RATE` | `16000` | Microphone sample rate in Hz (terminal only) |
| `VAD_THRESHOLD` | `0.5` | Silero-VAD speech confidence cutoff (terminal only) |
| `SILENCE_DURATION_S` | `1.8` | Seconds of silence that end a recording (terminal only) |
| `AUDIO_DEVICE` | `None` | Mic input device index (terminal only; `None` = system default) |
| `AUDIO_OUTPUT_DEVICE` | `2` | Speaker output device index (terminal only; find yours with `check_audio.py`) |

The data files (`vocabulary.md`, `profile.json`, `corrections.md`) are created on first
use and are git-ignored. Delete them to start from a blank learner.

### Choosing the LLM backend

The LLM backend and model live in `providers.py` and are controlled by environment
variables (no code edit needed):

| Variable | Default | Description |
|---|---|---|
| `LLM_BACKEND` | `ollama` | `ollama` (local daemon, incl. cloud tags) or `nvidia` (hosted API) |
| `OLLAMA_MODEL` | `qwen3.6:27b` | Ollama model tag (when `LLM_BACKEND=ollama`) |
| `NVIDIA_MODEL` | `meta/llama-3.3-70b-instruct` | NVIDIA catalog id (when `LLM_BACKEND=nvidia`) |
| `NVIDIA_API_KEY` | — | Required for the NVIDIA backend |
| `NVIDIA_BASE_URL` | `https://integrate.api.nvidia.com/v1` | OpenAI-compatible endpoint |

These are the code defaults; the `.bat` launchers override `OLLAMA_MODEL` with
`gemma4:31b-cloud`.

**Ollama Cloud models** (free tier): keep `LLM_BACKEND=ollama`, run `ollama signin` once,
then point `OLLAMA_MODEL` at a `-cloud` tag — nothing loads on your GPU. Verified free &
working: `gemma4:31b-cloud` and `gpt-oss:120b-cloud` (some tags such as `qwen3.5:cloud`
require a paid subscription). The free tier has session/weekly rate limits.

**NVIDIA hosted API.** On Windows, the simplest path is to set `LLM_BACKEND=nvidia` in
`launch.bat` and store the key once (persists for future terminals):

```bat
setx NVIDIA_API_KEY "nvapi-xxxxxxxx"
```

Reopen the terminal afterwards so the new value is picked up. To set the variables just
for one session instead:

```bat
REM Command Prompt
set LLM_BACKEND=nvidia
set NVIDIA_API_KEY=nvapi-xxxxxxxx
uvicorn api:app --port 8000
```

```powershell
# PowerShell
$env:LLM_BACKEND    = "nvidia"
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
> NVIDIA's free tier can be slow (tens of seconds per reply) and not every model in the
> catalog is callable on every account. If a request fails or times out (120 s), the bot
> replies with a short "connexion lente" apology instead of crashing.

All backends share the same system prompt and `ENABLE_THINKING` toggle.

### Speech backends (STT / TTS) — optional ElevenLabs

Server-side speech-to-text and text-to-speech default to the local **faster-whisper**
and **Piper** models. You can route either (or both) through the **ElevenLabs** API
instead. Config lives in `elevenlabs_api.py`; everything is env-driven:

| Variable | Default | Description |
|---|---|---|
| `STT_BACKEND` | `whisper` | `whisper` (local) or `elevenlabs` |
| `TTS_BACKEND` | `piper` | `piper` (local) or `elevenlabs` |
| `ELEVENLABS_API_KEY` | — | Required when either backend is `elevenlabs` |
| `ELEVENLABS_VOICE_FEMALE` | Sarah | Voice id used for female personas (free-tier voice) |
| `ELEVENLABS_VOICE_MALE` | Adam | Voice id used for male personas (free-tier voice) |
| `ELEVENLABS_TTS_MODEL` | `eleven_multilingual_v2` | TTS model |
| `ELEVENLABS_STT_MODEL` | `scribe_v1` | STT model |
| `ELEVENLABS_STT_LANG` | *(empty = auto-detect)* | ISO-639 language code for STT, e.g. `fr` |

Store the key once (Windows):

```bat
setx ELEVENLABS_API_KEY "sk_your_key_here"
```

Reopen the terminal, then enable the backend(s) — either change `STT_BACKEND` /
`TTS_BACKEND` in `launch.bat` / `run_server.bat`, or set them for one session:

```bat
set STT_BACKEND=elevenlabs
set TTS_BACKEND=elevenlabs
uvicorn api:app --port 8000
```

The default voice ids are generic multilingual voices; for the best French, replace them
with native-French voice ids from your ElevenLabs library. Male/female selection follows
each persona's `genre`.

If ElevenLabs TTS fails mid-session (out of credits, network error, …), synthesis
automatically **falls back to local Piper** — provided Piper is installed (see lite mode
below). If ElevenLabs STT fails, the turn is treated as "no speech" rather than crashing.

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

Access the UI from any browser, including a phone. Notes:

- An Ollama `-cloud` model still needs the Ollama daemon running locally (it proxies to
  the cloud). For a daemon-free thin client use `LLM_BACKEND=nvidia`.
- The automatic Piper fallback only works if Piper is installed, so a pure lite install
  trades that safety net for a smaller footprint.
- The terminal bot (`main.py`) is not supported in lite mode — its voice activity
  detection needs `torch` and `sounddevice`.

---

## Personas

The bot talks as a character inspired by a well-known person, always in French. Shipped
(shown by first name only): Marion (default), Juliette, Sophie, Shakira, Alex, Kylian.

Each persona is a JSON file in `personas/`, picked up automatically by the web app's
dropdown:

```json
{
  "id": "alex",
  "name": "Alex",
  "genre": "m",
  "wikidata_id": "Q2649225",
  "character": "calme, humble, concentré, passionné par la nature",
  "triples": [
    ["persona", "né_à", "Sacramento (Californie)"],
    ["persona", "aime", "escalade"],
    ["persona", "connu_pour", "le film Free Solo"]
  ]
}
```

- `id` must match the filename.
- `genre` is `"m"` or `"f"` (defaults to `"f"` if omitted). It drives French gender
  agreement in the system prompt (né/née, connu/connue) and which voice speaks.
- `wikidata_id` (optional) records where the facts came from.
- `triples` are `["persona", relation, value]` facts. Recognised relations: `né_à`/`née_à`,
  `né_le`/`née_le`, `habite_à`, `profession`, `parle`, `aime`, `a_famille`, `a_un_enfant`,
  `conjoint`, `connu_pour`/`connue_pour`, `a_reçu`, `engage_pour`, `grandi_à`/`grandie_à`,
  `a_écrit`, `a_réalisé`, `a_refusé`. Others are ignored.

> **Disclaimer.** The personas are a language-learning device only. Their biographical
> facts are publicly available information taken from Wikipedia / Wikidata, and may be
> incomplete or out of date. Replies are generated by an AI playing a character: they are
> not written, reviewed, or endorsed by the people the personas are based on, and do not
> reflect their views. This project is not affiliated with any of them. It is a free, non-commercial hobby
> project: nothing is sold, and no persona is used for advertising or any commercial purpose.

---

## How it works

```
Terminal:  mic → silero-VAD → STT → LLM → TTS → speakers

Web:       ☁️  browser mic → POST /turn       → STT → LLM → TTS → WAV → browser
           🎙️  browser mic → POST /turn_stt   → STT → LLM → text → device voice
           📱  device STT  → POST /turn_text  →       LLM → text → device voice

           STT = faster-whisper | ElevenLabs      LLM = Ollama | NVIDIA
           TTS = Piper | ElevenLabs (falls back to Piper)

           every turn reads/writes: vocabulary.md · profile.json · corrections.md
```

0. **Opening** — the bot greets the learner and asks the first question (on launch, and
   after any persona/level switch or new session) via `/greeting` or `/greeting_text`.
1. **VAD** (terminal only) — Silero records in 512-sample chunks, requiring several
   consecutive frames above threshold before committing to speech, to filter background
   noise; recording stops after `SILENCE_DURATION_S` of silence.
2. **STT** — French audio is transcribed; with Whisper, low-confidence segments are
   dropped (and, in the terminal, clips under 0.8 s).
3. **LLM** — the model replies in character at the selected level and appends tags for
   new vocabulary, grammar corrections, and personal facts shared by the learner. The
   last 8 exchanges are kept as context.
4. **Memory** — the learner profile is a typed knowledge graph (`profile.json`), e.g.
   `["apprenant", "habite_à", "Rio de Janeiro"]`. New facts are extracted from each
   turn and a natural-language summary is injected into the system prompt.
5. **Vocabulary** — new words are logged to `vocabulary.md` in dictionary format
   (word — Portuguese translation, French example), grouped by date.
6. **Corrections** — shown in Portuguese and logged to `corrections.md` with a
   colour-coded error type; never spoken.
7. **TTS** — only the French reply is spoken. The `fr_FR-upmc-medium` Piper voice is
   multi-speaker, so male personas get *pierre* and female personas *jessica*
   (ElevenLabs and device voices follow the same `genre` split).

### Project layout

| File | Role |
|---|---|
| `main.py` | Terminal bot entry point |
| `api.py` | FastAPI web backend (serves `static/index.html`) |
| `static/index.html` | Single-page web UI |
| `llm.py` | System prompt, level constraints, history, tag parsing, opening greeting |
| `providers.py` | LLM backend dispatch (Ollama / NVIDIA) |
| `stt.py` / `tts.py` | Speech-to-text / text-to-speech dispatch |
| `elevenlabs_api.py` | ElevenLabs STT/TTS client |
| `vad.py` | Silero voice activity detection (terminal) |
| `persona.py` + `personas/` | Persona loading and prompt summary |
| `memory.py` | Learner knowledge graph |
| `vocab.py` / `corrections.py` | Vocabulary and corrections logs |
| `check_audio.py` | Audio device diagnostic |
| `launch.bat` / `run_server.bat` | Windows launchers (desktop / phone via Tailscale) |

---

## Known limitations

- **No voice activity detection in the web app.** Everything recorded while the mic
  button is held is transcribed. Whisper can turn pure background noise into invented
  text (e.g. "Sous-titrage ST'501"), which is then sent to the bot as if you had said it.
  The terminal bot is not affected (it uses Silero VAD).
- **One conversation at a time.** The web server keeps a single shared conversation,
  persona and level in memory, so it is meant for one learner, not several at once.

---

## License

MIT — see [LICENSE](LICENSE).
