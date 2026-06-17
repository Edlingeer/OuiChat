# OuiChat — French A1 Conversation Bot

An offline French language learning bot for complete beginners (CEFR A1). It listens to
you speak French, transcribes it, replies in French, displays the reply, and speaks it
aloud — all running locally on your GPU.

Two ways to use it: a **terminal bot** (classic) or a **web interface** (browser-based).

---

## Prerequisites

| Tool | Install |
|---|---|
| Ollama | https://ollama.com/download |
| CUDA Toolkit 12.4+ | https://developer.nvidia.com/cuda-downloads |
| Python 3.11+ | https://python.org |

### Pull the LLM

```bash
ollama pull gemma4:12b
```

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

### Terminal bot (microphone + speakers)

```bash
python main.py
```

Press **Ctrl+C** to exit.

### Web interface (browser-based)

```bash
uvicorn api:app --port 8000
```

Then open `http://localhost:8000` in your browser. Hold the microphone button to speak,
release to send. The top-right buttons open live panels for vocabulary, learner profile,
and correction history.

---

## Configuration

Both `main.py` and `api.py` share the same constants at the top of each file:

| Constant | Default | Description |
|---|---|---|
| `OLLAMA_MODEL` | `gemma4:12b` | Ollama model tag |
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
| `AUDIO_DEVICE` | `None` | Mic input device index (terminal only; None = system default) |
| `AUDIO_OUTPUT_DEVICE` | `2` | Speaker output device index (terminal only) |
| `ENABLE_THINKING` | `False` | Enable Gemma 4 chain-of-thought (slower) |

---

## How it works

```
Microphone → silero-VAD → faster-whisper → Ollama/Gemma 4 → Piper TTS → Speaker
  (terminal)                                      ↕
                               vocabulary.md · profile.json · corrections.md

Browser mic → POST /turn → faster-whisper → Ollama/Gemma 4 → Piper TTS → WAV → Browser
  (web)                                              ↕
                               vocabulary.md · profile.json · corrections.md
```

1. **VAD** — Silero records in 512-sample chunks, requiring several consecutive frames above
   threshold before committing to speech, to filter background noise (terminal only).
2. **STT** — faster-whisper transcribes French audio; low-confidence and short clips are dropped.
3. **LLM** — Gemma 4 replies at A1 level, tags new vocabulary, grammar corrections, and
   personal facts shared by the learner.
4. **Memory** — The learner profile is stored as a typed knowledge graph (`profile.json`),
   e.g. `["apprenant", "habite_à", "Rio de Janeiro"]`. The bot extracts new facts from each
   turn and injects a natural-language summary into the system prompt.
5. **Vocabulary** — New words are logged to `vocabulary.md` in dictionary format, grouped by
   session date.
6. **TTS** — Piper synthesises only the French reply; corrections are display-only.
