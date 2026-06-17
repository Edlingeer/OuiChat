# OuiChat — French A1 Conversation Bot

An offline French language learning bot for complete beginners (CEFR A1). It listens to
you speak French, transcribes it, replies in French, displays the reply, and speaks it
aloud — all running locally on your GPU.

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

Download both files and place them in the repo root (next to `main.py`):

```
https://huggingface.co/rhasspy/piper-voices/resolve/main/fr/fr_FR/upmc/medium/fr_FR-upmc-medium.onnx
https://huggingface.co/rhasspy/piper-voices/resolve/main/fr/fr_FR/upmc/medium/fr_FR-upmc-medium.onnx.json
```

---

## Installation

```bash
# Create and activate the conda environment
conda env create -f environment.yml
conda activate french_teacher

# Or with pip — install PyTorch with CUDA 12.4 first
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
```

---

## Running

```bash
python main.py
```

Press **Ctrl+C** to exit.

---

## Configuration (`main.py`)

| Constant | Default | Description |
|---|---|---|
| `OLLAMA_MODEL` | `gemma4:12b` | Ollama model tag |
| `WHISPER_MODEL` | `medium` | Whisper model size |
| `WHISPER_DEVICE` | `cuda` | `cuda` or `cpu` |
| `SAMPLE_RATE` | `16000` | Microphone sample rate (Hz) |
| `VAD_THRESHOLD` | `0.5` | Silero-VAD speech confidence cutoff |
| `SILENCE_DURATION_S` | `1.2` | Seconds of silence that end a recording |
| `PIPER_VOICE_PATH` | `fr_FR-upmc-medium.onnx` | Path to Piper voice model |
| `VOCAB_FILE` | `vocabulary.md` | Vocabulary tracking file |
| `PROFILE_FILE` | `profile.md` | Learner profile (remembered facts) |
| `CORRECTIONS_FILE` | `corrections.md` | Grammar correction log |
| `NEW_WORDS_PER_TURN` | `1` | Max new French words per bot reply |
| `AUDIO_DEVICE` | `None` | Mic input device index (None = system default) |
| `AUDIO_OUTPUT_DEVICE` | `None` | Speaker output device index (None = system default) |
| `ENABLE_THINKING` | `False` | Enable Gemma 4 chain-of-thought reasoning |

---

## How it works

```
Microphone → silero-VAD → faster-whisper → Ollama/Gemma 4 → Piper TTS → Speaker
                                                   ↕
                                    vocabulary.md · profile.md · corrections.md
```

1. **VAD** — Silero records in 512-sample chunks, requiring several consecutive frames above
   threshold before committing to speech, to filter background noise.
2. **STT** — faster-whisper transcribes French audio; low-confidence and short clips are dropped.
3. **LLM** — Gemma 4 replies at A1 level, tags new vocabulary, grammar corrections, and
   personal facts shared by the learner.
4. **Memory** — Vocabulary, corrections, and learner profile are persisted across sessions
   and injected back into the system prompt on next run.
5. **TTS** — Piper synthesises only the French reply body; corrections are display-only.
