"""FastAPI backend — web alternative to main.py.
Start with:  uvicorn api:app --reload --port 8000
Then open:   http://localhost:8000
"""

import base64
import os
import tempfile
from datetime import date

from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles

import providers
from corrections import parse_correction, save_correction
from llm import chat, opening, build_system_prompt, list_levels
from memory import get_graph, summarize_for_prompt
from persona import list_personas, load_persona, is_male
from stt import transcribe_file
from tts import synthesize_to_bytes
from vocab import get_known_words, add_words

# ── Configuration (mirrors main.py) ──────────────────────────────────────────
# LLM backend & model are configured in providers.py (or via env vars:
#   LLM_BACKEND="ollama"|"nvidia", OLLAMA_MODEL, NVIDIA_MODEL, NVIDIA_API_KEY).
WHISPER_MODEL      = "medium"
WHISPER_DEVICE     = "cuda"
PIPER_VOICE_PATH   = "fr_FR-upmc-medium.onnx"
# fr_FR-upmc-medium is multi-speaker: 0 = jessica (féminine), 1 = pierre (masculine).
PIPER_SPEAKER_FEMALE = 0
PIPER_SPEAKER_MALE   = 1
VOCAB_FILE         = "vocabulary.md"
CORRECTIONS_FILE   = "corrections.md"
PROFILE_FILE       = "profile.json"
NEW_WORDS_PER_TURN = 1
ENABLE_THINKING    = False
DEFAULT_PERSONA    = "marion"
DEFAULT_LEVEL      = "a1"

_HERE = os.path.dirname(os.path.abspath(__file__))
PIPER_VOICE_PATH = os.path.join(_HERE, PIPER_VOICE_PATH)
VOCAB_FILE       = os.path.join(_HERE, VOCAB_FILE)
CORRECTIONS_FILE = os.path.join(_HERE, CORRECTIONS_FILE)
PROFILE_FILE     = os.path.join(_HERE, PROFILE_FILE)
# ─────────────────────────────────────────────────────────────────────────────

app = FastAPI(title="OuiChat")

import elevenlabs_api
print(f"  OuiChat backends -> LLM: {providers.backend_label()}  |  "
      f"STT: {elevenlabs_api.STT_BACKEND}  |  TTS: {elevenlabs_api.TTS_BACKEND}", flush=True)

_current_persona: dict = load_persona(DEFAULT_PERSONA)
_current_level: str = DEFAULT_LEVEL


def _speaker_id() -> int:
    """Pick the TTS voice matching the current persona's gender."""
    return PIPER_SPEAKER_MALE if is_male(_current_persona) else PIPER_SPEAKER_FEMALE


@app.post("/turn")
async def turn(audio: UploadFile = File(...)):
    """Receive browser audio → STT → LLM → TTS → return JSON."""
    ext = os.path.splitext(audio.filename or "")[1] or ".webm"
    with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as tmp:
        tmp.write(await audio.read())
        tmp_path = tmp.name

    try:
        user_text = transcribe_file(tmp_path, WHISPER_MODEL, WHISPER_DEVICE)
    finally:
        os.unlink(tmp_path)

    if not user_text:
        return JSONResponse({"error": "no_speech"})

    french_reply, correcao, new_words = chat(
        user_text=user_text,
        vocab_file=VOCAB_FILE,
        profile_file=PROFILE_FILE,
        new_words_per_turn=NEW_WORDS_PER_TURN,
        enable_thinking=ENABLE_THINKING,
        persona_data=_current_persona,
        level=_current_level,
    )

    correction_text = None
    if correcao:
        _, correction_text = parse_correction(correcao)
        save_correction(CORRECTIONS_FILE, user_text, correcao)

    wav_bytes = synthesize_to_bytes(french_reply, PIPER_VOICE_PATH, speaker_id=_speaker_id())

    return {
        "user_text":  user_text,
        "bot_reply":  french_reply,
        "correction": correction_text,
        "new_words":  new_words,
        "audio_b64":  base64.b64encode(wav_bytes).decode(),
    }


@app.post("/greeting")
def greeting():
    """Bot opens the conversation (called when a new session starts)."""
    french_reply, new_words = opening(
        vocab_file=VOCAB_FILE,
        profile_file=PROFILE_FILE,
        new_words_per_turn=NEW_WORDS_PER_TURN,
        enable_thinking=ENABLE_THINKING,
        persona_data=_current_persona,
        level=_current_level,
    )
    wav_bytes = synthesize_to_bytes(french_reply, PIPER_VOICE_PATH, speaker_id=_speaker_id())
    return {
        "bot_reply":  french_reply,
        "new_words":  new_words,
        "audio_b64":  base64.b64encode(wav_bytes).decode(),
    }


# ── Text-only endpoints for on-device speech (phone STT/TTS via Web Speech API) ──
# The browser transcribes/synthesises; the server only handles text + the LLM, so
# no audio crosses the wire and no server-side Whisper/Piper/ElevenLabs is used.

@app.post("/turn_text")
def turn_text(body: dict):
    """Receive already-transcribed text → LLM → return text (no audio)."""
    user_text = (body.get("text") or "").strip()
    if not user_text:
        return JSONResponse({"error": "no_text"}, status_code=400)

    french_reply, correcao, new_words = chat(
        user_text=user_text,
        vocab_file=VOCAB_FILE,
        profile_file=PROFILE_FILE,
        new_words_per_turn=NEW_WORDS_PER_TURN,
        enable_thinking=ENABLE_THINKING,
        persona_data=_current_persona,
        level=_current_level,
    )

    correction_text = None
    if correcao:
        _, correction_text = parse_correction(correcao)
        save_correction(CORRECTIONS_FILE, user_text, correcao)

    return {
        "user_text":  user_text,
        "bot_reply":  french_reply,
        "correction": correction_text,
        "new_words":  new_words,
    }


@app.post("/greeting_text")
def greeting_text():
    """Bot opens the conversation, text only (browser speaks it)."""
    french_reply, new_words = opening(
        vocab_file=VOCAB_FILE,
        profile_file=PROFILE_FILE,
        new_words_per_turn=NEW_WORDS_PER_TURN,
        enable_thinking=ENABLE_THINKING,
        persona_data=_current_persona,
        level=_current_level,
    )
    return {"bot_reply": french_reply, "new_words": new_words}


@app.post("/turn_stt")
async def turn_stt(audio: UploadFile = File(...)):
    """Hybrid: browser audio → server STT (Whisper) → LLM → TEXT out (no server
    TTS — the browser speaks the reply with its own voice)."""
    ext = os.path.splitext(audio.filename or "")[1] or ".webm"
    with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as tmp:
        tmp.write(await audio.read())
        tmp_path = tmp.name
    try:
        user_text = transcribe_file(tmp_path, WHISPER_MODEL, WHISPER_DEVICE)
    finally:
        os.unlink(tmp_path)

    if not user_text:
        return JSONResponse({"error": "no_speech"})

    french_reply, correcao, new_words = chat(
        user_text=user_text,
        vocab_file=VOCAB_FILE,
        profile_file=PROFILE_FILE,
        new_words_per_turn=NEW_WORDS_PER_TURN,
        enable_thinking=ENABLE_THINKING,
        persona_data=_current_persona,
        level=_current_level,
    )

    correction_text = None
    if correcao:
        _, correction_text = parse_correction(correcao)
        save_correction(CORRECTIONS_FILE, user_text, correcao)

    return {
        "user_text":  user_text,
        "bot_reply":  french_reply,
        "correction": correction_text,
        "new_words":  new_words,
    }


@app.get("/personas")
def get_personas():
    """List available personas."""
    return {"personas": list_personas()}


@app.post("/set_persona")
def set_persona(body: dict):
    """Switch persona and reset conversation history."""
    global _current_persona
    persona_id = body.get("id", "").strip()
    if not persona_id:
        return JSONResponse({"error": "no_id"}, status_code=400)
    _current_persona = load_persona(persona_id)
    from llm import _history
    _history.clear()
    return {"ok": True, "name": _current_persona["name"],
            "genre": _current_persona.get("genre", "f")}


@app.get("/levels")
def get_levels():
    """List available levels."""
    return {"levels": list_levels()}


@app.post("/set_level")
def set_level(body: dict):
    """Switch level and reset conversation history."""
    global _current_level
    level_id = body.get("id", "").strip()
    if level_id not in ("a1", "a2", "b1"):
        return JSONResponse({"error": "invalid_level"}, status_code=400)
    _current_level = level_id
    from llm import _history
    _history.clear()
    return {"ok": True}


@app.post("/reset")
def reset():
    """Clear conversation history (start a new session)."""
    from llm import _history
    _history.clear()
    return {"ok": True}


@app.post("/define")
async def define_word(body: dict):
    """Look up a word via the LLM and add it to vocabulary."""
    word    = body.get("word", "").strip()
    context = body.get("context", "").strip()
    if not word:
        return JSONResponse({"error": "no_word"}, status_code=400)

    prompt = (
        f"Traduis le mot français « {word} » en portugais brésilien. "
        f"Contexte : « {context} ». "
        f"Réponds UNIQUEMENT avec ce format, sans rien d'autre : traduction|exemple_en_français"
    )
    raw  = providers.generate(prompt).strip().splitlines()[0]  # take first line only
    parts      = [p.strip() for p in raw.split("|", 1)]
    traduction = parts[0] if parts else raw
    exemple    = parts[1] if len(parts) > 1 else ""

    add_words(VOCAB_FILE, [{"mot": word, "traduction": traduction,
                            "exemple": exemple, "date": str(date.today())}])
    return {"word": word, "traduction": traduction, "exemple": exemple}


@app.get("/vocabulary")
def vocabulary():
    if not os.path.exists(VOCAB_FILE):
        return {"content": ""}
    with open(VOCAB_FILE, encoding="utf-8") as f:
        return {"content": f.read()}


@app.get("/profile")
def profile():
    graph = get_graph(PROFILE_FILE)
    return {
        "summary": summarize_for_prompt(graph),
        "triples": graph.get("triples", []),
    }


@app.get("/corrections")
def corrections():
    if not os.path.exists(CORRECTIONS_FILE):
        return {"content": ""}
    with open(CORRECTIONS_FILE, encoding="utf-8") as f:
        return {"content": f.read()}


@app.get("/")
def index():
    """Serve the app shell with no-cache headers so browsers never run a stale
    index.html (which would miss new frontend features like the opening greeting)."""
    return FileResponse(
        os.path.join(_HERE, "static", "index.html"),
        headers={"Cache-Control": "no-cache, no-store, must-revalidate"},
    )


# Serve the frontend — must be last so API routes take priority
app.mount("/", StaticFiles(directory=os.path.join(_HERE, "static"), html=True), name="static")
