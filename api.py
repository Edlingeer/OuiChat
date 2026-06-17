"""FastAPI backend — web alternative to main.py.
Start with:  uvicorn api:app --reload --port 8000
Then open:   http://localhost:8000
"""

import base64
import os
import tempfile
from datetime import date

import ollama
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from corrections import parse_correction, save_correction
from llm import chat, build_system_prompt
from memory import get_graph, summarize_for_prompt
from stt import transcribe_file
from tts import synthesize_to_bytes
from vocab import get_known_words, add_words

# ── Configuration (mirrors main.py) ──────────────────────────────────────────
OLLAMA_MODEL       = "gemma4:12b"
WHISPER_MODEL      = "medium"
WHISPER_DEVICE     = "cuda"
PIPER_VOICE_PATH   = "fr_FR-upmc-medium.onnx"
VOCAB_FILE         = "vocabulary.md"
CORRECTIONS_FILE   = "corrections.md"
PROFILE_FILE       = "profile.json"
NEW_WORDS_PER_TURN = 1
ENABLE_THINKING    = False

_HERE = os.path.dirname(os.path.abspath(__file__))
PIPER_VOICE_PATH = os.path.join(_HERE, PIPER_VOICE_PATH)
VOCAB_FILE       = os.path.join(_HERE, VOCAB_FILE)
CORRECTIONS_FILE = os.path.join(_HERE, CORRECTIONS_FILE)
PROFILE_FILE     = os.path.join(_HERE, PROFILE_FILE)
# ─────────────────────────────────────────────────────────────────────────────

app = FastAPI(title="OuiChat")


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
        model=OLLAMA_MODEL,
        vocab_file=VOCAB_FILE,
        profile_file=PROFILE_FILE,
        new_words_per_turn=NEW_WORDS_PER_TURN,
        enable_thinking=ENABLE_THINKING,
    )

    correction_text = None
    if correcao:
        _, correction_text = parse_correction(correcao)
        save_correction(CORRECTIONS_FILE, user_text, correcao)

    wav_bytes = synthesize_to_bytes(french_reply, PIPER_VOICE_PATH)

    return {
        "user_text":  user_text,
        "bot_reply":  french_reply,
        "correction": correction_text,
        "new_words":  new_words,
        "audio_b64":  base64.b64encode(wav_bytes).decode(),
    }


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
    resp = ollama.generate(model=OLLAMA_MODEL, prompt=prompt, think=False)
    raw  = resp["response"].strip().splitlines()[0]  # take first line only
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


# Serve the frontend — must be last so API routes take priority
app.mount("/", StaticFiles(directory=os.path.join(_HERE, "static"), html=True), name="static")
