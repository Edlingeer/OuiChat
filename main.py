"""French A1 language learning bot — entry point."""

# ── Configuration ──────────────────────────────────────────────────────────────
OLLAMA_MODEL       = "gemma4:12b"   # or "gemma4:26b" (set WHISPER_DEVICE="cpu")
WHISPER_MODEL      = "medium"
WHISPER_DEVICE     = "cuda"         # set to "cpu" when using gemma4:26b
SAMPLE_RATE        = 16000
VAD_THRESHOLD      = 0.5
SILENCE_DURATION_S = 1.8
PIPER_VOICE_PATH   = "fr_FR-upmc-medium.onnx"
VOCAB_FILE         = "vocabulary.md"
CORRECTIONS_FILE   = "corrections.md"
PROFILE_FILE       = "profile.json"
NEW_WORDS_PER_TURN = 1
ENABLE_THINKING    = False          # True → <|think|> chain-of-thought (slower)
DEFAULT_PERSONA    = "marion_cotillard"  # id from personas/ folder
DEFAULT_LEVEL      = "a1"               # a1 | a2 | b1
AUDIO_DEVICE        = None  # mic input  — set to index from check_audio.py if needed
AUDIO_OUTPUT_DEVICE = 2 # speaker out — set to index from check_audio.py if needed
# ───────────────────────────────────────────────────────────────────────────────

import os
import time

# Resolve all data files relative to this script's directory
_HERE = os.path.dirname(os.path.abspath(__file__))
PIPER_VOICE_PATH = os.path.join(_HERE, PIPER_VOICE_PATH)
VOCAB_FILE       = os.path.join(_HERE, VOCAB_FILE)
CORRECTIONS_FILE = os.path.join(_HERE, CORRECTIONS_FILE)
PROFILE_FILE     = os.path.join(_HERE, PROFILE_FILE)

from vad import record_until_silence
from stt import transcribe
from llm import chat
from tts import speak
from vocab import get_known_words
from corrections import save_correction, parse_correction
from persona import load_persona
from llm import list_levels


def _banner() -> None:
    known = get_known_words(VOCAB_FILE)
    print("━" * 52)
    print("  🇫🇷  French A1 Conversation Bot")
    print("━" * 52)
    print(f"  LLM    : {OLLAMA_MODEL}")
    print(f"  Whisper: {WHISPER_MODEL}  [{WHISPER_DEVICE.upper()}]")
    print(f"  Vocab  : {len(known)} mots appris")
    print("━" * 52)
    print("  Appuyez sur Ctrl+C pour quitter.\n")


def main() -> None:
    _banner()
    persona = load_persona(DEFAULT_PERSONA)
    levels  = {l["id"]: l["name"] for l in list_levels()}
    print(f"  Persona : {persona['name']}")
    print(f"  Niveau  : {levels.get(DEFAULT_LEVEL, DEFAULT_LEVEL)}\n")

    while True:
        # 1. Record
        audio = record_until_silence(
            sample_rate=SAMPLE_RATE,
            vad_threshold=VAD_THRESHOLD,
            silence_duration_s=SILENCE_DURATION_S,
            device=AUDIO_DEVICE,
        )

        # 2. Transcribe
        user_text = transcribe(audio, WHISPER_MODEL, WHISPER_DEVICE, SAMPLE_RATE)
        if not user_text:
            continue  # false VAD trigger — loop silently

        print(f"Vous : {user_text}")

        # 3. LLM reply (chat() returns cleaned text; we print it here)
        french_reply, correcao, new_words = chat(
            user_text=user_text,
            model=OLLAMA_MODEL,
            vocab_file=VOCAB_FILE,
            profile_file=PROFILE_FILE,
            new_words_per_turn=NEW_WORDS_PER_TURN,
            enable_thinking=ENABLE_THINKING,
            persona_data=persona,
            level=DEFAULT_LEVEL,
        )

        print(f"Bot : {french_reply}")

        # 4. Correction block (display only, never spoken)
        if correcao:
            _, clean_msg = parse_correction(correcao)
            print("  ─────────────────────────────────────────")
            print("  Correção (português):")
            print(f"  {clean_msg}")
            print("  ─────────────────────────────────────────")
            save_correction(CORRECTIONS_FILE, user_text, correcao)

        # 5. Speak the French reply, then pause before reopening the mic
        speak(french_reply, PIPER_VOICE_PATH, device=AUDIO_OUTPUT_DEVICE)
        time.sleep(0.6)

        # 6. Vocabulary count
        if new_words:
            total = len(get_known_words(VOCAB_FILE))
            print(f"Vocabulaire : {total} mots appris au total.")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nAu revoir ! Bonne continuation !")
