"""Optional ElevenLabs backend for speech-to-text (Scribe) and text-to-speech.

This is an alternative to the local faster-whisper (STT) and Piper (TTS). Enable
each independently via environment variables; both default OFF (local models).

    STT_BACKEND=elevenlabs      # else "whisper" (default, local)
    TTS_BACKEND=elevenlabs      # else "piper"   (default, local)

╔══════════════════════════════════════════════════════════════════════════╗
║  PUT YOUR API KEY IN THE  ELEVENLABS_API_KEY  ENVIRONMENT VARIABLE.        ║
║  Windows (persists):   setx ELEVENLABS_API_KEY "sk_your_key_here"          ║
║  then reopen the terminal. (Or set it in launch.bat — see that file.)      ║
╚══════════════════════════════════════════════════════════════════════════╝

Optional overrides (all have sensible defaults):
    ELEVENLABS_TTS_MODEL       default "eleven_multilingual_v2"
    ELEVENLABS_STT_MODEL       default "scribe_v1"
    ELEVENLABS_VOICE_FEMALE    voice id used for female personas
    ELEVENLABS_VOICE_MALE      voice id used for male personas
    ELEVENLABS_STT_LANG        ISO-639 code (default "" = auto-detect)

Uses httpx (already installed as an openai dependency); imported lazily so
local-only setups need nothing from this file.
"""

import io
import os
import wave

STT_BACKEND  = os.environ.get("STT_BACKEND", "whisper").strip().lower()
TTS_BACKEND  = os.environ.get("TTS_BACKEND", "piper").strip().lower()
API_KEY      = os.environ.get("ELEVENLABS_API_KEY", "")

TTS_MODEL    = os.environ.get("ELEVENLABS_TTS_MODEL", "eleven_multilingual_v2")
STT_MODEL    = os.environ.get("ELEVENLABS_STT_MODEL", "scribe_v1")
STT_LANG     = os.environ.get("ELEVENLABS_STT_LANG", "").strip()   # "" → auto-detect

# Default voices — multilingual voices available on the free tier (some classic
# voices like Rachel/Charlotte are "library" voices that require a paid plan).
# Override with any voice ids from your library — ideally native French speakers.
VOICE_FEMALE = os.environ.get("ELEVENLABS_VOICE_FEMALE", "EXAVITQu4vr4xnSDxMaL")  # Sarah
VOICE_MALE   = os.environ.get("ELEVENLABS_VOICE_MALE",   "pNInz6obpgDQGcFmaJgB")  # Adam

BASE_URL   = "https://api.elevenlabs.io/v1"
_OUTPUT_SR = 16000   # request 16 kHz PCM and wrap it in a WAV container


def use_stt() -> bool:
    return STT_BACKEND == "elevenlabs"


def use_tts() -> bool:
    return TTS_BACKEND == "elevenlabs"


def _client():
    import httpx
    if not API_KEY:
        raise RuntimeError(
            "ElevenLabs backend enabled but ELEVENLABS_API_KEY is not set. "
            "Set it, e.g.  setx ELEVENLABS_API_KEY \"sk_...\"  then reopen the terminal."
        )
    return httpx.Client(base_url=BASE_URL, headers={"xi-api-key": API_KEY}, timeout=60)


def transcribe_bytes(audio_bytes: bytes, filename: str = "audio.wav") -> str:
    """Speech-to-text via ElevenLabs Scribe. Accepts any container ffmpeg-free
    (wav/webm/mp4/mp3). Returns the recognised text."""
    data = {"model_id": STT_MODEL}
    if STT_LANG:
        data["language_code"] = STT_LANG
    with _client() as c:
        r = c.post(
            "/speech-to-text",
            data=data,
            files={"file": (filename, audio_bytes, "application/octet-stream")},
        )
        r.raise_for_status()
        return (r.json().get("text") or "").strip()


def tts_wav(text: str, male: bool) -> bytes:
    """Text-to-speech via ElevenLabs. Returns 16-bit mono WAV bytes at 16 kHz."""
    voice = VOICE_MALE if male else VOICE_FEMALE
    with _client() as c:
        r = c.post(
            f"/text-to-speech/{voice}",
            params={"output_format": f"pcm_{_OUTPUT_SR}"},
            json={"text": text, "model_id": TTS_MODEL},
        )
        r.raise_for_status()
        pcm = r.content
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(_OUTPUT_SR)
        w.writeframes(pcm)
    return buf.getvalue()
