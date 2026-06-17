"""Piper TTS synthesis and sounddevice playback. CPU only, no WAV files."""

import numpy as np
import sounddevice as sd
from piper.voice import PiperVoice


_voice: PiperVoice | None = None


def _load_voice(model_path: str) -> PiperVoice:
    global _voice
    if _voice is None:
        _voice = PiperVoice.load(model_path)
    return _voice


def speak(text: str, model_path: str, device=None) -> None:
    if not text.strip():
        return
    voice = _load_voice(model_path)

    chunks: list[np.ndarray] = []
    for audio_chunk in voice.synthesize(text):
        chunks.append(audio_chunk.audio_float_array)

    if not chunks:
        print("  [TTS] warning: no audio produced", flush=True)
        return

    audio = np.concatenate(chunks).astype(np.float32)
    sd.play(audio, samplerate=voice.config.sample_rate, device=device)
    sd.wait()


def synthesize_to_bytes(text: str, model_path: str) -> bytes:
    """Return WAV bytes for the text — used by the web API (no audio playback)."""
    import io
    import wave as _wave
    if not text.strip():
        return b""
    voice = _load_voice(model_path)
    chunks: list[np.ndarray] = []
    for audio_chunk in voice.synthesize(text):
        chunks.append(audio_chunk.audio_float_array)
    if not chunks:
        return b""
    audio = np.concatenate(chunks)
    pcm = (audio * 32767).clip(-32768, 32767).astype(np.int16)
    buf = io.BytesIO()
    with _wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(voice.config.sample_rate)
        wf.writeframes(pcm.tobytes())
    return buf.getvalue()
