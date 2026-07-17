"""Text-to-speech: Piper (default, local) or ElevenLabs (optional, via
TTS_BACKEND=elevenlabs). Playback through sounddevice."""

import io
import wave

import numpy as np
import sounddevice as sd
from piper.voice import PiperVoice
from piper import SynthesisConfig

import elevenlabs_api


_voice: PiperVoice | None = None


def _wav_to_float(wav_bytes: bytes) -> tuple[np.ndarray, int]:
    """Decode 16-bit mono WAV bytes to a float32 [-1, 1] array + sample rate."""
    with wave.open(io.BytesIO(wav_bytes), "rb") as w:
        sr = w.getframerate()
        pcm = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16)
    return (pcm.astype(np.float32) / 32768.0), sr


def _load_voice(model_path: str) -> PiperVoice:
    global _voice
    if _voice is None:
        _voice = PiperVoice.load(model_path)
    return _voice


def _syn_config(speaker_id: int | None) -> SynthesisConfig | None:
    return SynthesisConfig(speaker_id=speaker_id) if speaker_id is not None else None


def speak(text: str, model_path: str, device=None, speaker_id: int | None = None) -> None:
    if not text.strip():
        return

    if elevenlabs_api.use_tts():
        audio, sr = _wav_to_float(elevenlabs_api.tts_wav(text, male=(speaker_id == 1)))
        sd.play(audio, samplerate=sr, device=device)
        sd.wait()
        return

    voice = _load_voice(model_path)

    chunks: list[np.ndarray] = []
    for audio_chunk in voice.synthesize(text, syn_config=_syn_config(speaker_id)):
        chunks.append(audio_chunk.audio_float_array)

    if not chunks:
        print("  [TTS] warning: no audio produced", flush=True)
        return

    audio = np.concatenate(chunks).astype(np.float32)
    sd.play(audio, samplerate=voice.config.sample_rate, device=device)
    sd.wait()


def synthesize_to_bytes(text: str, model_path: str, speaker_id: int | None = None) -> bytes:
    """Return WAV bytes for the text — used by the web API (no audio playback)."""
    import io
    import wave as _wave
    if not text.strip():
        return b""

    if elevenlabs_api.use_tts():
        return elevenlabs_api.tts_wav(text, male=(speaker_id == 1))

    voice = _load_voice(model_path)
    chunks: list[np.ndarray] = []
    for audio_chunk in voice.synthesize(text, syn_config=_syn_config(speaker_id)):
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
