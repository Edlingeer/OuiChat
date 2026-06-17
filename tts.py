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
