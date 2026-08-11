"""Text-to-speech: Piper (default, local) or ElevenLabs (optional, via
TTS_BACKEND=elevenlabs).

Piper and sounddevice are imported lazily (only when actually used), so an all-cloud
setup (TTS_BACKEND=elevenlabs) doesn't need them installed. When ElevenLabs is the
backend but a call fails (e.g. out of credits, network error), synthesis falls back to
local Piper if it's available."""

import io
import wave

import numpy as np

import elevenlabs_api


_voice = None   # piper.voice.PiperVoice, created lazily on first local synthesis


def _wav_to_float(wav_bytes: bytes) -> tuple[np.ndarray, int]:
    """Decode 16-bit mono WAV bytes to a float32 [-1, 1] array + sample rate."""
    with wave.open(io.BytesIO(wav_bytes), "rb") as w:
        sr = w.getframerate()
        pcm = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16)
    return (pcm.astype(np.float32) / 32768.0), sr


def _float_to_wav_bytes(audio: np.ndarray, sample_rate: int) -> bytes:
    pcm = (audio * 32767).clip(-32768, 32767).astype(np.int16)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm.tobytes())
    return buf.getvalue()


def _load_voice(model_path: str):
    global _voice
    if _voice is None:
        from piper.voice import PiperVoice   # lazy: heavy dep, local backend only
        _voice = PiperVoice.load(model_path)
    return _voice


def _piper_audio(text: str, model_path: str, speaker_id: int | None):
    """Synthesize with Piper. Returns (float32 array, sample_rate) or (None, None)."""
    from piper import SynthesisConfig   # lazy
    voice = _load_voice(model_path)
    syn = SynthesisConfig(speaker_id=speaker_id) if speaker_id is not None else None
    chunks = [c.audio_float_array for c in voice.synthesize(text, syn_config=syn)]
    if not chunks:
        return None, None
    return np.concatenate(chunks).astype(np.float32), voice.config.sample_rate


def _elevenlabs_audio(text: str, speaker_id: int | None):
    """ElevenLabs TTS -> (float32 array, sample_rate), or (None, None) on failure."""
    try:
        return _wav_to_float(elevenlabs_api.tts_wav(text, male=(speaker_id == 1)))
    except Exception as e:
        print(f"  [ElevenLabs TTS error -> falling back to Piper] "
              f"{type(e).__name__}: {str(e)[:160]}", flush=True)
        return None, None


def speak(text: str, model_path: str, device=None, speaker_id: int | None = None) -> None:
    if not text.strip():
        return

    audio = sr = None
    if elevenlabs_api.use_tts():
        audio, sr = _elevenlabs_audio(text, speaker_id)
    if audio is None:                       # local backend, or ElevenLabs failed
        try:
            audio, sr = _piper_audio(text, model_path, speaker_id)
        except Exception as e:
            print(f"  [Piper TTS error] {type(e).__name__}: {str(e)[:160]}", flush=True)
            return
    if audio is None:
        print("  [TTS] warning: no audio produced", flush=True)
        return

    import sounddevice as sd                 # lazy: only the terminal app plays audio
    sd.play(audio, samplerate=sr, device=device)
    sd.wait()


def synthesize_to_bytes(text: str, model_path: str, speaker_id: int | None = None) -> bytes:
    """Return WAV bytes for the text — used by the web API (no audio playback)."""
    if not text.strip():
        return b""

    if elevenlabs_api.use_tts():
        try:
            return elevenlabs_api.tts_wav(text, male=(speaker_id == 1))
        except Exception as e:
            print(f"  [ElevenLabs TTS error -> falling back to Piper] "
                  f"{type(e).__name__}: {str(e)[:160]}", flush=True)

    try:
        audio, sr = _piper_audio(text, model_path, speaker_id)
    except Exception as e:
        print(f"  [Piper TTS error] {type(e).__name__}: {str(e)[:160]}", flush=True)
        return b""
    if audio is None:
        return b""
    return _float_to_wav_bytes(audio, sr)
