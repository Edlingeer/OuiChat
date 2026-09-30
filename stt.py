"""Speech-to-text wrapper around faster-whisper (default) or ElevenLabs Scribe
(optional, via STT_BACKEND=elevenlabs).

faster-whisper is imported lazily (only when the local backend actually runs), so an
all-cloud setup (STT_BACKEND=elevenlabs) doesn't need it installed."""

import io
import wave

import numpy as np

import elevenlabs_api


_model = None   # faster_whisper.WhisperModel, created lazily on first local transcription


def _elevenlabs_stt(audio_bytes: bytes, filename: str = "audio.wav") -> str:
    """Call ElevenLabs STT, degrading to '' (treated as no speech) on error rather
    than crashing the turn — e.g. if the API key lacks the speech_to_text scope."""
    try:
        return elevenlabs_api.transcribe_bytes(audio_bytes, filename=filename)
    except Exception as e:
        print(f"  [ElevenLabs STT error] {type(e).__name__}: {str(e)[:200]}", flush=True)
        return ""


def _pcm16_wav_bytes(audio: np.ndarray, sample_rate: int) -> bytes:
    """Pack a float32 [-1, 1] mono array into 16-bit WAV bytes (for upload)."""
    pcm = (np.clip(audio, -1.0, 1.0) * 32767).astype(np.int16)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sample_rate)
        w.writeframes(pcm.tobytes())
    return buf.getvalue()


def _load_model(model_size: str, device: str):
    global _model
    if _model is None:
        from faster_whisper import WhisperModel   # lazy: heavy dep, local backend only
        compute_type = "float16" if device == "cuda" else "int8"
        _model = WhisperModel(model_size, device=device, compute_type=compute_type)
    return _model


def transcribe(
    audio: np.ndarray,
    model_size: str,
    device: str,
    sample_rate: int,
) -> str:
    if audio.size == 0:
        return ""

    # Reject clips shorter than 0.8 s — likely a false VAD trigger
    if len(audio) < int(0.8 * sample_rate):
        return ""

    if elevenlabs_api.use_stt():
        return _elevenlabs_stt(_pcm16_wav_bytes(audio, sample_rate))

    model = _load_model(model_size, device)
    segments, _ = model.transcribe(
        audio,
        language="fr",
        beam_size=5,
        temperature=0,                    # greedy decoding — reduces hallucination
        condition_on_previous_text=False, # don't feed prior output back as context
        no_speech_threshold=0.6,          # drop segments Whisper itself doubts
        compression_ratio_threshold=2.4,  # drop repetitive/hallucinated segments
    )

    # Also filter by confidence score
    words = [seg.text.strip() for seg in segments if seg.avg_logprob > -0.8]
    return " ".join(words).strip()


def transcribe_file(path: str, model_size: str, device: str) -> str:
    """Transcribe from a file path — used by the web API.
    faster-whisper decodes WebM/MP4/etc. via PyAV (bundled FFmpeg libraries)."""
    if elevenlabs_api.use_stt():
        import os
        with open(path, "rb") as f:
            return _elevenlabs_stt(f.read(), filename=os.path.basename(path))

    # Decode first so an empty/corrupt upload (e.g. a very quick tap on the mic
    # button) is treated as "no speech" instead of crashing the request.
    from faster_whisper.audio import decode_audio   # lazy: local backend only
    try:
        audio = decode_audio(path, sampling_rate=16000)
    except Exception as e:
        print(f"  [STT] could not decode upload: {type(e).__name__}: {str(e)[:120]}", flush=True)
        return ""
    if audio.size == 0:
        return ""

    model = _load_model(model_size, device)
    segments, _ = model.transcribe(
        audio,
        language="fr",
        beam_size=5,
        temperature=0,
        condition_on_previous_text=False,
        no_speech_threshold=0.6,
        compression_ratio_threshold=2.4,
    )
    words = [seg.text.strip() for seg in segments if seg.avg_logprob > -0.8]
    return " ".join(words).strip()
