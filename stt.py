"""Speech-to-text wrapper around faster-whisper."""

import numpy as np
from faster_whisper import WhisperModel


_model: WhisperModel | None = None


def _load_model(model_size: str, device: str) -> WhisperModel:
    global _model
    if _model is None:
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

    # Reject audio shorter than 0.8 s — likely a false VAD trigger
    if len(audio) < int(0.8 * sample_rate):
        return ""

    model = _load_model(model_size, device)
    segments, _ = model.transcribe(audio, language="fr", beam_size=5)

    # avg_logprob < -1.0 indicates Whisper is guessing on noise; discard those segments
    words = [seg.text.strip() for seg in segments if seg.avg_logprob > -1.0]
    return " ".join(words).strip()
