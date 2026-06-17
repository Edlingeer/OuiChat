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

    # Reject clips shorter than 0.8 s — likely a false VAD trigger
    if len(audio) < int(0.8 * sample_rate):
        return ""

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
    faster-whisper calls ffmpeg internally to decode WebM/MP4/etc."""
    model = _load_model(model_size, device)
    segments, _ = model.transcribe(
        path,
        language="fr",
        beam_size=5,
        temperature=0,
        condition_on_previous_text=False,
        no_speech_threshold=0.6,
        compression_ratio_threshold=2.4,
    )
    words = [seg.text.strip() for seg in segments if seg.avg_logprob > -0.8]
    return " ".join(words).strip()
