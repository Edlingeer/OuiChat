"""Voice activity detection — records until SILENCE_DURATION_S of silence."""

import numpy as np
import sounddevice as sd
import torch


_model = None
_utils = None


def _load_model():
    global _model, _utils
    if _model is None:
        _model, _utils = torch.hub.load(
            repo_or_dir="snakers4/silero-vad",
            model="silero_vad",
            force_reload=False,
            trust_repo=True,
        )
    return _model, _utils


def record_until_silence(
    sample_rate: int,
    vad_threshold: float,
    silence_duration_s: float,
    chunk_size: int = 512,
    device=None,
    confirm_chunks: int = 4,   # consecutive frames above threshold to confirm real speech
) -> np.ndarray:
    model, _ = _load_model()
    model.reset_states()

    silence_chunks = int(silence_duration_s * sample_rate / chunk_size)
    buffer: list[np.ndarray] = []
    pre_buffer: list[np.ndarray] = []  # holds chunks during confirmation window
    silent_count = 0
    has_speech = False
    confirm_count = 0

    print("En écoute...", flush=True)

    with sd.InputStream(samplerate=sample_rate, channels=1, dtype="float32",
                        blocksize=chunk_size, device=device) as stream:
        while True:
            audio_chunk, _ = stream.read(chunk_size)
            chunk = audio_chunk[:, 0]

            tensor = torch.from_numpy(chunk.copy()).unsqueeze(0)
            confidence = model(tensor, sample_rate).item()

            if not has_speech:
                if confidence >= vad_threshold:
                    confirm_count += 1
                    pre_buffer.append(chunk)
                    if confirm_count >= confirm_chunks:
                        has_speech = True
                        buffer.extend(pre_buffer)
                        pre_buffer.clear()
                else:
                    confirm_count = 0
                    pre_buffer.clear()
            else:
                buffer.append(chunk)
                if confidence < vad_threshold:
                    silent_count += 1
                    if silent_count >= silence_chunks:
                        break
                else:
                    silent_count = 0

    return np.concatenate(buffer) if buffer else np.array([], dtype=np.float32)
