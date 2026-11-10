"""OpenAI Whisper via Hugging Face `transformers` (PyTorch)."""

import numpy as np

from src.config import LANGUAGES, MODELS, get_device
from src.models.base import ASRModelWrapper


class WhisperWrapper(ASRModelWrapper):
    def __init__(self, model_name: str):
        import torch
        from transformers import pipeline

        self.name = model_name
        spec = MODELS[model_name]
        device = get_device()
        self.pipe = pipeline(
            task="automatic-speech-recognition",
            model=spec["checkpoint"],
            device=0 if device == "cuda" else -1,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
            chunk_length_s=30,   # Whisper's window; longer clips are chunked
        )

    def transcribe(self, audio_array: np.ndarray, sample_rate: int, language_key: str) -> str:
        # Force language + task. Otherwise Whisper guesses the language from
        # the first seconds of audio, adding an error source that has
        # nothing to do with transcription quality.
        out = self.pipe(
            {"raw": audio_array, "sampling_rate": sample_rate},
            generate_kwargs={"language": LANGUAGES[language_key]["whisper_name"],
                             "task": "transcribe"},
        )
        return out["text"]

    def close(self) -> None:
        del self.pipe
