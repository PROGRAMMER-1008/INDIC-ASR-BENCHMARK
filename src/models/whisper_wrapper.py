"""
Whisper ASR wrapper, using the `transformers` pipeline API.

Chosen over the `openai-whisper` package because:
  - `transformers` gives us a consistent API surface with how we'd load any
    other HF model (including indic-conformer), so the two wrappers in this
    project look structurally similar -- easier to read side by side.
  - Device/dtype handling is uniform with the rest of the transformers
    ecosystem (`.to(device)`, `torch_dtype=...`).
"""

import numpy as np
import torch
from transformers import pipeline

from src.config import DEVICE, TORCH_DTYPE, WHISPER_CHECKPOINT
from src.models.base import ASRModelWrapper

# Whisper's spoken-language names for the `generate_kwargs={"language": ...}`
# forcing argument. Whisper's tokenizer expects the *English name* of the
# language here (e.g. "hindi", "tamil"), not an ISO code -- this is a known
# quirk of Whisper's language-forcing API.
WHISPER_LANGUAGE_NAMES = {
    "hindi": "hindi",
    "tamil": "tamil",
}


class WhisperWrapper(ASRModelWrapper):
    name = "whisper-tiny"

    def __init__(self, checkpoint: str = WHISPER_CHECKPOINT):
        self.checkpoint = checkpoint
        # device index: transformers pipeline wants -1 for CPU, 0 for first GPU
        device_index = 0 if DEVICE == "cuda" else -1
        self.pipe = pipeline(
            task="automatic-speech-recognition",
            model=checkpoint,
            device=device_index,
            torch_dtype=TORCH_DTYPE,
        )

    def transcribe(self, audio_array: np.ndarray, sample_rate: int, language_code: str) -> str:
        whisper_lang = WHISPER_LANGUAGE_NAMES.get(language_code, language_code)

        # generate_kwargs forces the decoder to a known language + the
        # "transcribe" task (as opposed to "translate" into English), which
        # both improves accuracy and makes results comparable across
        # languages -- without forcing, Whisper auto-detects language from
        # the first few seconds of audio, which is an extra source of error
        # unrelated to transcription quality that we don't want polluting
        # the WER numbers.
        result = self.pipe(
            audio_array,
            generate_kwargs={"language": whisper_lang, "task": "transcribe"},
        )
        return result["text"]
