"""
faster-whisper: the same Whisper weights converted to CTranslate2, which is
much faster on CPU (int8 quantization) and uses less memory. Same family,
different inference engine: a good "does the engine change accuracy?" test.
"""

import numpy as np

from src.config import LANGUAGES, MODELS, get_device
from src.models.base import ASRModelWrapper


class FasterWhisperWrapper(ASRModelWrapper):
    def __init__(self, model_name: str):
        from faster_whisper import WhisperModel

        self.name = model_name
        device = get_device()
        self.model = WhisperModel(
            MODELS[model_name]["checkpoint"],
            device=device,
            compute_type="float16" if device == "cuda" else "int8",
        )

    def transcribe(self, audio_array: np.ndarray, sample_rate: int, language_key: str) -> str:
        segments, _info = self.model.transcribe(
            audio_array,
            language=LANGUAGES[language_key]["iso"],
            task="transcribe",
            beam_size=5,
            vad_filter=False,
        )
        # `segments` is a lazy generator: the actual decoding happens here.
        return " ".join(seg.text.strip() for seg in segments)

    def close(self) -> None:
        del self.model
