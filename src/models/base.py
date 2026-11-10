"""
The one interface every ASR model implements.

The benchmark loop only ever calls `transcribe(...)`. Everything model-
specific (feature extraction, decoding mode, language codes) stays inside
the model's own wrapper, so adding a model = one new file + one registry line.

Contract:
  audio_array : mono float32 waveform in [-1, 1], 16 kHz
  language_key: our key ("hindi"); the wrapper looks up whatever code its
                own model needs in config.LANGUAGES
  returns     : the RAW transcript. Never normalize here -- normalization
                lives in exactly one place (src/normalize.py).
"""

from abc import ABC, abstractmethod

import numpy as np


class ASRModelWrapper(ABC):
    name: str = "unnamed-model"

    @abstractmethod
    def transcribe(self, audio_array: np.ndarray, sample_rate: int, language_key: str) -> str:
        raise NotImplementedError

    def close(self) -> None:
        """Free memory (GPU/RAM). Override if the model holds big resources."""
        return None
