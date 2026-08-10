"""
Common interface every ASR model wrapper must implement.

DESIGN RATIONALE
-----------------
The benchmark loop (run_benchmark.py) never talks to Whisper or
indic-conformer APIs directly. It only calls `wrapper.transcribe(...)` on
whatever wrapper objects are registered in MODEL_REGISTRY. This means:

  - Adding model #3 means writing ONE new class with ONE method
    (`transcribe`) and adding one line to MODEL_REGISTRY. Nothing else in
    the pipeline, scoring, or reporting code needs to change.
  - Each model's loading, preprocessing, and decoding quirks (Whisper wants
    log-mel features via a processor; indic-conformer wants a raw waveform
    tensor + language code + decoding mode) are fully contained inside that
    model's own wrapper file. The rest of the codebase doesn't need to know
    or care how a given model works internally.
  - Swapping whisper-tiny for whisper-small is a one-line change in
    config.py, not a code change here.

Every wrapper must implement:

    transcribe(audio_array: np.ndarray, sample_rate: int, language_code: str) -> str

- audio_array: mono float32 waveform, values in [-1, 1]
- sample_rate: sample rate of audio_array in Hz (FLEURS is 16kHz, but a
  wrapper should resample if it needs a different rate rather than assuming)
- language_code: the model-specific language identifier. We pass FLEURS'
  language *name* (e.g. "hindi") from the benchmark loop and let each
  wrapper map it to whatever code its own API expects (see config.py
  LANGUAGES dict for the per-model code mappings), since different models
  use different language code conventions (Whisper wants "hindi" as a
  spoken-language string or "hi" as an ISO code depending on API surface;
  indic-conformer wants its own short codes).

Returns: the raw (non-normalized) hypothesis transcript string. Normalization
happens once, centrally, in normalize.py -- wrappers should NOT normalize
their own output, so we have one auditable place where normalization logic
lives.
"""

from abc import ABC, abstractmethod

import numpy as np


class ASRModelWrapper(ABC):
    """Abstract base class for an ASR model under benchmark."""

    #: Short display name used in results tables / report. Override in subclass.
    name: str = "unnamed-model"

    @abstractmethod
    def transcribe(self, audio_array: np.ndarray, sample_rate: int, language_code: str) -> str:
        """
        Transcribe a single audio sample.

        Must raise on unrecoverable errors (caller catches and logs/skips).
        Must NOT apply text normalization -- return raw model output.
        """
        raise NotImplementedError
