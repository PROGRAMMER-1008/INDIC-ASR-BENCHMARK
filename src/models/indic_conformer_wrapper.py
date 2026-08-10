"""
AI4Bharat indic-conformer-600m-multilingual wrapper.

Based on the usage pattern documented on the model's HF model card
(trust_remote_code=True, raw waveform tensor input, language code + decoding
mode string arguments to a custom `forward`/`transcribe` method exposed by
the model's own remote code).

IMPORTANT CAVEAT (read before running):
This wrapper is written to match AI4Bharat's documented model card API as of
last time it was checked, but `trust_remote_code=True` models can change
their exact calling convention between revisions since the code lives in the
model repo, not in a versioned library. If this wrapper throws an
AttributeError or TypeError on the exact method call, open the model card at
https://huggingface.co/ai4bharat/indic-conformer-600m-multilingual and check
the current "How to use" snippet -- the fix is almost always a one-line
change to the call in `transcribe()` below, not a structural rewrite.

This is also a 600M-parameter model. On CPU it will be dramatically slower
than whisper-tiny (~39M params) per sample -- budget for that. If it's too
slow to finish even a 20-sample run in reasonable time, see README "Known
Limitations" for the whisper-base fallback substitution.
"""

import numpy as np
import torch
from transformers import AutoModel

from src.config import DEVICE, INDIC_CONFORMER_CHECKPOINT, INDIC_CONFORMER_DECODING
from src.models.base import ASRModelWrapper


class IndicConformerWrapper(ASRModelWrapper):
    name = "indic-conformer-600m"

    def __init__(self, checkpoint: str = INDIC_CONFORMER_CHECKPOINT, decoding: str = INDIC_CONFORMER_DECODING):
        self.decoding = decoding
        self.model = AutoModel.from_pretrained(checkpoint, trust_remote_code=True)
        self.model = self.model.to(DEVICE)
        self.model.eval()

    def transcribe(self, audio_array: np.ndarray, sample_rate: int, language_code: str) -> str:
        # indic-conformer's remote code expects a raw waveform tensor. We
        # keep resampling responsibility centralized in data_loader.py
        # (FLEURS ships 16kHz audio, which is what this model expects), so
        # this wrapper asserts rather than silently resampling -- if audio
        # arrives at the wrong rate, that's a bug upstream we want to know
        # about, not paper over here.
        assert sample_rate == 16000, (
            f"indic-conformer expects 16kHz audio, got {sample_rate}Hz. "
            "Resample in data_loader.py before calling this wrapper."
        )

        wav_tensor = torch.tensor(audio_array, dtype=torch.float32).unsqueeze(0).to(DEVICE)

        # Map our internal language name (e.g. "hindi") to AI4Bharat's short
        # code (e.g. "hi") via config.LANGUAGES -- done in run_benchmark.py,
        # so `language_code` arriving here is already the model-specific code.
        with torch.no_grad():
            transcript = self.model(wav_tensor, language_code, self.decoding)

        # The model card indicates this returns a string directly for a
        # single-utterance batch of size 1; if a future revision returns a
        # list, take the first element defensively.
        if isinstance(transcript, list):
            transcript = transcript[0]

        return transcript
