"""
AI4Bharat IndicConformer (600M, multilingual).

A Conformer encoder (convolution + self-attention) trained on Indian
languages, decoded with CTC (default) or RNN-T. Unlike Whisper it is NOT
an encoder-decoder language model: CTC emits characters/tokens frame by
frame, which makes it fast and much less prone to "hallucinating" text
that was never spoken.

The model uses `trust_remote_code=True`: its calling code lives in the model
repo, so the exact call can change between revisions. If `transcribe()`
raises a TypeError/AttributeError, compare with the "How to use" snippet on
https://huggingface.co/ai4bharat/indic-conformer-600m-multilingual . If the
download fails with a 401/403, accept the model terms on that page and run
`huggingface-cli login`.
"""

import numpy as np

from src.config import LANGUAGES, MODELS, get_device
from src.models.base import ASRModelWrapper


class IndicConformerWrapper(ASRModelWrapper):
    def __init__(self, model_name: str = "indic-conformer-600m"):
        from transformers import AutoModel

        self.name = model_name
        spec = MODELS[model_name]
        self.device = get_device()
        self.decoding = spec.get("decoding", "ctc")
        self.model = AutoModel.from_pretrained(spec["checkpoint"], trust_remote_code=True)
        try:
            self.model = self.model.to(self.device)
        except Exception:
            self.device = "cpu"   # remote-code models sometimes are not .to()-able
        try:
            self.model.eval()
        except Exception:
            pass

    def transcribe(self, audio_array: np.ndarray, sample_rate: int, language_key: str) -> str:
        import torch

        assert sample_rate == 16000, f"IndicConformer expects 16 kHz, got {sample_rate}"
        wav = torch.tensor(audio_array, dtype=torch.float32).unsqueeze(0)
        if self.device == "cuda":
            wav = wav.to("cuda")
        with torch.no_grad():
            out = self.model(wav, LANGUAGES[language_key]["iso"], self.decoding)
        if isinstance(out, (list, tuple)):
            out = out[0]
        return str(out)

    def close(self) -> None:
        del self.model
