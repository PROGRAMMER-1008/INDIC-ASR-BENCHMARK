"""
Model registry: name -> constructor.

Adding model #N:  (1) add an entry to config.MODELS,  (2) write a wrapper
class that subclasses ASRModelWrapper,  (3) map its `family` here.
Wrappers are imported lazily, so a missing optional package (e.g.
faster-whisper) only breaks the models that need it.
"""

from src.config import MODELS
from src.models.base import ASRModelWrapper


def build_model(model_name: str) -> ASRModelWrapper:
    if model_name not in MODELS:
        raise KeyError(f"Unknown model '{model_name}'. Known: {', '.join(MODELS)}")
    family = MODELS[model_name]["family"]
    if family == "whisper":
        from src.models.whisper_wrapper import WhisperWrapper
        return WhisperWrapper(model_name)
    if family == "faster-whisper":
        from src.models.faster_whisper_wrapper import FasterWhisperWrapper
        return FasterWhisperWrapper(model_name)
    if family == "indic-conformer":
        from src.models.indic_conformer_wrapper import IndicConformerWrapper
        return IndicConformerWrapper(model_name)
    raise ValueError(f"No wrapper for family '{family}'")
