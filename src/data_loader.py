"""
Loads evaluation samples from google/fleurs for the configured languages.

NOT EXECUTED IN THIS SESSION -- see top-level README "How This Project Was
Built" for why. Written against the documented `datasets` library API for
FLEURS (a standard, well-established HF dataset loading pattern), but you
should treat the very first run of this file as a real test, not a formality.
"""

from dataclasses import dataclass

import numpy as np
from datasets import load_dataset

from src.config import DATASET_NAME, DATASET_SPLIT, LANGUAGES, SAMPLES_PER_LANGUAGE


@dataclass
class EvalSample:
    language: str          # our internal short name, e.g. "hindi"
    sample_id: str         # unique id for this sample within its language
    audio_array: np.ndarray
    sample_rate: int
    reference_text: str


def load_language_samples(language_key: str, n_samples: int = SAMPLES_PER_LANGUAGE) -> list[EvalSample]:
    """
    Load up to n_samples from FLEURS test split for one language.

    Wrapped in error handling per-item: FLEURS occasionally has corrupt or
    zero-length audio entries in some language configs. We skip and log
    rather than crash the whole load, consistent with the same
    skip-and-log policy used during inference (see run_benchmark.py).
    """
    lang_config = LANGUAGES[language_key]
    fleurs_config = lang_config["fleurs_config"]

    print(f"[data_loader] Loading {DATASET_NAME} config={fleurs_config} split={DATASET_SPLIT} ...")

    try:
        # streaming=False: dataset is small enough (a few hundred MB per
        # language at most for 20-50 samples) to just materialize. Streaming
        # would save disk but adds complexity we don't need at this scale.
        ds = load_dataset(DATASET_NAME, fleurs_config, split=DATASET_SPLIT)
    except Exception as e:
        print(f"[data_loader] FAILED to load FLEURS config '{fleurs_config}': {e}")
        raise

    samples: list[EvalSample] = []
    skipped = 0

    for i, row in enumerate(ds):
        if len(samples) >= n_samples:
            break
        try:
            audio = row["audio"]
            audio_array = np.asarray(audio["array"], dtype=np.float32)
            sample_rate = audio["sampling_rate"]
            reference_text = row["transcription"]

            if audio_array.size == 0 or not reference_text.strip():
                skipped += 1
                continue

            samples.append(
                EvalSample(
                    language=language_key,
                    sample_id=f"{language_key}_{i}",
                    audio_array=audio_array,
                    sample_rate=sample_rate,
                    reference_text=reference_text,
                )
            )
        except Exception as e:
            skipped += 1
            print(f"[data_loader] Skipped sample index {i} for {language_key}: {e}")
            continue

    print(f"[data_loader] Loaded {len(samples)} samples for {language_key} (skipped {skipped}).")
    return samples


def load_all_samples() -> list[EvalSample]:
    """Load samples for every language configured in config.LANGUAGES."""
    all_samples: list[EvalSample] = []
    for language_key in LANGUAGES:
        all_samples.extend(load_language_samples(language_key))
    return all_samples
