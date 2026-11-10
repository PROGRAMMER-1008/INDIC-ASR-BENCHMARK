"""
Evaluation data.

HOW IT WORKS
------------
1. The first time you ask for N Hindi samples, we stream them from Google
   FLEURS (test split) and export them to  data_cache/hindi/*.wav  plus a
   manifest.csv  (sample_id, wav, text, duration_s).
2. Every later run reads straight from that folder: no re-download, fully
   offline, identical samples each time.
3. Because the cache is just "wav files + manifest.csv", you can benchmark
   YOUR OWN audio too: create  data_cache/<language>/manifest.csv  with the
   same columns and run with --no-download.

The reference text is FLEURS' `raw_transcription` (original casing and
punctuation); our own normalizer is applied to reference AND hypothesis at
scoring time, so every model is treated identically.
"""

from __future__ import annotations

import csv
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from src.audio import to_mono_16k
from src.config import DATA_CACHE_DIR, DATASET_NAME, DATASET_SPLIT, LANGUAGES


@dataclass
class EvalSample:
    language: str
    sample_id: str
    audio_array: np.ndarray      # mono float32, 16 kHz
    sample_rate: int
    reference_text: str

    @property
    def audio_seconds(self) -> float:
        return len(self.audio_array) / self.sample_rate


MANIFEST_FIELDS = ["sample_id", "wav", "text", "duration_s"]


def _lang_dir(language_key: str, root: str) -> Path:
    return Path(root) / language_key


def _read_manifest(language_key: str, root: str) -> list[dict]:
    path = _lang_dir(language_key, root) / "manifest.csv"
    if not path.exists():
        return []
    with open(path, encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def export_fleurs(language_key: str, n_samples: int, root: str = DATA_CACHE_DIR) -> None:
    """Backward-compatible FLEURS exporter."""
    export_huggingface_dataset(
        language_key, n_samples, root=root, dataset_name=DATASET_NAME,
        dataset_config=LANGUAGES[language_key]["fleurs_config"], split=DATASET_SPLIT,
        text_field="raw_transcription", fallback_text_field="transcription",
    )


def export_huggingface_dataset(language_key: str, n_samples: int, root: str = DATA_CACHE_DIR,
                               dataset_name: str = DATASET_NAME, dataset_config: str | None = None,
                               split: str = DATASET_SPLIT, audio_field: str = "audio",
                               text_field: str = "text", fallback_text_field: str | None = None) -> None:
    """Stream an HF ASR dataset into the standard local WAV/manifest format.

    The adapter expects the selected audio field to decode to an ``array`` and
    ``sampling_rate`` mapping, and the text field to contain the transcript.
    This lets Kathbath, Vistaar, Common Voice, or private datasets be added
    without changing the scorer; dataset-specific fields are supplied by CLI.
    """
    import soundfile as sf
    from datasets import load_dataset

    print(f"[data] Downloading {n_samples} samples of {dataset_name}/{dataset_config or ''} ({split}) ...")

    try:
        kwargs = {"split": split, "streaming": True, "trust_remote_code": True}
        ds = load_dataset(dataset_name, dataset_config, **kwargs) if dataset_config else load_dataset(dataset_name, **kwargs)
    except TypeError:
        kwargs.pop("trust_remote_code", None)
        ds = load_dataset(dataset_name, dataset_config, **kwargs) if dataset_config else load_dataset(dataset_name, **kwargs)
    except Exception as e:
        raise RuntimeError(
            f"Could not load dataset '{dataset_name}' config '{dataset_config}' split '{split}': {e}\n"
            "Check its Hugging Face dataset card, field names, access terms, and `datasets` version."
        ) from e

    out_dir = _lang_dir(language_key, root)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows, skipped = [], 0

    for i, row in enumerate(ds):
        if len(rows) >= n_samples:
            break
        try:
            audio_value = row[audio_field]
            audio, sr = to_mono_16k(audio_value["array"], audio_value["sampling_rate"])
            text_value = row.get(text_field)
            if not text_value and fallback_text_field:
                text_value = row.get(fallback_text_field)
            text = (text_value or "").strip()
            if audio.size == 0 or not text:
                skipped += 1
                continue
            raw_id = str(row.get("id", row.get("path", i)))
            safe_id = re.sub(r"[^A-Za-z0-9_.-]+", "_", raw_id).strip("._") or str(i)
            sample_id = f"{language_key}_{safe_id}"
            wav_name = f"{sample_id}.wav"
            sf.write(out_dir / wav_name, audio, sr, subtype="PCM_16")
            rows.append({"sample_id": sample_id, "wav": wav_name, "text": text,
                         "duration_s": round(len(audio) / sr, 3)})
        except Exception as e:  # one corrupt row must not kill the export
            skipped += 1
            print(f"[data] skipped row {i} ({language_key}): {e}")

    with open(out_dir / "manifest.csv", "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=MANIFEST_FIELDS)
        w.writeheader()
        w.writerows(rows)
    print(f"[data] Cached {len(rows)} samples for {language_key} (skipped {skipped}) in {out_dir}")


def load_samples(language_key: str, n_samples: int, root: str = DATA_CACHE_DIR,
                 allow_download: bool = True, dataset_name: str = DATASET_NAME,
                 dataset_config: str | None = None, dataset_split: str = DATASET_SPLIT,
                 audio_field: str = "audio", text_field: str = "raw_transcription",
                 fallback_text_field: str | None = "transcription") -> list[EvalSample]:
    """Load up to n_samples for a language from the local cache (downloading if needed)."""
    import soundfile as sf

    manifest = _read_manifest(language_key, root)
    if len(manifest) < n_samples and allow_download:
        if dataset_name == DATASET_NAME and dataset_config is None:
            dataset_config = LANGUAGES[language_key]["fleurs_config"]
        export_huggingface_dataset(language_key, n_samples, root, dataset_name, dataset_config,
                                   dataset_split, audio_field, text_field, fallback_text_field)
        manifest = _read_manifest(language_key, root)
    if not manifest:
        raise FileNotFoundError(
            f"No samples for '{language_key}' in {root}/{language_key}/manifest.csv "
            "(and downloading is disabled or failed)."
        )

    samples = []
    for row in manifest[:n_samples]:
        audio, sr = sf.read(_lang_dir(language_key, root) / row["wav"], dtype="float32")
        audio, sr = to_mono_16k(audio, sr)
        samples.append(EvalSample(language_key, row["sample_id"], audio, sr, row["text"]))
    print(f"[data] {language_key}: {len(samples)} samples, "
          f"{sum(s.audio_seconds for s in samples) / 60:.1f} min of audio")
    return samples
