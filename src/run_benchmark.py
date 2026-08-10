"""
Main entry point: runs every registered model against every configured
language's samples, scores each, and writes raw results to CSV + JSON.

NOT EXECUTED END-TO-END IN THIS SESSION because this sandbox cannot reach
huggingface.co (dataset + model downloads are blocked at the network level
here). Every piece that doesn't need that network access (normalize.py,
scoring.py) WAS actually run and verified -- see those files' docstrings.
This file wires those tested pieces together with the untested-here dataset
loading and model inference calls.

Run this yourself with:  python3 -m src.run_benchmark
"""

import json
import time
import traceback

import pandas as pd

from src.config import (
    LANGUAGES,
    N_WORST_FAILURES,
    RESULTS_CSV,
    RESULTS_JSON,
    SUMMARY_CSV,
)
from src.data_loader import load_language_samples
from src.models.indic_conformer_wrapper import IndicConformerWrapper
from src.models.whisper_wrapper import WhisperWrapper
from src.scoring import score_pair

# ---------------------------------------------------------------------------
# Model registry -- THIS is the line you add to when adding model #3.
# Each entry: display key -> zero-arg callable that constructs the wrapper.
# Kept as callables (not instances) so a model that fails to load doesn't
# prevent the others from being attempted.
# ---------------------------------------------------------------------------
MODEL_REGISTRY = {
    "whisper-tiny": WhisperWrapper,
    "indic-conformer-600m": IndicConformerWrapper,
}


def run_benchmark() -> list[dict]:
    results = []

    # Load all language samples once up front, so a slow/failed model load
    # doesn't cost us a repeated dataset download.
    samples_by_language = {}
    for language_key in LANGUAGES:
        try:
            samples_by_language[language_key] = load_language_samples(language_key)
        except Exception as e:
            print(f"[run_benchmark] FAILED to load samples for {language_key}, skipping this language entirely: {e}")
            samples_by_language[language_key] = []

    for model_key, model_ctor in MODEL_REGISTRY.items():
        print(f"\n[run_benchmark] Loading model: {model_key} ...")
        try:
            model = model_ctor()
        except Exception as e:
            print(f"[run_benchmark] FAILED to load model '{model_key}': {e}")
            print(traceback.format_exc())
            print(f"[run_benchmark] Skipping '{model_key}' entirely and continuing with remaining models.")
            continue

        for language_key, samples in samples_by_language.items():
            lang_conf = LANGUAGES[language_key]
            # Each wrapper expects a different language-code convention;
            # config.LANGUAGES stores the right code per model type. Whisper
            # wrapper internally maps our language_key -> its own name, so
            # we pass language_key straight through for Whisper. For
            # indic-conformer we must pass its short code explicitly.
            if model_key == "indic-conformer-600m":
                model_language_code = lang_conf["indic_conformer_code"]
            else:
                model_language_code = language_key

            print(f"[run_benchmark] Running {model_key} on {language_key} ({len(samples)} samples) ...")

            for sample in samples:
                try:
                    start = time.time()
                    hypothesis = model.transcribe(
                        sample.audio_array, sample.sample_rate, model_language_code
                    )
                    latency = time.time() - start
                except Exception as e:
                    print(f"[run_benchmark] Inference FAILED for {model_key}/{sample.sample_id}: {e}")
                    continue  # skip this sample, don't crash the run

                score = score_pair(sample.reference_text, hypothesis)

                results.append(
                    {
                        "model": model_key,
                        "language": language_key,
                        "sample_id": sample.sample_id,
                        "reference": sample.reference_text,
                        "hypothesis": hypothesis,
                        "reference_normalized": score["reference_normalized"],
                        "hypothesis_normalized": score["hypothesis_normalized"],
                        "wer": score["wer"],
                        "cer": score["cer"],
                        "latency_seconds": latency,
                        "scoring_error": score["scoring_error"],
                    }
                )

    return results


def save_results(results: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame(results)
    df.to_csv(RESULTS_CSV, index=False)
    with open(RESULTS_JSON, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"[run_benchmark] Saved {len(df)} raw results to {RESULTS_CSV} and {RESULTS_JSON}")
    return df


def build_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Model x language -> mean WER, mean CER, mean latency, sample count."""
    scoreable = df[df["wer"].notna()]
    summary = (
        scoreable.groupby(["model", "language"])
        .agg(
            mean_wer=("wer", "mean"),
            mean_cer=("cer", "mean"),
            mean_latency_seconds=("latency_seconds", "mean"),
            n_samples=("sample_id", "count"),
        )
        .reset_index()
    )
    summary.to_csv(SUMMARY_CSV, index=False)
    print(f"[run_benchmark] Saved summary to {SUMMARY_CSV}")
    return summary


def worst_failures(df: pd.DataFrame, n: int = N_WORST_FAILURES) -> pd.DataFrame:
    """Top-n highest-WER samples per model, for the 'where does it break' view."""
    scoreable = df[df["wer"].notna()]
    return (
        scoreable.sort_values("wer", ascending=False)
        .groupby("model")
        .head(n)
        .reset_index(drop=True)
    )


if __name__ == "__main__":
    raw_results = run_benchmark()
    if not raw_results:
        print("[run_benchmark] No results produced -- every model or dataset load failed. "
              "Check the errors printed above.")
    else:
        df = save_results(raw_results)
        summary_df = build_summary(df)
        print("\n=== SUMMARY ===")
        print(summary_df.to_string(index=False))

        failures_df = worst_failures(df)
        failures_df.to_csv("results/worst_failures.csv", index=False)
        print(f"\n[run_benchmark] Saved worst failures to results/worst_failures.csv")
