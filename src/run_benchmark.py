"""
Main entry point.

    python -m src.run_benchmark --profile quick
    python -m src.run_benchmark --profile standard
    python -m src.run_benchmark --models whisper-small,indic-conformer-600m \
                                --languages hindi,tamil --samples 30 --conditions clean,snr10

Design points
-------------
* RESUMABLE: every scored utterance is appended to results/raw_results.jsonl
  immediately. If the run is interrupted (Ctrl-C, sleep, crash) just run the
  same command again; finished (model, language, condition, sample) combos
  are skipped.
* FAIR TIMING: each model is warmed up on one sample before timing starts,
  so the one-off cost of the first call is not blamed on the model.
* ISOLATED FAILURES: a model that fails to load, or a clip that fails to
  transcribe, is logged and skipped; it never kills the whole run.
* REPRODUCIBLE: run_metadata.json records hardware, library versions, and
  exact settings.
"""

from __future__ import annotations

import argparse
import hashlib
import gc
import json
import time
import traceback
from pathlib import Path

import pandas as pd

from src import config
from src.audio import add_background_noise, add_noise
from src.config import BACKGROUND_CONDITIONS, CONDITIONS, LANGUAGES, MODELS, PROFILES
from src.data_loader import EvalSample, load_samples
from src.metrics import (build_leaderboard, build_summary, score_utterance)
from src.models.registry import build_model


# ---------------------------------------------------------------------------
# Resume support
# ---------------------------------------------------------------------------
def _key(model, language, condition, sample_id) -> tuple:
    return (model, language, condition, sample_id)


def _condition_audio(audio, condition: str, sample_id: str, noise_dir: str | None):
    if condition in BACKGROUND_CONDITIONS:
        if not noise_dir:
            raise ValueError(f"Condition '{condition}' requires --noise-dir with licensed .wav noise files")
        return add_background_noise(audio, BACKGROUND_CONDITIONS[condition], sample_id, noise_dir)
    snr = CONDITIONS[condition]
    return add_noise(audio, snr, key=sample_id) if snr is not None else audio


def load_existing(path: str = config.RAW_RESULTS_JSONL) -> list[dict]:
    p = Path(path)
    if not p.exists():
        return []
    rows = []
    with open(p, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    pass  # half-written last line from an interrupted run
    return rows


def append_row(row: dict, path: str = config.RAW_RESULTS_JSONL) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


# ---------------------------------------------------------------------------
# Core loop
# ---------------------------------------------------------------------------
def _rss_mb() -> float:
    import psutil
    return psutil.Process().memory_info().rss / 1e6


def _gpu_peak_mb() -> float | None:
    try:
        import torch
        if torch.cuda.is_available():
            return torch.cuda.max_memory_allocated() / 1e6
    except ImportError:
        pass
    return None


def run(models: list[str], languages: list[str], n_samples: int, conditions: list[str],
        allow_download: bool = True, data_dir: str = config.DATA_CACHE_DIR,
        noise_dir: str | None = None, dataset_name: str = config.DATASET_NAME,
        dataset_config: str | None = None, dataset_split: str = config.DATASET_SPLIT,
        audio_field: str = "audio", text_field: str = "raw_transcription",
        fallback_text_field: str | None = "transcription") -> dict:
    """Run the benchmark; returns per-model runtime stats (load time, memory)."""
    if any(c in BACKGROUND_CONDITIONS for c in conditions):
        if not noise_dir or not Path(noise_dir).is_dir() or not list(Path(noise_dir).glob("*.wav")):
            raise ValueError("Background-noise conditions require --noise-dir containing at least one .wav file")
    # Failed attempts remain visible in the log but are retryable on resume.
    done = {_key(r["model"], r["language"], r["condition"], r["sample_id"])
            for r in load_existing()
            if r.get("status") != "inference_failed" and r.get("wer") is not None}
    print(f"[run] {len(done)} results already on disk (will be skipped)")

    samples_by_lang: dict[str, list[EvalSample]] = {}
    for lang in languages:
        try:
            samples_by_lang[lang] = load_samples(
                lang, n_samples, data_dir, allow_download, dataset_name, dataset_config,
                dataset_split, audio_field, text_field, fallback_text_field,
            )
            if len(samples_by_lang[lang]) < n_samples:
                print(f"[run] WARNING: {lang} requested {n_samples} samples; "
                      f"cache provides {len(samples_by_lang[lang])}.")
        except Exception as e:
            print(f"[run] Could not load data for {lang}: {e}  -> skipping this language")

    runtime_stats: dict[str, dict] = {}

    for model_name in models:
        todo = [(lang, cond, s) for lang, ss in samples_by_lang.items()
                for cond in conditions for s in ss
                if _key(model_name, lang, cond, s.sample_id) not in done]
        if not todo:
            print(f"[run] {model_name}: nothing left to do")
            continue

        print(f"\n[run] === {model_name}: loading ({len(todo)} utterances to transcribe) ===")
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()
        except ImportError:
            pass
        rss_before, t0 = _rss_mb(), time.perf_counter()
        try:
            model = build_model(model_name)
        except Exception as e:
            print(f"[run] FAILED to load {model_name}: {e}\n{traceback.format_exc()}")
            print(f"[run] Skipping {model_name}; continuing with the others.")
            continue
        load_s = time.perf_counter() - t0
        runtime_stats[model_name] = {
            "load_seconds": round(load_s, 2),
            "ram_delta_mb_approx": round(_rss_mb() - rss_before, 1),
            "params_m": MODELS[model_name].get("params_m"),
        }

        # Warm-up: one untimed call so first-call overhead is not counted.
        try:
            first = todo[0][2]
            warm_audio = _condition_audio(first.audio_array, todo[0][1], first.sample_id, noise_dir)
            model.transcribe(warm_audio, first.sample_rate, first.language)
        except Exception as e:
            print(f"[run] warm-up failed for {model_name}: {e}")

        for i, (lang, cond, s) in enumerate(todo, 1):
            audio = s.audio_array
            try:
                audio = _condition_audio(audio, cond, s.sample_id, noise_dir)
                t = time.perf_counter()
                hyp = model.transcribe(audio, s.sample_rate, lang)
                latency = time.perf_counter() - t
            except Exception as e:
                print(f"[run] inference FAILED {model_name}/{s.sample_id}/{cond}: {e}")
                append_row({
                    "model": model_name, "language": lang, "condition": cond,
                    "sample_id": s.sample_id, "reference": s.reference_text,
                    "audio_seconds": round(s.audio_seconds, 3),
                    "latency_seconds": round(time.perf_counter() - t, 4),
                    "status": "inference_failed", "inference_error": f"{type(e).__name__}: {e}",
                })
                continue

            row = {
                "model": model_name, "language": lang, "condition": cond,
                "sample_id": s.sample_id, "reference": s.reference_text,
                "hypothesis": hyp, "audio_seconds": round(s.audio_seconds, 3),
                "latency_seconds": round(latency, 4),
                "status": "scored",
                **score_utterance(s.reference_text, hyp, LANGUAGES[lang]["script"]),
            }
            append_row(row)
            if i % 10 == 0 or i == len(todo):
                w = row.get("wer")
                print(f"[run] {model_name} {i}/{len(todo)}  last: {lang}/{cond} "
                      f"WER={w if w is None else round(w, 3)}  {latency:.2f}s")

        runtime_stats[model_name]["gpu_peak_mb"] = _gpu_peak_mb()
        try:
            model.close()
        except Exception:
            pass
        del model
        gc.collect()
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass

    return runtime_stats


def write_metadata(args_dict: dict, runtime_stats: dict, data_dir: str = config.DATA_CACHE_DIR) -> None:
    import platform
    import sys
    import psutil

    versions = {}
    for pkg in ["torch", "transformers", "datasets", "faster_whisper", "jiwer", "numpy", "pandas"]:
        try:
            versions[pkg] = __import__(pkg).__version__
        except Exception:
            versions[pkg] = None
    gpu = None
    try:
        import torch
        if torch.cuda.is_available():
            gpu = torch.cuda.get_device_name(0)
    except ImportError:
        pass

    # merge runtime stats with those from earlier (resumed) runs
    meta_path = Path(config.RUN_METADATA_JSON)
    old = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
    merged_runtime = {**old.get("model_runtime", {}), **runtime_stats}

    meta = {
        "finished_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "settings": args_dict,
        "hardware": {
            "device": config.get_device(),
            "gpu": gpu,
            "cpu": platform.processor() or platform.machine(),
            "cpu_threads": psutil.cpu_count(logical=True),
            "ram_gb": round(psutil.virtual_memory().total / 1e9, 1),
            "os": platform.platform(),
            "python": sys.version.split()[0],
        },
        "library_versions": versions,
        "model_runtime": merged_runtime,
        "model_checkpoints": {name: MODELS[name]["checkpoint"] for name in args_dict.get("models", [])},
        "data_manifests": _manifest_fingerprints(data_dir, args_dict.get("languages", [])),
        "scoring_protocol": {
            "normalization_version": "indic-asr-normalization-v1",
            "bootstrap_resamples": config.BOOTSTRAP_RESAMPLES,
            "random_seed": config.RANDOM_SEED,
        },
    }
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")


def _manifest_fingerprints(data_dir: str, languages: list[str]) -> dict[str, dict]:
    """Record the exact local manifest used for each language, when present."""
    out = {}
    for language in languages:
        manifest = Path(data_dir) / language / "manifest.csv"
        if manifest.is_file():
            payload = manifest.read_bytes()
            out[language] = {
                "path": str(manifest.as_posix()),
                "sha256": hashlib.sha256(payload).hexdigest(),
                "bytes": len(payload),
            }
    return out


def finalize() -> pd.DataFrame:
    """Rebuild summary/leaderboard CSVs from everything in raw_results.jsonl."""
    rows = load_existing()
    if not rows:
        print("[run] No results on disk.")
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    summary = build_summary(df)
    summary.to_csv(config.SUMMARY_CSV, index=False)
    board = build_leaderboard(summary)
    board.to_csv(config.LEADERBOARD_CSV, index=False)
    print("\n=== LEADERBOARD (macro average over languages) ===")
    show = board.copy()
    for c in ["macro_wer", "macro_cer", "hallucination_rate", "wrong_script_rate"]:
        show[c] = (show[c] * 100).round(1)
    show["rtfx"] = show["rtfx"].round(2)
    print(show.to_string(index=False))
    return df


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Indic ASR benchmark")
    p.add_argument("--profile", choices=list(PROFILES), default=None,
                   help="preset run size (default: quick, unless other flags are given)")
    p.add_argument("--models", help=f"comma list. choices: {', '.join(MODELS)}")
    p.add_argument("--languages", help=f"comma list. choices: {', '.join(LANGUAGES)}")
    p.add_argument("--samples", type=int, help="utterances per language")
    p.add_argument("--conditions", help=f"comma list. choices: {', '.join([*CONDITIONS, *BACKGROUND_CONDITIONS])}")
    p.add_argument("--data-dir", default=config.DATA_CACHE_DIR)
    p.add_argument("--noise-dir", help="directory of licensed background-noise WAVs for background_snr* conditions")
    p.add_argument("--dataset-name", default=config.DATASET_NAME,
                   help="Hugging Face dataset repository; local manifest cache is always supported")
    p.add_argument("--dataset-config", help="dataset config/subset (defaults to the language's FLEURS config)")
    p.add_argument("--dataset-split", default=config.DATASET_SPLIT)
    p.add_argument("--audio-field", default="audio", help="dataset field yielding array + sampling_rate")
    p.add_argument("--text-field", default="raw_transcription")
    p.add_argument("--fallback-text-field", default="transcription")
    p.add_argument("--no-download", action="store_true",
                   help="use only audio already in --data-dir (e.g. your own manifest)")
    p.add_argument("--list", action="store_true", help="list models/languages/profiles and exit")
    return p.parse_args()


def resolve_settings(a: argparse.Namespace) -> dict:
    base = PROFILES[a.profile or config.DEFAULT_PROFILE]
    s = {
        "models": a.models.split(",") if a.models else base["models"],
        "languages": a.languages.split(",") if a.languages else base["languages"],
        "samples": a.samples or base["samples"],
        "conditions": a.conditions.split(",") if a.conditions else base["conditions"],
        "noise_dir": a.noise_dir,
        "dataset_name": a.dataset_name,
        "dataset_config": a.dataset_config,
        "dataset_split": a.dataset_split,
        "audio_field": a.audio_field,
        "text_field": a.text_field,
        "fallback_text_field": a.fallback_text_field,
    }
    for m in s["models"]:
        if m not in MODELS:
            raise SystemExit(f"Unknown model '{m}'. Use --list.")
    for l in s["languages"]:
        if l not in LANGUAGES:
            raise SystemExit(f"Unknown language '{l}'. Use --list.")
    for c in s["conditions"]:
        if c not in CONDITIONS and c not in BACKGROUND_CONDITIONS:
            raise SystemExit(f"Unknown condition '{c}'. Use --list.")
    if any(c in BACKGROUND_CONDITIONS for c in s["conditions"]) and not s["noise_dir"]:
        raise SystemExit("Background-noise conditions require --noise-dir with licensed WAV recordings.")
    if any(c in BACKGROUND_CONDITIONS for c in s["conditions"]):
        noise_path = Path(s["noise_dir"])
        if not noise_path.is_dir() or not list(noise_path.glob("*.wav")):
            raise SystemExit(f"No .wav background-noise recordings found in '{noise_path}'.")
    return s


def main() -> None:
    a = parse_args()
    if a.list:
        print("models   :", ", ".join(MODELS))
        print("languages:", ", ".join(LANGUAGES))
        print("conditions:", ", ".join([*CONDITIONS, *BACKGROUND_CONDITIONS]))
        for name, p in PROFILES.items():
            print(f"profile {name}: {p}")
        return
    s = resolve_settings(a)
    print(f"[run] device={config.get_device()}  settings={s}")
    stats = run(s["models"], s["languages"], s["samples"], s["conditions"],
                allow_download=not a.no_download, data_dir=a.data_dir,
                noise_dir=s["noise_dir"], dataset_name=s["dataset_name"],
                dataset_config=s["dataset_config"], dataset_split=s["dataset_split"],
                audio_field=s["audio_field"], text_field=s["text_field"],
                fallback_text_field=s["fallback_text_field"])
    write_metadata(s, stats, data_dir=a.data_dir)
    finalize()
    print("\n[run] Done. Now run:  python -m src.generate_report")


if __name__ == "__main__":
    main()
