"""
Central configuration for the Indic ASR Benchmark.

Everything you might want to change for an experiment lives here: which
languages, which models, how many samples, which noise levels. Heavy
libraries (torch etc.) are NOT imported at module level, so the scoring,
reporting and test code run on a bare-bones install.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Languages
# ---------------------------------------------------------------------------
# key -> everything any model or scorer needs to know about the language.
#   fleurs_config : config name in the google/fleurs dataset
#   iso           : ISO 639-1 code (faster-whisper and AI4Bharat both use it)
#   whisper_name  : English name Whisper's `language=` argument accepts
#   script        : Unicode-name prefix of the expected writing system;
#                   used to detect "model answered in the wrong script"
LANGUAGES: dict[str, dict] = {
    "hindi":   {"fleurs_config": "hi_in", "display_name": "Hindi",   "iso": "hi", "whisper_name": "hindi",   "script": "DEVANAGARI"},
    "tamil":   {"fleurs_config": "ta_in", "display_name": "Tamil",   "iso": "ta", "whisper_name": "tamil",   "script": "TAMIL"},
    "bengali": {"fleurs_config": "bn_in", "display_name": "Bengali", "iso": "bn", "whisper_name": "bengali", "script": "BENGALI"},
    "telugu":  {"fleurs_config": "te_in", "display_name": "Telugu",  "iso": "te", "whisper_name": "telugu",  "script": "TELUGU"},
    "marathi": {"fleurs_config": "mr_in", "display_name": "Marathi", "iso": "mr", "whisper_name": "marathi", "script": "DEVANAGARI"},
}

# ---------------------------------------------------------------------------
# Models  (name -> spec; the factory lives in src/models/registry.py)
# ---------------------------------------------------------------------------
MODELS: dict[str, dict] = {
    "whisper-tiny":                 {"family": "whisper",        "checkpoint": "openai/whisper-tiny",           "params_m": 39},
    "whisper-base":                 {"family": "whisper",        "checkpoint": "openai/whisper-base",           "params_m": 74},
    "whisper-small":                {"family": "whisper",        "checkpoint": "openai/whisper-small",          "params_m": 244},
    "whisper-large-v3-turbo":       {"family": "whisper",        "checkpoint": "openai/whisper-large-v3-turbo", "params_m": 809},
    "faster-whisper-small":         {"family": "faster-whisper", "checkpoint": "small",                         "params_m": 244},
    "faster-whisper-large-v3-turbo":{"family": "faster-whisper", "checkpoint": "large-v3-turbo",                "params_m": 809},
    "indic-conformer-600m":         {"family": "indic-conformer","checkpoint": "ai4bharat/indic-conformer-600m-multilingual", "params_m": 600, "decoding": "ctc"},
}

# ---------------------------------------------------------------------------
# Profiles: pick how big a run you want with `--profile`.
# Times are rough guides for a mid-range laptop CPU; a GPU is much faster.
# ---------------------------------------------------------------------------
PROFILES: dict[str, dict] = {
    # Smoke test: confirms the whole pipeline works (minutes).
    "quick": {
        "languages": ["hindi", "tamil"],
        "samples": 10,
        "models": ["whisper-tiny", "faster-whisper-small", "indic-conformer-600m"],
        "conditions": ["clean"],
    },
    # Good default for a first real result (about an hour on CPU).
    "standard": {
        "languages": ["hindi", "tamil", "bengali"],
        "samples": 50,
        "models": ["whisper-small", "faster-whisper-small", "indic-conformer-600m"],
        "conditions": ["clean", "snr10"],
    },
    # The README-quality run: 5 languages, 100 samples, 4 models, noise sweep.
    # Realistically needs a GPU or an overnight CPU run.
    "full": {
        "languages": ["hindi", "tamil", "bengali", "telugu", "marathi"],
        "samples": 100,
        "models": ["whisper-small", "faster-whisper-small", "whisper-large-v3-turbo", "indic-conformer-600m"],
        "conditions": ["clean", "snr20", "snr10", "snr5"],
    },
}
DEFAULT_PROFILE = "quick"

# Noise conditions: name -> SNR in dB (None = untouched audio).
CONDITIONS: dict[str, float | None] = {
    "clean": None,
    "snr20": 20.0,
    "snr10": 10.0,
    "snr5": 5.0,
}
BACKGROUND_CONDITIONS: dict[str, float] = {
    "background_snr20": 20.0,
    "background_snr10": 10.0,
    "background_snr5": 5.0,
}

# ---------------------------------------------------------------------------
# Data, statistics, paths
# ---------------------------------------------------------------------------
DATASET_NAME = "google/fleurs"
DATASET_SPLIT = "test"          # held-out split: right choice for evaluation only
TARGET_SAMPLE_RATE = 16000      # every model here expects 16 kHz mono

BOOTSTRAP_RESAMPLES = 1000      # for 95% confidence intervals
RANDOM_SEED = 1234              # noise generation + bootstrap are reproducible
N_WORST_FAILURES = 5
HALLUCINATION_WER = 1.0         # utterance WER above this => "hallucination"
WRONG_SCRIPT_RATIO = 0.5        # < 50% letters in expected script => wrong script

DATA_CACHE_DIR = "data_cache"   # FLEURS audio is exported here once, then reused
RESULTS_DIR = "results"
RAW_RESULTS_JSONL = f"{RESULTS_DIR}/raw_results.jsonl"
SUMMARY_CSV = f"{RESULTS_DIR}/summary.csv"
LEADERBOARD_CSV = f"{RESULTS_DIR}/leaderboard.csv"
RUN_METADATA_JSON = f"{RESULTS_DIR}/run_metadata.json"
RESULTS_MD = f"{RESULTS_DIR}/RESULTS.md"
REPORT_HTML = "report/report.html"


def get_device() -> str:
    """'cuda' if a CUDA GPU is usable, else 'cpu'. Imported lazily."""
    try:
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"
