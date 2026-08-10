"""
Central configuration for the Indic ASR Benchmark.

Everything that a reviewer might want to tweak (which languages, how many
samples, which model checkpoints) lives here so the rest of the codebase
doesn't need to be touched to run a different-sized experiment.
"""

# ---------------------------------------------------------------------------
# Device
# ---------------------------------------------------------------------------
# Single source of truth for device selection. Every model wrapper imports
# DEVICE/TORCH_DTYPE from here instead of re-checking torch.cuda.is_available()
# independently, so the whole pipeline behaves consistently whether you're
# on a CPU-only machine or a free-tier Colab GPU.
#
# torch is imported inside a try/except here (not at top-of-file) so that
# modules which only need path/language config -- e.g. generate_report.py,
# which never touches a model -- don't require torch to be installed just to
# import this config file. Model wrapper files need torch regardless and
# import it directly themselves.
try:
    import torch
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    # Whisper via transformers pipeline wants torch.float16 on GPU for speed,
    # but float16 on CPU is either unsupported or slower than float32 on many
    # ops -- so we branch dtype on device too.
    TORCH_DTYPE = torch.float16 if DEVICE == "cuda" else torch.float32
except ImportError:
    DEVICE = "cpu"
    TORCH_DTYPE = None


# ---------------------------------------------------------------------------
# Languages
# ---------------------------------------------------------------------------
# FLEURS language codes: https://huggingface.co/datasets/google/fleurs
# Each entry maps our internal short code -> (FLEURS config name, human name,
# AI4Bharat indic-conformer language code). AI4Bharat uses ISO 639-1-ish
# short codes (e.g. "hi", "ta") which differ from FLEURS' "hi_in"/"ta_in".
LANGUAGES = {
    "hindi": {
        "fleurs_config": "hi_in",
        "display_name": "Hindi",
        "indic_conformer_code": "hi",
    },
    "tamil": {
        "fleurs_config": "ta_in",
        "display_name": "Tamil",
        "indic_conformer_code": "ta",
    },
}

# ---------------------------------------------------------------------------
# Sample size
# ---------------------------------------------------------------------------
# Start small. On CPU, whisper-tiny + indic-conformer-600m on 20 samples/lang
# (40 total per model, 80 total inferences across 2 models) is a reasonable
# few-minutes-to-~30-minutes run depending on hardware. Raise this once you've
# confirmed the pipeline runs cleanly end-to-end on a handful of samples.
SAMPLES_PER_LANGUAGE = 20

# FLEURS "test" split is used per the assignment -- it's held out and not
# used for training any of the models we're benchmarking (to the best of
# public documentation), which is the right split for an eval-only benchmark.
DATASET_NAME = "google/fleurs"
DATASET_SPLIT = "test"

# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------
WHISPER_CHECKPOINT = "openai/whisper-tiny"
# NOTE: the original spec asked for whisper-small/base. On CPU, tiny is the
# honest choice for a bounded run -- see README "Known Limitations" for the
# full tradeoff discussion and how to switch to small/base if you have GPU
# access or more time.

INDIC_CONFORMER_CHECKPOINT = "ai4bharat/indic-conformer-600m-multilingual"
INDIC_CONFORMER_DECODING = "ctc"  # alternative: "rnnt"

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
RESULTS_DIR = "results"
RESULTS_CSV = f"{RESULTS_DIR}/results.csv"
RESULTS_JSON = f"{RESULTS_DIR}/results.json"
SUMMARY_CSV = f"{RESULTS_DIR}/summary.csv"
REPORT_HTML = "report/report.html"

# How many worst-failure examples to surface per model in the report.
N_WORST_FAILURES = 5
