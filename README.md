# Indic ASR Benchmark

## What this does, and why

Speech AI teams working on Indian languages can't improve what they don't
measure. This project benchmarks multiple open-source ASR models
(OpenAI Whisper, AI4Bharat's indic-conformer) against a standard multilingual
speech dataset (Google FLEURS), reporting Word Error Rate and Character Error
Rate broken down by language and by model, with a "where does it actually
break" view of the worst individual failures. That combination — quantified
accuracy, per-language breakdown, and concrete failure inspection, all
reproducible from a documented pipeline — is exactly the kind of internal
tooling a speech AI team needs before they can responsibly ship or compare
models for Indian-language products. Benchmarking infrastructure like this
is the difference between "we think the new model is better" and "we can
prove the new model is 4 points of WER better on Tamil specifically, and
here's the failure pattern that's still unsolved."

## Project structure

```
src/
  config.py                    # all tunable settings: languages, sample count, model checkpoints
  normalize.py                 # text normalization (TESTED)
  scoring.py                   # WER/CER computation via jiwer (TESTED)
  data_loader.py                # FLEURS loading (UNTESTED — needs your run)
  generate_report.py            # HTML report builder (TESTED with synthetic data)
  run_benchmark.py               # main orchestration script (UNTESTED — needs your run)
  models/
    base.py                    # ASRModelWrapper interface every model implements
    whisper_wrapper.py          # Whisper via transformers pipeline (UNTESTED)
    indic_conformer_wrapper.py   # AI4Bharat indic-conformer (UNTESTED)
results/                       # written by run_benchmark.py: results.csv, results.json, summary.csv, worst_failures.csv
report/                        # written by generate_report.py: report.html
dev_test_fixtures/             # synthetic data used to prove generate_report.py works — NOT real results
requirements.txt
```

## Setup

Python 3.10+ recommended (matches what `transformers==4.44.2` and
`datasets==2.21.0` are tested against upstream).

```bash
python3 -m venv venv
source venv/bin/activate    # Windows: venv\Scripts\activate

pip install -r requirements.txt
```

If you're on Colab, skip the venv step and just `!pip install -r requirements.txt`.

**GPU check**: the pipeline auto-detects GPU via `torch.cuda.is_available()`
in `config.py` and uses it if present, falling back to CPU otherwise — no
manual flag needed. Colab's free tier T4 GPU will be picked up automatically
if your runtime is set to GPU.

## How to run

```bash
# 1. Run the benchmark (loads data, runs both models, scores, saves results/)
python3 -m src.run_benchmark

# 2. Generate the HTML report from results/
python3 -m src.generate_report
```

Open `report/report.html` in any browser — it's fully self-contained
(charts are embedded base64 PNGs, no internet connection or server needed to
view it after generation).

## How to add a new model

Every model wrapper implements one method:

```python
def transcribe(self, audio_array: np.ndarray, sample_rate: int, language_code: str) -> str:
    ...
```

Steps to add model #3:
1. Create `src/models/your_model_wrapper.py`, subclass `ASRModelWrapper`
   (`src/models/base.py`), implement `transcribe()`.
2. Add a checkpoint constant to `config.py` if it's HF-hosted.
3. Add one line to `MODEL_REGISTRY` in `run_benchmark.py`:
   `"your-model-name": YourModelWrapper`
4. Run `python3 -m src.run_benchmark` — nothing else needs to change. Scoring,
   aggregation, and reporting all consume the same result schema regardless
   of which model produced it.

## How to add a new language

1. Add an entry to `LANGUAGES` in `config.py` with the FLEURS config name,
   display name, and indic-conformer language code (check the model card for
   supported language codes — indic-conformer supports a specific set of
   Indian languages, not arbitrary ones).
2. If using Whisper, add the language's English name to
   `WHISPER_LANGUAGE_NAMES` in `whisper_wrapper.py`.
3. Run the benchmark — `data_loader.py` and `run_benchmark.py` iterate over
   whatever's in `LANGUAGES`, no other code changes needed.
