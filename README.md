# Indic ASR Benchmark

## ⚠️ Status: code complete, real numbers NOT yet in this README — read this first

This project was built in a sandboxed environment with **no network access to
huggingface.co** (only PyPI/npm/GitHub package registries were reachable).
That means I could not download `google/fleurs`, `openai/whisper-tiny`, or
`ai4bharat/indic-conformer-600m-multilingual`, and therefore could not
actually execute the full pipeline end-to-end.

**What I actually did test in this environment** (see each file's docstring
for specifics):
- `src/normalize.py` — ran directly, verified output on Hindi/Tamil/English samples.
- `src/scoring.py` — ran directly against real `jiwer==4.0.0`, verified WER/CER
  math including the empty-reference edge case.
- `src/generate_report.py` — ran end-to-end against **synthetic** data shaped
  like real output, verified it produces valid HTML with two real embedded
  matplotlib charts and two populated tables. That synthetic-data test output
  lives in `dev_test_fixtures/` — **it is not real benchmark data, do not
  quote numbers from it.**

**What is written but unverified**: `data_loader.py`, `whisper_wrapper.py`,
`indic_conformer_wrapper.py`, and the full `run_benchmark.py` orchestration.
These are built against well-documented, standard APIs (HF `datasets` load
pattern, `transformers.pipeline` for ASR, HF model-card-documented usage for
indic-conformer) but the very first time you run them **is** the real test.

**The results table below is a placeholder.** Run the benchmark yourself
(instructions below), then replace the placeholder table with your actual
`results/summary.csv` output before this goes in your portfolio. Do not
present the placeholder numbers as real — that would defeat the entire point
of a benchmarking project.

---

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

### First-run checklist (things likely to need attention)

Because `data_loader.py` and the model wrappers were never executed against
the real network in development, here's what to check on your first run,
roughly in order of likelihood:

1. **`indic-conformer` API mismatch**: this model uses `trust_remote_code=True`,
   meaning its exact calling convention lives in the model repo and can
   change between revisions. If `model.transcribe()` / `model(wav, lang, mode)`
   throws `TypeError` or `AttributeError`, check the current "How to use"
   snippet at https://huggingface.co/ai4bharat/indic-conformer-600m-multilingual
   and adjust the one call inside `IndicConformerWrapper.transcribe()`
   (`src/models/indic_conformer_wrapper.py`) accordingly. This should be a
   small, localized fix, not a rewrite.
2. **FLEURS language config names**: double check `hi_in` / `ta_in` are
   still the current config names for the FLEURS dataset on the Hub — these
   are stable and well-established, but dataset configs occasionally get
   renamed.
3. **CPU runtime**: see "Known Limitations" below for realistic timing
   expectations, especially for indic-conformer-600m.
4. **Whisper language forcing**: Whisper's `generate_kwargs={"language": ...}`
   expects the English *name* of the language (e.g. `"hindi"`), not an ISO
   code — this is handled in `WHISPER_LANGUAGE_NAMES` in
   `whisper_wrapper.py`, but worth knowing if you add a language.

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

## Results

**PLACEHOLDER — replace this section with your actual `results/summary.csv`
output after running the benchmark yourself.**

| Model | Language | Mean WER | Mean CER | Mean Latency | N Samples |
|---|---|---|---|---|---|
| _(run the benchmark)_ | | | | | |

Worst failure cases: see `results/worst_failures.csv` after running, and the
"Worst Failures" table in `report/report.html`.

## Known limitations

- **Model substitution from original spec**: whisper-tiny was used instead
  of whisper-small/base, specifically because this project was built in a
  CPU-only, no-persistent-session sandbox where a small/base model across
  even 40 samples risked an unacceptably long run. If you have GPU access,
  switching is a one-line change: set `WHISPER_CHECKPOINT = "openai/whisper-small"`
  in `config.py`. Expect meaningfully better WER from small/base — tiny is
  the fastest/least accurate point in Whisper's size range, so don't treat
  a tiny-based WER number as representative of "Whisper's" ceiling.
- **Small sample size**: `SAMPLES_PER_LANGUAGE = 20` by default (configurable
  in `config.py`). This is enough to see directional patterns and find real
  failure cases, but not enough for tight statistical confidence — a model
  scoring 2 points of WER better than another on 20 samples could plausibly
  reverse on a larger set. Increase sample count for anything beyond a
  portfolio demo.
- **indic-conformer-600m is a much larger model than whisper-tiny** (600M vs
  ~39M parameters). Comparing their latency numbers directly is comparing
  different weight classes, not just "model A vs model B" — call this out
  explicitly if presenting latency results, since a size-matched comparison
  (e.g. indic-conformer vs whisper-small/medium) would be fairer for latency
  specifically, even though it's not what accuracy-focused benchmarking
  usually optimizes for.
- **CPU vs GPU timing**: any latency numbers in this report are only
  meaningful in the context of what hardware produced them — always report
  device (CPU model or GPU type) alongside latency numbers, and don't
  compare CPU-measured latency to another paper's GPU-measured latency.
- **Normalization affects WER comparability**: see `normalize.py` docstring
  for the full rationale. In short — normalized WER measures "did the model
  get the words right," not "did the model format the transcript the way
  the reference does." A model optimized for verbatim, punctuated output
  (useful for subtitles) could show similar normalized WER to a model with
  much less usable raw output. If your use case cares about punctuation/
  casing, add an unnormalized scoring pass — the `score_pair()` function in
  `scoring.py` is a natural place to add a second scoring mode.
- **FLEURS test-split assumption**: this benchmark assumes FLEURS `test` was
  not used in training any of the benchmarked models. This is standard
  practice and documented for Whisper, but is not independently re-verified
  here for every model version — worth a footnote if this benchmark result
  is used in anything more formal than a portfolio piece.

## Design decisions worth being able to explain in an interview

- **Why a common `transcribe()` interface**: keeps the benchmark loop,
  scoring, and reporting code completely decoupled from any individual
  model's API quirks. Adding a model never means touching the aggregation
  or reporting layer.
- **Why normalize once, centrally**: prevents each model wrapper from
  quietly applying its own normalization (which would make WER numbers
  incomparable across models without anyone noticing) — normalization
  policy lives in exactly one auditable function.
- **Why skip-and-log instead of fail-fast on bad samples**: a single corrupt
  audio file or decoding error shouldn't invalidate an otherwise-successful
  40+ sample run. Every skip is printed to stdout so you can see exactly
  what was dropped and why, in `results.json`'s absence for that sample_id.
- **Why HTML report over Streamlit**: zero runtime dependency to view results
  — a hiring manager can open a single file in any browser, no `streamlit run`
  step, no server, no port to forward from Colab.
