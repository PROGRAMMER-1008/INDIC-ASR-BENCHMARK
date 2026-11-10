# Indic ASR Benchmark

## What this does

This project measures how open-source automatic speech recognition (ASR) models
perform on Hindi, Tamil, Bengali, Telugu, and Marathi. It uses the Google FLEURS
test split by default and reports word and character error rates, confidence
intervals, speed, noise robustness, and transcript failure patterns.

The benchmark is designed to make comparisons reproducible: models receive the
same audio, references and hypotheses use one shared normalizer, and every run
records its settings and data-manifest fingerprints. It does not publish
benchmark scores until you run it on your own machine.

## Models

| Model family | Implementation |
|---|---|
| OpenAI Whisper | Hugging Face Transformers |
| faster-whisper | CTranslate2, including CPU int8 inference |
| AI4Bharat IndicConformer | Multilingual Conformer with CTC decoding |

## What it measures

- Corpus-level WER and CER, with utterance-bootstrap 95% confidence intervals.
- Paired model comparisons on common utterances.
- RTFx after a warm-up call.
- White-noise conditions and optional user-supplied background-noise recordings.
- Empty output, wrong-script output, high-error utterances, digit-bearing references, and S/D/I counts.
- Attempted/scored coverage, including visible inference failures that can be retried on resume.

## Setup

Python 3.10–3.12 is recommended. Install the PyTorch build suitable for your
machine first, then install the project requirements.

```powershell
python -m venv venv
venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m src.doctor
```

On Linux or macOS, activate the environment with `source venv/bin/activate`.
The FLEURS loader uses dataset scripts, so the runtime requirement pins
`datasets<4`.

## Run

```bash
# Small smoke test
python -m src.run_benchmark --profile quick

# Build HTML and Markdown reports
python -m src.generate_report

# Choose models, languages, sample count, and noise conditions
python -m src.run_benchmark --models whisper-small,indic-conformer-600m \
    --languages hindi,telugu --samples 30 --conditions clean,snr10
```

Run `python -m src.run_benchmark --list` for available models, languages,
profiles, and conditions. Each completed utterance is appended to
`results/raw_results.jsonl`. Rerunning the same command skips completed work;
failed inferences stay visible and are retried.

## Add a dataset

The default source is FLEURS. A Hugging Face ASR dataset with an audio field
that decodes to `array` and `sampling_rate`, plus a text transcript field, can
use the generic adapter:

```bash
python -m src.run_benchmark --models whisper-small --languages hindi --samples 30 \
    --dataset-name YOUR_ORG/YOUR_DATASET --dataset-config hindi \
    --dataset-split test --audio-field audio --text-field transcript
```

Set `--fallback-text-field ""` if there is no fallback transcript column.
Check dataset terms, speaker overlap, the evaluation split, and transcript
quality before comparing results. You can also provide local WAV files and a
`data_cache/<language>/manifest.csv` with columns
`sample_id,wav,text,duration_s`, then run with `--no-download`.

## Add recorded background noise

Place licensed mono or stereo WAV recordings in a directory and pass it with
`--noise-dir`. Use `background_snr20`, `background_snr10`, or
`background_snr5` in `--conditions`. Noise selection is deterministic by sample
ID, and mixing preserves the target SNR without clipping.

```bash
python -m src.run_benchmark --models whisper-small --languages hindi --samples 30 \
    --conditions clean,background_snr10 --noise-dir noise_wavs --no-download
```

## Outputs

```text
results/raw_results.jsonl   per-attempt transcripts, scores, latency, and failures
results/summary.csv         per model/language/condition metrics and coverage
results/leaderboard.csv     macro scores over the shared language cohort
results/run_metadata.json   hardware, versions, settings, model IDs, manifest hashes
results/RESULTS.md          Markdown report generated from actual run data
report/report.html          self-contained HTML report
```

Use `python -m src.generate_report --demo` only to preview the layout. Demo
values are synthetic and must never be presented as benchmark results.

## Development

```bash
python -m pip install -r requirements-ci.txt
python -m pytest -q
```

The offline test suite covers normalization, audio processing, scoring,
aggregation, dataset-field mapping, report generation, and resumability.
