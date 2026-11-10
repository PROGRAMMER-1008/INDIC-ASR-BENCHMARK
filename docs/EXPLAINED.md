# Indic ASR Benchmark, explained from scratch

Read this once, then you can defend every line of the project.

## 1. The problem in one paragraph
ASR (automatic speech recognition) turns audio into text. Many models exist
(Whisper, IndicConformer, ...), and vendors quote numbers on English. For
Indian languages the picture is different, so before a team picks a model it
needs a fair, repeatable comparison on **that** language. This project is the
measuring stick: same audio, same scoring rules, every model.

## 2. The pipeline (follow one sentence through it)
1. **Data** (`data_loader.py`): a Hindi clip + its correct text from Google
   FLEURS (read speech, many languages, held-out *test* split).
2. **Noise** (`audio.py`, optional): add random noise at a chosen strength.
3. **Model** (`models/*`): audio in, text out.
4. **Normalize** (`normalize.py`): clean both texts the same way.
5. **Score** (`metrics.py`): count errors, compute WER/CER, flag failures.
6. **Aggregate + statistics**: corpus WER, confidence intervals, comparisons.
7. **Report** (`generate_report.py`): tables and charts.

## 3. Metrics
**WER = (S + D + I) / N.** S = words replaced, D = words missing, I = extra
words, N = words in the correct text. Example: correct "a b c d", model says
"a x c": one substitution (b->x), one deletion (d) -> WER 2/4 = 50%.
WER can exceed 100% when the model adds many extra words.

**CER** is the same on characters. In Hindi/Tamil/Bengali one word can be long
and carry grammar, so one wrong letter makes a whole word wrong; CER is more
forgiving and more informative.

**Corpus vs per-sentence WER.** Add all errors, divide by all words
(corpus). Averaging each sentence's WER lets a 3-word sentence weigh as much
as a 40-word sentence. Test `test_corpus_wer_differs_from_mean_of_sentence_wer`
shows 2% vs 50% on the same data.

**RTFx** = audio seconds / compute seconds. RTFx 5 means 1 minute of audio is
transcribed in 12 seconds. Higher is faster. We run one untimed warm-up call
first because the first call pays one-off setup costs.

## 4. Why normalization matters
A model that writes "नमस्ते, आप कैसे हैं।" and one that writes
"नमस्ते आप कैसे हैं" said the same thing, but raw WER would punish the
first for punctuation. We apply **one** function to reference and output:
Unicode NFC (same letter, one encoding), remove punctuation including the
danda "।", remove invisible joiners, convert native digits (४) to ASCII (4),
lowercase, collapse spaces. Honest limitation: we do not convert spoken
numbers ("दो सौ") to digits, so some errors are formatting, not hearing.

## 5. Statistics: why confidence intervals
With 50 sentences, WER = 31% could easily be 27% or 35% on another sample.
**Bootstrap:** resample the sentences with replacement 1000 times, recompute
WER each time, and take the middle 95% -> an interval. The **paired
bootstrap** does this for the *difference* between two models on the same
sentences: if the interval excludes 0, the gap is unlikely to be luck.
This is what lets you say "model A is better by X points" and mean it.

## 6. The models, in plain words
* **Whisper** (OpenAI): encoder-decoder transformer trained on 680k hours of
  web audio. The decoder is a language model, so it writes fluent text, but it
  can **hallucinate** (invent plausible text) on noise or silence. Sizes:
  tiny (39M) ... large (1.5B). Weak on low-resource Indian languages at small sizes.
* **faster-whisper:** the same Whisper weights run by a faster engine
  (CTranslate2, int8 quantization). Tests whether the engine changes accuracy.
* **IndicConformer** (AI4Bharat): Conformer encoder (convolutions for local
  sound patterns + self-attention for context) trained on Indian languages,
  decoded with **CTC**: it predicts a token for each audio frame and merges
  repeats. No free-running language-model decoder -> fast and rarely
  hallucinates, but fewer "smart guesses".

## 7. Failure analysis (the part interviewers like)
* **Hallucination:** utterance WER > 100%.
* **Wrong script:** asked for Hindi, the model answers in Urdu/Latin letters.
  Detected via Unicode character names.
* **Empty output**, **utterances with digits** (formatting vs recognition),
  and the S/D/I split (mostly substitutions = wrong words; mostly deletions =
  model drops speech; insertions = invented speech).

## 8. Engineering choices you can defend
* **Resumable JSONL log:** long runs survive crashes; adding a model only runs new work.
* **Model wrappers + registry:** adding a model touches one file.
* **Deterministic noise** (seeded per sample): every model hears identical noise.
* **Failures are visible and retryable:** failed inference attempts are logged,
  counted in coverage, and retried on resume; they are not converted to fabricated text.
* **Comparable macro scores:** each leaderboard comparison uses the same set of
  languages across models; the report displays that shared cohort.
* **Run metadata:** hardware, versions, model IDs, scoring protocol, and SHA-256
  hashes of local manifests are saved with the results.
* **Extensible data/noise inputs:** compatible Hugging Face transcript/audio fields,
  local manifests, and user-supplied licensed background WAVs use the same scorer.
* **37 tests**, including resumability, coverage, noise SNR, and aggregation checks.

## 9. Interview Q&A
**Why corpus WER?** Weights by words, so long sentences count proportionally.
**Why CER too?** Indic words are long/agglutinative; WER over-penalises tiny errors.
**Why force Whisper's language?** Auto-detection errors would pollute WER with a non-recognition failure.
**What does the confidence interval tell you?** How much the number could move with a different sample; non-overlap/paired-significant gaps are trustworthy.
**What is RTFx and why warm up?** Speed relative to real time; first call includes loading/compilation costs.
**Why might a smaller specialised model beat a bigger general one?** Training data match: IndicConformer saw Indian-language speech; Whisper-small saw little.
**What are the limitations?** Read speech only, synthetic white noise, no spoken-number normalization, hardware-dependent speed.
**What would you do next?** Add validated adapters for conversational datasets (e.g. Vistaar/Kathbath), curate a licensed background-noise pack, evaluate code-mixed Hinglish, and compare a separately documented spoken-number normalization track.
**How do you know your scoring is right?** Unit tests on hand-computed examples (S/D/I counts, empty hypothesis, hallucination, digit and Unicode normalization).

## 10. Your learning checklist (do these before interviews)
- [ ] Compute WER by hand for three sentence pairs and check with `score_utterance`.
- [ ] Run `quick`, open `report.html`, and explain every chart.
- [ ] Look at the 5 worst failures per model and write one sentence on each pattern.
- [ ] Explain why bootstrap intervals shrink with more samples (run `--samples 10` vs `50`).
- [ ] Explain CTC vs encoder-decoder in 30 seconds without notes.
