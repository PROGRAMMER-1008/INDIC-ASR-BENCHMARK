"""
Scoring and statistics.

KEY IDEAS (all worth being able to explain)
-------------------------------------------
* WER  = (Substitutions + Deletions + Insertions) / words in the reference.
* CER  = same formula on characters. Better than WER for Indian languages,
         where one "word" can carry lots of grammar and a single wrong
         letter makes the whole word wrong.
* CORPUS-level WER: sum all errors, divide by sum of all reference words.
  The alternative -- averaging each sentence's WER -- lets a 3-word sentence
  count as much as a 40-word one, so it is not the standard.
* A 95% CONFIDENCE INTERVAL via bootstrap: resample the utterances with
  replacement many times, recompute corpus WER each time, take the middle
  95%. With only dozens of utterances this is the honest way to show how
  much a number could move by chance.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.config import (
    BOOTSTRAP_RESAMPLES,
    HALLUCINATION_WER,
    RANDOM_SEED,
    WRONG_SCRIPT_RATIO,
)
from src.normalize import normalize_text, script_ratio


# ---------------------------------------------------------------------------
# Per-utterance scoring
# ---------------------------------------------------------------------------
def score_utterance(reference: str, hypothesis: str, script: str | None = None) -> dict:
    """Score one reference/hypothesis pair. Normalizes both identically first."""
    import jiwer

    ref_n = normalize_text(reference)
    hyp_n = normalize_text(hypothesis)

    base = {
        "reference_normalized": ref_n,
        "hypothesis_normalized": hyp_n,
        "scoring_error": None,
    }
    if not ref_n:
        return {**base, "scoring_error": "empty reference after normalization"}

    w = jiwer.process_words(ref_n, hyp_n)
    c = jiwer.process_characters(ref_n, hyp_n)

    ref_words = w.hits + w.substitutions + w.deletions
    ref_chars = c.hits + c.substitutions + c.deletions
    word_errors = w.substitutions + w.deletions + w.insertions
    char_errors = c.substitutions + c.deletions + c.insertions
    wer = word_errors / ref_words

    ratio = script_ratio(hyp_n, script) if script else None

    return {
        **base,
        "ref_words": ref_words,
        "ref_chars": ref_chars,
        "word_errors": word_errors,
        "char_errors": char_errors,
        "substitutions": w.substitutions,
        "deletions": w.deletions,
        "insertions": w.insertions,
        "wer": wer,
        "cer": char_errors / ref_chars,
        # error categories used in the failure analysis
        "is_empty": hyp_n == "",
        "is_hallucination": wer > HALLUCINATION_WER,
        "script_ratio": ratio,
        "is_wrong_script": (ratio is not None and ratio < WRONG_SCRIPT_RATIO),
        "ref_has_digits": any(ch.isdigit() for ch in ref_n),
    }


# ---------------------------------------------------------------------------
# Bootstrap statistics
# ---------------------------------------------------------------------------
def _rng(seed: int | None = None) -> np.random.Generator:
    return np.random.default_rng(RANDOM_SEED if seed is None else seed)


def bootstrap_ci(errors, denominators, n_resamples: int = BOOTSTRAP_RESAMPLES,
                 seed: int | None = None, alpha: float = 0.05) -> tuple[float, float]:
    """95% CI of sum(errors)/sum(denominators) by resampling utterances."""
    errors = np.asarray(errors, dtype=float)
    denominators = np.asarray(denominators, dtype=float)
    n = len(errors)
    if n == 0:
        return (float("nan"), float("nan"))
    idx = _rng(seed).integers(0, n, size=(n_resamples, n))
    rates = errors[idx].sum(axis=1) / denominators[idx].sum(axis=1)
    lo, hi = np.percentile(rates, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(lo), float(hi)


def paired_bootstrap_diff(err_a, err_b, denominators, n_resamples: int = BOOTSTRAP_RESAMPLES,
                          seed: int | None = None) -> dict:
    """
    Paired comparison of two models scored on the SAME utterances.
    Returns WER(A) - WER(B) with a 95% CI. If the CI excludes 0 the
    difference is unlikely to be chance (given this sample).
    """
    err_a = np.asarray(err_a, dtype=float)
    err_b = np.asarray(err_b, dtype=float)
    den = np.asarray(denominators, dtype=float)
    n = len(den)
    diff = err_a.sum() / den.sum() - err_b.sum() / den.sum()
    idx = _rng(seed).integers(0, n, size=(n_resamples, n))
    d = (err_a[idx].sum(axis=1) - err_b[idx].sum(axis=1)) / den[idx].sum(axis=1)
    lo, hi = np.percentile(d, [2.5, 97.5])
    return {"diff": float(diff), "ci_lo": float(lo), "ci_hi": float(hi),
            "significant": bool(lo > 0 or hi < 0)}


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------
def scoreable(df: pd.DataFrame) -> pd.DataFrame:
    """Rows that were actually scored (no inference or scoring failure)."""
    if "wer" not in df.columns:
        return df.iloc[0:0]
    return df[df["wer"].notna()].copy()


def summarize_group(g: pd.DataFrame) -> dict:
    """Corpus-level metrics for one (model, language, condition) group."""
    n = len(g)
    ref_words = g["ref_words"].sum()
    ref_chars = g["ref_chars"].sum()
    lo, hi = bootstrap_ci(g["word_errors"], g["ref_words"])
    audio_s = g["audio_seconds"].sum()
    infer_s = g["latency_seconds"].sum()
    return {
        "n": n,
        "n_attempted": int(g.attrs.get("n_attempted", n)),
        "n_inference_failures": int(g.attrs.get("n_inference_failures", 0)),
        "n_scoring_errors": int(g.attrs.get("n_scoring_errors", 0)),
        "coverage": n / g.attrs.get("n_attempted", n) if g.attrs.get("n_attempted", n) else float("nan"),
        "corpus_wer": g["word_errors"].sum() / ref_words,
        "wer_ci_lo": lo,
        "wer_ci_hi": hi,
        "corpus_cer": g["char_errors"].sum() / ref_chars,
        "sub_rate": g["substitutions"].sum() / ref_words,
        "del_rate": g["deletions"].sum() / ref_words,
        "ins_rate": g["insertions"].sum() / ref_words,
        "empty_rate": g["is_empty"].mean(),
        "hallucination_rate": g["is_hallucination"].mean(),
        "wrong_script_rate": g["is_wrong_script"].mean(),
        "audio_seconds": audio_s,
        "infer_seconds": infer_s,
        "rtfx": audio_s / infer_s if infer_s > 0 else float("nan"),
        "mean_latency_s": g["latency_seconds"].mean(),
    }


def build_summary(df: pd.DataFrame) -> pd.DataFrame:
    """One row per (model, language, condition)."""
    rows = []
    keys = ["model", "language", "condition"]
    for (model, language, condition), attempted in df.groupby(keys):
        g = scoreable(attempted)
        scoring_errors = int(attempted.get("scoring_error", pd.Series(index=attempted.index, dtype=object)).notna().sum())
        if g.empty:
            failed = int((attempted.get("status", pd.Series(index=attempted.index, dtype=object)) == "inference_failed").sum())
            rows.append({
                "model": model, "language": language, "condition": condition,
                "n": 0, "n_attempted": len(attempted), "n_inference_failures": failed,
                "n_scoring_errors": scoring_errors, "coverage": 0.0,
                **{field: float("nan") for field in (
                    "corpus_wer", "wer_ci_lo", "wer_ci_hi", "corpus_cer", "sub_rate",
                    "del_rate", "ins_rate", "empty_rate", "hallucination_rate",
                    "wrong_script_rate", "audio_seconds", "infer_seconds", "rtfx",
                    "mean_latency_s",
                )},
            })
            continue
        g.attrs["n_attempted"] = len(attempted)
        g.attrs["n_inference_failures"] = int((attempted.get("status", pd.Series(index=attempted.index, dtype=object)) == "inference_failed").sum())
        g.attrs["n_scoring_errors"] = scoring_errors
        rows.append({"model": model, "language": language, "condition": condition,
                     **summarize_group(g)})
    return pd.DataFrame(rows)


def build_leaderboard(summary: pd.DataFrame) -> pd.DataFrame:
    """
    One row per (model, condition). Languages get EQUAL weight (macro
    average) so a language with more audio does not dominate the ranking.
    """
    if summary.empty:
        return summary
    rows = []
    for condition, condition_rows in summary.groupby("condition"):
        # Keep macro averages comparable: every model in this condition is
        # scored over the same language cohort.
        languages_by_model = {
            model: set(g.loc[g["corpus_wer"].notna(), "language"])
            for model, g in condition_rows.groupby("model")
        }
        shared_languages = set.intersection(*languages_by_model.values()) if languages_by_model else set()
        for model, all_rows in condition_rows.groupby("model"):
            g = all_rows[all_rows["language"].isin(shared_languages)]
            if g.empty:
                continue
            rows.append({
                "model": model,
                "condition": condition,
                "n_languages": g["language"].nunique(),
                "languages": ",".join(sorted(g["language"].unique())),
                "n_utterances": int(g["n"].sum()),
                "n_attempted": int(g.get("n_attempted", g["n"]).sum()),
                "coverage": (g["n"].sum() / g.get("n_attempted", g["n"]).sum()
                             if g.get("n_attempted", g["n"]).sum() else float("nan")),
                "macro_wer": g["corpus_wer"].mean(),
                "macro_cer": g["corpus_cer"].mean(),
                "rtfx": (g["audio_seconds"].sum() / g["infer_seconds"].sum()
                         if g["infer_seconds"].sum() > 0 else float("nan")),
                "hallucination_rate": np.average(g["hallucination_rate"], weights=g["n"]),
                "wrong_script_rate": np.average(g["wrong_script_rate"], weights=g["n"]),
            })
    if not rows:
        return pd.DataFrame(columns=["model", "condition", "n_languages", "languages", "n_utterances",
                                     "n_attempted", "coverage", "macro_wer", "macro_cer", "rtfx",
                                     "hallucination_rate", "wrong_script_rate"])
    return pd.DataFrame(rows).sort_values(["condition", "macro_wer"]).reset_index(drop=True)


def pairwise_comparison(df: pd.DataFrame, condition: str = "clean") -> pd.DataFrame:
    """
    Head-to-head, per language: for every pair of models, WER(A)-WER(B)
    on the utterances both models transcribed, with a bootstrap CI.
    """
    ok = scoreable(df)
    ok = ok[ok["condition"] == condition]
    rows = []
    for language, lg in ok.groupby("language"):
        models = sorted(lg["model"].unique())
        for i, a in enumerate(models):
            for b in models[i + 1:]:
                ga = lg[lg["model"] == a].set_index("sample_id")
                gb = lg[lg["model"] == b].set_index("sample_id")
                common = ga.index.intersection(gb.index)
                if len(common) < 5:
                    continue
                ga, gb = ga.loc[common], gb.loc[common]
                res = paired_bootstrap_diff(ga["word_errors"], gb["word_errors"], ga["ref_words"])
                rows.append({"language": language, "model_a": a, "model_b": b,
                             "n_common": len(common), **res})
    return pd.DataFrame(rows)


def failure_categories(df: pd.DataFrame) -> pd.DataFrame:
    """
    Where do models break? Mean WER for utterances in each error category,
    per model (clean condition only, so noise does not muddy the picture).
    """
    ok = scoreable(df)
    ok = ok[ok["condition"] == "clean"]
    rows = []
    for model, g in ok.groupby("model"):
        digits = g[g["ref_has_digits"]]
        no_digits = g[~g["ref_has_digits"]]
        rows.append({
            "model": model,
            "n": len(g),
            "hallucination_pct": 100 * g["is_hallucination"].mean(),
            "wrong_script_pct": 100 * g["is_wrong_script"].mean(),
            "empty_pct": 100 * g["is_empty"].mean(),
            "n_with_digits": len(digits),
            "wer_with_digits": digits["wer"].mean() if len(digits) else float("nan"),
            "wer_without_digits": no_digits["wer"].mean() if len(no_digits) else float("nan"),
        })
    return pd.DataFrame(rows)


def worst_failures(df: pd.DataFrame, n: int = 5) -> pd.DataFrame:
    """Top-n highest-WER utterances per model (clean condition)."""
    ok = scoreable(df)
    ok = ok[ok["condition"] == "clean"]
    if ok.empty:
        return ok
    return (ok.sort_values("wer", ascending=False)
              .groupby("model").head(n).reset_index(drop=True))
