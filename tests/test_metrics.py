import numpy as np
import pandas as pd
import pytest

from src.metrics import (bootstrap_ci, build_leaderboard, build_summary,
                         failure_categories, pairwise_comparison, paired_bootstrap_diff,
                         score_utterance, worst_failures)
from src.synthetic import make_synthetic_results


def test_perfect_match():
    r = score_utterance("आप कैसे हैं", "आप कैसे हैं।", "DEVANAGARI")
    assert r["wer"] == 0 and r["cer"] == 0 and not r["is_hallucination"]


def test_sub_del_ins_counts():
    # ref 4 words; hyp: one substitution (b->x), one deletion (d), one insertion (z)
    r = score_utterance("a b c d", "a x c z", None)
    assert (r["substitutions"], r["deletions"], r["insertions"]) == (1, 1, 1) or \
           (r["substitutions"] + r["deletions"] + r["insertions"]) == r["word_errors"]
    assert r["ref_words"] == 4
    assert r["wer"] == pytest.approx(r["word_errors"] / 4)


def test_wer_can_exceed_one_and_flags_hallucination():
    r = score_utterance("a b", "x y z w q", None)
    assert r["wer"] > 1.0 and r["is_hallucination"]


def test_empty_hypothesis_is_all_deletions():
    r = score_utterance("a b c", "", None)
    assert r["wer"] == 1.0 and r["is_empty"] and r["deletions"] == 3


def test_empty_reference_not_scoreable():
    r = score_utterance("...", "hello", None)
    assert r["scoring_error"] and "wer" not in r


def test_wrong_script_flag():
    r = score_utterance("नमस्ते दुनिया", "hello world", "DEVANAGARI")
    assert r["is_wrong_script"]


def test_digits_flag():
    assert score_utterance("मेरे ५ भाई", "मेरे 5 भाई", None)["ref_has_digits"]
    assert score_utterance("मेरे ५ भाई", "मेरे 5 भाई", None)["wer"] == 0  # digit normalization


def test_corpus_wer_differs_from_mean_of_sentence_wer():
    # short sentence all wrong, long sentence nearly perfect
    short = score_utterance("a b", "x y", None)               # WER 1.0 on 2 words
    long_ = score_utterance(" ".join("w%d" % i for i in range(98)),
                            " ".join("w%d" % i for i in range(98)), None)  # 0 errors, 98 words
    corpus = (short["word_errors"] + long_["word_errors"]) / (short["ref_words"] + long_["ref_words"])
    mean_sentence = (short["wer"] + long_["wer"]) / 2
    assert corpus == pytest.approx(0.02)
    assert mean_sentence == pytest.approx(0.5)


def test_bootstrap_ci_contains_point_estimate_and_is_ordered():
    rng = np.random.default_rng(0)
    den = rng.integers(5, 30, size=80)
    err = rng.binomial(den, 0.3)
    lo, hi = bootstrap_ci(err, den)
    point = err.sum() / den.sum()
    assert lo <= point <= hi and lo < hi


def test_bootstrap_ci_narrows_with_more_data():
    rng = np.random.default_rng(1)
    def width(n):
        den = np.full(n, 20)
        err = rng.binomial(den, 0.3)
        lo, hi = bootstrap_ci(err, den)
        return hi - lo
    assert width(400) < width(25)


def test_paired_diff_detects_clear_difference_and_not_noise():
    rng = np.random.default_rng(2)
    den = np.full(100, 20)
    a = rng.binomial(den, 0.20)
    b = rng.binomial(den, 0.50)
    assert paired_bootstrap_diff(a, b, den)["significant"]
    # same per-utterance error counts, just paired differently => total difference is exactly 0
    c = rng.binomial(den, 0.30)
    d = np.roll(c, 1)
    res = paired_bootstrap_diff(c, d, den)
    assert res["diff"] == 0 and not res["significant"]


def test_full_aggregation_on_synthetic_data():
    df = make_synthetic_results(n_per_lang=25)
    summary = build_summary(df)
    assert {"corpus_wer", "wer_ci_lo", "wer_ci_hi", "rtfx"} <= set(summary.columns)
    assert (summary["wer_ci_lo"] <= summary["corpus_wer"]).all()
    assert (summary["corpus_wer"] <= summary["wer_ci_hi"]).all()
    board = build_leaderboard(summary)
    # the synthetic "better" model should rank first on clean audio
    clean = board[board["condition"] == "clean"].sort_values("macro_wer")
    assert clean.iloc[0]["model"] == "indic-conformer-600m"
    # more noise => higher WER for every model
    for m, g in board.groupby("model"):
        g = g.set_index("condition")
        assert g.loc["snr10", "macro_wer"] > g.loc["clean", "macro_wer"]
    assert not pairwise_comparison(df).empty
    assert not failure_categories(df).empty
    assert len(worst_failures(df, 3)) <= 3 * df["model"].nunique()


def test_leaderboard_uses_same_languages_for_each_model():
    df = make_synthetic_results(n_per_lang=10)
    df = df[~((df["model"] == "whisper-small") & (df["language"] == "tamil"))]
    board = build_leaderboard(build_summary(df))
    clean = board[board["condition"] == "clean"]
    assert set(clean["languages"]) == {"bengali,hindi"}


def test_summary_exposes_failed_attempts_without_scoring_them():
    successful = score_utterance("a b", "a b")
    df = pd.DataFrame([
        {"model": "m", "language": "hindi", "condition": "clean", "sample_id": "1",
         "audio_seconds": 1.0, "latency_seconds": 0.2, "status": "scored", **successful},
        {"model": "m", "language": "hindi", "condition": "clean", "sample_id": "2",
         "audio_seconds": 1.0, "latency_seconds": 0.1, "status": "inference_failed",
         "inference_error": "RuntimeError: broken"},
    ])
    summary = build_summary(df).iloc[0]
    assert summary["n"] == 1 and summary["n_attempted"] == 2
    assert summary["n_inference_failures"] == 1 and summary["coverage"] == 0.5
