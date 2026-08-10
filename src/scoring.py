"""
WER/CER scoring, built on top of `jiwer`.

TESTED in this environment against jiwer==4.0.0 (see requirements.txt) --
this is one of the few modules in this project that has no dependency on
huggingface.co, so it was actually run and verified, not just written from
documentation. If you're on a different jiwer version and something here
breaks, `jiwer.wer` / `jiwer.cer` are the two functions this module depends
on -- check those signatures first.
"""

import jiwer

from src.normalize import normalize_text


def score_pair(reference: str, hypothesis: str) -> dict:
    """
    Compute normalized WER and CER for one reference/hypothesis pair.

    Both strings are normalized identically (see normalize.py for why)
    before scoring. Returns a dict with wer, cer, and the normalized
    strings themselves (kept for debugging/display -- e.g. the "worst
    failures" report wants to show what was actually compared, not the
    raw pre-normalization text, so a reader can see exactly why a WER was
    high).

    Edge case: if the normalized reference is empty (rare, but FLEURS can
    have a reference that becomes empty after normalization if it was
    punctuation-only), jiwer.wer will raise a ZeroDivisionError-adjacent
    error. We catch this and return wer=None to signal "not scoreable"
    rather than crashing the whole batch on one bad sample.
    """
    ref_norm = normalize_text(reference)
    hyp_norm = normalize_text(hypothesis)

    if not ref_norm:
        return {
            "wer": None,
            "cer": None,
            "reference_normalized": ref_norm,
            "hypothesis_normalized": hyp_norm,
            "scoring_error": "empty reference after normalization",
        }

    try:
        wer = jiwer.wer(ref_norm, hyp_norm)
        cer = jiwer.cer(ref_norm, hyp_norm)
    except Exception as e:
        return {
            "wer": None,
            "cer": None,
            "reference_normalized": ref_norm,
            "hypothesis_normalized": hyp_norm,
            "scoring_error": str(e),
        }

    return {
        "wer": wer,
        "cer": cer,
        "reference_normalized": ref_norm,
        "hypothesis_normalized": hyp_norm,
        "scoring_error": None,
    }


if __name__ == "__main__":
    # Manual sanity check -- run directly with `python3 src/scoring.py`
    tests = [
        ("Hello, world! This is a test.", "hello world this is test"),
        ("नमस्ते, आप कैसे हैं?", "नमस्ते आप कैसे हैं"),
        ("completely different", "nothing alike here at all"),
        ("...", "hello"),  # empty-after-normalization edge case
    ]
    for ref, hyp in tests:
        result = score_pair(ref, hyp)
        print(f"ref={ref!r} hyp={hyp!r}")
        print(f"  -> wer={result['wer']}, cer={result['cer']}, error={result['scoring_error']}")
