"""
SYNTHETIC data generator -- used ONLY to preview the report layout and to
exercise the pipeline in tests. Nothing here is a real benchmark result, and
any report built from it carries a large SYNTHETIC banner.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.config import LANGUAGES
from src.metrics import score_utterance

_WORDS = {
    "hindi": "मैं आप वह यह घर पानी किताब स्कूल बाजार सरकार शहर लोग समय काम दिन रात खाना सड़क देश बच्चे".split(),
    "tamil": "நான் நீங்கள் அவர் வீடு தண்ணீர் புத்தகம் பள்ளி சந்தை அரசு நகரம் மக்கள் நேரம் வேலை நாள் இரவு உணவு சாலை நாடு".split(),
    "bengali": "আমি আপনি সে বাড়ি জল বই স্কুল বাজার সরকার শহর মানুষ সময় কাজ দিন রাত খাবার রাস্তা দেশ".split(),
}
# (word error prob, rtfx, hallucination prob) -- invented numbers
_MODEL_PROFILES = {
    "whisper-small":        (0.38, 2.5, 0.06),
    "faster-whisper-small": (0.37, 6.0, 0.05),
    "indic-conformer-600m": (0.22, 4.0, 0.00),
}
_NOISE_PENALTY = {"clean": 0.0, "snr20": 0.03, "snr10": 0.10, "snr5": 0.22}


def make_synthetic_results(n_per_lang: int = 40, languages=("hindi", "tamil", "bengali"),
                           conditions=("clean", "snr10"), seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for lang in languages:
        vocab = _WORDS[lang]
        script = LANGUAGES[lang]["script"]
        for i in range(n_per_lang):
            ref_words = list(rng.choice(vocab, size=int(rng.integers(5, 26))))
            reference = " ".join(ref_words)
            audio_s = len(ref_words) * 0.55
            for model, (p_err, rtfx, p_hall) in _MODEL_PROFILES.items():
                for cond in conditions:
                    p = min(0.9, p_err + _NOISE_PENALTY[cond])
                    hyp_words = []
                    for w in ref_words:
                        r = rng.random()
                        if r < p * 0.55:
                            hyp_words.append(str(rng.choice(vocab)))
                        elif r < p * 0.80:
                            continue
                        else:
                            hyp_words.append(w)
                        if rng.random() < p * 0.12:
                            hyp_words.append(str(rng.choice(vocab)))
                    hyp = " ".join(hyp_words)
                    if rng.random() < p_hall * (1 + _NOISE_PENALTY[cond] * 5):
                        hyp = " ".join(["thank you for watching"] * int(rng.integers(3, 9)))
                    rows.append({
                        "model": model, "language": lang, "condition": cond,
                        "sample_id": f"{lang}_{i}", "reference": reference, "hypothesis": hyp,
                        "audio_seconds": round(audio_s, 3),
                        "latency_seconds": round(audio_s / rtfx * rng.uniform(0.8, 1.25), 4),
                        **score_utterance(reference, hyp, script),
                    })
    return pd.DataFrame(rows)


SYNTHETIC_RUNTIME = {
    "whisper-small": {"load_seconds": 4.1, "ram_delta_mb_approx": 1200.0, "params_m": 244},
    "faster-whisper-small": {"load_seconds": 2.3, "ram_delta_mb_approx": 520.0, "params_m": 244},
    "indic-conformer-600m": {"load_seconds": 9.8, "ram_delta_mb_approx": 2600.0, "params_m": 600},
}
