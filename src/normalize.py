"""
Text normalization applied identically to reference and hypothesis before
scoring.

WHY THIS EXISTS
---------------
WER/CER are edit distances. Two transcripts that a human reads as identical
can score very differently because of formatting that has nothing to do with
recognition quality: punctuation, casing, double spaces, digits written in
different scripts, invisible joiner characters, or two different Unicode
encodings of the same letter. One shared normalizer, applied to every model
and to the reference, keeps the comparison fair.

WHAT IT DOES NOT DO (be upfront about this in interviews)
---------------------------------------------------------
It does not convert spoken numbers to digits ("दो सौ" vs "200"), does not
fix spelling variants, and does not judge punctuation quality. Those are
real error sources and are reported separately as error categories.
"""

import re
import unicodedata

# Invisible characters that change how text is encoded but not how it reads.
_INVISIBLE = {
    "\u200b",  # zero-width space
    "\u200c",  # zero-width non-joiner
    "\u200d",  # zero-width joiner
    "\u2060",  # word joiner
    "\ufeff",  # byte-order mark
}
_WS = re.compile(r"\s+")


def _is_punct_or_symbol(ch: str) -> bool:
    # Unicode category P* = punctuation (includes Devanagari danda '।',
    # curly quotes, em dashes, ellipsis ...). S* = symbols (+, =, $ ...).
    cat = unicodedata.category(ch)
    return cat[0] in ("P", "S")


def normalize_text(text: str | None) -> str:
    """
    Pipeline:
      1. NFC Unicode normalization. This also makes letters like क़ (U+0958)
         and क+़ (U+0915 U+093C) the same sequence.
      2. Drop invisible joiner / zero-width characters.
      3. Lowercase (matters for Latin loan-words).
      4. Native digits -> ASCII digits (Devanagari ४ -> 4, Tamil ௪ -> 4 ...).
      5. Punctuation and symbols -> space.
      6. Collapse whitespace.
    """
    if not text:
        return ""
    text = unicodedata.normalize("NFC", text)

    out = []
    for ch in text:
        if ch in _INVISIBLE:
            continue
        if unicodedata.category(ch) == "Nd":          # any script's decimal digit
            out.append(str(unicodedata.decimal(ch)))
        elif _is_punct_or_symbol(ch):
            out.append(" ")
        else:
            out.append(ch.lower())
    return _WS.sub(" ", "".join(out)).strip()


def script_ratio(text: str, script_prefix: str) -> float | None:
    """
    Fraction of *letters* in `text` that belong to the expected script
    (e.g. "DEVANAGARI"). Returns None when the text has no letters.
    Used to catch a classic multilingual failure: asked for Hindi, the
    model answers in Urdu/Latin script (or a different Indic script).
    """
    letters = [c for c in text if unicodedata.category(c).startswith("L")]
    if not letters:
        return None
    hits = 0
    for c in letters:
        try:
            if unicodedata.name(c).startswith(script_prefix):
                hits += 1
        except ValueError:
            pass
    return hits / len(letters)
