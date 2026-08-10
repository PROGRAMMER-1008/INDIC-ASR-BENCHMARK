"""
Text normalization applied identically to reference and hypothesis
transcripts before scoring.

WHY NORMALIZATION MATTERS FOR FAIR COMPARISON
-----------------------------------------------
WER/CER are computed as edit distance over tokens/characters. Two models can
produce *semantically identical* transcriptions that score very differently
on raw WER purely because of surface formatting choices that have nothing to
do with transcription accuracy:

  - Whisper tends to emit punctuation and capitalization (it was trained on
    web text with natural formatting). AI4Bharat's indic-conformer, and the
    FLEURS reference transcripts themselves, are typically lowercase/
    punctuation-light.
  - FLEURS references may have inconsistent whitespace (double spaces,
    trailing spaces) inherited from the source transcription process.
  - Indian-language text can have visually-equivalent strings represented
    with different Unicode normalization forms (e.g. composed vs decomposed
    diacritics), which edit-distance scoring treats as different characters
    even though a human reader sees identical text.

If we score raw model output against raw references, a model that happens to
match the reference's punctuation/casing style gets an unfair WER advantage
that has nothing to do with how well it actually transcribed the speech. So
we apply ONE normalization function to both reference and hypothesis, for
every model, before computing WER/CER. This doesn't make WER "true accuracy"
(no metric fully captures that) but it does make the comparison *fair*
across models -- which is the whole point of a benchmark.

Trade-off to be upfront about: normalizing away punctuation/casing means the
benchmark does NOT measure whether a model gets punctuation or capitalization
right. If that matters for your use case (e.g. subtitle generation), you'd
want a second, unnormalized scoring pass. See README "Known Limitations".
"""

import re
import unicodedata


def normalize_text(text: str) -> str:
    """
    Apply a consistent normalization pipeline to a transcript string.

    Steps (in order, each documented):
      1. Unicode NFC normalization -- collapses composed/decomposed
         equivalent representations of the same character (important for
         Devanagari/Tamil script where diacritics can be encoded either way).
      2. Lowercase -- removes casing as a scoring factor. Indic scripts
         mostly don't have case, but this matters for any Latin-script
         numerals/loanwords that show up in transcripts.
      3. Strip punctuation -- removes ASCII and common Unicode punctuation
         (including Devanagari danda '।' and double danda '॥') so
         punctuation-style differences between models don't count as errors.
      4. Normalize whitespace -- collapse multiple spaces/tabs/newlines to
         a single space, strip leading/trailing whitespace.

    Returns the normalized string. Empty/None input returns "".
    """
    if not text:
        return ""

    # Step 1: Unicode normalization form C (canonical composition)
    text = unicodedata.normalize("NFC", text)

    # Step 2: lowercase
    text = text.lower()

    # Step 3: strip punctuation. We explicitly include Devanagari danda
    # (U+0964) and double danda (U+0965) since Python's string.punctuation
    # and most regex \W classes are Latin-script-centric and miss these.
    punctuation_pattern = r"[।॥!\"#$%&'()*+,\-./:;<=>?@\[\\\]^_`{|}~]"
    text = re.sub(punctuation_pattern, " ", text)

    # Step 4: collapse whitespace
    text = re.sub(r"\s+", " ", text).strip()

    return text


if __name__ == "__main__":
    # Quick manual sanity check
    samples = [
        "  नमस्ते,  आप कैसे हैं?  ",
        "Hello, WORLD!!",
        "வணக்கம்।  எப்படி இருக்கிறீர்கள்॥",
    ]
    for s in samples:
        print(repr(s), "->", repr(normalize_text(s)))
