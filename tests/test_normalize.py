from src.normalize import normalize_text, script_ratio


def test_punctuation_case_whitespace():
    assert normalize_text("  Hello,  WORLD!! ") == "hello world"


def test_devanagari_danda_removed():
    assert normalize_text("नमस्ते, आप कैसे हैं।") == "नमस्ते आप कैसे हैं"


def test_tamil_double_danda_and_spacing():
    assert normalize_text("வணக்கம்।  எப்படி இருக்கிறீர்கள்॥") == "வணக்கம் எப்படி இருக்கிறீர்கள்"


def test_curly_quotes_and_dashes_removed():
    assert normalize_text("“quoted” – text… done") == "quoted text done"


def test_native_digits_become_ascii():
    assert normalize_text("४५ और ௪௫") == "45 और 45"


def test_zero_width_joiners_removed():
    assert normalize_text("क\u200dष") == normalize_text("कष")


def test_nukta_forms_equal():
    assert normalize_text("\u0958") == normalize_text("\u0915\u093c")


def test_empty_and_none():
    assert normalize_text("") == ""
    assert normalize_text(None) == ""


def test_script_ratio():
    assert script_ratio("नमस्ते दुनिया", "DEVANAGARI") == 1.0
    assert script_ratio("hello world", "DEVANAGARI") == 0.0
    assert script_ratio("1234 ...", "DEVANAGARI") is None
    mixed = script_ratio("नमस्ते hello", "DEVANAGARI")
    assert 0.0 < mixed < 1.0
