"""End-to-end test of the runner with a fake model: no downloads, no GPU."""

import csv
import json

import numpy as np
import pandas as pd
import soundfile as sf

from src import run_benchmark as rb
from src.data_loader import load_samples
from src.metrics import build_summary
from src.models.base import ASRModelWrapper


class EchoModel(ASRModelWrapper):
    """Returns the reference text (looked up by audio length) -> WER 0 on clean audio."""
    name = "echo"
    lookup: dict = {}
    calls = 0

    def transcribe(self, audio_array, sample_rate, language_key):
        EchoModel.calls += 1
        return EchoModel.lookup.get(len(audio_array), "")


class FlakyModel(ASRModelWrapper):
    name = "flaky"
    calls = 0

    def transcribe(self, audio_array, sample_rate, language_key):
        FlakyModel.calls += 1
        if FlakyModel.calls == 2:  # warm-up succeeds; first scored attempt fails
            raise RuntimeError("temporary inference problem")
        return EchoModel.lookup.get(len(audio_array), "")


def _make_manifest(root, lang="hindi", n=4):
    d = root / lang
    d.mkdir(parents=True)
    rows = []
    for i in range(n):
        length = 16000 + i * 800                      # unique length per sample
        audio = (0.1 * np.sin(np.linspace(0, 400, length))).astype(np.float32)
        sf.write(d / f"{lang}_{i}.wav", audio, 16000)
        text = f"यह वाक्य संख्या {i} है।"
        EchoModel.lookup[length] = text
        rows.append({"sample_id": f"{lang}_{i}", "wav": f"{lang}_{i}.wav", "text": text,
                     "duration_s": length / 16000})
    with open(d / "manifest.csv", "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["sample_id", "wav", "text", "duration_s"])
        w.writeheader()
        w.writerows(rows)


def test_manifest_loader_roundtrip(tmp_path):
    _make_manifest(tmp_path)
    s = load_samples("hindi", 3, root=str(tmp_path), allow_download=False)
    assert len(s) == 3 and s[0].sample_rate == 16000 and s[0].reference_text.startswith("यह")


def test_huggingface_adapter_maps_configured_audio_and_text_fields(tmp_path, monkeypatch):
    import sys
    import types
    from src.data_loader import export_huggingface_dataset

    captured = {}
    def fake_load_dataset(name, config, **kwargs):
        captured.update(name=name, config=config, **kwargs)
        return [{"id": "fixture-1", "speech": {"array": np.ones(1600, dtype=np.float32) * 0.1,
                                                   "sampling_rate": 16000},
                 "sentence": "नमूना वाक्य"}]
    monkeypatch.setitem(sys.modules, "datasets", types.SimpleNamespace(load_dataset=fake_load_dataset))
    export_huggingface_dataset("hindi", 1, str(tmp_path), dataset_name="fixture/dataset",
                               dataset_config="hi", split="test", audio_field="speech",
                               text_field="sentence", fallback_text_field=None)
    loaded = load_samples("hindi", 1, root=str(tmp_path), allow_download=False)
    assert loaded[0].reference_text == "नमूना वाक्य"
    assert captured["name"] == "fixture/dataset" and captured["config"] == "hi"
    assert captured["split"] == "test" and captured["streaming"] is True


def test_run_is_resumable_and_scores(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    _make_manifest(tmp_path / "cache")
    monkeypatch.setattr(rb, "build_model", lambda name: EchoModel())
    monkeypatch.setitem(rb.MODELS, "echo", {"family": "x", "checkpoint": "x", "params_m": 1})

    EchoModel.calls = 0
    rb.run(["echo"], ["hindi"], 4, ["clean"], allow_download=False, data_dir=str(tmp_path / "cache"))
    rows = rb.load_existing()
    assert len(rows) == 4 and all(r["wer"] == 0 for r in rows)
    first_calls = EchoModel.calls

    # second identical run: everything already done -> no new inference at all
    rb.run(["echo"], ["hindi"], 4, ["clean"], allow_download=False, data_dir=str(tmp_path / "cache"))
    assert len(rb.load_existing()) == 4 and EchoModel.calls == first_calls

    # adding a noise condition only runs the NEW combinations
    rb.run(["echo"], ["hindi"], 4, ["clean", "snr10"], allow_download=False, data_dir=str(tmp_path / "cache"))
    rows = rb.load_existing()
    assert len(rows) == 8
    noisy = [r for r in rows if r["condition"] == "snr10"]
    assert len(noisy) == 4   # only the new condition was run

    df = rb.finalize()
    assert (tmp_path / "results" / "summary.csv").exists()
    assert set(pd.read_csv(tmp_path / "results" / "summary.csv")["condition"]) == {"clean", "snr10"}


def test_truncated_last_line_is_ignored(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "results").mkdir()
    (tmp_path / "results" / "raw_results.jsonl").write_text(
        json.dumps({"a": 1}) + "\n" + '{"broken": ', encoding="utf-8")
    assert rb.load_existing() == [{"a": 1}]


def test_inference_failure_is_visible_and_retried_on_resume(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    _make_manifest(tmp_path / "cache", n=1)
    monkeypatch.setattr(rb, "build_model", lambda name: FlakyModel())
    monkeypatch.setitem(rb.MODELS, "flaky", {"family": "x", "checkpoint": "x", "params_m": 1})
    FlakyModel.calls = 0

    rb.run(["flaky"], ["hindi"], 1, ["clean"], allow_download=False, data_dir=str(tmp_path / "cache"))
    first = rb.load_existing()
    assert len(first) == 1 and first[0]["status"] == "inference_failed"

    rb.run(["flaky"], ["hindi"], 1, ["clean"], allow_download=False, data_dir=str(tmp_path / "cache"))
    rows = rb.load_existing()
    assert len(rows) == 2 and rows[-1]["status"] == "scored" and rows[-1]["wer"] == 0
    summary = rb.finalize()
    cell = build_summary(summary).iloc[0]
    assert cell["n_attempted"] == 2 and cell["n_inference_failures"] == 1 and cell["coverage"] == 0.5


def test_manifest_fingerprint_is_recorded(tmp_path):
    manifest = tmp_path / "hindi" / "manifest.csv"
    manifest.parent.mkdir()
    manifest.write_text("sample_id,wav,text,duration_s\n", encoding="utf-8")
    record = rb._manifest_fingerprints(str(tmp_path), ["hindi"])["hindi"]
    assert len(record["sha256"]) == 64 and record["bytes"] == manifest.stat().st_size


def test_report_builds_from_synthetic():
    from src.generate_report import build_html, build_markdown
    from src.synthetic import SYNTHETIC_RUNTIME, make_synthetic_results
    df = make_synthetic_results(n_per_lang=15)
    html = build_html(df, {"model_runtime": SYNTHETIC_RUNTIME}, synthetic=True)
    assert "SYNTHETIC" in html and "<table>" in html and "base64" in html
    assert "| Model |" in build_markdown(df)
