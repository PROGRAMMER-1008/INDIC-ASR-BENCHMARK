"""
Builds a single self-contained HTML report (charts embedded as base64 PNGs)
plus a Markdown results file you can paste into the README.

    python -m src.generate_report          # real results from results/
    python -m src.generate_report --demo   # SYNTHETIC preview (clearly labelled)
"""

from __future__ import annotations

import argparse
import base64
import html
import io
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src import config
from src.metrics import (build_leaderboard, build_summary, failure_categories,
                         pairwise_comparison, scoreable, worst_failures)

CONDITION_ORDER = ["clean", "snr20", "snr10", "snr5"]


# ---------------------------------------------------------------------------
# Charts
# ---------------------------------------------------------------------------
def _b64(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=130)
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode()


def chart_wer_by_language(summary: pd.DataFrame) -> str | None:
    s = summary[summary["condition"] == "clean"]
    if s.empty:
        return None
    langs = sorted(s["language"].unique())
    models = sorted(s["model"].unique())
    x = np.arange(len(langs))
    width = 0.8 / max(len(models), 1)
    fig, ax = plt.subplots(figsize=(8, 4.2))
    for i, m in enumerate(models):
        g = s[s["model"] == m].set_index("language").reindex(langs)
        wer = g["corpus_wer"].values * 100
        err = np.vstack([wer - g["wer_ci_lo"].values * 100, g["wer_ci_hi"].values * 100 - wer])
        ax.bar(x + i * width, wer, width, yerr=np.nan_to_num(err), capsize=3, label=m)
    ax.set_xticks(x + width * (len(models) - 1) / 2)
    ax.set_xticklabels([l.title() for l in langs])
    ax.set_ylabel("Corpus WER (%)  -  lower is better")
    ax.set_title("Word error rate by language (clean audio, 95% bootstrap CI)")
    ax.legend(fontsize=8)
    return _b64(fig)


def chart_noise(board: pd.DataFrame) -> str | None:
    if board.empty or board["condition"].nunique() < 2:
        return None
    conds = [c for c in CONDITION_ORDER if c in set(board["condition"])]
    fig, ax = plt.subplots(figsize=(7, 4))
    for m, g in board.groupby("model"):
        g = g.set_index("condition").reindex(conds)
        ax.plot(conds, g["macro_wer"].values * 100, marker="o", label=m)
    ax.set_ylabel("Macro WER (%)")
    ax.set_xlabel("Condition (snrX = X dB signal-to-noise; lower SNR = noisier)")
    ax.set_title("Robustness to added noise")
    ax.legend(fontsize=8)
    return _b64(fig)


def chart_error_types(df: pd.DataFrame) -> str | None:
    ok = scoreable(df)
    ok = ok[ok["condition"] == "clean"]
    if ok.empty:
        return None
    g = ok.groupby("model")[["substitutions", "deletions", "insertions", "ref_words"]].sum()
    rates = g[["substitutions", "deletions", "insertions"]].div(g["ref_words"], axis=0) * 100
    fig, ax = plt.subplots(figsize=(7, 4))
    rates.plot(kind="bar", stacked=True, ax=ax, color=["#d95f02", "#7570b3", "#1b9e77"])
    ax.set_ylabel("Errors per 100 reference words")
    ax.set_title("What kind of errors? (substitution / deletion / insertion)")
    plt.xticks(rotation=15)
    return _b64(fig)


def chart_speed_accuracy(board: pd.DataFrame) -> str | None:
    b = board[board["condition"] == "clean"]
    if b.empty:
        return None
    fig, ax = plt.subplots(figsize=(6.5, 4))
    ax.scatter(b["rtfx"], b["macro_wer"] * 100, s=70)
    for _, r in b.iterrows():
        ax.annotate(r["model"], (r["rtfx"], r["macro_wer"] * 100), fontsize=8,
                    xytext=(5, 5), textcoords="offset points")
    ax.set_xscale("log")
    ax.set_xlabel("RTFx (audio seconds per compute second, log scale)  -  right = faster")
    ax.set_ylabel("Macro WER (%)  -  down = better")
    ax.set_title("Accuracy vs speed (bottom-right is best)")
    return _b64(fig)


# ---------------------------------------------------------------------------
# HTML helpers
# ---------------------------------------------------------------------------
def _pct(x, d=1):
    return "-" if pd.isna(x) else f"{100 * x:.{d}f}%"


def _table(headers: list[str], rows: list[list]) -> str:
    th = "".join(f"<th>{html.escape(h)}</th>" for h in headers)
    body = "".join("<tr>" + "".join(f"<td>{c}</td>" for c in r) + "</tr>" for r in rows)
    return f"<table><thead><tr>{th}</tr></thead><tbody>{body}</tbody></table>"


def _img(b64: str | None, alt: str) -> str:
    return f'<img alt="{alt}" src="data:image/png;base64,{b64}">' if b64 else ""


CSS = """
body{font-family:-apple-system,Segoe UI,Roboto,sans-serif;max-width:980px;margin:2rem auto;padding:0 1rem;color:#222;line-height:1.5}
h1{margin-bottom:.2rem} h2{margin-top:2.2rem;border-bottom:2px solid #eee;padding-bottom:.3rem}
table{border-collapse:collapse;width:100%;font-size:.88rem;margin:.8rem 0}
th,td{border:1px solid #ddd;padding:6px 8px;text-align:left;vertical-align:top} th{background:#f5f5f5}
img{max-width:100%;margin:.5rem 0} .note{background:#f7f7f2;border-left:4px solid #bbb;padding:.6rem 1rem;font-size:.9rem}
.banner{background:#b00020;color:#fff;padding:1rem;font-weight:700;text-align:center;font-size:1.1rem;margin-bottom:1rem}
.ref{color:#0a6} .hyp{color:#a30} code{background:#f2f2f2;padding:1px 4px}
"""


def build_html(df: pd.DataFrame, meta: dict, synthetic: bool = False) -> str:
    summary = build_summary(df)
    board = build_leaderboard(summary)
    pairs = pairwise_comparison(df, "clean")
    cats = failure_categories(df)
    worst = worst_failures(df, config.N_WORST_FAILURES)

    parts = [f"<html><head><meta charset='utf-8'><title>Indic ASR Benchmark</title><style>{CSS}</style></head><body>"]
    if synthetic:
        parts.append("<div class='banner'>SYNTHETIC DEMO DATA - these numbers are invented to preview the report layout. NOT real results.</div>")
    parts.append("<h1>Indic ASR Benchmark</h1>")
    hw = meta.get("hardware", {})
    ok = scoreable(df)
    parts.append(
        f"<p>{ok['model'].nunique()} models &middot; {ok['language'].nunique()} languages &middot; "
        f"{ok['sample_id'].nunique()} unique utterances &middot; device: "
        f"<code>{html.escape(str(hw.get('gpu') or hw.get('device', 'unknown')))}</code> &middot; "
        f"{html.escape(str(hw.get('cpu_threads', '?')))} CPU threads, {html.escape(str(hw.get('ram_gb', '?')))} GB RAM</p>")

    # Leaderboard
    parts.append("<h2>Leaderboard</h2><p class='note'>Macro average: every language counts equally. "
                 "WER/CER lower is better; RTFx higher is faster. Hallucination = utterance WER above 100%; "
                 "wrong script = output mostly in a different writing system than the language's own.</p>")
    rows = [[html.escape(r.model), r.condition, r.n_languages, r.n_utterances,
             _pct(getattr(r, "coverage", float("nan"))), _pct(r.macro_wer),
             _pct(r.macro_cer), f"{r.rtfx:.2f}x", _pct(r.hallucination_rate), _pct(r.wrong_script_rate)]
            for r in board.itertuples()]
    parts.append(_table(["Model", "Condition", "Langs", "Utts", "Coverage", "Macro WER", "Macro CER", "RTFx",
                         "Halluc.", "Wrong script"], rows))
    parts.append(_img(chart_speed_accuracy(board), "speed vs accuracy"))

    # Per language
    parts.append("<h2>Per-language results (with 95% confidence intervals)</h2>")
    parts.append(_img(chart_wer_by_language(summary), "WER by language"))
    s = summary.sort_values(["condition", "language", "corpus_wer"])
    rows = [[r.language.title(), html.escape(r.model), r.condition, r.n,
             getattr(r, "n_attempted", r.n), getattr(r, "n_inference_failures", 0),
             getattr(r, "n_scoring_errors", 0),
             _pct(r.coverage) if hasattr(r, "coverage") else "100.0%", _pct(r.corpus_wer),
             f"{_pct(r.wer_ci_lo)} - {_pct(r.wer_ci_hi)}", _pct(r.corpus_cer),
             _pct(r.sub_rate), _pct(r.del_rate), _pct(r.ins_rate)] for r in s.itertuples()]
    parts.append(_table(["Language", "Model", "Cond.", "Scored", "Attempted", "Inference failures",
                         "Scoring errors", "Coverage",
                         "WER", "95% CI", "CER", "Sub", "Del", "Ins"], rows))

    # Head-to-head
    if not pairs.empty:
        parts.append("<h2>Head-to-head (paired bootstrap, clean audio)</h2><p class='note'>"
                     "WER(A) minus WER(B) on the same utterances. Negative = A is better. "
                     "&ldquo;Significant&rdquo; means the 95% interval excludes zero.</p>")
        rows = [[r.language.title(), html.escape(r.model_a), html.escape(r.model_b), r.n_common,
                 f"{100 * r.diff:+.1f} pts", f"{100 * r.ci_lo:+.1f} to {100 * r.ci_hi:+.1f}",
                 "yes" if r.significant else "no"] for r in pairs.itertuples()]
        parts.append(_table(["Language", "A", "B", "N", "WER(A)-WER(B)", "95% CI", "Significant"], rows))

    # Noise + error types
    nb = chart_noise(board)
    if nb:
        parts.append("<h2>Robustness to noise</h2>" + _img(nb, "noise robustness"))
    parts.append("<h2>Error analysis</h2>" + _img(chart_error_types(df), "error types"))
    if not cats.empty:
        rows = [[html.escape(r.model), r.n, f"{r.hallucination_pct:.1f}%", f"{r.wrong_script_pct:.1f}%",
                 f"{r.empty_pct:.1f}%", r.n_with_digits, _pct(r.wer_with_digits), _pct(r.wer_without_digits)]
                for r in cats.itertuples()]
        parts.append(_table(["Model", "N", "Hallucination", "Wrong script", "Empty output",
                             "Utts w/ digits", "Mean WER w/ digits", "Mean WER w/o digits"], rows))

    # Worst failures
    parts.append("<h2>Worst individual failures</h2>")
    if worst.empty:
        parts.append("<p><em>No data.</em></p>")
    else:
        rows = [[html.escape(r.model), html.escape(r.sample_id), _pct(r.wer, 0),
                 f"<span class='ref'>{html.escape(r.reference_normalized)}</span>",
                 f"<span class='hyp'>{html.escape(r.hypothesis_normalized)}</span>"]
                for r in worst.itertuples()]
        parts.append(_table(["Model", "Sample", "WER", "Reference (normalized)", "Model output (normalized)"], rows))

    # Reproducibility
    parts.append("<h2>Reproducibility</h2><pre>" + html.escape(json.dumps(
        {k: meta.get(k) for k in ["settings", "hardware", "library_versions", "model_runtime"]},
        indent=2, ensure_ascii=False)) + "</pre>")
    n_min = int(summary["n"].min()) if not summary.empty else 0
    parts.append(f"<p class='note'>Smallest cell has {n_min} utterances. Small samples give wide confidence "
                 "intervals: trust differences only when intervals do not overlap or the head-to-head says significant.</p>")
    parts.append("</body></html>")
    return "\n".join(parts)


def build_markdown(df: pd.DataFrame) -> str:
    summary = build_summary(df)
    board = build_leaderboard(summary)
    lines = ["## Leaderboard (macro average over shared languages)", "",
             "| Model | Condition | Languages | Coverage | Macro WER | Macro CER | RTFx | Hallucination | Wrong script |",
             "|---|---|---|---:|---:|---:|---:|---:|---:|"]
    for r in board.itertuples():
        lines.append(f"| {r.model} | {r.condition} | {r.languages} | {_pct(r.coverage)} | {_pct(r.macro_wer)} | {_pct(r.macro_cer)} | "
                     f"{r.rtfx:.2f}x | {_pct(r.hallucination_rate)} | {_pct(r.wrong_script_rate)} |")
    lines += ["", "## Per language (clean audio, 95% CI)", "",
              "| Language | Model | N | WER | 95% CI | CER |", "|---|---|---|---|---|---|"]
    for r in summary[summary["condition"] == "clean"].sort_values(["language", "corpus_wer"]).itertuples():
        lines.append(f"| {r.language.title()} | {r.model} | {r.n} | {_pct(r.corpus_wer)} | "
                     f"{_pct(r.wer_ci_lo)} - {_pct(r.wer_ci_hi)} | {_pct(r.corpus_cer)} |")
    return "\n".join(lines) + "\n"


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--demo", action="store_true", help="use SYNTHETIC data to preview the layout")
    a = p.parse_args()

    if a.demo:
        from src.synthetic import SYNTHETIC_RUNTIME, make_synthetic_results
        df = make_synthetic_results()
        meta = {"settings": {"note": "synthetic"}, "hardware": {"device": "synthetic"},
                "model_runtime": SYNTHETIC_RUNTIME}
        out = Path("report/demo_report_SYNTHETIC.html")
        out.parent.mkdir(exist_ok=True)
        out.write_text(build_html(df, meta, synthetic=True), encoding="utf-8")
        print(f"Wrote {out}  (SYNTHETIC data - layout preview only)")
        return

    from src.run_benchmark import load_existing
    rows = load_existing()
    if not rows:
        raise SystemExit("No results found. Run:  python -m src.run_benchmark --profile quick")
    df = pd.DataFrame(rows)
    meta_path = Path(config.RUN_METADATA_JSON)
    meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
    Path(config.REPORT_HTML).parent.mkdir(exist_ok=True)
    Path(config.REPORT_HTML).write_text(build_html(df, meta), encoding="utf-8")
    Path(config.RESULTS_MD).write_text(build_markdown(df), encoding="utf-8")
    print(f"Wrote {config.REPORT_HTML} and {config.RESULTS_MD}")


if __name__ == "__main__":
    main()
