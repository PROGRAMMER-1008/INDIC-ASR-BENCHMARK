"""
Generates a single self-contained HTML report from results.csv + summary.csv.

Charts are rendered with matplotlib and embedded as base64 PNGs directly in
the HTML -- no external JS/CSS CDN, no server, no internet connection needed
to view the report after generation. Open report/report.html in any browser.

This module WAS tested end-to-end in this session using synthetic
placeholder data (see the __main__ block and results/README_TEST_DATA.md)
to confirm the HTML actually renders correctly, since it has no dependency
on huggingface.co. When you run the real benchmark, this script will consume
the real results.csv/summary.csv and produce a report with real numbers.
"""

import base64
import io

import matplotlib
matplotlib.use("Agg")  # no display backend needed
import matplotlib.pyplot as plt
import pandas as pd

from src.config import REPORT_HTML, RESULTS_CSV, SUMMARY_CSV


def _fig_to_base64(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=120)
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("utf-8")


def _wer_chart(summary_df: pd.DataFrame) -> str:
    fig, ax = plt.subplots(figsize=(7, 4))
    pivot = summary_df.pivot(index="language", columns="model", values="mean_wer")
    pivot.plot(kind="bar", ax=ax)
    ax.set_ylabel("Mean WER")
    ax.set_title("Word Error Rate by Model / Language")
    ax.legend(title="Model")
    plt.xticks(rotation=0)
    return _fig_to_base64(fig)


def _latency_chart(summary_df: pd.DataFrame) -> str:
    fig, ax = plt.subplots(figsize=(7, 4))
    latency_by_model = summary_df.groupby("model")["mean_latency_seconds"].mean()
    latency_by_model.plot(kind="bar", ax=ax, color="darkorange")
    ax.set_ylabel("Mean Latency (seconds/sample)")
    ax.set_title("Inference Latency by Model")
    plt.xticks(rotation=0)
    return _fig_to_base64(fig)


def _failures_table_html(failures_df: pd.DataFrame) -> str:
    if failures_df.empty:
        return "<p><em>No failure data available.</em></p>"
    rows = []
    for _, row in failures_df.iterrows():
        rows.append(
            f"<tr>"
            f"<td>{row['model']}</td>"
            f"<td>{row['language']}</td>"
            f"<td>{row['wer']:.2f}</td>"
            f"<td>{row['reference']}</td>"
            f"<td>{row['hypothesis']}</td>"
            f"</tr>"
        )
    return f"""
    <table>
      <thead>
        <tr><th>Model</th><th>Language</th><th>WER</th><th>Reference</th><th>Hypothesis</th></tr>
      </thead>
      <tbody>
        {''.join(rows)}
      </tbody>
    </table>
    """


def _summary_table_html(summary_df: pd.DataFrame) -> str:
    rows = []
    for _, row in summary_df.iterrows():
        rows.append(
            f"<tr>"
            f"<td>{row['model']}</td>"
            f"<td>{row['language']}</td>"
            f"<td>{row['mean_wer']:.3f}</td>"
            f"<td>{row['mean_cer']:.3f}</td>"
            f"<td>{row['mean_latency_seconds']:.2f}s</td>"
            f"<td>{int(row['n_samples'])}</td>"
            f"</tr>"
        )
    return f"""
    <table>
      <thead>
        <tr><th>Model</th><th>Language</th><th>Mean WER</th><th>Mean CER</th><th>Mean Latency</th><th>N Samples</th></tr>
      </thead>
      <tbody>
        {''.join(rows)}
      </tbody>
    </table>
    """


HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Indic ASR Benchmark Report</title>
<style>
  body {{ font-family: -apple-system, Segoe UI, Roboto, sans-serif; max-width: 960px; margin: 40px auto; padding: 0 20px; color: #1a1a1a; }}
  h1 {{ font-size: 28px; margin-bottom: 4px; }}
  h2 {{ font-size: 20px; margin-top: 40px; border-bottom: 2px solid #eee; padding-bottom: 6px; }}
  .subtitle {{ color: #666; margin-top: 0; }}
  table {{ border-collapse: collapse; width: 100%; margin-top: 16px; font-size: 14px; }}
  th, td {{ border: 1px solid #ddd; padding: 8px 10px; text-align: left; vertical-align: top; }}
  th {{ background: #f5f5f5; }}
  tr:nth-child(even) {{ background: #fafafa; }}
  img {{ max-width: 100%; margin-top: 16px; }}
  .charts {{ display: flex; gap: 24px; flex-wrap: wrap; }}
  .chart-box {{ flex: 1; min-width: 320px; }}
</style>
</head>
<body>
  <h1>Indic ASR Benchmark Report</h1>
  <p class="subtitle">Comparing open-source ASR models on Indian language speech (FLEURS benchmark)</p>

  <h2>Summary: Model &times; Language</h2>
  {summary_table}

  <h2>Charts</h2>
  <div class="charts">
    <div class="chart-box">
      <img src="data:image/png;base64,{wer_chart}" alt="WER by model and language">
    </div>
    <div class="chart-box">
      <img src="data:image/png;base64,{latency_chart}" alt="Latency by model">
    </div>
  </div>

  <h2>Worst Failures (highest WER per model)</h2>
  {failures_table}

</body>
</html>
"""


def generate_report(results_csv: str = RESULTS_CSV, summary_csv: str = SUMMARY_CSV, output_path: str = REPORT_HTML):
    df = pd.read_csv(results_csv)
    summary_df = pd.read_csv(summary_csv)

    scoreable = df[df["wer"].notna()]
    failures_df = (
        scoreable.sort_values("wer", ascending=False)
        .groupby("model")
        .head(5)
        .reset_index(drop=True)
    )

    html = HTML_TEMPLATE.format(
        summary_table=_summary_table_html(summary_df),
        wer_chart=_wer_chart(summary_df),
        latency_chart=_latency_chart(summary_df),
        failures_table=_failures_table_html(failures_df),
    )

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"[generate_report] Wrote report to {output_path}")


if __name__ == "__main__":
    generate_report()
