"""
descriptive_stats.py

Plots basic descriptive statistics for each magazine in the dataset:
  - Number of issues
  - Issue frequency (average issues per year)
  - Temporal coverage (start year – end year)

Also produces a LaTeX table summarising all of the above.

Outputs (all written to --output_dir):
  coverage_gantt.png      – Gantt-style temporal-coverage chart with per-year issue counts
  summary_bars.png        – Side-by-side bar charts (total issues, avg issues/year)
  dataset_summary.tex     – LaTeX booktabs table

Usage:
    python magazines/scripts/descriptive_stats.py
    python magazines/scripts/descriptive_stats.py --output_dir meeting_results/plots
"""

import argparse
import re
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "meeting_results" / "plots"

ISSUE_META_CSV = PROJECT_ROOT / "data" / "issue_metadata_df.csv"

# Predictions CSV (needed for Esquire, which lacks pre-built issue metadata)
try:
    from magazineads.paths import paths
    PREDICTIONS_CSV = paths.predictions_dir / "predictions_all.csv"
except Exception:
    PREDICTIONS_CSV = Path("/data/laviniad/magazines/page_classification_predictions/predictions_all.csv")

# ---------------------------------------------------------------------------
# Display names & colours
# ---------------------------------------------------------------------------
DISPLAY = {
    "forbes":  "Forbes",
    "life":    "Life",
    "time":    "Time",
    "variety": "Variety",
    "esquire": "Esquire",
}

# Order for all plots (chronological first-issue)
MAG_ORDER = ["time", "esquire", "life", "forbes", "variety"]

COLOURS = {
    "forbes":  "#2166AC",
    "life":    "#D6604D",
    "time":    "#4DAC26",
    "variety": "#762A83",
    "esquire": "#E08214",
}

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_issue_meta_magazines() -> pd.DataFrame:
    """Load Forbes/Life/Time/Variety from the pre-built issue metadata CSV."""
    df = pd.read_csv(ISSUE_META_CSV, parse_dates=["date"])
    # one row = one issue
    return df[["magazine", "year", "num_pages"]].copy()


def load_esquire_issues() -> pd.DataFrame:
    """Derive Esquire issues from the predictions CSV (no issue metadata CSV exists)."""
    preds = pd.read_csv(PREDICTIONS_CSV, usecols=["filepath"])
    esq = preds[preds["filepath"].str.contains("/esquire/")].copy()
    esq["date_str"] = esq["filepath"].str.extract(r"/esquire/(\d{4}-\d{2}-\d{2})")
    # Each unique date = one issue
    issues = (
        esq.dropna(subset=["date_str"])
           .groupby("date_str")
           .size()
           .reset_index(name="num_pages")
    )
    issues["year"] = issues["date_str"].str[:4].astype(int)
    issues["magazine"] = "esquire"
    return issues[["magazine", "year", "num_pages"]]


def build_issue_table() -> pd.DataFrame:
    """Return one row per issue with (magazine, year, num_pages)."""
    meta = load_issue_meta_magazines()
    esq  = load_esquire_issues()
    return pd.concat([meta, esq], ignore_index=True)


# ---------------------------------------------------------------------------
# Summary statistics
# ---------------------------------------------------------------------------

def compute_summary(issues: pd.DataFrame) -> pd.DataFrame:
    """Return a per-magazine summary DataFrame."""
    rows = []
    for mag in MAG_ORDER:
        sub = issues[issues["magazine"] == mag]
        if sub.empty:
            continue
        n_issues    = len(sub)
        start_year  = int(sub["year"].min())
        end_year    = int(sub["year"].max())
        years_span  = end_year - start_year + 1
        avg_per_yr  = n_issues / years_span
        total_pages = int(sub["num_pages"].sum())
        rows.append({
            "magazine":    mag,
            "display":     DISPLAY[mag],
            "start_year":  start_year,
            "end_year":    end_year,
            "years_span":  years_span,
            "n_issues":    n_issues,
            "avg_per_yr":  avg_per_yr,
            "total_pages": total_pages,
        })
    return pd.DataFrame(rows)


def compute_issues_per_year(issues: pd.DataFrame) -> pd.DataFrame:
    """Return (magazine, year, count) for every magazine × year combination."""
    return (
        issues.groupby(["magazine", "year"])
              .size()
              .reset_index(name="count")
    )


# ---------------------------------------------------------------------------
# Plot 1: Summary bar charts
# ---------------------------------------------------------------------------

def plot_summary_bars(summary: pd.DataFrame, output_dir: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    labels   = [DISPLAY[m] for m in summary["magazine"]]
    colours  = [COLOURS[m]  for m in summary["magazine"]]
    x        = np.arange(len(labels))
    bar_w    = 0.55

    # --- total issues ---
    ax = axes[0]
    bars = ax.bar(x, summary["n_issues"], width=bar_w, color=colours, edgecolor="white",
                  linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=11)
    ax.set_ylabel("Number of issues", fontsize=11)
    ax.set_title("Total issues per magazine", fontsize=12, fontweight="bold")
    ax.spines[["top", "right"]].set_visible(False)
    ax.yaxis.grid(True, linestyle="--", alpha=0.5)
    ax.set_axisbelow(True)
    for bar, v in zip(bars, summary["n_issues"]):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 20,
                f"{v:,}", ha="center", va="bottom", fontsize=9)

    # --- avg issues / year ---
    ax = axes[1]
    bars = ax.bar(x, summary["avg_per_yr"], width=bar_w, color=colours, edgecolor="white",
                  linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=11)
    ax.set_ylabel("Average issues per year", fontsize=11)
    ax.set_title("Issue frequency (avg issues / year)", fontsize=12, fontweight="bold")
    ax.spines[["top", "right"]].set_visible(False)
    ax.yaxis.grid(True, linestyle="--", alpha=0.5)
    ax.set_axisbelow(True)
    for bar, v in zip(bars, summary["avg_per_yr"]):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                f"{v:.1f}", ha="center", va="bottom", fontsize=9)

    fig.tight_layout(pad=2.0)
    out = output_dir / "summary_bars.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {out}")


# ---------------------------------------------------------------------------
# Plot 2: Gantt-style coverage chart with per-year issue counts
# ---------------------------------------------------------------------------

def plot_coverage_gantt(summary: pd.DataFrame, ipy: pd.DataFrame, output_dir: Path) -> None:
    """Gantt rows, one per magazine, filled with per-year issue-count heatmap bars."""
    mags     = list(summary["magazine"])
    n_mags   = len(mags)
    all_years = np.arange(ipy["year"].min(), ipy["year"].max() + 1)

    fig, ax = plt.subplots(figsize=(14, 4))

    row_h   = 0.7   # height of each magazine strip
    gap     = 0.35  # vertical gap between strips
    y_step  = row_h + gap

    # Normalise issue counts within each magazine for shading intensity
    max_counts = {
        m: ipy[ipy["magazine"] == m]["count"].max()
        for m in mags
    }

    for i, mag in enumerate(mags):
        y_bottom = i * y_step
        sub = ipy[ipy["magazine"] == mag].set_index("year")["count"]
        smeta = summary[summary["magazine"] == mag].iloc[0]

        # Background span (light grey) for the full covered range
        ax.barh(y_bottom + row_h / 2,
                smeta["end_year"] - smeta["start_year"] + 1,
                left=smeta["start_year"],
                height=row_h,
                color="#e8e8e8", edgecolor="none", zorder=1)

        # Per-year bars, height proportional to issue count
        base_col = np.array(plt.matplotlib.colors.to_rgb(COLOURS[mag]))
        for yr in all_years:
            if yr < smeta["start_year"] or yr > smeta["end_year"]:
                continue
            cnt = sub.get(yr, 0)
            if cnt == 0:
                continue
            alpha = 0.25 + 0.75 * (cnt / max_counts[mag])
            ax.bar(yr, row_h, bottom=y_bottom,
                   width=0.85, color=base_col, alpha=float(alpha),
                   edgecolor="none", zorder=2)

        # Label on the left
        ax.text(all_years[0] - 1.5, y_bottom + row_h / 2,
                DISPLAY[mag], ha="right", va="center", fontsize=11, fontweight="bold",
                color=COLOURS[mag])

    # Axis formatting
    ax.set_xlim(all_years[0] - 8, all_years[-1] + 2)
    ax.set_ylim(-gap, n_mags * y_step)
    ax.set_yticks([])
    decade_ticks = np.arange((all_years[0] // 10) * 10, all_years[-1] + 1, 10)
    ax.set_xticks(decade_ticks)
    ax.set_xticklabels([str(y) for y in decade_ticks], fontsize=10)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.xaxis.grid(True, linestyle="--", alpha=0.3, zorder=0)
    ax.set_axisbelow(True)
    ax.set_title("Temporal coverage and issue frequency by magazine\n"
                 "(bar opacity ~ issues that year)", fontsize=12, fontweight="bold")

    fig.tight_layout()
    out = output_dir / "coverage_gantt.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {out}")


# ---------------------------------------------------------------------------
# LaTeX table
# ---------------------------------------------------------------------------

LATEX_TEMPLATE = r"""\begin{{table}}[h]
\centering
\caption{{Descriptive statistics for each magazine corpus.
  \emph{{Issues/yr}} is computed as total issues divided by the number of years
  in the covered range (inclusive).}}
\label{{tab:corpus-stats}}
\begin{{tabular}}{{lrrrrr}}
\toprule
\textbf{{Magazine}} & \textbf{{Start}} & \textbf{{End}}
  & \textbf{{Issues}} & \textbf{{Issues/yr}} & \textbf{{Pages}} \\
\midrule
{rows}
\bottomrule
\end{{tabular}}
\end{{table}}
"""


def write_latex_table(summary: pd.DataFrame, output_dir: Path) -> None:
    row_strs = []
    for _, r in summary.iterrows():
        row_strs.append(
            f"{r['display']} & {r['start_year']} & {r['end_year']} "
            f"& {r['n_issues']:,} & {r['avg_per_yr']:.1f} "
            f"& {r['total_pages']:,} \\\\"
        )
    latex = LATEX_TEMPLATE.format(rows="\n".join(row_strs))
    out = output_dir / "dataset_summary.tex"
    out.write_text(latex)
    print(f"  saved {out}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT_DIR,
                   help="Directory for output plots and LaTeX table "
                        f"(default: {DEFAULT_OUTPUT_DIR})")
    p.add_argument("--issue_meta_csv", type=Path, default=ISSUE_META_CSV,
                   help="Path to issue_metadata_df.csv")
    p.add_argument("--predictions_csv", type=Path, default=PREDICTIONS_CSV,
                   help="Path to predictions_all.csv (used for Esquire)")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    global ISSUE_META_CSV, PREDICTIONS_CSV
    ISSUE_META_CSV   = args.issue_meta_csv
    PREDICTIONS_CSV  = args.predictions_csv

    args.output_dir.mkdir(parents=True, exist_ok=True)

    print("Loading issue data …")
    issues  = build_issue_table()
    summary = compute_summary(issues)
    ipy     = compute_issues_per_year(issues)

    print("\nSummary statistics:")
    print(summary[["display", "start_year", "end_year", "years_span",
                    "n_issues", "avg_per_yr", "total_pages"]].to_string(index=False))

    print("\nGenerating plots …")
    plot_summary_bars(summary, args.output_dir)
    plot_coverage_gantt(summary, ipy, args.output_dir)

    print("\nWriting LaTeX table …")
    write_latex_table(summary, args.output_dir)

    print("\nDone.")


if __name__ == "__main__":
    main()
