"""Spread prevalence and partial-ad prevalence plots for the paper.

Outputs (all to paperplots/):
  fig_S01_spread_prevalence_over_time.{pdf,png,csv}
      Fraction of adjacent full-page-ad pairs predicted as two-page spreads,
      per magazine per year. Smoothed with a 5-year rolling mean. One line per
      magazine (magazine colors).

  fig_S02_spread_rate_by_magazine.{pdf,png,csv}
      Overall predicted spread rate per magazine (bar chart).

  fig_S03_partial_ad_fraction_over_time.{pdf,png,csv}
      Fraction of all pages classified as partial ads (Mixed, label=2) per
      magazine per year. Smoothed with a 5-year rolling mean. Complements
      fig_S01 by showing partial-ad presence alongside spread prevalence.

Inputs:
  spread_predictions_v5.csv  — left_page, right_page, spread_prob, pred_spread
  predictions_all.csv        — filepath, pred_label, ad_prob  (for fig_S03)

Usage:
    python -m magazines.scripts.plot_spread_prevalence
    python -m magazines.scripts.plot_spread_prevalence \\
        --spread_csv /data/laviniad/magazines/page_classification_predictions/spread_predictions_v5.csv \\
        --predictions_csv /data/laviniad/magazines/page_classification_predictions/predictions_all.csv \\
        --output_dir paperplots
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SPREAD_CSV = Path(
    "/data/laviniad/magazines/page_classification_predictions/spread_predictions_v5.csv"
)
DEFAULT_PREDICTIONS_CSV = Path(
    "/data/laviniad/magazines/page_classification_predictions/predictions_all.csv"
)
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "paperplots"

MAG_ORDER = ["time", "esquire", "life", "forbes", "variety"]
DISPLAY = {
    "time": "Time", "esquire": "Esquire", "life": "Life",
    "forbes": "Forbes", "variety": "Variety",
}

try:
    from magazineads.analysis.plot_utils import (
        apply_style, fix_fonts, NUM_FONT, MAGAZINE_PALETTE,
    )
except Exception:
    def apply_style() -> None: pass
    def fix_fonts(fig) -> None: pass
    NUM_FONT = "Arial"
    MAGAZINE_PALETTE = {
        "life": "#C03A2B", "time": "#E67E22", "esquire": "#2980B9",
        "forbes": "#27AE60", "variety": "#8E44AD",
    }

DPI = 600
SMOOTH_WINDOW = 5   # years


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_spread(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df["magazine"] = df["left_page"].str.extract(r"/images/([^/]+)/")
    df["year"] = (
        df["left_page"]
        .str.extract(r"/(\d{4})-")[0]
        .astype(float)
        .astype("Int64")
    )
    df = df[df["magazine"].isin(MAG_ORDER)].copy()
    df = df.dropna(subset=["year"])
    return df


# ---------------------------------------------------------------------------
# Fig S01 — spread prevalence over time
# ---------------------------------------------------------------------------

def fig_spread_prevalence_over_time(df: pd.DataFrame, outdir: Path) -> None:
    apply_style()

    # Rate = fraction of pairs predicted as spread, per magazine per year
    rate = (
        df.groupby(["magazine", "year"])["pred_spread"]
        .mean()
        .reset_index(name="spread_rate")
    )

    fig, ax = plt.subplots(figsize=(8, 4))

    for mag in MAG_ORDER:
        grp = rate[rate["magazine"] == mag].sort_values("year")
        if grp.empty:
            continue
        # Fill in missing years so rolling mean doesn't skip gaps
        all_years = pd.DataFrame({"year": range(int(grp["year"].min()), int(grp["year"].max()) + 1)})
        grp = all_years.merge(grp, on="year", how="left")
        smoothed = grp["spread_rate"].rolling(window=SMOOTH_WINDOW, center=True, min_periods=1).mean()
        color = MAGAZINE_PALETTE.get(mag, "#888888")
        ax.plot(
            grp["year"], smoothed,
            label=DISPLAY[mag],
            color=color,
            linewidth=1.8,
        )
        # Faint raw line underneath
        ax.plot(
            grp["year"], grp["spread_rate"],
            color=color, linewidth=0.5, alpha=0.25,
        )

    ax.set_xlabel("Year")
    ax.set_ylabel("Fraction of adjacent pairs\npredicted as spread")
    ax.set_ylim(0, 1.05)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1, decimals=0))
    ax.xaxis.set_major_formatter(mticker.FormatStrFormatter("%d"))
    ax.legend(fontsize=9)

    fix_fonts(fig)
    stem = outdir / "fig_S01_spread_prevalence_over_time"
    fig.savefig(str(stem) + ".pdf", dpi=DPI)
    fig.savefig(str(stem) + ".png", dpi=150)
    plt.close(fig)
    rate.to_csv(str(stem) + ".csv", index=False)
    print(f"Saved {stem}.pdf")


# ---------------------------------------------------------------------------
# Fig S02 — overall spread rate by magazine
# ---------------------------------------------------------------------------

def fig_spread_rate_by_magazine(df: pd.DataFrame, outdir: Path) -> None:
    apply_style()

    rate = (
        df.groupby("magazine")["pred_spread"]
        .agg(n_pairs="count", n_spread="sum")
        .reset_index()
    )
    rate["spread_rate"] = rate["n_spread"] / rate["n_pairs"]
    # Respect MAG_ORDER for x-axis
    rate["_order"] = rate["magazine"].map({m: i for i, m in enumerate(MAG_ORDER)})
    rate = rate.sort_values("_order")

    fig, ax = plt.subplots(figsize=(5, 4))

    colors = [MAGAZINE_PALETTE.get(m, "#888888") for m in rate["magazine"]]
    bars = ax.bar(
        range(len(rate)), rate["spread_rate"],
        color=colors, alpha=0.9, width=0.6,
    )

    for bar, (_, row) in zip(bars, rate.iterrows()):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            row["spread_rate"] + 0.015,
            f"{row['spread_rate']:.0%}\n(n={row['n_pairs']:,})",
            ha="center", va="bottom", fontsize=8,
            fontfamily=NUM_FONT,
        )

    ax.set_xticks(range(len(rate)))
    ax.set_xticklabels([DISPLAY[m] for m in rate["magazine"]], fontsize=11)
    ax.set_ylabel("Predicted spread rate")
    ax.set_ylim(0, 1.18)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1, decimals=0))

    fix_fonts(fig)
    stem = outdir / "fig_S02_spread_rate_by_magazine"
    fig.savefig(str(stem) + ".pdf", dpi=DPI)
    fig.savefig(str(stem) + ".png", dpi=150)
    plt.close(fig)
    rate.drop(columns="_order").to_csv(str(stem) + ".csv", index=False)
    print(f"Saved {stem}.pdf")


# ---------------------------------------------------------------------------
# Fig S03 — partial ad fraction over time
# ---------------------------------------------------------------------------

def fig_partial_ad_fraction_over_time(predictions_csv: Path, outdir: Path) -> None:
    apply_style()

    preds = pd.read_csv(predictions_csv, usecols=["filepath", "pred_label"])
    preds["magazine"] = preds["filepath"].str.extract(r"/images/([^/]+)/")
    preds["year"] = (
        preds["filepath"]
        .str.extract(r"/(\d{4})-")[0]
        .astype(float)
        .astype("Int64")
    )
    preds = preds[preds["magazine"].isin(MAG_ORDER)].dropna(subset=["year"])

    counts = (
        preds.groupby(["magazine", "year", "pred_label"])
        .size()
        .reset_index(name="n")
    )
    totals = preds.groupby(["magazine", "year"]).size().reset_index(name="total")
    mixed = counts[counts["pred_label"] == 2].copy()
    mixed = mixed.merge(totals, on=["magazine", "year"])
    mixed["frac_mixed"] = mixed["n"] / mixed["total"]

    fig, ax = plt.subplots(figsize=(8, 4))

    for mag in MAG_ORDER:
        grp = mixed[mixed["magazine"] == mag].sort_values("year")
        if grp.empty:
            continue
        all_years = pd.DataFrame({"year": range(int(grp["year"].min()), int(grp["year"].max()) + 1)})
        grp = all_years.merge(grp, on="year", how="left")
        smoothed = grp["frac_mixed"].rolling(window=SMOOTH_WINDOW, center=True, min_periods=1).mean()
        color = MAGAZINE_PALETTE.get(mag, "#888888")
        ax.plot(grp["year"], smoothed, label=DISPLAY[mag], color=color, linewidth=1.8)
        ax.plot(grp["year"], grp["frac_mixed"], color=color, linewidth=0.5, alpha=0.25)

    ax.set_xlabel("Year")
    ax.set_ylabel("Fraction of pages\nclassified as partial ad")
    ax.set_ylim(0, None)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1, decimals=0))
    ax.xaxis.set_major_formatter(mticker.FormatStrFormatter("%d"))
    ax.legend(fontsize=9)

    fix_fonts(fig)
    stem = outdir / "fig_S03_partial_ad_fraction_over_time"
    fig.savefig(str(stem) + ".pdf", dpi=DPI)
    fig.savefig(str(stem) + ".png", dpi=150)
    plt.close(fig)
    mixed[["magazine", "year", "n", "total", "frac_mixed"]].to_csv(str(stem) + ".csv", index=False)
    print(f"Saved {stem}.pdf")


# ---------------------------------------------------------------------------
# Fig S04 — partial ads per issue over time
# ---------------------------------------------------------------------------

BB_CSV = Path("/data/laviniad/magazines/bounding_boxes/mixed_page_boxes.csv")


def fig_partial_ads_per_issue(predictions_csv: Path, outdir: Path,
                              bb_csv: Path = BB_CSV) -> None:
    apply_style()

    # Total boxes per issue from the BB inference CSV
    bb = pd.read_csv(bb_csv, usecols=["stem", "magazine", "n_boxes"])
    bb = bb[bb["magazine"].isin(MAG_ORDER)]
    bb["issue"] = bb["stem"].str.extract(r"^(\d{4}-\d{2}-\d{2})")
    bb["year"] = bb["stem"].str.extract(r"^(\d{4})")[0].astype(float).astype("Int64")
    bb = bb.dropna(subset=["year", "issue"])

    # Sum boxes per page, then sum per issue, then average across issues in a year
    boxes_per_issue = (
        bb.groupby(["magazine", "year", "issue", "stem"])["n_boxes"]
        .first()                        # one row per page (take first box_idx row)
        .reset_index()
        .groupby(["magazine", "year", "issue"])["n_boxes"]
        .sum()
        .reset_index(name="total_boxes")
    )

    # Denominator: all issues that exist in predictions_all (not just those with mixed pages)
    preds = pd.read_csv(predictions_csv, usecols=["filepath"])
    preds["magazine"] = preds["filepath"].str.extract(r"/images/([^/]+)/")
    stem_col = preds["filepath"].str.extract(r"/([^/]+)\.jpg$")[0]
    preds["issue"] = stem_col.str.extract(r"^(\d{4}-\d{2}-\d{2})")
    preds["year"] = stem_col.str.extract(r"^(\d{4})")[0].astype(float).astype("Int64")
    all_issues = (
        preds[preds["magazine"].isin(MAG_ORDER)]
        .dropna(subset=["year", "issue"])
        .drop_duplicates(["magazine", "year", "issue"])[["magazine", "year", "issue"]]
    )

    per_issue = all_issues.merge(boxes_per_issue, on=["magazine", "year", "issue"], how="left").fillna({"total_boxes": 0})
    avg = (
        per_issue.groupby(["magazine", "year"])["total_boxes"]
        .mean()
        .reset_index(name="avg_ads_per_issue")
    )

    fig, ax = plt.subplots(figsize=(8, 4))

    for mag in MAG_ORDER:
        grp = avg[avg["magazine"] == mag].sort_values("year")
        if grp.empty:
            continue
        all_years = pd.DataFrame({"year": range(int(grp["year"].min()), int(grp["year"].max()) + 1)})
        grp = all_years.merge(grp, on="year", how="left")
        smoothed = grp["avg_ads_per_issue"].rolling(window=SMOOTH_WINDOW, center=True, min_periods=1).mean()
        color = MAGAZINE_PALETTE.get(mag, "#888888")
        ax.plot(grp["year"], smoothed, label=DISPLAY[mag], color=color, linewidth=1.8)
        ax.plot(grp["year"], grp["avg_ads_per_issue"], color=color, linewidth=0.5, alpha=0.25)

    ax.set_xlabel("Year")
    ax.set_ylabel("Partial ads per issue")
    ax.xaxis.set_major_formatter(mticker.FormatStrFormatter("%d"))
    ax.legend(fontsize=9)

    fix_fonts(fig)
    stem_out = outdir / "fig_S04_partial_ads_per_issue"
    fig.savefig(str(stem_out) + ".pdf", dpi=DPI)
    fig.savefig(str(stem_out) + ".png", dpi=150)
    plt.close(fig)
    avg.to_csv(str(stem_out) + ".csv", index=False)
    print(f"Saved {stem_out}.pdf")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Spread and partial-ad prevalence plots for paper.")
    parser.add_argument(
        "--spread_csv", type=Path, default=DEFAULT_SPREAD_CSV,
        help="Path to spread_predictions_v5.csv",
    )
    parser.add_argument(
        "--predictions_csv", type=Path, default=DEFAULT_PREDICTIONS_CSV,
        help="Path to predictions_all.csv (for fig_S03, fig_S04)",
    )
    parser.add_argument(
        "--bb_csv", type=Path, default=BB_CSV,
        help="Path to mixed_page_boxes.csv (for fig_S04)",
    )
    parser.add_argument(
        "--output_dir", type=Path, default=DEFAULT_OUTPUT_DIR,
        help="Directory for output files (default: paperplots/)",
    )
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    print("Loading spread predictions …")
    df = load_spread(args.spread_csv)
    print(f"  {len(df):,} pairs across {df['magazine'].nunique()} magazines "
          f"({df['pred_spread'].sum():,} predicted spreads)")

    print("Generating fig_S01_spread_prevalence_over_time …")
    fig_spread_prevalence_over_time(df, args.output_dir)

    print("Generating fig_S02_spread_rate_by_magazine …")
    fig_spread_rate_by_magazine(df, args.output_dir)

    print("Generating fig_S03_partial_ad_fraction_over_time …")
    fig_partial_ad_fraction_over_time(args.predictions_csv, args.output_dir)

    print("Generating fig_S04_partial_ads_per_issue …")
    fig_partial_ads_per_issue(args.predictions_csv, args.output_dir, bb_csv=args.bb_csv)


if __name__ == "__main__":
    main()
