"""Ad-type composition over time, per magazine.

For each magazine, shows the fraction of ad pages that are partial (mixed),
single full-page, or two-page spread, by year with a 5-year rolling mean.
One subplot per magazine; stacked area chart using GENERAL_PALETTE.

Outputs (all to paperplots/):
  fig_S06_ad_type_composition.{pdf,png,csv}
  fig_S06b_ad_type_counts.{pdf,png,csv}

Inputs:
  predictions_all.csv        — filepath, pred_label  (1=full-page, 2=mixed)
  spread_predictions_v5.csv  — left_page, right_page, spread_prob, pred_spread

Usage:
    python -m magazines.scripts.plot_ad_type_composition
    python -m magazines.scripts.plot_ad_type_composition \
        --predictions_csv /data/laviniad/magazines/page_classification_predictions/predictions_all.csv \
        --spread_csv /data/laviniad/magazines/page_classification_predictions/spread_predictions_v5.csv \
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
DEFAULT_PREDICTIONS_CSV = Path(
    "/data/laviniad/magazines/page_classification_predictions/predictions_all.csv"
)
DEFAULT_SPREAD_CSV = Path(
    "/data/laviniad/magazines/page_classification_predictions/spread_predictions_v5.csv"
)
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "paperplots"

MAG_ORDER = ["time", "esquire", "life", "forbes", "variety"]
DISPLAY = {
    "time": "Time", "esquire": "Esquire", "life": "Life",
    "forbes": "Forbes", "variety": "Variety",
}

SMOOTH_WINDOW = 5  # years
DPI = 600

try:
    from magazineads.analysis.plot_utils import (
        apply_style, fix_fonts, NUM_FONT, GENERAL_PALETTE,
    )
except Exception:
    def apply_style() -> None: pass
    def fix_fonts(fig) -> None: pass
    NUM_FONT = "Arial"
    GENERAL_PALETTE = ["#542344", "#bfd1e5", "#d8bfaa", "#ebf5ee", "#808080"]

# Ad-type colors from GENERAL_PALETTE:
#   [0] dark burgundy  → two-page spread (most visually distinctive category)
#   [1] light blue     → single full-page
#   [2] tan            → partial (mixed)
TYPE_COLORS = {
    "spread": GENERAL_PALETTE[0],
    "single_fullpage": GENERAL_PALETTE[1],
    "partial": GENERAL_PALETTE[2],
}
TYPE_LABELS = {
    "spread": "Two-page spread",
    "single_fullpage": "Single full-page",
    "partial": "Partial ad",
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _extract_magazine_year(filepath_series: pd.Series) -> pd.DataFrame:
    """Extract magazine name and year from filepath column."""
    magazine = filepath_series.str.extract(r"/images/([^/]+)/")[0]
    year = filepath_series.str.extract(r"/(\d{4})-")[0].astype(float).astype("Int64")
    return pd.DataFrame({"magazine": magazine, "year": year})


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_data(predictions_csv: Path, spread_csv: Path) -> pd.DataFrame:
    """Build a per-page DataFrame with columns: magazine, year, ad_type.

    ad_type is one of: 'spread', 'single_fullpage', 'partial'.
    Only pages that are ads (pred_label in {1, 2}) are included.
    """
    # --- predictions ---
    preds = pd.read_csv(predictions_csv, usecols=["filepath", "pred_label"])
    preds = preds[preds["pred_label"].isin([1, 2])].copy()
    meta = _extract_magazine_year(preds["filepath"])
    preds["magazine"] = meta["magazine"]
    preds["year"] = meta["year"]
    preds = preds[preds["magazine"].isin(MAG_ORDER)].dropna(subset=["year"]).copy()

    # --- spread predictions: identify pages that belong to a predicted spread ---
    spreads = pd.read_csv(spread_csv, usecols=["left_page", "right_page", "pred_spread"])
    spreads = spreads[spreads["pred_spread"] == 1]
    spread_pages: set[str] = set(spreads["left_page"]) | set(spreads["right_page"])

    # --- assign ad_type ---
    def _ad_type(row):
        if row["pred_label"] == 2:
            return "partial"
        # pred_label == 1 (full-page ad)
        if row["filepath"] in spread_pages:
            return "spread"
        return "single_fullpage"

    preds["ad_type"] = preds.apply(_ad_type, axis=1)
    return preds[["magazine", "year", "ad_type"]].copy()


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------

def plot_ad_type_composition(df: pd.DataFrame, outdir: Path) -> pd.DataFrame:
    apply_style()

    # Count pages per (magazine, year, ad_type), then pivot to fractions
    counts = (
        df.groupby(["magazine", "year", "ad_type"])
        .size()
        .reset_index(name="n")
    )
    totals = df.groupby(["magazine", "year"]).size().reset_index(name="total")
    counts = counts.merge(totals, on=["magazine", "year"])
    counts["frac"] = counts["n"] / counts["total"]

    # One subplot per magazine
    fig, axes = plt.subplots(
        1, len(MAG_ORDER), figsize=(3.2 * len(MAG_ORDER), 4),
        sharey=True,
    )

    type_order = ["spread", "single_fullpage", "partial"]  # bottom to top
    all_csv_rows = []

    for ax, mag in zip(axes, MAG_ORDER):
        mag_data = counts[counts["magazine"] == mag]
        if mag_data.empty:
            ax.set_visible(False)
            continue

        year_min = int(mag_data["year"].min())
        year_max = int(mag_data["year"].max())
        all_years = np.arange(year_min, year_max + 1)

        # Build matrix: rows = years, cols = ad types (fractions)
        fracs = {}
        for ad_type in type_order:
            sub = mag_data[mag_data["ad_type"] == ad_type].set_index("year")["frac"]
            series = sub.reindex(all_years).fillna(0)
            smoothed = series.rolling(
                window=SMOOTH_WINDOW, center=True, min_periods=1
            ).mean()
            fracs[ad_type] = smoothed.values

        # Renormalize so stacked areas sum to 1 after smoothing
        total = sum(fracs.values())
        total = np.where(total == 0, 1, total)  # avoid division by zero
        for ad_type in type_order:
            fracs[ad_type] = fracs[ad_type] / total

        # Stacked area
        bottom = np.zeros(len(all_years))
        for ad_type in type_order:
            ax.fill_between(
                all_years, bottom, bottom + fracs[ad_type],
                color=TYPE_COLORS[ad_type],
                label=TYPE_LABELS[ad_type],
                alpha=0.85,
                linewidth=0.5,
                edgecolor="white",
            )
            bottom = bottom + fracs[ad_type]

        ax.set_xlim(year_min, year_max)
        ax.set_ylim(0, 1)
        ax.set_xlabel("Year")
        ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1, decimals=0))
        ax.xaxis.set_major_formatter(mticker.FormatStrFormatter("%d"))
        ax.xaxis.set_major_locator(mticker.MultipleLocator(20))
        ax.text(
            0.5, 0.97, DISPLAY[mag],
            transform=ax.transAxes, ha="center", va="top",
            fontsize=12, fontweight="bold",
        )

        # CSV rows (raw, unsmoothed fractions)
        for ad_type in type_order:
            sub = mag_data[mag_data["ad_type"] == ad_type]
            for _, row in sub.iterrows():
                all_csv_rows.append({
                    "magazine": mag,
                    "year": int(row["year"]),
                    "ad_type": ad_type,
                    "n": int(row["n"]),
                    "total": int(row["total"]),
                    "frac": row["frac"],
                })

    axes[0].set_ylabel("Fraction of ad pages")

    # Single legend at the bottom
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles, labels,
        loc="lower center",
        ncol=len(type_order),
        fontsize=10,
        bbox_to_anchor=(0.5, -0.02),
        frameon=False,
    )

    fig.tight_layout(rect=[0, 0.06, 1, 1])
    fix_fonts(fig)

    stem = outdir / "fig_S06_ad_type_composition"
    fig.savefig(str(stem) + ".pdf", dpi=DPI)
    fig.savefig(str(stem) + ".png", dpi=150)
    plt.close(fig)

    csv_df = pd.DataFrame(all_csv_rows)
    csv_df.to_csv(str(stem) + ".csv", index=False)
    print(f"Saved {stem}.pdf")

    return csv_df


# ---------------------------------------------------------------------------
# Count version (S06b)
# ---------------------------------------------------------------------------

def plot_ad_type_counts(df: pd.DataFrame, outdir: Path) -> None:
    apply_style()

    counts = (
        df.groupby(["magazine", "year", "ad_type"])
        .size()
        .reset_index(name="n")
    )

    type_order = ["spread", "single_fullpage", "partial"]

    fig, axes = plt.subplots(
        1, len(MAG_ORDER), figsize=(3.2 * len(MAG_ORDER), 4),
        sharey=True,
    )

    csv_rows = []

    for ax, mag in zip(axes, MAG_ORDER):
        mag_data = counts[counts["magazine"] == mag]
        if mag_data.empty:
            ax.set_visible(False)
            continue

        year_min = int(mag_data["year"].min())
        year_max = int(mag_data["year"].max())
        all_years = np.arange(year_min, year_max + 1)

        smoothed = {}
        for ad_type in type_order:
            sub = mag_data[mag_data["ad_type"] == ad_type].set_index("year")["n"]
            series = sub.reindex(all_years).fillna(0)
            smoothed[ad_type] = series.rolling(
                window=SMOOTH_WINDOW, center=True, min_periods=1
            ).mean().values

        bottom = np.zeros(len(all_years))
        for ad_type in type_order:
            ax.fill_between(
                all_years, bottom, bottom + smoothed[ad_type],
                color=TYPE_COLORS[ad_type],
                label=TYPE_LABELS[ad_type],
                alpha=0.85,
                linewidth=0.5,
                edgecolor="white",
            )
            bottom = bottom + smoothed[ad_type]

        ax.set_xlim(year_min, year_max)
        ax.set_xlabel("Year")
        ax.xaxis.set_major_formatter(mticker.FormatStrFormatter("%d"))
        ax.xaxis.set_major_locator(mticker.MultipleLocator(20))
        ax.text(
            0.5, 0.97, DISPLAY[mag],
            transform=ax.transAxes, ha="center", va="top",
            fontsize=12, fontweight="bold",
        )

        for ad_type in type_order:
            sub = mag_data[mag_data["ad_type"] == ad_type]
            for _, row in sub.iterrows():
                csv_rows.append({
                    "magazine": mag,
                    "year": int(row["year"]),
                    "ad_type": ad_type,
                    "n": int(row["n"]),
                })

    axes[0].set_ylabel("Ad pages per year")

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles, labels,
        loc="lower center",
        ncol=len(type_order),
        fontsize=10,
        bbox_to_anchor=(0.5, -0.02),
        frameon=False,
    )

    fig.tight_layout(rect=[0, 0.06, 1, 1])
    fix_fonts(fig)

    stem = outdir / "fig_S06b_ad_type_counts"
    fig.savefig(str(stem) + ".pdf", dpi=DPI)
    fig.savefig(str(stem) + ".png", dpi=150)
    plt.close(fig)

    pd.DataFrame(csv_rows).to_csv(str(stem) + ".csv", index=False)
    print(f"Saved {stem}.pdf")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Ad-type composition (partial / single full-page / spread) over time."
    )
    parser.add_argument(
        "--predictions_csv", type=Path, default=DEFAULT_PREDICTIONS_CSV,
        help="Path to predictions_all.csv",
    )
    parser.add_argument(
        "--spread_csv", type=Path, default=DEFAULT_SPREAD_CSV,
        help="Path to spread_predictions_v5.csv",
    )
    parser.add_argument(
        "--output_dir", type=Path, default=DEFAULT_OUTPUT_DIR,
        help="Directory for output files (default: paperplots/)",
    )
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    print("Loading data ...")
    df = load_data(args.predictions_csv, args.spread_csv)
    print(f"  {len(df):,} ad pages across {df['magazine'].nunique()} magazines")
    for t in ["spread", "single_fullpage", "partial"]:
        n = (df["ad_type"] == t).sum()
        print(f"    {t}: {n:,} ({n/len(df):.1%})")

    print("Generating fig_S06_ad_type_composition ...")
    plot_ad_type_composition(df, args.output_dir)

    print("Generating fig_S06b_ad_type_counts ...")
    plot_ad_type_counts(df, args.output_dir)


if __name__ == "__main__":
    main()
