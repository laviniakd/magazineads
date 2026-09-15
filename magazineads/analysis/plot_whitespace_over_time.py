"""Plot whitespace decomposition over time for Time and Esquire.

Reads the CSV produced by compute_whitespace.py.  Two plot types:

1. **Stacked area** — shows how the three fractions (whitespace, text, visual)
   compose the full page area over decades, one subplot per magazine.
2. **Whitespace trend** — line plot of whitespace_frac alone with ±1 std band,
   both magazines on a single axis for comparison.

Usage:
    python -m magazines.scripts.plot_whitespace_over_time \
        --csv data/whitespace.csv \
        --out-stacked data/whitespace_stacked.png \
        --out-trend data/whitespace_trend.png
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from magazineads.analysis.plot_utils import (
    apply_style, fix_fonts, MAGAZINE_PALETTE, GENERAL_PALETTE,
)

MAGAZINES = ["time", "esquire"]
MAGAZINE_DISPLAY = {
    "time":    "Time",
    "esquire": "Esquire",
}

# Colors for the three composition bands
COMP_COLORS = {
    "whitespace": "#FFFFFF",       # white (bordered)
    "text":       GENERAL_PALETTE[0],  # dark burgundy
    "visual":     GENERAL_PALETTE[1],  # light blue
}


def _plot_stacked(df: pd.DataFrame, out_path: Path) -> None:
    """Stacked area chart: whitespace + text + visual = 1, per magazine."""
    magazines = [m for m in MAGAZINES if m in df["magazine"].unique()]
    n_mags = len(magazines)

    fig, axes = plt.subplots(1, n_mags, figsize=(4.5 * n_mags, 5), sharey=True)
    if n_mags == 1:
        axes = [axes]

    for ax, mag in zip(axes, magazines):
        sub = df[df["magazine"] == mag]
        agg = sub.groupby("decade")[["whitespace_frac", "text_frac", "visual_frac"]].mean()
        agg = agg.sort_index()
        decades = agg.index.values

        # Stack order bottom-to-top: visual, text, whitespace
        ax.fill_between(decades, 0, agg["visual_frac"],
                        color=COMP_COLORS["visual"], alpha=0.85,
                        label="Visual content")
        ax.fill_between(decades,
                        agg["visual_frac"],
                        agg["visual_frac"] + agg["text_frac"],
                        color=COMP_COLORS["text"], alpha=0.85,
                        label="Text")
        ax.fill_between(decades,
                        agg["visual_frac"] + agg["text_frac"],
                        1.0,
                        color=COMP_COLORS["whitespace"], alpha=0.85,
                        edgecolor="#CCCCCC", linewidth=0.8,
                        label="Whitespace")

        mag_color = MAGAZINE_PALETTE.get(mag, "#888888")
        ax.set_title(MAGAZINE_DISPLAY.get(mag, mag), fontsize=13,
                     fontweight="bold", color=mag_color)
        ax.set_xlabel("Decade", fontsize=10)
        if ax is axes[0]:
            ax.set_ylabel("Fraction of page area", fontsize=10)
        ax.set_ylim(0, 1)
        ax.set_xlim(decades.min() - 2, decades.max() + 2)
        ax.yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1.0))
        ax.tick_params(labelsize=9)

        # Sample count annotation
        n_per_decade = sub.groupby("decade").size()
        for dec, n in n_per_decade.items():
            ax.text(dec, 0.01, f"n={n}", ha="center", va="bottom",
                    fontsize=6, color="#AAAAAA", rotation=90)

    # Shared legend
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3,
               fontsize=9, bbox_to_anchor=(0.5, -0.06),
               framealpha=0.9, edgecolor="#CCCCCC")

    plt.tight_layout()
    fix_fonts(fig)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, bbox_inches="tight", dpi=200)
    plt.close(fig)
    print(f"Saved stacked plot: {out_path}")


def _plot_trend(df: pd.DataFrame, out_path: Path) -> None:
    """Line plot of whitespace_frac over time, one line per magazine."""
    magazines = [m for m in MAGAZINES if m in df["magazine"].unique()]

    fig, ax = plt.subplots(figsize=(10, 5.5))

    for mag in magazines:
        sub = df[df["magazine"] == mag]
        agg = sub.groupby("decade")["whitespace_frac"].agg(["mean", "std", "count"]).reset_index()
        decades = agg["decade"].values
        mu = agg["mean"].values
        se = (agg["std"].fillna(0) / np.sqrt(agg["count"])).values
        color = MAGAZINE_PALETTE.get(mag, "#888888")

        ax.plot(decades, mu, color=color, lw=2.2,
                label=MAGAZINE_DISPLAY.get(mag, mag), zorder=3)
        ax.fill_between(decades, mu - 1.96 * se, mu + 1.96 * se,
                        color=color, alpha=0.15, zorder=1)

    ax.set_xlabel("Decade", fontsize=11)
    ax.set_ylabel("Whitespace fraction", fontsize=11)
    ax.yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1.0))
    ax.legend(fontsize=9, framealpha=0.9, edgecolor="#CCCCCC",
              loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=2)
    ax.tick_params(labelsize=9)

    plt.tight_layout()
    fix_fonts(fig)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, bbox_inches="tight", dpi=200)
    plt.close(fig)
    print(f"Saved trend plot: {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot whitespace decomposition over time."
    )
    parser.add_argument("--csv", required=True, help="Input CSV from compute_whitespace.py")
    parser.add_argument("--out-stacked", required=True, help="Output path for stacked area chart")
    parser.add_argument("--out-trend", required=True, help="Output path for trend line chart")
    args = parser.parse_args()

    apply_style()

    df = pd.read_csv(args.csv)
    df = df[df["magazine"].isin(MAGAZINES)]
    print(f"Loaded {len(df):,} rows from {args.csv} (Time + Esquire)")

    _plot_stacked(df, Path(args.out_stacked))
    _plot_trend(df, Path(args.out_trend))


if __name__ == "__main__":
    main()
