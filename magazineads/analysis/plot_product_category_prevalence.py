"""Plot product category prevalence by magazine and over time.

Reads corpus-wide predictions and produces:
  1. Stacked bar chart: category share by magazine
  2. Line plots: top categories over time (5-year bins)
  3. Per-magazine small multiples: top categories over time

Usage:
    python -m magazines.scripts.plot_product_category_prevalence \
        --predictions results/product_category/corpus_predictions/product_category_predictions.csv \
        --out meeting_results/jun17 \
        --min_confidence 0.5
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd

from magazineads.analysis.plot_utils import (
    apply_style, fix_fonts,
    MAGAZINE_PALETTE, GENERAL_PALETTE,
)


# Top-level categories to highlight (rest grouped as "Other")
TOP_N = 10
# More categories for diachronic plots (+ Other line)
DIACHRONIC_N = 15

# Warm categorical palette for product categories (distinct from magazine palette)
CATEGORY_COLORS = [
    "#C05C58",  # dusty coral
    "#5C9E88",  # sage
    "#C4963C",  # mustard
    "#28788C",  # teal
    "#542344",  # burgundy
    "#B5A080",  # taupe
    "#7B68A0",  # muted purple
    "#D4816B",  # salmon
    "#4A7C59",  # forest
    "#8B6C42",  # brown
    "#E88D67",  # peach
    "#3D5A80",  # steel blue
    "#A44A3F",  # brick
    "#6B8E6B",  # olive green
    "#C9A959",  # gold
    "#999999",  # grey (Other)
]

MAGAZINE_ORDER = ["life", "time", "esquire", "forbes", "variety"]
MAGAZINE_DISPLAY = {
    "life": "Life", "time": "Time", "esquire": "Esquire",
    "forbes": "Forbes", "variety": "Variety",
}


def load_and_filter(path: Path, min_conf: float) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["year"] = df["unit_id"].apply(lambda s: int(s.split("-")[0]))
    if min_conf > 0:
        low_conf = df["confidence"] < min_conf
        n_low = low_conf.sum()
        df.loc[low_conf, "predicted_category"] = "Other"
        print(f"Reclassified {n_low}/{len(df)} ({n_low/len(df)*100:.1f}%) "
              f"low-confidence predictions (< {min_conf}) as 'Other'")
    return df


def get_top_categories(df: pd.DataFrame, n: int = TOP_N) -> list[str]:
    counts = df["predicted_category"].value_counts()
    counts = counts.drop("Other", errors="ignore")
    return counts.head(n).index.tolist()


def plot_by_magazine(df: pd.DataFrame, top_cats: list[str], out: Path,
                     paper: bool = False):
    """Stacked horizontal bar: category share within each magazine.

    When *paper* is True, follows PLOTS.md: no title, 600 DPI PDF,
    companion CSV, Canela + Arial fonts.
    """
    fig, ax = plt.subplots(figsize=(10, 4))

    mags = [m for m in MAGAZINE_ORDER if m in df["magazine"].unique()]
    cat_plus_other = top_cats + ["Other"]
    df2 = df.copy()
    df2.loc[~df2["predicted_category"].isin(top_cats), "predicted_category"] = "Other"

    # Compute proportions
    ct = pd.crosstab(df2["magazine"], df2["predicted_category"], normalize="index")
    ct = ct.reindex(index=mags, columns=cat_plus_other, fill_value=0)

    lefts = np.zeros(len(mags))
    for i, cat in enumerate(cat_plus_other):
        vals = ct[cat].values
        color = CATEGORY_COLORS[i] if i < len(CATEGORY_COLORS) else "#CCCCCC"
        ax.barh(range(len(mags)), vals, left=lefts, color=color,
                label=cat, edgecolor="white", linewidth=0.5)
        lefts += vals

    ax.set_yticks(range(len(mags)))
    ax.set_yticklabels([MAGAZINE_DISPLAY.get(m, m) for m in mags])
    ax.set_xlabel("Share of full-page ads")
    ax.set_xlim(0, 1)
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda v, _: f"{v*100:.0f}%"))
    # Reverse legend order so top-of-stack comes first visually
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles[::-1], labels[::-1],
              bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=9)

    if not paper:
        ax.set_title("Product Category Distribution by Magazine", fontweight="bold")

    fig.tight_layout()
    fix_fonts(fig)

    stem = "product_category_by_magazine"
    if paper:
        fig.savefig(out / f"{stem}.pdf", dpi=600)
        # Companion CSV
        csv_df = ct.copy()
        csv_df.index.name = "magazine"
        n_per_mag = df2.groupby("magazine").size().reindex(mags)
        csv_df.insert(0, "n_ads", n_per_mag.values)
        csv_df.to_csv(out / f"{stem}.csv")
        print(f"Saved: {out / stem}.pdf + .csv")
    else:
        fig.savefig(out / f"{stem}.png")
        print(f"Saved: {out / stem}.png")
    plt.close(fig)


def plot_over_time(df: pd.DataFrame, top_cats: list[str], out: Path):
    """Line chart: top category proportions over time (5-year bins).

    Shows DIACHRONIC_N named categories plus an "Other" aggregate.
    """
    diachronic_cats = top_cats[:DIACHRONIC_N]

    df2 = df.copy()
    df2["period"] = (df2["year"] // 5) * 5
    # Filter to periods with reasonable coverage
    period_counts = df2["period"].value_counts()
    valid_periods = period_counts[period_counts >= 50].index
    df2 = df2[df2["period"].isin(valid_periods)]

    fig, ax = plt.subplots(figsize=(14, 7))

    periods = sorted(df2["period"].unique())
    totals = df2.groupby("period").size()

    for i, cat in enumerate(diachronic_cats):
        cat_counts = df2[df2["predicted_category"] == cat].groupby("period").size()
        props = (cat_counts / totals).reindex(periods, fill_value=0)
        color = CATEGORY_COLORS[i] if i < len(CATEGORY_COLORS) else "#999"
        ax.plot(periods, props.values, marker="o", markersize=4,
                color=color, label=cat, linewidth=1.8)

    # Other line: everything not in the named categories
    other_counts = df2[~df2["predicted_category"].isin(diachronic_cats)].groupby("period").size()
    other_props = (other_counts / totals).reindex(periods, fill_value=0)
    ax.plot(periods, other_props.values, marker="s", markersize=4,
            color=CATEGORY_COLORS[-1], label="Other", linewidth=1.8,
            linestyle="--")

    ax.set_xlabel("Year")
    ax.set_ylabel("Share of full-page ads")
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda v, _: f"{v*100:.0f}%"))
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=9)
    ax.set_title("Product Category Prevalence Over Time, All Magazines",
                 fontweight="bold")

    fig.tight_layout()
    fix_fonts(fig)
    fig.savefig(out / "product_category_over_time.png")
    plt.close(fig)
    print(f"Saved: {out / 'product_category_over_time.png'}")


def plot_per_magazine_over_time(df: pd.DataFrame, top_cats: list[str], out: Path):
    """Small multiples: top categories over time, one subplot per magazine."""
    mags = [m for m in MAGAZINE_ORDER if m in df["magazine"].unique()]
    n_mags = len(mags)

    # Use same DIACHRONIC_N categories + Other as the aggregate plot
    cats_sm = top_cats[:DIACHRONIC_N]

    fig, axes = plt.subplots(2, 3, figsize=(18, 11), sharey=True)
    axes = axes.flatten()

    df2 = df.copy()
    df2["period"] = (df2["year"] // 5) * 5

    for idx, mag in enumerate(mags):
        ax = axes[idx]
        mag_df = df2[df2["magazine"] == mag]
        period_counts = mag_df["period"].value_counts()
        valid_periods = period_counts[period_counts >= 20].index
        mag_df = mag_df[mag_df["period"].isin(valid_periods)]
        periods = sorted(mag_df["period"].unique())
        totals = mag_df.groupby("period").size()

        for i, cat in enumerate(cats_sm):
            cat_counts = mag_df[mag_df["predicted_category"] == cat].groupby("period").size()
            props = (cat_counts / totals).reindex(periods, fill_value=0)
            color = CATEGORY_COLORS[i]
            ax.plot(periods, props.values, marker="o", markersize=3,
                    color=color, label=cat, linewidth=1.4)

        # Other line
        other_counts = mag_df[~mag_df["predicted_category"].isin(cats_sm)].groupby("period").size()
        other_props = (other_counts / totals).reindex(periods, fill_value=0)
        ax.plot(periods, other_props.values, marker="s", markersize=3,
                color=CATEGORY_COLORS[-1], label="Other", linewidth=1.4,
                linestyle="--")

        ax.set_title(MAGAZINE_DISPLAY.get(mag, mag), fontweight="bold")
        ax.yaxis.set_major_formatter(
            ticker.FuncFormatter(lambda v, _: f"{v*100:.0f}%"))
        ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
        ax.tick_params(axis="x", rotation=45)

    # Hide unused subplot
    for idx in range(len(mags), len(axes)):
        axes[idx].set_visible(False)

    axes[0].set_ylabel("Share of full-page ads")
    axes[3].set_ylabel("Share of full-page ads")

    # Single legend in the empty subplot slot
    handles, labels = axes[0].get_legend_handles_labels()
    legend_ax = axes[len(mags)]
    legend_ax.set_visible(True)
    legend_ax.axis("off")
    legend_ax.legend(handles, labels, loc="center", fontsize=9, frameon=False,
                     ncol=2)

    fig.suptitle("Product Category Prevalence by Magazine Over Time",
                 fontweight="bold", fontsize=13)
    fig.tight_layout()
    fix_fonts(fig)
    fig.savefig(out / "product_category_per_magazine_over_time.png")
    plt.close(fig)
    print(f"Saved: {out / 'product_category_per_magazine_over_time.png'}")


def plot_individual_magazine(df: pd.DataFrame, mag: str, out: Path,
                             top_n: int = 10):
    """Stacked area plot per magazine: top-N categories + Other, by proportion.

    Categories that ever appear in the top-N across any period are shown;
    everything else is grouped as "Other".

    Follows PLOTS.md conventions: no title, 600 DPI PDF, companion CSV,
    Canela Text Trial for words, Arial for numerics.
    """
    mag_df = df[df["magazine"] == mag].copy()
    if len(mag_df) == 0:
        return

    mag_df["period"] = (mag_df["year"] // 5) * 5
    period_counts = mag_df["period"].value_counts()
    valid_periods = period_counts[period_counts >= 20].index
    mag_df = mag_df[mag_df["period"].isin(valid_periods)]
    periods = sorted(mag_df["period"].unique())
    totals = mag_df.groupby("period").size()

    # Find named categories that appear in the top N of any period
    # (exclude "Other" — it's always added at the end)
    named_df = mag_df[mag_df["predicted_category"] != "Other"]
    ever_topn = set()
    for period in periods:
        period_df = named_df[named_df["period"] == period]
        topn = period_df["predicted_category"].value_counts().head(top_n).index
        ever_topn.update(topn)

    # Sort by overall prevalence for consistent stacking order
    cat_totals = named_df[named_df["predicted_category"].isin(ever_topn)] \
        .groupby("predicted_category").size().sort_values(ascending=False)
    cats = cat_totals.index.tolist()

    # Build proportion matrix: rows = periods, cols = categories + Other
    prop_data = {}
    for cat in cats:
        cat_counts = mag_df[mag_df["predicted_category"] == cat] \
            .groupby("period").size()
        prop_data[cat] = (cat_counts.reindex(periods, fill_value=0) / totals).values

    # Other = 1 - sum of named categories (always present now that
    # low-confidence predictions are reclassified as Other)
    named_sum = np.sum(list(prop_data.values()), axis=0)
    prop_data["Other"] = 1.0 - named_sum
    all_cats = cats + ["Other"]
    Y = np.array([prop_data[c] for c in all_cats])

    fig, ax = plt.subplots(figsize=(12, 6))

    colors = [CATEGORY_COLORS[i % (len(CATEGORY_COLORS) - 1)]
              for i in range(len(cats))]
    colors.append(CATEGORY_COLORS[-1])  # grey for Other

    ax.stackplot(periods, Y, labels=all_cats, colors=colors,
                 edgecolor="white", linewidth=0.3)

    ax.set_xlabel("Year")
    ax.set_ylabel("Share of full-page ads")
    ax.set_xlim(periods[0], periods[-1])
    ax.set_ylim(0, 1)
    ax.yaxis.set_major_formatter(
        ticker.FuncFormatter(lambda v, _: f"{v*100:.0f}%"))
    ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    # Reverse legend order so it matches visual stacking (top to bottom)
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles[::-1], labels[::-1],
              bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=9)

    # No title — captions go in the paper (PLOTS.md convention)
    fig.tight_layout()
    fix_fonts(fig)
    stem = f"product_category_{mag}_over_time"
    fig.savefig(out / f"{stem}.pdf", dpi=600)
    plt.close(fig)

    # Companion CSV — proportions per period per category
    csv_rows = []
    for pi, period in enumerate(periods):
        row = {"period": period, "n_ads": int(totals.loc[period])}
        for ci, cat in enumerate(all_cats):
            row[cat] = float(Y[ci, pi])
        csv_rows.append(row)
    pd.DataFrame(csv_rows).to_csv(out / f"{stem}.csv", index=False)

    display = MAGAZINE_DISPLAY.get(mag, mag)
    print(f"Saved: {out / stem}.pdf + .csv  "
          f"({len(cats)} named categories for {display})")


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--min_confidence", type=float, default=0.5,
                        help="Minimum prediction confidence (default: 0.5)")
    parser.add_argument("--individual", action="store_true",
                        help="Plot individual per-magazine category timelines")
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    apply_style()

    df = load_and_filter(args.predictions, args.min_confidence)

    if args.individual:
        # Use the 8 reliable categories for paper plots
        top_cats = get_top_categories(df, n=TOP_N)
        plot_by_magazine(df, top_cats, args.out, paper=True)
        mags = [m for m in MAGAZINE_ORDER if m in df["magazine"].unique()]
        for mag in mags:
            plot_individual_magazine(df, mag, args.out)
        return

    top_cats = get_top_categories(df, n=max(TOP_N, DIACHRONIC_N))
    print(f"Top {len(top_cats)} categories: {top_cats}")

    plot_by_magazine(df, top_cats, args.out)
    plot_over_time(df, top_cats, args.out)
    plot_per_magazine_over_time(df, top_cats, args.out)


if __name__ == "__main__":
    main()
