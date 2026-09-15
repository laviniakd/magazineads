"""
paper_plots.py

Unified script for all publication-quality figures that go to finalplots/.
Shared style constants (figsize, font sizes) ensure visual consistency.

Usage
-----
    # All plots (requires all data inputs):
    python magazines/scripts/paper_plots.py --section all

    # Just the coverage + text plots (no people data needed):
    python magazines/scripts/paper_plots.py --section coverage

    # Just people plots:
    python magazines/scripts/paper_plots.py --section people

Sections
--------
    coverage      fig_00 coverage Gantts, fig_T01 median words,
                  fig_WS01 whitespace, fig_03d ad footprint,
                  fig_PC product category over time
    classifiers   fig_04 page classifier confusion, fig_S05 spread confusion
    people        fig_PF07 area by magazine, fig_PF08 area by category,
                  fig_people_prevalence_ppipp (standalone + with recall)
    all           everything
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]

try:
    from magazineads.paths import paths as _paths
    _ISSUE_META_CSV  = _paths.local_data_dir / "issue_metadata_df.csv"
    _PREDICTIONS_CSV = _paths.predictions_dir / "predictions_all.csv"
except Exception:
    _ISSUE_META_CSV  = PROJECT_ROOT / "data" / "issue_metadata_df.csv"
    _PREDICTIONS_CSV = Path("/data/laviniad/magazines/page_classification_predictions/predictions_all.csv")

from magazineads.analysis.plot_utils import (
    apply_style, fix_fonts, NUM_FONT,
    MAGAZINE_PALETTE, GENERAL_PALETTE, magazine_label,
    plot_confusion_matrix,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger(__name__)

# ═══════════════════════════════════════════════════════════════════════════
# Shared style constants — change here to update ALL figures at once
# ═══════════════════════════════════════════════════════════════════════════

MAG_ORDER = ["time", "esquire", "life", "forbes", "variety"]
DISPLAY   = {m: magazine_label(m) for m in MAG_ORDER}

FIG_W, FIG_H = 14, 5            # matched aspect ratio for all plots
DPI           = 600

LABEL_SIZE     = 24             # axis labels
TICK_SIZE      = 22             # tick labels
LEGEND_SIZE    = 20             # legend entries
MAG_LABEL_SIZE = 22             # magazine name annotations
RCPARAMS_FONT  = 20            # global matplotlib default font size

# Universal x-axis range for time-series plots
YEAR_MIN, YEAR_MAX = 1925, 2015

# "All magazines" aggregate line style
AGG_COLOR     = "#333333"
AGG_LW        = 3.5
AGG_LINESTYLE = "--"

# Product-category palette (for PF08)
CATEGORY_PALETTE: dict[str, str] = {
    "Alcohol":                    "#542344",
    "Vehicles":                   "#28788C",
    "Clothing and Accessories":   "#C4963C",
    "Finance and Insurance":      "#5C9E88",
    "Media":                      "#B5A080",
    "Tobacco":                    "#C05C58",
    "Travel and Tourism":         "#3A6EA5",
    "Food and Beverage Services": "#8B5E3C",
    "Consumer Electronics":       "#7B68AE",
    "Business and Industrial":    "#808080",
}

RELIABLE_CATEGORIES = list(CATEGORY_PALETTE.keys())

# Product-category over-time palette (16 colors: 15 named + Other grey)
PRODCAT_COLORS = [
    "#C05C58",  "#5C9E88",  "#C4963C",  "#28788C",  "#542344",
    "#B5A080",  "#7B68A0",  "#D4816B",  "#4A7C59",  "#8B6C42",
    "#E88D67",  "#3D5A80",  "#A44A3F",  "#6B8E6B",  "#C9A959",
    "#999999",  # grey (Other)
]
DIACHRONIC_N = 15

# Ad footprint type colours
TYPE_COLORS = {
    "full-page": GENERAL_PALETTE[1],   # light blue
    "partial":   GENERAL_PALETTE[2],   # tan
}

# Classifier labels
LABEL_MAP: dict[str, int] = {
    "Not Ad": 0, "Ad": 1, "Mixed": 2, "Two-page": 1, "Two-Page": 1,
}
CLASSIFIER_CLASS_NAMES = {0: "Non-Ad", 1: "Ad", 2: "Mixed"}

# Spread classifier
SPREAD_CLASS_NAMES = ["Not spread", "Spread"]
SPREAD_CLASS_KEYS  = ["not_spread", "spread"]
SPREAD_MAGAZINE_ORDER = ["esquire", "forbes", "life", "time", "variety"]

# Topics from MALLET LDA that correspond to pharmaceutical / medical warning text
# (FDA-mandated prescribing info, side effects, adverse events).
# Ads dominated by these topics are excluded from text-quantity plots.
PHARMA_TOPICS = {15, 31, 44, 48, 96}
_MALLET_DOC_TOPICS = PROJECT_ROOT / "paperplots" / "text" / "mallet" / "doc_topics.txt"

_pharma_stems_cache: set[str] | None = None


def _pharma_stems() -> set[str]:
    """Return set of unit_id stems whose dominant LDA topic is pharmaceutical."""
    global _pharma_stems_cache
    if _pharma_stems_cache is not None:
        return _pharma_stems_cache
    stems: set[str] = set()
    with open(_MALLET_DOC_TOPICS) as f:
        next(f)  # skip header
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) < 4:
                continue
            doc_id = parts[1]
            dominant_topic = int(parts[2])
            if dominant_topic in PHARMA_TOPICS:
                stems.add(doc_id)
    _pharma_stems_cache = stems
    log.info("Pharma-dominant ads to exclude from text plots: %d", len(stems))
    return stems


def _save(fig: plt.Figure, path: Path) -> None:
    """Save figure as PDF + PNG. *path* suffix is replaced."""
    fig.tight_layout(pad=1.5)
    fix_fonts(fig)
    for suffix in (".pdf", ".png"):
        out = path.with_suffix(suffix)
        fig.savefig(out, dpi=DPI, bbox_inches="tight", pad_inches=0.3)
    plt.close(fig)
    log.info("saved %s (.pdf + .png)", path.stem)


def _bootstrap_ci(series: pd.Series, n_boot: int = 1000, ci: float = 0.95,
                   rng: np.random.Generator | None = None,
                   statistic: str = "median") -> tuple[float, float]:
    if rng is None:
        rng = np.random.default_rng(42)
    vals = series.values
    n = len(vals)
    if n < 3:
        return (np.nan, np.nan)
    func = np.mean if statistic == "mean" else np.median
    boots = np.array([func(rng.choice(vals, size=n, replace=True))
                      for _ in range(n_boot)])
    alpha = (1 - ci) / 2
    return (float(np.percentile(boots, 100 * alpha)),
            float(np.percentile(boots, 100 * (1 - alpha))))


# ═══════════════════════════════════════════════════════════════════════════
# Data loading helpers
# ═══════════════════════════════════════════════════════════════════════════

def _load_issue_meta() -> pd.DataFrame:
    return pd.read_csv(_ISSUE_META_CSV, parse_dates=["date"])[
        ["magazine", "year", "num_pages"]
    ]


def _load_esquire_issues(predictions_csv: Path) -> pd.DataFrame:
    preds = pd.read_csv(predictions_csv, usecols=["filepath"])
    esq = preds[preds["filepath"].str.contains("/esquire/")].copy()
    esq["date_str"] = esq["filepath"].str.extract(r"/esquire/(\d{4}-\d{2}-\d{2})")
    issues = (
        esq.dropna(subset=["date_str"])
           .groupby("date_str")
           .size()
           .reset_index(name="num_pages")
    )
    issues["year"] = issues["date_str"].str[:4].astype(int)
    issues["magazine"] = "esquire"
    return issues[["magazine", "year", "num_pages"]]


def _build_issue_table(predictions_csv: Path) -> pd.DataFrame:
    meta = _load_issue_meta()
    esq  = _load_esquire_issues(predictions_csv)
    return pd.concat([meta, esq], ignore_index=True)


def _compute_summary(issues: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for mag in MAG_ORDER:
        sub = issues[issues["magazine"] == mag]
        if sub.empty:
            continue
        rows.append({
            "magazine":   mag,
            "start_year": int(sub["year"].min()),
            "end_year":   int(sub["year"].max()),
            "n_issues":   len(sub),
            "total_pages": int(sub["num_pages"].sum()),
        })
    return pd.DataFrame(rows)


def _load_preds_yearly(predictions_csv: Path) -> pd.DataFrame:
    """Load predictions CSV and parse magazine + year."""
    preds = pd.read_csv(predictions_csv, usecols=["filepath", "pred_label"])
    preds["magazine"] = preds["filepath"].str.extract(r"/images/([^/]+)/")
    preds["year"] = pd.to_numeric(
        preds["filepath"].str.extract(r"/(\d{4})")[0], errors="coerce"
    )
    preds = preds.dropna(subset=["year"])
    preds["year"] = preds["year"].astype(int)
    return preds


# ═══════════════════════════════════════════════════════════════════════════
# fig_00: Dataset coverage Gantt (pages or ads)
# ═══════════════════════════════════════════════════════════════════════════

def _plot_coverage_gantt(
    summary: pd.DataFrame,
    hue_data: pd.DataFrame,
    output_dir: Path,
    hue_label: str,
    legend_max: int,
    suffix: str,
) -> None:
    from matplotlib.colors import LinearSegmentedColormap

    all_years = np.arange(
        hue_data["year"].min(),
        max(hue_data["year"].max(), summary["end_year"].max()) + 1,
    )
    mags = list(summary["magazine"])
    n_mags = len(mags)

    fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))
    row_h, gap = 0.65, 0.35
    y_step = row_h + gap

    max_vals = {}
    for m in mags:
        sub = hue_data[hue_data["magazine"] == m]
        max_vals[m] = sub["value"].max() if not sub.empty else 1

    for i, mag in enumerate(mags):
        y_bottom = i * y_step
        sub = hue_data[hue_data["magazine"] == mag].set_index("year")["value"]
        smeta = summary[summary["magazine"] == mag].iloc[0]
        base_col = np.array(plt.matplotlib.colors.to_rgb(MAGAZINE_PALETTE[mag]))

        ax.barh(y_bottom + row_h / 2,
                smeta["end_year"] - smeta["start_year"] + 1,
                left=smeta["start_year"],
                height=row_h,
                color="#eeeeee", edgecolor="none", zorder=1)

        for yr in all_years:
            if yr < smeta["start_year"] or yr > smeta["end_year"]:
                continue
            val = sub.get(yr, 0)
            if val == 0:
                continue
            alpha = 0.20 + 0.80 * (val / max_vals[mag])
            ax.bar(yr, row_h, bottom=y_bottom, width=0.85,
                   color=base_col, alpha=float(alpha), edgecolor="none", zorder=2)

    decade_ticks = np.arange((all_years[0] // 10) * 10, all_years[-1] + 1, 10)
    label_x = decade_ticks[0] - 2

    ax.set_xlim(label_x - 10, all_years[-1] + 2)
    ax.set_ylim(-gap, n_mags * y_step)
    ax.set_yticks([])
    ax.set_xticks(decade_ticks)
    ax.set_xticklabels([str(y) for y in decade_ticks], fontsize=TICK_SIZE)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.xaxis.grid(True, linestyle="--", alpha=0.25, zorder=0)
    ax.set_axisbelow(True)

    for i, mag in enumerate(mags):
        y_bottom = i * y_step
        ax.text(label_x, y_bottom + row_h / 2,
                DISPLAY[mag], ha="right", va="center",
                fontsize=MAG_LABEL_SIZE, fontweight="bold",
                color=MAGAZINE_PALETTE[mag])

    # Hue-intensity legend bar
    cmap = LinearSegmentedColormap.from_list(
        "intensity", [(1, 1, 1), (0.25, 0.25, 0.25)], N=256
    )
    cbar_ax = fig.add_axes([0.72, 0.88, 0.18, 0.025])
    norm = plt.matplotlib.colors.Normalize(vmin=0, vmax=legend_max)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cb = fig.colorbar(sm, cax=cbar_ax, orientation="horizontal")
    cb.set_label(hue_label, fontsize=LABEL_SIZE - 6)
    cb.ax.tick_params(labelsize=TICK_SIZE - 4)
    cb.set_ticks([0, legend_max])
    cb.set_ticklabels(["0", f"{legend_max:,}"])

    fname = f"fig_00_dataset_coverage{suffix}"
    _save(fig, output_dir / fname)

    hue_data.to_csv(output_dir / f"{fname}.csv", index=False)


def plot_coverage_pages(issues, summary, output_dir):
    hue_data = (
        issues.groupby(["magazine", "year"])["num_pages"]
              .sum()
              .reset_index(name="value")
    )
    _plot_coverage_gantt(summary, hue_data, output_dir,
                         "Pages per year", 10_000, "_pages")


def plot_coverage_ads(issues, summary, predictions_csv, output_dir):
    preds = _load_preds_yearly(predictions_csv)
    hue_data = (
        preds[(preds["pred_label"] == 1) & preds["magazine"].isin(MAG_ORDER)]
        .groupby(["magazine", "year"])
        .size()
        .reset_index(name="value")
    )
    _plot_coverage_gantt(summary, hue_data, output_dir,
                         "Full-page ads per year", 4_000, "_ads")


# ═══════════════════════════════════════════════════════════════════════════
# fig_01: Ad prevalence over time (fraction of pages that are ads)
# ═══════════════════════════════════════════════════════════════════════════

def plot_ad_prevalence(predictions_csv: Path, output_dir: Path) -> None:
    """Two-panel line chart: percentage of pages that are full-page ads / mixed."""
    import matplotlib.ticker as mticker

    preds = _load_preds_yearly(predictions_csv)
    preds = preds[preds["magazine"].isin(MAG_ORDER)].copy()
    preds = preds[(preds["year"] >= 1900) & (preds["year"] <= 2025)]

    rows = []
    for (mag, year), grp in preds.groupby(["magazine", "year"]):
        n = len(grp)
        rows.append({
            "magazine": mag, "year": year, "n_pages": n,
            "frac_fullpage": (grp["pred_label"] == 1).sum() / n,
            "frac_mixed": (grp["pred_label"] == 2).sum() / n,
        })
    agg = pd.DataFrame(rows)

    fig, axes = plt.subplots(2, 1, figsize=(FIG_W, FIG_H * 2), sharex=True)
    panel_configs = [
        ("frac_fullpage", "Full-page ads", axes[0]),
        ("frac_mixed", "Mixed", axes[1]),
    ]

    for panel_idx, (col, panel_label, ax) in enumerate(panel_configs):
        for mag in MAG_ORDER:
            sub = agg[agg["magazine"] == mag].sort_values("year")
            if sub.empty:
                continue
            color = MAGAZINE_PALETTE.get(mag, "#808080")
            ts = sub.set_index("year")
            rolled = ts[col].rolling(SMOOTH_WINDOW, center=True, min_periods=2).mean()
            lbl = DISPLAY.get(mag, mag) if panel_idx == 0 else None
            ax.plot(rolled.index, rolled.values,
                    color=color, linewidth=3, label=lbl, zorder=2)

        ax.set_ylim(bottom=0)
        ax.set_xlim(YEAR_MIN, YEAR_MAX)
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v*100:.0f}%"))
        ax.tick_params(labelsize=TICK_SIZE)
        ax.spines[["top", "right"]].set_visible(False)
        ax.yaxis.grid(True, linestyle="--", alpha=0.3)
        ax.set_axisbelow(True)
        ax.text(0.02, 0.95, panel_label, transform=ax.transAxes,
                fontsize=MAG_LABEL_SIZE, fontweight="bold", va="top")

    fig.supylabel("Percentage of pages", fontsize=LABEL_SIZE, x=0.02)
    axes[-1].set_xlabel("Year", fontsize=LABEL_SIZE)
    axes[0].tick_params(axis="x", labelbottom=False)
    axes[0].legend(fontsize=LEGEND_SIZE, loc="upper left",
                   bbox_to_anchor=(1.01, 1), borderaxespad=0)

    _save(fig, output_dir / "fig_01_ad_prevalence")
    agg.to_csv(output_dir / "fig_01_ad_prevalence.csv", index=False)


# ═══════════════════════════════════════════════════════════════════════════
# fig_T01: Median words per ad over time
# ═══════════════════════════════════════════════════════════════════════════

def plot_median_words(cache_csv: Path, output_dir: Path) -> None:
    df = pd.read_csv(cache_csv)
    df = df[df["magazine"].isin(MAG_ORDER)].copy()
    exclude = _pharma_stems()
    before = len(df)
    df = df[~df["stem"].isin(exclude)]
    log.info("T01: excluded %d / %d pharma-warning ads", before - len(df), before)

    rng = np.random.default_rng(42)

    rows = []
    for (mag, year), grp in df.groupby(["magazine", "year"]):
        lo, hi = _bootstrap_ci(grp["n_words"], rng=rng)
        rows.append({
            "magazine": mag, "year": year,
            "median_words": grp["n_words"].median(),
            "ci_lo": lo, "ci_hi": hi,
            "n_pages": len(grp),
        })
    agg = pd.DataFrame(rows)

    all_rows = []
    for year, grp in df.groupby("year"):
        lo, hi = _bootstrap_ci(grp["n_words"], rng=rng)
        all_rows.append({
            "year": year,
            "median_words": grp["n_words"].median(),
            "ci_lo": lo, "ci_hi": hi,
        })
    agg_all = pd.DataFrame(all_rows).sort_values("year").set_index("year")

    fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))

    for mag in MAG_ORDER:
        sub = agg[agg["magazine"] == mag].sort_values("year")
        if sub.empty:
            continue
        color = MAGAZINE_PALETTE.get(mag, "#808080")

        ax.plot(sub["year"], sub["median_words"],
                color=color, alpha=0.25, linewidth=0.8, zorder=1)

        ts = sub.set_index("year")
        rolled = ts["median_words"].rolling(5, center=True, min_periods=2).mean()
        ax.plot(rolled.index, rolled.values,
                color=color, linewidth=3, label=DISPLAY.get(mag, mag), zorder=2)

        lo_rolled = ts["ci_lo"].rolling(5, center=True, min_periods=2).mean()
        hi_rolled = ts["ci_hi"].rolling(5, center=True, min_periods=2).mean()
        ax.fill_between(lo_rolled.index, lo_rolled.values, hi_rolled.values,
                        color=color, alpha=0.12, zorder=1)

    rolled_all = agg_all["median_words"].rolling(5, center=True, min_periods=2).mean()
    ax.plot(rolled_all.index, rolled_all.values,
            color=AGG_COLOR, linewidth=AGG_LW, linestyle=AGG_LINESTYLE,
            label="All magazines", zorder=3)
    lo_all = agg_all["ci_lo"].rolling(5, center=True, min_periods=2).mean()
    hi_all = agg_all["ci_hi"].rolling(5, center=True, min_periods=2).mean()
    ax.fill_between(lo_all.index, lo_all.values, hi_all.values,
                    color=AGG_COLOR, alpha=0.08, zorder=1)

    ax.set_xlabel("Year", fontsize=LABEL_SIZE)
    ax.set_ylabel("Median words per ad", fontsize=LABEL_SIZE)
    ax.set_title("")
    ax.set_xlim(YEAR_MIN, YEAR_MAX)
    ax.tick_params(labelsize=TICK_SIZE)
    ax.spines[["top", "right"]].set_visible(False)
    ax.yaxis.grid(True, linestyle="--", alpha=0.3)
    ax.set_axisbelow(True)
    ax.legend(fontsize=LEGEND_SIZE, loc="upper left",
              bbox_to_anchor=(1.01, 1), borderaxespad=0)

    _save(fig, output_dir / "fig_T01_words_per_ad_over_time")

    agg.to_csv(output_dir / "fig_T01_words_per_ad_over_time.csv", index=False)


# ═══════════════════════════════════════════════════════════════════════════
# fig_WS01: Whitespace trend over time
# ═══════════════════════════════════════════════════════════════════════════

DEFAULT_WHITESPACE_CSV = PROJECT_ROOT / "data" / "whitespace_full.csv"


def plot_whitespace_trend(whitespace_csv: Path, output_dir: Path) -> None:
    import matplotlib.ticker as mticker

    df = pd.read_csv(whitespace_csv)
    df = df[df["magazine"].isin(MAG_ORDER)].copy()
    df["year"] = df["year"].astype(int)

    rng = np.random.default_rng(42)

    # Per-magazine-year aggregation
    rows = []
    for (mag, year), grp in df.groupby(["magazine", "year"]):
        vals = grp["whitespace_frac"]
        lo, hi = _bootstrap_ci(vals, rng=rng, statistic="mean")
        rows.append({
            "magazine": mag, "year": year,
            "mean_ws": vals.mean(),
            "ci_lo": lo, "ci_hi": hi,
            "n": len(grp),
        })
    agg = pd.DataFrame(rows)

    # All-magazines aggregate
    all_rows = []
    for year, grp in df.groupby("year"):
        vals = grp["whitespace_frac"]
        lo, hi = _bootstrap_ci(vals, rng=rng, statistic="mean")
        all_rows.append({
            "year": year,
            "mean_ws": vals.mean(),
            "ci_lo": lo, "ci_hi": hi,
        })
    agg_all = pd.DataFrame(all_rows).sort_values("year").set_index("year")

    fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))

    for mag in MAG_ORDER:
        sub = agg[agg["magazine"] == mag].sort_values("year")
        if sub.empty:
            continue
        color = MAGAZINE_PALETTE.get(mag, "#808080")

        ax.plot(sub["year"], sub["mean_ws"],
                color=color, alpha=0.25, linewidth=0.8, zorder=1)

        ts = sub.set_index("year")
        rolled = ts["mean_ws"].rolling(5, center=True, min_periods=2).mean()
        ax.plot(rolled.index, rolled.values,
                color=color, linewidth=3, label=DISPLAY.get(mag, mag), zorder=2)

        lo_rolled = ts["ci_lo"].rolling(5, center=True, min_periods=2).mean()
        hi_rolled = ts["ci_hi"].rolling(5, center=True, min_periods=2).mean()
        ax.fill_between(lo_rolled.index, lo_rolled.values, hi_rolled.values,
                        color=color, alpha=0.12, zorder=1)

    rolled_all = agg_all["mean_ws"].rolling(5, center=True, min_periods=2).mean()
    ax.plot(rolled_all.index, rolled_all.values,
            color=AGG_COLOR, linewidth=AGG_LW, linestyle=AGG_LINESTYLE,
            label="All magazines", zorder=3)
    lo_all = agg_all["ci_lo"].rolling(5, center=True, min_periods=2).mean()
    hi_all = agg_all["ci_hi"].rolling(5, center=True, min_periods=2).mean()
    ax.fill_between(lo_all.index, lo_all.values, hi_all.values,
                    color=AGG_COLOR, alpha=0.08, zorder=1)

    ax.set_xlabel("Year", fontsize=LABEL_SIZE)
    ax.set_ylabel("Mean whitespace percentage", fontsize=LABEL_SIZE)
    ax.set_title("")
    ax.set_xlim(YEAR_MIN, YEAR_MAX)
    ax.set_ylim(0.10, 0.50)
    ax.yaxis.set_major_locator(mticker.MultipleLocator(0.10))
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:.0%}"))
    ax.tick_params(labelsize=TICK_SIZE)
    ax.spines[["top", "right"]].set_visible(False)
    ax.yaxis.grid(True, linestyle="--", alpha=0.3)
    ax.set_axisbelow(True)
    ax.legend(fontsize=LEGEND_SIZE, loc="upper left",
              bbox_to_anchor=(1.01, 1), borderaxespad=0)

    _save(fig, output_dir / "fig_WS01_whitespace_over_time")

    agg.to_csv(output_dir / "fig_WS01_whitespace_over_time.csv", index=False)


# ═══════════════════════════════════════════════════════════════════════════
# fig_ED01: Edge density (text-masked) over time
# ═══════════════════════════════════════════════════════════════════════════

DEFAULT_EDGE_DENSITY_CSV = PROJECT_ROOT / "data" / "text_masked_complexity_full.csv"


def plot_edge_density_trend(edge_csv: Path, output_dir: Path) -> None:
    import matplotlib.ticker as mticker

    df = pd.read_csv(edge_csv)
    df = df[df["magazine"].isin(MAG_ORDER)].copy()
    df["year"] = df["year"].astype(int)

    rng = np.random.default_rng(42)
    col = "edge_density_gauss"

    # Per-magazine-year aggregation
    rows = []
    for (mag, year), grp in df.groupby(["magazine", "year"]):
        vals = grp[col]
        lo, hi = _bootstrap_ci(vals, rng=rng, statistic="mean")
        rows.append({
            "magazine": mag, "year": year,
            "mean_ed": vals.mean(),
            "ci_lo": lo, "ci_hi": hi,
            "n": len(grp),
        })
    agg = pd.DataFrame(rows)

    # All-magazines aggregate
    all_rows = []
    for year, grp in df.groupby("year"):
        vals = grp[col]
        lo, hi = _bootstrap_ci(vals, rng=rng, statistic="mean")
        all_rows.append({
            "year": year,
            "mean_ed": vals.mean(),
            "ci_lo": lo, "ci_hi": hi,
        })
    agg_all = pd.DataFrame(all_rows).sort_values("year").set_index("year")

    fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))

    for mag in MAG_ORDER:
        sub = agg[agg["magazine"] == mag].sort_values("year")
        if sub.empty:
            continue
        color = MAGAZINE_PALETTE.get(mag, "#808080")

        ax.plot(sub["year"], sub["mean_ed"],
                color=color, alpha=0.25, linewidth=0.8, zorder=1)

        ts = sub.set_index("year")
        rolled = ts["mean_ed"].rolling(5, center=True, min_periods=2).mean()
        ax.plot(rolled.index, rolled.values,
                color=color, linewidth=3, label=DISPLAY.get(mag, mag), zorder=2)

        lo_rolled = ts["ci_lo"].rolling(5, center=True, min_periods=2).mean()
        hi_rolled = ts["ci_hi"].rolling(5, center=True, min_periods=2).mean()
        ax.fill_between(lo_rolled.index, lo_rolled.values, hi_rolled.values,
                        color=color, alpha=0.12, zorder=1)

    rolled_all = agg_all["mean_ed"].rolling(5, center=True, min_periods=2).mean()
    ax.plot(rolled_all.index, rolled_all.values,
            color=AGG_COLOR, linewidth=AGG_LW, linestyle=AGG_LINESTYLE,
            label="All magazines", zorder=3)
    lo_all = agg_all["ci_lo"].rolling(5, center=True, min_periods=2).mean()
    hi_all = agg_all["ci_hi"].rolling(5, center=True, min_periods=2).mean()
    ax.fill_between(lo_all.index, lo_all.values, hi_all.values,
                    color=AGG_COLOR, alpha=0.08, zorder=1)

    ax.set_xlabel("Year", fontsize=LABEL_SIZE)
    ax.set_ylabel("Mean edge density", fontsize=LABEL_SIZE)
    ax.set_title("")
    ax.set_xlim(YEAR_MIN, YEAR_MAX)
    ax.tick_params(labelsize=TICK_SIZE)
    ax.spines[["top", "right"]].set_visible(False)
    ax.yaxis.grid(True, linestyle="--", alpha=0.3)
    ax.set_axisbelow(True)
    ax.legend(fontsize=LEGEND_SIZE, loc="upper left",
              bbox_to_anchor=(1.01, 1), borderaxespad=0)

    _save(fig, output_dir / "fig_ED01_edge_density_over_time")

    agg.to_csv(output_dir / "fig_ED01_edge_density_over_time.csv", index=False)


# ═══════════════════════════════════════════════════════════════════════════
# fig_WSED: Composite whitespace + edge density (2×1)
# ═══════════════════════════════════════════════════════════════════════════

def plot_whitespace_edge_composite(whitespace_csv: Path, edge_csv: Path,
                                   output_dir: Path) -> None:
    """Two-panel plot: whitespace fraction (top) and edge density (bottom)."""
    import matplotlib.ticker as mticker

    ws = pd.read_csv(whitespace_csv)
    ws = ws[ws["magazine"].isin(MAG_ORDER)].copy()
    ws["year"] = ws["year"].astype(int)

    ed = pd.read_csv(edge_csv)
    ed = ed[ed["magazine"].isin(MAG_ORDER)].copy()
    ed["year"] = ed["year"].astype(int)

    rng = np.random.default_rng(42)

    def _agg_metric(df, col):
        rows = []
        for (mag, year), grp in df.groupby(["magazine", "year"]):
            vals = grp[col]
            lo, hi = _bootstrap_ci(vals, rng=rng, statistic="mean")
            rows.append({"magazine": mag, "year": year,
                         "mean": vals.mean(), "ci_lo": lo, "ci_hi": hi})
        agg = pd.DataFrame(rows)
        all_rows = []
        for year, grp in df.groupby("year"):
            vals = grp[col]
            lo, hi = _bootstrap_ci(vals, rng=rng, statistic="mean")
            all_rows.append({"year": year, "mean": vals.mean(),
                             "ci_lo": lo, "ci_hi": hi})
        agg_all = pd.DataFrame(all_rows).sort_values("year").set_index("year")
        return agg, agg_all

    ws_agg, ws_all = _agg_metric(ws, "whitespace_frac")
    ed_agg, ed_all = _agg_metric(ed, "edge_density_gauss")

    fig, axes = plt.subplots(1, 2, figsize=(FIG_W * 2, FIG_H * 1.3), sharey=False)

    panels = [
        (axes[0], ws_agg, ws_all, "Mean whitespace fraction"),
        (axes[1], ed_agg, ed_all, "Mean edge density"),
    ]

    for panel_idx, (ax, agg, agg_all, panel_label) in enumerate(panels):
        is_last = (panel_idx == len(panels) - 1)
        for mag in MAG_ORDER:
            sub = agg[agg["magazine"] == mag].sort_values("year")
            if sub.empty:
                continue
            color = MAGAZINE_PALETTE.get(mag, "#808080")
            ax.plot(sub["year"], sub["mean"],
                    color=color, alpha=0.25, linewidth=0.8, zorder=1)
            ts = sub.set_index("year")
            rolled = ts["mean"].rolling(5, center=True, min_periods=2).mean()
            lbl = DISPLAY.get(mag, mag) if is_last else None
            ax.plot(rolled.index, rolled.values,
                    color=color, linewidth=3, label=lbl, zorder=2)
            lo_r = ts["ci_lo"].rolling(5, center=True, min_periods=2).mean()
            hi_r = ts["ci_hi"].rolling(5, center=True, min_periods=2).mean()
            ax.fill_between(lo_r.index, lo_r.values, hi_r.values,
                            color=color, alpha=0.12, zorder=1)

        rolled_all = agg_all["mean"].rolling(5, center=True, min_periods=2).mean()
        ax.plot(rolled_all.index, rolled_all.values,
                color=AGG_COLOR, linewidth=AGG_LW, linestyle=AGG_LINESTYLE,
                label="All magazines" if is_last else None, zorder=3)
        lo_a = agg_all["ci_lo"].rolling(5, center=True, min_periods=2).mean()
        hi_a = agg_all["ci_hi"].rolling(5, center=True, min_periods=2).mean()
        ax.fill_between(lo_a.index, lo_a.values, hi_a.values,
                        color=AGG_COLOR, alpha=0.08, zorder=1)

        ax.set_ylabel(panel_label, fontsize=LABEL_SIZE + 2)
        ax.set_xlabel("Year", fontsize=LABEL_SIZE + 2)
        ax.set_xlim(YEAR_MIN, YEAR_MAX)
        decade_ticks = list(range(1930, 2020, 10))
        ax.set_xticks(decade_ticks)
        ax.set_xticklabels([str(y) for y in decade_ticks],
                           fontfamily=NUM_FONT, fontsize=TICK_SIZE + 2)
        ax.tick_params(axis="y", labelsize=TICK_SIZE + 2)
        ax.spines[["top", "right"]].set_visible(False)
        ax.yaxis.grid(True, linestyle="--", alpha=0.3)
        ax.set_axisbelow(True)

    axes[1].legend(fontsize=LEGEND_SIZE + 2, loc="upper left",
                   bbox_to_anchor=(1.01, 1), borderaxespad=0)

    _save(fig, output_dir / "fig_WSED_whitespace_edge_density")

    ws_agg.to_csv(output_dir / "fig_WSED_whitespace.csv", index=False)
    ed_agg.to_csv(output_dir / "fig_WSED_edge_density.csv", index=False)


# ═══════════════════════════════════════════════════════════════════════════
# fig_NV01: Non-visual (text + whitespace) fraction over time
# fig_TO01: Text-only ad prevalence over time (binary threshold)
# ═══════════════════════════════════════════════════════════════════════════

TEXT_ONLY_THRESHOLD = 0.20


def _plot_time_series(df: pd.DataFrame, col: str, ylabel: str,
                      fig_stem: str, output_dir: Path,
                      ylim: tuple[float, float] | None = None) -> pd.DataFrame:
    """Shared helper: per-magazine + all-magazines time series with CIs."""
    import matplotlib.ticker as mticker

    rng = np.random.default_rng(42)

    rows = []
    for (mag, year), grp in df.groupby(["magazine", "year"]):
        vals = grp[col]
        lo, hi = _bootstrap_ci(vals, rng=rng, statistic="mean")
        rows.append({"magazine": mag, "year": year,
                     "mean": vals.mean(), "ci_lo": lo, "ci_hi": hi,
                     "n": len(grp)})
    agg = pd.DataFrame(rows)

    all_rows = []
    for year, grp in df.groupby("year"):
        vals = grp[col]
        lo, hi = _bootstrap_ci(vals, rng=rng, statistic="mean")
        all_rows.append({"year": year, "mean": vals.mean(),
                         "ci_lo": lo, "ci_hi": hi})
    agg_all = pd.DataFrame(all_rows).sort_values("year").set_index("year")

    fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))

    for mag in MAG_ORDER:
        sub = agg[agg["magazine"] == mag].sort_values("year")
        if sub.empty:
            continue
        color = MAGAZINE_PALETTE.get(mag, "#808080")
        ax.plot(sub["year"], sub["mean"],
                color=color, alpha=0.25, linewidth=0.8, zorder=1)
        ts = sub.set_index("year")
        rolled = ts["mean"].rolling(5, center=True, min_periods=2).mean()
        ax.plot(rolled.index, rolled.values,
                color=color, linewidth=3, label=DISPLAY.get(mag, mag), zorder=2)
        lo_r = ts["ci_lo"].rolling(5, center=True, min_periods=2).mean()
        hi_r = ts["ci_hi"].rolling(5, center=True, min_periods=2).mean()
        ax.fill_between(lo_r.index, lo_r.values, hi_r.values,
                        color=color, alpha=0.12, zorder=1)

    rolled_all = agg_all["mean"].rolling(5, center=True, min_periods=2).mean()
    ax.plot(rolled_all.index, rolled_all.values,
            color=AGG_COLOR, linewidth=AGG_LW, linestyle=AGG_LINESTYLE,
            label="All magazines", zorder=3)
    lo_a = agg_all["ci_lo"].rolling(5, center=True, min_periods=2).mean()
    hi_a = agg_all["ci_hi"].rolling(5, center=True, min_periods=2).mean()
    ax.fill_between(lo_a.index, lo_a.values, hi_a.values,
                    color=AGG_COLOR, alpha=0.08, zorder=1)

    ax.set_xlabel("Year", fontsize=LABEL_SIZE)
    ax.set_ylabel(ylabel, fontsize=LABEL_SIZE)
    ax.set_xlim(YEAR_MIN, YEAR_MAX)
    if ylim is not None:
        ax.set_ylim(*ylim)
    else:
        ax.set_ylim(bottom=0)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v*100:.0f}%"))
    ax.tick_params(labelsize=TICK_SIZE)
    ax.spines[["top", "right"]].set_visible(False)
    ax.yaxis.grid(True, linestyle="--", alpha=0.3)
    ax.set_axisbelow(True)
    ax.legend(fontsize=LEGEND_SIZE, loc="upper left",
              bbox_to_anchor=(1.01, 1), borderaxespad=0)

    _save(fig, output_dir / fig_stem)
    agg.to_csv(output_dir / f"{fig_stem}.csv", index=False)
    return agg


def plot_nonvisual_fraction(whitespace_csv: Path, output_dir: Path) -> None:
    df = pd.read_csv(whitespace_csv)
    df = df[df["magazine"].isin(MAG_ORDER)].copy()
    df["year"] = df["year"].astype(int)
    df["nonvisual_frac"] = 1.0 - df["visual_frac"]

    _plot_time_series(df, "nonvisual_frac",
                      "Mean non-visual fraction", "fig_NV01_nonvisual_fraction",
                      output_dir)


def plot_text_only_prevalence(whitespace_csv: Path, output_dir: Path,
                              threshold: float = TEXT_ONLY_THRESHOLD) -> None:
    df = pd.read_csv(whitespace_csv)
    df = df[df["magazine"].isin(MAG_ORDER)].copy()
    df["year"] = df["year"].astype(int)
    df["text_only"] = (df["visual_frac"] < threshold).astype(float)

    n_text = int(df["text_only"].sum())
    log.info("Text-only ads (visual_frac < %.2f): %d / %d (%.1f%%)",
             threshold, n_text, len(df), n_text / len(df) * 100)

    _plot_time_series(df, "text_only",
                      "Percentage of text-only ads", "fig_TO01_text_only_ads",
                      output_dir)


def plot_text_area_fraction(whitespace_csv: Path, output_dir: Path) -> None:
    """Mean fraction of ad area occupied by text, over time."""
    df = pd.read_csv(whitespace_csv)
    df = df[df["magazine"].isin(MAG_ORDER)].copy()
    df["year"] = df["year"].astype(int)
    exclude = _pharma_stems()
    df["_stem"] = df["filepath"].apply(lambda p: Path(p).stem)
    before = len(df)
    df = df[~df["_stem"].isin(exclude)]
    log.info("TF01: excluded %d / %d pharma-warning ads", before - len(df), before)
    df = df.drop(columns=["_stem"])

    _plot_time_series(df, "text_frac",
                      "Mean percentage of ad\noccupied by text", "fig_TF01_text_area_fraction",
                      output_dir)


# ═══════════════════════════════════════════════════════════════════════════
# fig_PF07 / PF08: Person/face area by magazine and by category
# ═══════════════════════════════════════════════════════════════════════════

MIN_N = 50


def _load_people_faces(people_csv: Path, faces_csv: Path,
                       min_year: int, max_year: int) -> pd.DataFrame:
    """Merge people + face detections into a single DataFrame."""
    ppl = pd.read_csv(people_csv)
    face = pd.read_csv(faces_csv)
    log.info("People CSV: %d rows; Face CSV: %d rows", len(ppl), len(face))

    df = ppl.merge(
        face[["image_path", "n_faces", "face_confs", "face_boxes", "max_conf", "mean_conf"]],
        on="image_path", how="inner", suffixes=("", "_face"),
    )

    if "year" not in df.columns or df["year"].isna().any():
        df["year"] = df["image_path"].apply(
            lambda p: int(Path(p).stem.split("-")[0])
            if Path(p).stem.split("-")[0].isdigit() else None
        )
    df = df.dropna(subset=["year"])
    df["year"] = df["year"].astype(int)
    df = df[(df["year"] >= min_year) & (df["year"] <= max_year)]
    df["decade"] = (df["year"] // 10 * 10).astype(int)
    df["magazine"] = df["magazine"].str.lower().str.strip()

    df["n_persons"] = df["n_persons"].fillna(0).astype(int)
    df["n_faces"] = df["n_faces"].fillna(0).astype(int)
    df["has_person"] = df["n_persons"] > 0
    df["has_face"] = df["n_faces"] > 0

    def _box_area(box_str):
        if pd.isna(box_str) or box_str in ("[]", ""):
            return 0.0
        import ast
        try:
            boxes = ast.literal_eval(box_str)
        except Exception:
            return 0.0
        total = 0.0
        for b in boxes:
            if len(b) == 4:
                total += abs((b[2] - b[0]) * (b[3] - b[1]))
        return total

    df["person_area"] = df["person_boxes"].apply(_box_area)
    df["face_area"] = df["face_boxes"].apply(_box_area)
    df["area_per_person"] = np.where(df["n_persons"] > 0, df["person_area"] / df["n_persons"], 0.0)
    df["area_per_face"] = np.where(df["n_faces"] > 0, df["face_area"] / df["n_faces"], 0.0)

    log.info("People+faces: %d images, years %d–%d", len(df), df["year"].min(), df["year"].max())
    return df


def _join_prodcat(df: pd.DataFrame, prodcat_csv: Path,
                  min_confidence: float = 0.7) -> pd.DataFrame:
    df["stem"] = df["image_path"].apply(lambda p: Path(p).stem)
    pc = pd.read_csv(prodcat_csv)
    pc = pc[pc["confidence"] >= min_confidence].copy()
    out = df.merge(pc[["unit_id", "predicted_category"]],
                   left_on="stem", right_on="unit_id", how="inner")
    out = out[out["predicted_category"].isin(RELIABLE_CATEGORIES)].copy()
    log.info("Joined with prodcat: %d rows", len(out))
    return out


def _agg_area(df: pd.DataFrame, group_col: str) -> pd.DataFrame:
    rows = []
    for (grp, decade), sub in df.groupby([group_col, "decade"]):
        n = len(sub)
        if n < MIN_N:
            continue
        for metric, col in [("Average space per person", "area_per_person"),
                            ("Average space per face", "area_per_face")]:
            vals = sub[col].values.astype(float)
            rows.append({"group": grp, "decade": decade, "metric": metric,
                         "n": n, "mean": float(vals.mean()), "std": float(vals.std()),
                         "se": float(vals.std() / np.sqrt(n))})
    return pd.DataFrame(rows)


def _plot_area_composite(df: pd.DataFrame, group_col: str, palette: dict,
                         output_dir: Path, fig_id: str, group_label: str) -> None:
    """Side-by-side person + face area plots with shared legend on right."""
    agg = _agg_area(df, group_col)
    if agg.empty:
        return

    metrics = [("Average space per person", "Average fraction of ad\nper person"),
               ("Average space per face", "Average fraction of ad\nper face")]

    fig, axes = plt.subplots(1, 2, figsize=(FIG_W * 2, FIG_H), sharey=False)

    groups = sorted(agg["group"].unique(),
                    key=lambda g: -agg[agg["group"] == g]["n"].sum())
    decades = sorted(agg["decade"].unique())

    for panel_idx, (ax, (metric, ylabel)) in enumerate(zip(axes, metrics)):
        is_last = (panel_idx == len(metrics) - 1)
        msub = agg[agg["metric"] == metric]
        for grp in groups:
            sub = msub[msub["group"] == grp].sort_values("decade")
            if len(sub) < 2:
                continue
            color = palette.get(grp, "#808080")
            label_text = magazine_label(grp) if grp in MAGAZINE_PALETTE else grp
            lbl = label_text if is_last else None
            ax.fill_between(sub["decade"], sub["mean"] - sub["se"],
                            sub["mean"] + sub["se"],
                            color=color, alpha=0.10)
            ax.plot(sub["decade"], sub["mean"], color=color, lw=3,
                    marker="o", ms=4, label=lbl)

        ax.tick_params(labelsize=TICK_SIZE + 2)
        ax.spines[["top", "right"]].set_visible(False)
        ax.yaxis.grid(True, ls="--", alpha=0.3)
        ax.set_axisbelow(True)
        ax.set_xticks(decades)
        ax.set_xticklabels([f"{d}s" for d in decades], rotation=45, ha="right")
        ax.set_xlabel("Decade", fontsize=LABEL_SIZE + 2)
        ax.set_ylabel(ylabel, fontsize=LABEL_SIZE + 2)

    axes[1].legend(fontsize=LEGEND_SIZE + 2, loc="upper left",
                   bbox_to_anchor=(1.01, 1), borderaxespad=0)

    _save(fig, output_dir / f"fig_{fig_id}_area_by_{group_label}")
    agg.to_csv(output_dir / f"fig_{fig_id}_area_by_{group_label}.csv", index=False)


def plot_area_by_magazine(df: pd.DataFrame, output_dir: Path) -> None:
    df_tmp = df.copy()
    df_tmp["group"] = df_tmp["magazine"]
    _plot_area_composite(df_tmp, "group", MAGAZINE_PALETTE, output_dir,
                         "PF07", "magazine")


def plot_area_by_category(df: pd.DataFrame, output_dir: Path) -> None:
    df_tmp = df.copy()
    df_tmp["group"] = df_tmp["predicted_category"]
    _plot_area_composite(df_tmp, "group", CATEGORY_PALETTE, output_dir,
                         "PF08", "category")


# ═══════════════════════════════════════════════════════════════════════════
# fig_people_prevalence_ppipp: PPI++-corrected person prevalence
# ═══════════════════════════════════════════════════════════════════════════

def _wilson_ci(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (0.0, 1.0)
    p = k / n
    denom = 1 + z**2 / n
    centre = (p + z**2 / (2 * n)) / denom
    margin = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
    return (max(0.0, centre - margin), min(1.0, centre + margin))


def _load_val_for_ppi(validation_csv: Path) -> pd.DataFrame:
    vdf = pd.read_csv(validation_csv)
    vdf = vdf[vdf["ann_answered"].astype(bool)].copy()
    vdf["decade"] = (vdf["year"].astype(float) // 10 * 10).astype(int)
    vdf["gt"] = (vdf["ann_n_persons"] > 0).astype(int)
    vdf["yhat"] = vdf["yolo_detected"].astype(int)
    return vdf


def _compute_decade_prevalence_ppi(
    det_df: pd.DataFrame, val_df: pd.DataFrame,
    min_n_labeled: int = 5, min_n: int = 20,
) -> pd.DataFrame:
    from magazineads.ppi import get_ppi_estimate

    rows = []
    for decade, det_sub in det_df.groupby("decade"):
        N = len(det_sub)
        if N < min_n:
            continue

        Yu = det_sub["yhat"].values.astype(float)
        f = float(Yu.mean())

        val_sub = val_df[val_df["decade"] == decade]
        n = len(val_sub)

        raw_lo, raw_hi = _wilson_ci(int(Yu.sum()), N)

        if n < min_n_labeled:
            rows.append({
                "decade": decade, "n": N, "n_labeled": n,
                "raw_rate": f, "raw_lo": raw_lo, "raw_hi": raw_hi,
                "corrected": float("nan"), "corrected_lo": float("nan"),
                "corrected_hi": float("nan"), "lam": float("nan"),
                "ppi_available": False,
            })
            continue

        Y = val_sub["gt"].values.astype(float)
        Yhat = val_sub["yhat"].values.astype(float)
        n_det = int(Yhat.sum())
        n_undet = n - n_det
        if n_det > 0 and n_undet > 0:
            w = np.where(Yhat == 1, f / (n_det / n), (1 - f) / (n_undet / n))
        else:
            w = np.ones(n)

        ppi = get_ppi_estimate(Y, Yhat, Yu, weights=w, lam=None)
        rows.append({
            "decade": decade, "n": N, "n_labeled": n,
            "raw_rate": f, "raw_lo": raw_lo, "raw_hi": raw_hi,
            "corrected": ppi.estimate,
            "corrected_lo": ppi.ci_lo, "corrected_hi": ppi.ci_hi,
            "lam": ppi.lam, "ppi_available": True,
        })

    return pd.DataFrame(rows)


_LABEL_RAW = "Raw YOLO rate"
_LABEL_CI  = "PPI++ 95% CI"
_LABEL_EST = "PPI++ estimate"


def _draw_prevalence(ax: plt.Axes, prev: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    color_raw = "#999999"
    color_corr = GENERAL_PALETTE[0]

    idx = prev.set_index("decade")
    decades = prev["decade"].sort_values().values
    raw = idx["raw_rate"].reindex(decades).values * 100
    corr = idx["corrected"].reindex(decades).values * 100
    lo = idx["corrected_lo"].reindex(decades).values * 100
    hi = idx["corrected_hi"].reindex(decades).values * 100
    lams = idx["lam"].reindex(decades).values
    valid = ~np.isnan(corr)

    ax.plot(decades, raw, color=color_raw, lw=3, ls="--",
            label=_LABEL_RAW, alpha=0.7)
    if valid.any():
        ax.fill_between(decades[valid], lo[valid], hi[valid],
                        color=color_corr, alpha=0.15, label=_LABEL_CI)
        ax.plot(decades[valid], corr[valid], color=color_corr, lw=3,
                label=_LABEL_EST)

    offsets_y = [12, 12, 12, 12, -18, 12, 12, 12, -18, 12]
    for i, dec in enumerate(decades):
        if valid[i]:
            oy = offsets_y[i] if i < len(offsets_y) else 12
            va = "bottom" if oy > 0 else "top"
            ax.annotate(
                f"$\\lambda$={lams[i]:.2f}",
                xy=(dec, corr[i]),
                xytext=(0, oy),
                textcoords="offset points",
                ha="center", va=va,
                fontsize=TICK_SIZE - 4, fontfamily=NUM_FONT,
                color="#444444",
            )

    ax.set_ylabel("Percentage of ads containing\n≥1 person", fontsize=LABEL_SIZE)
    ax.set_ylim(0, 100)
    ax.tick_params(labelsize=TICK_SIZE)

    leg = ax.legend(fontsize=LEGEND_SIZE, loc="upper right")
    for text in leg.get_texts():
        text.set_fontfamily(NUM_FONT)

    return decades, valid


def plot_prevalence_standalone(prev: pd.DataFrame, output_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))
    decades, _ = _draw_prevalence(ax, prev)

    ax.set_xticks(decades)
    ax.set_xticklabels([f"{d}s" for d in decades], rotation=45, ha="right",
                       fontsize=TICK_SIZE)
    ax.set_xlabel("Decade", fontsize=LABEL_SIZE)
    ax.spines[["top", "right"]].set_visible(False)

    _save(fig, output_dir / "fig_people_prevalence_ppipp")


def _compute_recall_by_decade(val_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for decade, sub in val_df.groupby("decade"):
        tp = int(sub["is_tp"].sum())
        fn = int(sub["is_fn"].sum())
        n_pos = tp + fn
        recall = tp / n_pos if n_pos > 0 else float("nan")
        rows.append({"decade": decade, "recall": recall, "n_pos": n_pos})
    return pd.DataFrame(rows)


def plot_prevalence_with_recall(
    prev: pd.DataFrame, val_df: pd.DataFrame, output_dir: Path,
) -> None:
    color_recall = GENERAL_PALETTE[2]

    fig = plt.figure(figsize=(FIG_W, FIG_H + 1.5))
    gs = gridspec.GridSpec(2, 1, height_ratios=[3.5, 1], hspace=0.08, figure=fig)

    ax_top = fig.add_subplot(gs[0])
    decades, _ = _draw_prevalence(ax_top, prev)

    ax_top.set_xticks(decades)
    ax_top.tick_params(axis="x", labelbottom=False)
    xlim = (decades[0] - 5, decades[-1] + 5)
    ax_top.set_xlim(xlim)
    ax_top.spines[["top", "right"]].set_visible(False)

    ax_bot = fig.add_subplot(gs[1])
    recall_df = _compute_recall_by_decade(val_df).set_index("decade")

    rec_decades, rec_values = [], []
    for dec in decades:
        if dec in recall_df.index and not np.isnan(recall_df.loc[dec, "recall"]):
            rec_decades.append(dec)
            rec_values.append(recall_df.loc[dec, "recall"] * 100)
    rec_decades = np.array(rec_decades)
    rec_values = np.array(rec_values)

    ax_bot.plot(rec_decades, rec_values, color=color_recall, lw=3,
                marker="o", markersize=5, markeredgecolor="#a08070",
                markerfacecolor=color_recall, zorder=3)

    for dec, rv in zip(rec_decades, rec_values):
        ax_bot.text(dec, rv + 2.5, f"{rv:.0f}%", ha="center", va="bottom",
                    fontsize=TICK_SIZE - 4, fontfamily=NUM_FONT, color="#444444")

    ax_bot.set_ylabel("YOLO\nrecall (%)", fontsize=LABEL_SIZE)
    ax_bot.set_xlabel("Decade", fontsize=LABEL_SIZE)
    ax_bot.set_xticks(decades)
    ax_bot.set_xticklabels([f"{d}s" for d in decades], rotation=45, ha="right",
                           fontsize=TICK_SIZE)
    ax_bot.set_xlim(xlim)
    ax_bot.set_ylim(0, 110)
    ax_bot.tick_params(labelsize=TICK_SIZE)
    ax_bot.spines[["top", "right"]].set_visible(False)

    ax_top.yaxis.set_label_coords(-0.08, 0.5)
    ax_bot.yaxis.set_label_coords(-0.08, 0.5)

    _save(fig, output_dir / "fig_people_prevalence_ppipp_recall")


# ═══════════════════════════════════════════════════════════════════════════
# fig_03d: Ad footprint (area-weighted small multiples)
# ═══════════════════════════════════════════════════════════════════════════

SPREAD_THRESHOLD = 0.4
SMOOTH_WINDOW    = 5

DEFAULT_SPREAD_CSV = Path("/data/laviniad/magazines/page_classification_predictions/spread_predictions_v5.csv")
DEFAULT_BB_CSV     = Path("/data/laviniad/magazines/bounding_boxes/mixed_page_boxes.csv")


def _issue_date_from_stem(stem: str) -> str | None:
    parts = stem.split("-")
    try:
        return f"{int(parts[0]):04d}-{int(parts[1]):02d}-{int(parts[2]):02d}"
    except (IndexError, ValueError):
        return None


def _year_from_stem(stem: str) -> int | None:
    parts = stem.split("-")
    try:
        return int(parts[0])
    except (IndexError, ValueError):
        return None


def load_footprint_data(
    predictions_csv: Path, spread_csv: Path, bb_csv: Path, cache_csv: Path,
) -> pd.DataFrame:
    """Build per-issue DataFrame with ad area footprint.

    Delegates heavy partial-area computation to plot_ads_per_issue.compute_partial_ad_areas.
    """
    from magazineads.scripts.plot_ads_per_issue import compute_partial_ad_areas

    preds = pd.read_csv(predictions_csv, usecols=["filepath", "pred_label"])
    preds["stem"] = preds["filepath"].apply(lambda f: Path(f).stem)
    preds["magazine"] = preds["filepath"].str.extract(r"/images/([^/]+)/")[0]
    preds["issue_date"] = preds["stem"].apply(_issue_date_from_stem)
    preds["year"] = preds["stem"].apply(_year_from_stem)
    preds = preds[preds["magazine"].isin(MAG_ORDER)].dropna(subset=["issue_date"]).copy()

    fp = preds[preds["pred_label"] == 1]
    fp_footprint = (
        fp.groupby(["magazine", "issue_date"]).size()
        .reset_index(name="footprint_fullpage")
        .astype({"footprint_fullpage": float})
    )

    partial_areas = compute_partial_ad_areas(bb_csv, cache_csv)
    partial_areas["issue_date"] = partial_areas["stem"].apply(_issue_date_from_stem)
    partial_areas = partial_areas.dropna(subset=["issue_date"])
    partial_areas = partial_areas[partial_areas["magazine"].isin(MAG_ORDER)]
    partial_footprint = (
        partial_areas.groupby(["magazine", "issue_date"])["ad_area_frac"]
        .sum().reset_index(name="footprint_partial")
    )

    total_pages = preds.groupby(["magazine", "issue_date"]).size().reset_index(name="n_pages")
    issues = preds[["magazine", "issue_date", "year"]].drop_duplicates()
    df = issues.merge(fp_footprint, on=["magazine", "issue_date"], how="left")
    df = df.merge(partial_footprint, on=["magazine", "issue_date"], how="left")
    df = df.merge(total_pages, on=["magazine", "issue_date"], how="left")

    df["footprint_fullpage"] = df["footprint_fullpage"].fillna(0.0)
    df["footprint_partial"] = df["footprint_partial"].fillna(0.0)
    df["n_pages"] = df["n_pages"].fillna(0).astype(int)
    df["footprint_total"] = df["footprint_fullpage"] + df["footprint_partial"]
    return df.sort_values(["magazine", "issue_date"]).reset_index(drop=True)


def plot_ad_footprint(df: pd.DataFrame, output_dir: Path) -> None:
    import matplotlib.ticker as mticker

    df = df.copy()
    df["footprint_frac"] = df["footprint_total"] / df["n_pages"].replace(0, np.nan)
    df["footprint_fp_frac"] = df["footprint_fullpage"] / df["n_pages"].replace(0, np.nan)
    df["footprint_part_frac"] = df["footprint_partial"] / df["n_pages"].replace(0, np.nan)

    cols = ["footprint_frac", "footprint_fp_frac", "footprint_part_frac"]
    yearly = df.groupby(["magazine", "year"])[cols].mean().reset_index()

    subplot_w = FIG_W / len(MAG_ORDER)
    fig, axes = plt.subplots(1, len(MAG_ORDER),
                              figsize=(FIG_W, FIG_H), sharey=True)

    type_order = ["full-page", "partial"]
    type_labels = {"full-page": "Full-page ads", "partial": "Partial ads"}
    col_map = {"full-page": "footprint_fp_frac", "partial": "footprint_part_frac"}

    for i, mag in enumerate(MAG_ORDER):
        ax = axes[i]
        g = yearly[yearly["magazine"] == mag].sort_values("year")
        if g.empty:
            continue

        years = g["year"].values
        smoothed = {}
        for ad_type in type_order:
            raw = pd.Series(g[col_map[ad_type]].values)
            smoothed[ad_type] = raw.rolling(SMOOTH_WINDOW, center=True, min_periods=1).mean().values

        bottom = np.zeros(len(years))
        for ad_type in type_order:
            ax.fill_between(
                years, bottom, bottom + smoothed[ad_type],
                color=TYPE_COLORS[ad_type],
                label=type_labels[ad_type] if i == 0 else None,
                alpha=0.85, linewidth=0.5, edgecolor="white",
            )
            bottom += smoothed[ad_type]

        ax.set_xlim(YEAR_MIN, YEAR_MAX)
        ax.set_ylim(0, 1)
        ax.xaxis.set_major_locator(mticker.MultipleLocator(40))
        ax.xaxis.set_major_formatter(mticker.FormatStrFormatter("%d"))
        ax.text(0.5, 0.95, DISPLAY[mag], transform=ax.transAxes,
                ha="center", va="top", fontsize=MAG_LABEL_SIZE,
                fontweight="bold", color=MAGAZINE_PALETTE[mag])
        ax.set_xlabel("Year", fontsize=LABEL_SIZE)
        ax.tick_params(labelsize=TICK_SIZE - 6)

        if i > 0:
            ax.tick_params(labelleft=False)

    axes[0].set_ylabel("Ad footprint\n(% of issue space)", fontsize=LABEL_SIZE)
    axes[0].yaxis.set_major_locator(mticker.MultipleLocator(0.2))
    axes[0].yaxis.set_major_formatter(
        mticker.FuncFormatter(lambda x, _: f"{x * 100:.0f}%")
    )

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2,
               fontsize=LEGEND_SIZE, bbox_to_anchor=(0.5, -0.04), frameon=False)

    _save(fig, output_dir / "fig_03d_ad_footprint")

    csv_cols = ["magazine", "year"] + cols
    yearly[csv_cols].to_csv(output_dir / "fig_03d_ad_footprint.csv", index=False)


# ═══════════════════════════════════════════════════════════════════════════
# fig_04: Page classifier confusion matrix
# ═══════════════════════════════════════════════════════════════════════════

_NUM_ROUNDS = 9
_ANNOTATORS = ["lavinia", "sravani", "dallas"]


def _normalize_label(raw) -> int | None:
    if isinstance(raw, (int, float)):
        v = int(raw)
        return 1 if v == 3 else v
    if isinstance(raw, str):
        return LABEL_MAP.get(raw.strip())
    return None


def _page_key(fname: str) -> tuple[int, ...] | None:
    parts = Path(fname).stem.split("-")
    if len(parts) < 4:
        return None
    try:
        return (int(parts[0]), int(parts[1]), int(parts[2]), int(parts[-1]))
    except ValueError:
        return None


def load_annotations(annotation_dir: Path, magazines: list[str],
                     splits: list[str] = ("test",)) -> pd.DataFrame:
    rows = []
    for mag in magazines:
        for rnd in range(1, _NUM_ROUNDS + 1):
            for split in splits:
                for ann in _ANNOTATORS:
                    p = annotation_dir / f"round{rnd}_{mag}_{ann}.{split}.csv"
                    if not p.exists():
                        continue
                    try:
                        df = pd.read_csv(p)
                    except Exception:
                        continue
                    if "Label" not in df.columns:
                        continue
                    fname_col = df.columns[0]
                    for _, row in df.iterrows():
                        lbl = _normalize_label(row["Label"])
                        if lbl is not None:
                            rows.append({"filename": str(row[fname_col]).strip(),
                                         "magazine": mag, "gt_label": lbl})
                if mag == "life":
                    legacy = annotation_dir / f"round{rnd}_life.{split}.csv"
                    if legacy.exists():
                        try:
                            df = pd.read_csv(legacy)
                        except Exception:
                            continue
                        if "Label" not in df.columns:
                            continue
                        fname_col = df.columns[0]
                        for _, row in df.iterrows():
                            lbl = _normalize_label(row["Label"])
                            if lbl is not None:
                                rows.append({"filename": str(row[fname_col]).strip(),
                                             "magazine": "life", "gt_label": lbl})

    if not rows:
        log.error("No annotation rows loaded. Check annotation_dir.")
        return pd.DataFrame()

    ann = pd.DataFrame(rows).drop_duplicates(subset=["filename", "magazine"])
    log.info("Loaded %d annotated images.", len(ann))
    return ann


def _match_predictions(ann: pd.DataFrame, predictions_csv: Path) -> pd.DataFrame:
    preds = pd.read_csv(predictions_csv, usecols=["filepath", "pred_label"])
    preds["magazine"] = preds["filepath"].str.extract(r"/images/([^/]+)/")
    preds["_key"] = preds["filepath"].apply(lambda p: _page_key(Path(p).name))
    preds = preds.dropna(subset=["_key"]).drop_duplicates(subset=["magazine", "_key"])

    ann = ann.copy()
    ann["_key"] = ann["filename"].apply(_page_key)
    ann = ann.dropna(subset=["_key"])

    merged = ann.merge(
        preds[["magazine", "_key", "pred_label"]],
        on=["magazine", "_key"], how="inner",
    ).drop(columns=["_key"])

    n_missing = len(ann) - len(merged)
    if n_missing:
        log.warning("%d annotated images not found in predictions CSV.", n_missing)
    log.info("Matched %d annotated images to predictions.", len(merged))
    return merged


def plot_classifier_confusion(merged: pd.DataFrame, output_dir: Path) -> None:
    from sklearn.metrics import confusion_matrix

    present = sorted(set(merged["gt_label"]) | set(merged["pred_label"]))
    class_names = [CLASSIFIER_CLASS_NAMES[c] for c in present]

    cm = confusion_matrix(merged["gt_label"], merged["pred_label"], labels=present)

    fig, ax = plt.subplots(figsize=(7, 5.5))
    plot_confusion_matrix(cm, class_names, ax, normalize=False)
    ax.tick_params(labelsize=TICK_SIZE)

    _save(fig, output_dir / "fig_04_classifier_confusion")

    cm_df = pd.DataFrame(cm,
                         index=[f"true_{CLASSIFIER_CLASS_NAMES[c]}" for c in present],
                         columns=[f"pred_{CLASSIFIER_CLASS_NAMES[c]}" for c in present])
    cm_df.to_csv(output_dir / "fig_04_classifier_confusion.csv")


# ═══════════════════════════════════════════════════════════════════════════
# fig_S05: Spread classifier confusion matrix (2×3 grid)
# ═══════════════════════════════════════════════════════════════════════════

DEFAULT_SPREAD_METRICS = Path("/data/laviniad/magazines/models/spread_classifier_v5/diff/training_metrics.json")


def plot_spread_confusion(metrics_json: Path, output_dir: Path) -> None:
    import json
    from sklearn.metrics import confusion_matrix as skl_cm, f1_score

    with open(metrics_json) as f:
        metrics = json.load(f)

    opt_t = metrics["optimal_threshold"]
    per_mag = metrics["per_magazine"]

    val_probs = metrics["val_probs"]
    val_labels = metrics["val_labels"]
    preds = [1 if p > opt_t else 0 for p in val_probs]
    overall_cm = skl_cm(val_labels, preds, labels=[0, 1])
    overall_f1 = f1_score(val_labels, preds, pos_label=1)

    panels = [("Overall", overall_cm, overall_f1, len(val_labels), "#333333")]
    for mag in SPREAD_MAGAZINE_ORDER:
        info = per_mag[mag]
        cm = np.array(info["confusion_matrix_at_optimal"])
        f1 = info["spread_f1_at_optimal"]
        n = info["n_val"]
        color = MAGAZINE_PALETTE.get(mag, "#333333")
        panels.append((magazine_label(mag), cm, f1, n, color))

    fig, axes = plt.subplots(2, 3, figsize=(FIG_W, FIG_H * 2))

    csv_rows = []
    for idx, (label, cm, f1, n, color) in enumerate(panels):
        ax = axes[idx // 3][idx % 3]
        plot_confusion_matrix(cm, SPREAD_CLASS_NAMES, ax=ax, title="")
        ax.set_title(label, fontsize=MAG_LABEL_SIZE, fontweight="bold",
                     color=color, pad=24)
        ax.text(0.5, 1.04, f"F1 = {f1:.3f}   n = {n}",
                transform=ax.transAxes, ha="center", va="bottom",
                fontsize=TICK_SIZE - 4, fontfamily=NUM_FONT, color="#555555")
        ax.tick_params(labelsize=TICK_SIZE - 4)

        for i, actual in enumerate(SPREAD_CLASS_KEYS):
            for j, predicted in enumerate(SPREAD_CLASS_KEYS):
                csv_rows.append({
                    "panel": label.lower(), "actual": actual,
                    "predicted": predicted, "count": int(cm[i, j]),
                    "spread_f1": f1, "n_val": n, "threshold": opt_t,
                })

    fig.tight_layout(h_pad=3.5, w_pad=2.5)
    fix_fonts(fig)
    stem = output_dir / "fig_S05_spread_confusion"
    for suffix in (".pdf", ".png"):
        fig.savefig(str(stem) + suffix, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    log.info("saved %s (.pdf + .png)", stem.stem)

    pd.DataFrame(csv_rows).to_csv(str(stem) + ".csv", index=False)


# ═══════════════════════════════════════════════════════════════════════════
# fig_PC: Product category prevalence over time
# ═══════════════════════════════════════════════════════════════════════════

def _load_prodcat_for_time(prodcat_csv: Path, min_confidence: float) -> pd.DataFrame:
    df = pd.read_csv(prodcat_csv)
    df["year"] = df["unit_id"].apply(lambda s: int(s.split("-")[0]))
    low_conf = df["confidence"] < min_confidence
    n_low = low_conf.sum()
    df.loc[low_conf, "predicted_category"] = "Other"
    log.info("Reclassified %d/%d (%.1f%%) low-confidence as Other",
             n_low, len(df), n_low / len(df) * 100)
    return df


def plot_product_category_over_time(prodcat_csv: Path, min_confidence: float,
                                     output_dir: Path) -> None:
    import matplotlib.ticker as mticker

    df = _load_prodcat_for_time(prodcat_csv, min_confidence)

    # Top N categories by count (excluding Other)
    counts = df["predicted_category"].value_counts().drop("Other", errors="ignore")
    top_cats = counts.head(DIACHRONIC_N).index.tolist()

    df["period"] = (df["year"] // 5) * 5
    period_counts = df["period"].value_counts()
    valid_periods = period_counts[period_counts >= 50].index
    df = df[df["period"].isin(valid_periods)]

    fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))

    periods = sorted(df["period"].unique())
    totals = df.groupby("period").size()

    for i, cat in enumerate(top_cats):
        cat_counts = df[df["predicted_category"] == cat].groupby("period").size()
        props = (cat_counts / totals).reindex(periods, fill_value=0)
        color = PRODCAT_COLORS[i] if i < len(PRODCAT_COLORS) else "#999"
        ax.plot(periods, props.values, marker="o", markersize=5,
                color=color, label=cat, linewidth=2.5)

    # Other line
    other_counts = df[~df["predicted_category"].isin(top_cats)].groupby("period").size()
    other_props = (other_counts / totals).reindex(periods, fill_value=0)
    ax.plot(periods, other_props.values, marker="s", markersize=5,
            color=PRODCAT_COLORS[-1], label="Other", linewidth=3, linestyle="--")

    ax.set_xlabel("Year", fontsize=LABEL_SIZE)
    ax.set_ylabel("Percentage of full-page ads", fontsize=LABEL_SIZE)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v*100:.0f}%"))
    ax.tick_params(labelsize=TICK_SIZE)
    ax.spines[["top", "right"]].set_visible(False)
    ax.yaxis.grid(True, linestyle="--", alpha=0.3)
    ax.set_axisbelow(True)
    ax.legend(fontsize=LEGEND_SIZE - 6, bbox_to_anchor=(1.01, 1),
              loc="upper left", borderaxespad=0)

    _save(fig, output_dir / "fig_PC_product_category_over_time")

    # Companion CSV
    csv_rows = []
    for period in periods:
        row = {"period": period, "n_total": int(totals.loc[period])}
        for cat in top_cats + ["Other"]:
            if cat == "Other":
                c = other_counts.get(period, 0)
            else:
                c = df[df["predicted_category"] == cat].groupby("period").size().get(period, 0)
            row[cat] = float(c / totals.loc[period])
        csv_rows.append(row)
    pd.DataFrame(csv_rows).to_csv(
        output_dir / "fig_PC_product_category_over_time.csv", index=False)


# ═══════════════════════════════════════════════════════════════════════════
# fig_PC02: Product category prevalence by magazine (bar chart)
# ═══════════════════════════════════════════════════════════════════════════

def plot_product_category_by_magazine(prodcat_csv: Path, min_confidence: float,
                                      output_dir: Path) -> None:
    """Bar chart: category share within each magazine."""
    import matplotlib.ticker as mticker

    df = _load_prodcat_for_time(prodcat_csv, min_confidence)
    # Exclude Variety (dominated by Media, distorts scale)
    df = df[df["magazine"] != "variety"]

    # All categories sorted by overall count, with Other last
    counts = df["predicted_category"].value_counts()
    named_cats = [c for c in counts.index if c != "Other"] + ["Other"]

    mags = [m for m in MAG_ORDER if m != "variety" and m in df["magazine"].unique()]

    # Compute proportions: category share within each magazine
    ct = pd.crosstab(df["magazine"], df["predicted_category"], normalize="index")
    ct = ct.reindex(index=mags, fill_value=0)

    # Assign colours: use PRODCAT_COLORS for named categories, grey for Other
    cat_colors = {}
    for i, cat in enumerate(named_cats):
        if cat == "Other":
            cat_colors[cat] = "#999999"
        else:
            cat_colors[cat] = PRODCAT_COLORS[i] if i < len(PRODCAT_COLORS) else "#999999"

    n_cats = len(named_cats)
    n_mags = len(mags)
    x = np.arange(n_mags)
    bar_w = 0.8 / n_cats

    fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))

    for i, cat in enumerate(named_cats):
        vals = ct[cat].reindex(mags, fill_value=0).values
        ax.bar(x + i * bar_w, vals, bar_w,
               color=cat_colors[cat], label=cat,
               edgecolor="white", linewidth=0.3)

    ax.set_xticks(x + bar_w * (n_cats - 1) / 2)
    ax.set_xticklabels([DISPLAY[m] for m in mags], fontsize=TICK_SIZE)
    ax.set_ylabel("Percentage of full-page ads", fontsize=LABEL_SIZE)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v*100:.0f}%"))
    ax.tick_params(labelsize=TICK_SIZE)
    ax.spines[["top", "right"]].set_visible(False)
    ax.yaxis.grid(True, linestyle="--", alpha=0.3)
    ax.set_axisbelow(True)
    ax.legend(fontsize=LEGEND_SIZE - 4, loc="upper left",
              bbox_to_anchor=(1.01, 1), borderaxespad=0)

    _save(fig, output_dir / "fig_PC02_product_category_by_magazine")

    # Companion CSV
    csv_df = ct[named_cats].copy()
    csv_df.index.name = "magazine"
    n_per_mag = df.groupby("magazine").size().reindex(mags)
    csv_df.insert(0, "n_ads", n_per_mag.values)
    csv_df.to_csv(output_dir / "fig_PC02_product_category_by_magazine.csv")


# ═══════════════════════════════════════════════════════════════════════════
# fig_03: Dataset composition — page-type counts per magazine (grouped bar)
# ═══════════════════════════════════════════════════════════════════════════

# Colours keyed by pred_label int
_CLASS_COLOR = {
    0: GENERAL_PALETTE[2],   # tan  → Non-Ad
    1: GENERAL_PALETTE[0],   # dark burgundy → Ad
    2: GENERAL_PALETTE[1],   # light blue → Mixed
}


def plot_dataset_composition(predictions_csv: Path, output_dir: Path) -> None:
    """Grouped bar chart: page-type counts per magazine (was stacked bar)."""
    import matplotlib.ticker as mticker

    preds = pd.read_csv(predictions_csv, usecols=["filepath", "pred_label"])
    preds["magazine"] = preds["filepath"].str.extract(r"/images/([^/]+)/")
    ad_counts = (
        preds[preds["magazine"].isin(MAG_ORDER)]
        .groupby(["magazine", "pred_label"])
        .size()
        .unstack(fill_value=0)
        .reindex(MAG_ORDER, fill_value=0)
    )
    ad_counts.columns = [int(c) for c in ad_counts.columns]
    ad_counts = ad_counts.rename(columns={0: "non_ad", 1: "ad", 2: "mixed"})
    for col in ("non_ad", "ad", "mixed"):
        if col not in ad_counts.columns:
            ad_counts[col] = 0

    labels = [DISPLAY[m] for m in MAG_ORDER]
    x = np.arange(len(MAG_ORDER))
    n_groups = 3
    bar_w = 0.25

    fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))
    for i, (cls_key, label_text, label_int) in enumerate([
        ("non_ad", "Non-Ad", 0), ("mixed", "Mixed", 2), ("ad", "Ad", 1),
    ]):
        vals = ad_counts[cls_key].values
        ax.bar(x + i * bar_w, vals, width=bar_w,
               color=_CLASS_COLOR[label_int], label=label_text,
               edgecolor="white", linewidth=0.5)

    ax.set_xticks(x + bar_w)
    ax.set_xticklabels(labels, fontsize=TICK_SIZE)
    ax.set_ylabel("Pages", fontsize=LABEL_SIZE)
    ax.tick_params(labelsize=TICK_SIZE)
    ax.spines[["top", "right"]].set_visible(False)
    ax.yaxis.grid(True, linestyle="--", alpha=0.3)
    ax.set_axisbelow(True)
    ax.legend(fontsize=LEGEND_SIZE, loc="upper right")

    _save(fig, output_dir / "fig_03_dataset_composition")

    ad_counts.reset_index().to_csv(
        output_dir / "fig_03_dataset_composition.csv", index=False)


# ═══════════════════════════════════════════════════════════════════════════
# fig_person_count_categories: Person count distribution over time (stacked)
# ═══════════════════════════════════════════════════════════════════════════

PERSON_COUNT_COLORS = {
    "0":  "#4A6274",   # slate blue-grey
    "1":  "#C07A4E",   # warm terracotta
    "2+": "#7A9E5C",   # olive green
}


def _prepare_person_count_data(people_csv: Path):
    """Prepare smoothed person-count fractions. Returns (years, smoothed, ct, frac, totals)."""
    df = pd.read_csv(people_csv)
    df["year"] = df["year"].astype(int)
    df["category"] = np.where(
        df["n_persons"] == 0, "0",
        np.where(df["n_persons"] == 1, "1", "2+"),
    )

    ct = df.groupby(["year", "category"]).size().unstack(fill_value=0)
    for cat in ["0", "1", "2+"]:
        if cat not in ct.columns:
            ct[cat] = 0
    ct = ct[["0", "1", "2+"]]
    totals = ct.sum(axis=1)
    frac = ct.div(totals, axis=0)

    smoothed = {}
    for cat in ["0", "1", "2+"]:
        smoothed[cat] = frac[cat].rolling(SMOOTH_WINDOW, center=True, min_periods=2).mean()
    smooth_total = smoothed["0"] + smoothed["1"] + smoothed["2+"]
    smooth_total = smooth_total.replace(0, np.nan)
    for cat in ["0", "1", "2+"]:
        smoothed[cat] = smoothed[cat] / smooth_total

    return frac.index.values, smoothed, ct, frac, totals


def _draw_person_count(ax, years, smoothed, font_bump: int = 0) -> None:
    """Draw person count stacked area on the given axes."""
    import matplotlib.ticker as mticker

    cat_labels = {"0": "No people", "1": "One person", "2+": "Two or more"}

    ax.stackplot(
        years,
        smoothed["0"].values,
        smoothed["1"].values,
        smoothed["2+"].values,
        colors=[PERSON_COUNT_COLORS["0"], PERSON_COUNT_COLORS["1"],
                PERSON_COUNT_COLORS["2+"]],
        alpha=0.85,
    )

    # Place labels inside each band at the horizontal midpoint
    mid_x = int(np.median(years))
    mid_idx = np.argmin(np.abs(years - mid_x))
    bottom = 0.0
    for cat in ["0", "1", "2+"]:
        band_h = smoothed[cat].values[mid_idx]
        y_center = bottom + band_h / 2
        ax.text(mid_x, y_center, cat_labels[cat],
                ha="center", va="center", fontsize=LEGEND_SIZE + font_bump,
                fontweight="bold", color="white")
        bottom += band_h

    ax.set_xlim(YEAR_MIN, YEAR_MAX)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Year", fontsize=LABEL_SIZE + font_bump)
    ax.set_ylabel("Percentage of full-page ads", fontsize=LABEL_SIZE + font_bump)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v*100:.0f}%"))
    decade_ticks = list(range(1930, 2020, 10))
    ax.set_xticks(decade_ticks)
    ax.set_xticklabels([str(y) for y in decade_ticks],
                       fontfamily=NUM_FONT, fontsize=TICK_SIZE + font_bump)
    ax.tick_params(axis="y", labelsize=TICK_SIZE + font_bump)
    ax.spines[["top", "right"]].set_visible(False)


def _draw_person_area(ax, df: pd.DataFrame, palette: dict,
                      add_legend: bool = True, font_bump: int = 0) -> None:
    """Draw person area by group (magazine or category) on the given axes."""
    agg = _agg_area(df, "group")
    if agg.empty:
        return

    msub = agg[agg["metric"] == "Average space per person"]
    if msub.empty:
        return

    groups = sorted(msub["group"].unique(),
                    key=lambda g: -msub[msub["group"] == g]["n"].sum())
    decades = sorted(msub["decade"].unique())

    for grp in groups:
        sub = msub[msub["group"] == grp].sort_values("decade")
        if len(sub) < 2:
            continue
        color = palette.get(grp, "#808080")
        label_text = magazine_label(grp) if grp in MAGAZINE_PALETTE else grp
        ax.fill_between(sub["decade"], sub["mean"] - sub["se"],
                        sub["mean"] + sub["se"],
                        color=color, alpha=0.10)
        ax.plot(sub["decade"], sub["mean"], color=color, lw=3,
                marker="o", ms=4, label=label_text)

    ax.tick_params(labelsize=TICK_SIZE + font_bump)
    ax.spines[["top", "right"]].set_visible(False)
    ax.yaxis.grid(True, ls="--", alpha=0.3)
    ax.set_axisbelow(True)
    ax.set_xticks(decades)
    ax.set_xticklabels([f"{d}s" for d in decades], rotation=45, ha="right")
    ax.set_xlabel("Decade", fontsize=LABEL_SIZE + font_bump)
    ax.set_ylabel("Average fraction of ad\nper person", fontsize=LABEL_SIZE + font_bump)

    if add_legend:
        ax.legend(fontsize=LEGEND_SIZE + font_bump, loc="upper left",
                  bbox_to_anchor=(1.01, 1), borderaxespad=0)


def plot_person_count_categories(people_csv: Path, output_dir: Path) -> None:
    """Standalone person count stacked area plot."""
    years, smoothed, ct, frac, totals = _prepare_person_count_data(people_csv)

    fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))
    _draw_person_count(ax, years, smoothed)
    _save(fig, output_dir / "fig_person_count_categories")

    combined = ct.copy()
    combined.columns = [f"n_{c}" for c in combined.columns]
    combined["total"] = totals
    for cat in ["0", "1", "2+"]:
        combined[f"frac_{cat}"] = frac[cat]
    combined.to_csv(output_dir / "fig_person_count_categories.csv")


def plot_person_count_and_area(people_csv: Path, df_people: pd.DataFrame,
                               output_dir: Path) -> None:
    """Composite: person count (left) + person area by magazine (right)."""
    years, smoothed, ct, frac, totals = _prepare_person_count_data(people_csv)

    df_tmp = df_people.copy()
    df_tmp["group"] = df_tmp["magazine"]

    fig, axes = plt.subplots(1, 2, figsize=(FIG_W * 2, FIG_H * 1.3))

    _draw_person_count(axes[0], years, smoothed, font_bump=2)
    _draw_person_area(axes[1], df_tmp, MAGAZINE_PALETTE, add_legend=True, font_bump=2)

    _save(fig, output_dir / "fig_PCPA_person_count_and_area")

    combined = ct.copy()
    combined.columns = [f"n_{c}" for c in combined.columns]
    combined["total"] = totals
    for cat in ["0", "1", "2+"]:
        combined[f"frac_{cat}"] = frac[cat]
    combined.to_csv(output_dir / "fig_PCPA_person_count.csv")
    agg = _agg_area(df_tmp, "group")
    agg[agg["metric"] == "Average space per person"].to_csv(
        output_dir / "fig_PCPA_person_area.csv", index=False)


# ═══════════════════════════════════════════════════════════════════════════
# fig_S06: Ad-type composition over time (line, per-magazine small multiples)
# fig_S06b: Ad-type counts over time (line, per-magazine small multiples)
# ═══════════════════════════════════════════════════════════════════════════

AD_TYPE_COLORS = {
    "spread":          GENERAL_PALETTE[0],   # dark burgundy
    "single_fullpage": GENERAL_PALETTE[1],   # light blue
    "partial":         GENERAL_PALETTE[2],   # tan
}
AD_TYPE_LABELS = {
    "spread":          "Two-page spread",
    "single_fullpage": "Single full-page",
    "partial":         "Partial ad",
}


def _load_ad_type_data(predictions_csv: Path, spread_csv: Path) -> pd.DataFrame:
    """Build per-page DataFrame with magazine, year, ad_type."""
    preds = pd.read_csv(predictions_csv, usecols=["filepath", "pred_label"])
    preds = preds[preds["pred_label"].isin([1, 2])].copy()
    preds["magazine"] = preds["filepath"].str.extract(r"/images/([^/]+)/")[0]
    preds["year"] = pd.to_numeric(
        preds["filepath"].str.extract(r"/(\d{4})-")[0], errors="coerce"
    ).astype("Int64")
    preds = preds[preds["magazine"].isin(MAG_ORDER)].dropna(subset=["year"]).copy()

    spreads = pd.read_csv(spread_csv, usecols=["left_page", "right_page", "pred_spread"])
    spreads = spreads[spreads["pred_spread"] == 1]
    spread_pages: set[str] = set(spreads["left_page"]) | set(spreads["right_page"])

    def _ad_type(row):
        if row["pred_label"] == 2:
            return "partial"
        if row["filepath"] in spread_pages:
            return "spread"
        return "single_fullpage"

    preds["ad_type"] = preds.apply(_ad_type, axis=1)
    return preds[["magazine", "year", "ad_type"]].copy()


def plot_ad_type_composition(ad_type_df: pd.DataFrame, output_dir: Path) -> None:
    """Line chart small multiples: ad-type fractions over time (was stacked area)."""
    import matplotlib.ticker as mticker

    counts = (
        ad_type_df.groupby(["magazine", "year", "ad_type"])
        .size().reset_index(name="n")
    )
    totals = ad_type_df.groupby(["magazine", "year"]).size().reset_index(name="total")
    counts = counts.merge(totals, on=["magazine", "year"])
    counts["frac"] = counts["n"] / counts["total"]

    type_order = ["spread", "single_fullpage", "partial"]
    subplot_w = FIG_W / len(MAG_ORDER)

    fig, axes = plt.subplots(1, len(MAG_ORDER),
                              figsize=(FIG_W, FIG_H), sharey=True)

    csv_rows = []
    for i, mag in enumerate(MAG_ORDER):
        ax = axes[i]
        mag_data = counts[counts["magazine"] == mag]
        if mag_data.empty:
            ax.set_visible(False)
            continue

        year_min, year_max = int(mag_data["year"].min()), int(mag_data["year"].max())
        all_years = np.arange(year_min, year_max + 1)

        for ad_type in type_order:
            sub = mag_data[mag_data["ad_type"] == ad_type].set_index("year")["frac"]
            series = sub.reindex(all_years).fillna(0)
            smoothed = series.rolling(SMOOTH_WINDOW, center=True, min_periods=1).mean()

            ax.plot(smoothed.index, smoothed.values,
                    color=AD_TYPE_COLORS[ad_type], linewidth=3,
                    label=AD_TYPE_LABELS[ad_type] if i == 0 else None)

        ax.set_xlim(year_min, year_max)
        ax.set_ylim(0, 1)
        ax.set_xlabel("Year", fontsize=LABEL_SIZE)
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v*100:.0f}%"))
        ax.xaxis.set_major_locator(mticker.MultipleLocator(40))
        ax.xaxis.set_major_formatter(mticker.FormatStrFormatter("%d"))
        ax.text(0.5, 0.95, DISPLAY[mag], transform=ax.transAxes,
                ha="center", va="top", fontsize=MAG_LABEL_SIZE,
                fontweight="bold", color=MAGAZINE_PALETTE[mag])
        ax.tick_params(labelsize=TICK_SIZE - 6)
        if i > 0:
            ax.tick_params(labelleft=False)

        for ad_type in type_order:
            sub = mag_data[mag_data["ad_type"] == ad_type]
            for _, row in sub.iterrows():
                csv_rows.append({"magazine": mag, "year": int(row["year"]),
                                 "ad_type": ad_type, "n": int(row["n"]),
                                 "total": int(row["total"]), "frac": row["frac"]})

    axes[0].set_ylabel("Percentage of ad pages", fontsize=LABEL_SIZE)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=len(type_order),
               fontsize=LEGEND_SIZE, bbox_to_anchor=(0.5, -0.04), frameon=False)

    _save(fig, output_dir / "fig_S06_ad_type_composition")
    pd.DataFrame(csv_rows).to_csv(
        output_dir / "fig_S06_ad_type_composition.csv", index=False)


def plot_ad_type_counts(ad_type_df: pd.DataFrame, output_dir: Path) -> None:
    """Line chart small multiples: ad-type counts over time (was stacked area)."""
    import matplotlib.ticker as mticker

    counts = (
        ad_type_df.groupby(["magazine", "year", "ad_type"])
        .size().reset_index(name="n")
    )

    type_order = ["spread", "single_fullpage", "partial"]

    fig, axes = plt.subplots(1, len(MAG_ORDER),
                              figsize=(FIG_W, FIG_H), sharey=True)

    csv_rows = []
    for i, mag in enumerate(MAG_ORDER):
        ax = axes[i]
        mag_data = counts[counts["magazine"] == mag]
        if mag_data.empty:
            ax.set_visible(False)
            continue

        year_min, year_max = int(mag_data["year"].min()), int(mag_data["year"].max())
        all_years = np.arange(year_min, year_max + 1)

        for ad_type in type_order:
            sub = mag_data[mag_data["ad_type"] == ad_type].set_index("year")["n"]
            series = sub.reindex(all_years).fillna(0)
            smoothed = series.rolling(SMOOTH_WINDOW, center=True, min_periods=1).mean()

            ax.plot(smoothed.index, smoothed.values,
                    color=AD_TYPE_COLORS[ad_type], linewidth=3,
                    label=AD_TYPE_LABELS[ad_type] if i == 0 else None)

        ax.set_xlim(year_min, year_max)
        ax.set_xlabel("Year", fontsize=LABEL_SIZE)
        ax.xaxis.set_major_locator(mticker.MultipleLocator(40))
        ax.xaxis.set_major_formatter(mticker.FormatStrFormatter("%d"))
        ax.text(0.5, 0.95, DISPLAY[mag], transform=ax.transAxes,
                ha="center", va="top", fontsize=MAG_LABEL_SIZE,
                fontweight="bold", color=MAGAZINE_PALETTE[mag])
        ax.tick_params(labelsize=TICK_SIZE - 6)
        if i > 0:
            ax.tick_params(labelleft=False)

        for ad_type in type_order:
            sub = mag_data[mag_data["ad_type"] == ad_type]
            for _, row in sub.iterrows():
                csv_rows.append({"magazine": mag, "year": int(row["year"]),
                                 "ad_type": ad_type, "n": int(row["n"])})

    axes[0].set_ylabel("Ad pages per year", fontsize=LABEL_SIZE)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=len(type_order),
               fontsize=LEGEND_SIZE, bbox_to_anchor=(0.5, -0.04), frameon=False)

    _save(fig, output_dir / "fig_S06b_ad_type_counts")
    pd.DataFrame(csv_rows).to_csv(
        output_dir / "fig_S06b_ad_type_counts.csv", index=False)


# ═══════════════════════════════════════════════════════════════════════════
# fig_SP01: Spread frequency over time (share of ads that are two-page)
# ═══════════════════════════════════════════════════════════════════════════

def plot_spread_frequency(predictions_csv: Path, spread_csv: Path,
                          output_dir: Path) -> None:
    """Line chart: fraction of ads (full-page + mixed) that are two-page spreads."""
    import matplotlib.ticker as mticker

    preds = _load_preds_yearly(predictions_csv)
    preds = preds[preds["magazine"].isin(MAG_ORDER)].copy()
    preds = preds[(preds["year"] >= 1900) & (preds["year"] <= 2025)]
    preds = preds[preds["pred_label"].isin([1, 2])].copy()

    spreads = pd.read_csv(spread_csv, usecols=["left_page", "right_page", "pred_spread"])
    spreads = spreads[spreads["pred_spread"] == 1]
    spread_pages: set[str] = set(spreads["left_page"]) | set(spreads["right_page"])

    preds["is_spread"] = preds["filepath"].isin(spread_pages).astype(int)

    rng = np.random.default_rng(42)

    rows = []
    for (mag, year), grp in preds.groupby(["magazine", "year"]):
        vals = grp["is_spread"]
        lo, hi = _bootstrap_ci(vals, rng=rng, statistic="mean")
        rows.append({
            "magazine": mag, "year": year,
            "rate": vals.mean(), "ci_lo": lo, "ci_hi": hi,
            "n": len(grp),
        })
    agg = pd.DataFrame(rows)

    all_rows = []
    for year, grp in preds.groupby("year"):
        vals = grp["is_spread"]
        lo, hi = _bootstrap_ci(vals, rng=rng, statistic="mean")
        all_rows.append({
            "year": year, "rate": vals.mean(),
            "ci_lo": lo, "ci_hi": hi,
        })
    agg_all = pd.DataFrame(all_rows).sort_values("year").set_index("year")

    fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))

    for mag in MAG_ORDER:
        sub = agg[agg["magazine"] == mag].sort_values("year")
        if sub.empty:
            continue
        color = MAGAZINE_PALETTE.get(mag, "#808080")

        ax.plot(sub["year"], sub["rate"],
                color=color, alpha=0.25, linewidth=0.8, zorder=1)

        ts = sub.set_index("year")
        rolled = ts["rate"].rolling(SMOOTH_WINDOW, center=True, min_periods=2).mean()
        ax.plot(rolled.index, rolled.values,
                color=color, linewidth=3, label=DISPLAY.get(mag, mag), zorder=2)

        lo_rolled = ts["ci_lo"].rolling(SMOOTH_WINDOW, center=True, min_periods=2).mean()
        hi_rolled = ts["ci_hi"].rolling(SMOOTH_WINDOW, center=True, min_periods=2).mean()
        ax.fill_between(lo_rolled.index, lo_rolled.values, hi_rolled.values,
                        color=color, alpha=0.12, zorder=1)

    rolled_all = agg_all["rate"].rolling(SMOOTH_WINDOW, center=True, min_periods=2).mean()
    ax.plot(rolled_all.index, rolled_all.values,
            color=AGG_COLOR, linewidth=AGG_LW, linestyle=AGG_LINESTYLE,
            label="All magazines", zorder=3)
    lo_all = agg_all["ci_lo"].rolling(SMOOTH_WINDOW, center=True, min_periods=2).mean()
    hi_all = agg_all["ci_hi"].rolling(SMOOTH_WINDOW, center=True, min_periods=2).mean()
    ax.fill_between(lo_all.index, lo_all.values, hi_all.values,
                    color=AGG_COLOR, alpha=0.08, zorder=1)

    ax.set_xlabel("Year", fontsize=LABEL_SIZE)
    ax.set_ylabel("Percentage of ad pages\nin spreads", fontsize=LABEL_SIZE)
    ax.set_ylim(bottom=0)
    ax.set_xlim(YEAR_MIN, YEAR_MAX)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v*100:.0f}%"))
    ax.tick_params(labelsize=TICK_SIZE)
    ax.spines[["top", "right"]].set_visible(False)
    ax.yaxis.grid(True, linestyle="--", alpha=0.3)
    ax.set_axisbelow(True)
    ax.legend(fontsize=LEGEND_SIZE, loc="upper left",
              bbox_to_anchor=(1.01, 1), borderaxespad=0)

    _save(fig, output_dir / "fig_SP01_spread_frequency")
    agg.to_csv(output_dir / "fig_SP01_spread_frequency.csv", index=False)


# ═══════════════════════════════════════════════════════════════════════════
# Section runners
# ═══════════════════════════════════════════════════════════════════════════

def run_coverage(args) -> None:
    log.info("=== Section: coverage ===")
    issues  = _build_issue_table(args.predictions_csv)
    summary = _compute_summary(issues)

    log.info("Generating fig_00_dataset_coverage_pages …")
    plot_coverage_pages(issues, summary, args.output_dir)

    log.info("Generating fig_00_dataset_coverage_ads …")
    plot_coverage_ads(issues, summary, args.predictions_csv, args.output_dir)

    log.info("Generating fig_01_ad_prevalence …")
    plot_ad_prevalence(args.predictions_csv, args.output_dir)

    if not args.wordcount_cache.exists():
        log.warning("Word-count cache not found: %s — skipping fig_T01. "
                     "Run text_features_over_time.py --precompute first.",
                     args.wordcount_cache)
    else:
        log.info("Generating fig_T01_words_per_ad_over_time …")
        plot_median_words(args.wordcount_cache, args.output_dir)

    if args.whitespace_csv.exists():
        log.info("Generating fig_WS01_whitespace_over_time …")
        plot_whitespace_trend(args.whitespace_csv, args.output_dir)
    else:
        log.warning("Whitespace CSV not found: %s — skipping fig_WS01.",
                     args.whitespace_csv)

    if args.edge_density_csv.exists():
        log.info("Generating fig_ED01_edge_density_over_time …")
        plot_edge_density_trend(args.edge_density_csv, args.output_dir)
    else:
        log.warning("Edge density CSV not found: %s — skipping fig_ED01.",
                     args.edge_density_csv)

    if args.whitespace_csv.exists() and args.edge_density_csv.exists():
        log.info("Generating fig_WSED_whitespace_edge_density …")
        plot_whitespace_edge_composite(args.whitespace_csv, args.edge_density_csv,
                                       args.output_dir)

    if args.whitespace_csv.exists():
        log.info("Generating fig_NV01_nonvisual_fraction …")
        plot_nonvisual_fraction(args.whitespace_csv, args.output_dir)
        log.info("Generating fig_TO01_text_only_ads …")
        plot_text_only_prevalence(args.whitespace_csv, args.output_dir)
        log.info("Generating fig_TF01_text_area_fraction …")
        plot_text_area_fraction(args.whitespace_csv, args.output_dir)

    # Ad footprint small multiples
    if args.bb_csv.exists() and args.spread_csv.exists():
        log.info("Loading footprint data …")
        fp_df = load_footprint_data(
            args.predictions_csv, args.spread_csv, args.bb_csv, args.footprint_cache)
        log.info("Generating fig_03d_ad_footprint …")
        plot_ad_footprint(fp_df, args.output_dir)
    else:
        log.warning("BB CSV or spread CSV not found — skipping fig_03d.")

    # Product category prevalence
    if args.prodcat_csv.exists():
        log.info("Generating fig_PC_product_category_over_time …")
        plot_product_category_over_time(args.prodcat_csv, args.min_confidence,
                                        args.output_dir)
        log.info("Generating fig_PC02_product_category_by_magazine …")
        plot_product_category_by_magazine(args.prodcat_csv, args.min_confidence,
                                          args.output_dir)
    else:
        log.warning("Product category CSV not found — skipping fig_PC.")

    # Dataset composition (grouped bar)
    log.info("Generating fig_03_dataset_composition …")
    plot_dataset_composition(args.predictions_csv, args.output_dir)

    # Spread frequency over time
    if args.spread_csv.exists():
        log.info("Generating fig_SP01_spread_frequency …")
        plot_spread_frequency(args.predictions_csv, args.spread_csv, args.output_dir)

    # Ad-type composition & counts (line charts)
    if args.spread_csv.exists():
        log.info("Loading ad-type data …")
        at_df = _load_ad_type_data(args.predictions_csv, args.spread_csv)
        log.info("Generating fig_S06_ad_type_composition …")
        plot_ad_type_composition(at_df, args.output_dir)
        log.info("Generating fig_S06b_ad_type_counts …")
        plot_ad_type_counts(at_df, args.output_dir)
    else:
        log.warning("Spread CSV not found — skipping fig_S06/S06b/SP01.")


def run_classifiers(args) -> None:
    log.info("=== Section: classifiers ===")

    # Page classifier confusion matrix
    annotation_dir = args.annotation_dir
    if annotation_dir.exists():
        ann = load_annotations(annotation_dir, MAG_ORDER)
        if not ann.empty:
            merged = _match_predictions(ann, args.predictions_csv)
            if not merged.empty:
                log.info("Generating fig_04_classifier_confusion …")
                plot_classifier_confusion(merged, args.output_dir)
    else:
        log.warning("Annotation dir not found: %s — skipping fig_04.", annotation_dir)

    # Spread classifier confusion matrix
    if args.spread_metrics_json.exists():
        log.info("Generating fig_S05_spread_confusion …")
        plot_spread_confusion(args.spread_metrics_json, args.output_dir)
    else:
        log.warning("Spread metrics JSON not found: %s — skipping fig_S05.",
                     args.spread_metrics_json)


def run_people(args) -> None:
    log.info("=== Section: people ===")

    if not args.people_csv.exists() or not args.faces_csv.exists():
        log.error("People CSV (%s) or faces CSV (%s) not found.", args.people_csv, args.faces_csv)
        return

    df = _load_people_faces(args.people_csv, args.faces_csv,
                            args.min_year, args.max_year)

    log.info("Generating fig_person_count_categories …")
    plot_person_count_categories(args.people_csv, args.output_dir)

    log.info("Generating fig_PCPA_person_count_and_area …")
    plot_person_count_and_area(args.people_csv, df, args.output_dir)

    log.info("Generating fig_PF07_area_by_magazine …")
    plot_area_by_magazine(df, args.output_dir)

    if args.prodcat_csv is not None and args.prodcat_csv.exists():
        log.info("Generating fig_PF08_area_by_category …")
        df_cat = _join_prodcat(df, args.prodcat_csv, args.min_confidence)
        plot_area_by_category(df_cat, args.output_dir)
    else:
        log.warning("Product category CSV not found — skipping fig_PF08.")

    if args.validation_csv is not None and args.validation_csv.exists():
        log.info("Generating fig_people_prevalence_ppipp …")
        # Need detection predictions with yhat column
        det_df = df.copy()
        det_df["yhat"] = (det_df["n_persons"] > 0).astype(int)
        val_df = _load_val_for_ppi(args.validation_csv)
        prev = _compute_decade_prevalence_ppi(det_df, val_df)

        plot_prevalence_standalone(prev, args.output_dir)
        plot_prevalence_with_recall(prev, val_df, args.output_dir)

        prev.to_csv(args.output_dir / "fig_people_prevalence_ppipp.csv", index=False)
    else:
        log.warning("Validation CSV not provided — skipping PPI++ plots.")


# ═══════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════

DEFAULT_PRODCAT_CSV = PROJECT_ROOT / "results" / "product_category" / "corpus_predictions" / "product_category_predictions.csv"
DEFAULT_ANNOTATION_DIR = PROJECT_ROOT / "data"
DEFAULT_FOOTPRINT_CACHE = PROJECT_ROOT / "paperplots" / "partial_ad_area_cache.csv"


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--section",
                   choices=["coverage", "classifiers", "people", "all"],
                   default="all")
    p.add_argument("--output_dir", type=Path, default=PROJECT_ROOT / "plots_v2")
    p.add_argument("--predictions_csv", type=Path, default=_PREDICTIONS_CSV)
    p.add_argument("--wordcount_cache", type=Path,
                   default=PROJECT_ROOT / "paperplots" / "text_wordcount_cache.csv")
    p.add_argument("--whitespace_csv", type=Path, default=DEFAULT_WHITESPACE_CSV)
    p.add_argument("--edge_density_csv", type=Path, default=DEFAULT_EDGE_DENSITY_CSV)
    # Ad footprint inputs
    p.add_argument("--spread_csv", type=Path, default=DEFAULT_SPREAD_CSV)
    p.add_argument("--bb_csv", type=Path, default=DEFAULT_BB_CSV)
    p.add_argument("--footprint_cache", type=Path, default=DEFAULT_FOOTPRINT_CACHE)
    # Product category
    p.add_argument("--prodcat_csv", type=Path, default=DEFAULT_PRODCAT_CSV)
    p.add_argument("--min_confidence", type=float, default=0.5)
    # Classifier evaluation
    p.add_argument("--annotation_dir", type=Path, default=DEFAULT_ANNOTATION_DIR)
    p.add_argument("--spread_metrics_json", type=Path, default=DEFAULT_SPREAD_METRICS)
    # People
    p.add_argument("--people_csv", type=Path,
                   default=Path("/data/laviniad/magazines/people_detections/yolov8l/people_detections.csv"))
    p.add_argument("--faces_csv", type=Path,
                   default=Path("/data/laviniad/magazines/face_detections/scrfd/face_detections.csv"))
    p.add_argument("--validation_csv", type=Path,
                   default=Path("/data/laviniad/magazines/people_validation/validation_results_merged.csv"))
    p.add_argument("--min_year", type=int, default=1920)
    p.add_argument("--max_year", type=int, default=2017)
    args = p.parse_args()

    apply_style()
    plt.rcParams["font.size"] = RCPARAMS_FONT
    args.output_dir.mkdir(parents=True, exist_ok=True)

    if args.section in ("coverage", "all"):
        run_coverage(args)

    if args.section in ("classifiers", "all"):
        run_classifiers(args)

    if args.section in ("people", "all"):
        run_people(args)

    log.info("Done. Figures in %s", args.output_dir)


if __name__ == "__main__":
    main()
