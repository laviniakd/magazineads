"""
make_paper_plots.py

Generate publication-quality figures for the paper.
All outputs go to paperplots/ at the project root.

Sections
--------
dataset     Descriptive coverage and size figures (no GPU required)
classifier  Page classifier confusion matrix and per-class F1 (requires
            a pre-computed predictions CSV from the classifier and the
            annotation CSVs in --annotation_dir)
all         Run both sections

Usage
-----
    # Dataset figures only (fast, no GPU):
    python magazines/scripts/make_paper_plots.py --section dataset

    # Classifier figures — pass the canonical predictions CSV:
    python magazines/scripts/make_paper_plots.py --section classifier \\
        --predictions_csv /data/laviniad/magazines/page_classification_predictions/predictions_all.csv

    # Both:
    python magazines/scripts/make_paper_plots.py --section all \\
        --predictions_csv /data/laviniad/magazines/page_classification_predictions/predictions_all.csv

Output files (all in --output_dir, default: paperplots/)
---------------------------------------------------------
    fig_01_dataset_coverage.{png,csv}
    fig_02_dataset_pages.{png,csv}
    fig_03_classifier_confusion.{png,csv}
    fig_04_classifier_f1.{png,csv}
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "paperplots"

# ---------------------------------------------------------------------------
# Paths (with fallbacks)
# ---------------------------------------------------------------------------
try:
    from magazineads.paths import paths as _paths
    _ISSUE_META_CSV   = _paths.local_data_dir / "issue_metadata_df.csv"
    _PREDICTIONS_CSV  = _paths.predictions_dir / "predictions_all.csv"
    _ANNOTATION_DIR   = _paths.local_data_dir
except Exception:
    _ISSUE_META_CSV   = PROJECT_ROOT / "data" / "issue_metadata_df.csv"
    _PREDICTIONS_CSV  = Path("/data/laviniad/magazines/page_classification_predictions/predictions_all.csv")
    _ANNOTATION_DIR   = PROJECT_ROOT / "data"

# ---------------------------------------------------------------------------
# Shared style + palettes
# ---------------------------------------------------------------------------
from magazineads.analysis.plot_utils import (
    apply_style,
    fix_fonts,
    NUM_FONT,
    MAGAZINE_PALETTE,
    GENERAL_PALETTE,
    plot_confusion_matrix,
)

DISPLAY = {
    "time":    "Time",
    "esquire": "Esquire",
    "life":    "Life",
    "forbes":  "Forbes",
    "variety": "Variety",
}
MAG_ORDER = ["time", "esquire", "life", "forbes", "variety"]

# Page classifier label encoding
LABEL_MAP: dict[str, int] = {
    "Not Ad": 0, "Ad": 1, "Mixed": 2, "Two-page": 1, "Two-Page": 1,
}
CLASS_NAMES = {0: "Non-Ad", 1: "Ad", 2: "Mixed"}

# Map class int → GENERAL_PALETTE entry (hue does not index magazine here)
#   0 Non-Ad → tan (neutral / majority)
#   1 Ad     → dark burgundy (target / highlighted)
#   2 Mixed  → light blue
_CLASS_COLOR = {
    0: GENERAL_PALETTE[2],   # tan      — Non-Ad
    1: GENERAL_PALETTE[0],   # burgundy — Ad
    2: GENERAL_PALETTE[1],   # blue     — Mixed
}

DPI = 600   # print-quality

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _save(fig: plt.Figure, path: Path) -> None:
    """Save figure as PDF at DPI dpi. path may have any suffix — it is replaced with .pdf."""
    pdf_path = path.with_suffix(".pdf")
    fig.tight_layout()
    fix_fonts(fig)          # canvas.draw() + swap numeric tick labels to Arial
    fig.savefig(pdf_path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {pdf_path}")


# ---------------------------------------------------------------------------
# Section 1: Dataset descriptive
# ---------------------------------------------------------------------------

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
        n_issues   = len(sub)
        start_year = int(sub["year"].min())
        end_year   = int(sub["year"].max())
        years_span = end_year - start_year + 1
        rows.append({
            "magazine":   mag,
            "start_year": start_year,
            "end_year":   end_year,
            "years_span": years_span,
            "n_issues":   n_issues,
            "avg_per_yr": n_issues / years_span,
            "total_pages": int(sub["num_pages"].sum()),
        })
    return pd.DataFrame(rows)


def plot_dataset_coverage(issues: pd.DataFrame, summary: pd.DataFrame,
                          output_dir: Path) -> None:
    """Fig 1 — Gantt-style temporal coverage."""
    ipy = (
        issues.groupby(["magazine", "year"])
              .size()
              .reset_index(name="count")
    )
    all_years = np.arange(ipy["year"].min(), ipy["year"].max() + 1)
    mags = list(summary["magazine"])
    n_mags = len(mags)

    fig, ax = plt.subplots(figsize=(14, 3.5))
    row_h, gap = 0.65, 0.35
    y_step = row_h + gap

    max_counts = {
        m: ipy[ipy["magazine"] == m]["count"].max() for m in mags
    }

    for i, mag in enumerate(mags):
        y_bottom = i * y_step
        sub = ipy[ipy["magazine"] == mag].set_index("year")["count"]
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
            cnt = sub.get(yr, 0)
            if cnt == 0:
                continue
            alpha = 0.25 + 0.75 * (cnt / max_counts[mag])
            ax.bar(yr, row_h, bottom=y_bottom, width=0.85,
                   color=base_col, alpha=float(alpha), edgecolor="none", zorder=2)

    decade_ticks = np.arange((all_years[0] // 10) * 10, all_years[-1] + 1, 10)
    # Pin all labels at the same x: 2 years left of the first decade gridline,
    # so they are vertically aligned and fully to its left.
    label_x = decade_ticks[0] - 2

    ax.set_xlim(label_x - 10, all_years[-1] + 2)
    ax.set_ylim(-gap, n_mags * y_step)
    ax.set_yticks([])
    ax.set_xticks(decade_ticks)
    ax.set_xticklabels([str(y) for y in decade_ticks], fontsize=10)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.xaxis.grid(True, linestyle="--", alpha=0.25, zorder=0)
    ax.set_axisbelow(True)
    ax.set_title("")   # no title — description lives in PLOTS.md

    for i, mag in enumerate(mags):
        y_bottom = i * y_step
        ax.text(label_x, y_bottom + row_h / 2,
                DISPLAY[mag], ha="right", va="center",
                fontsize=11, fontweight="bold", color=MAGAZINE_PALETTE[mag])

    _save(fig, output_dir / "fig_01_dataset_coverage.pdf")

    # Companion CSV
    ipy.to_csv(output_dir / "fig_01_dataset_coverage.csv", index=False)
    print(f"  saved {output_dir / 'fig_01_dataset_coverage.csv'}")


# Display names for final plots (LIFE/TIME in caps)
DISPLAY_FINAL = {
    "time":    "TIME",
    "esquire": "Esquire",
    "life":    "LIFE",
    "forbes":  "Forbes",
    "variety": "Variety",
}


def _build_yearly_hue_data(
    issues: pd.DataFrame,
    predictions_csv: Path,
    hue_by: str,
) -> pd.DataFrame:
    """Return DataFrame with (magazine, year, value) for hue mapping.

    hue_by="pages"  → total pages per magazine-year (from issue metadata)
    hue_by="ads"    → full-page ad count per magazine-year (pred_label==1)
    """
    if hue_by == "pages":
        return (
            issues.groupby(["magazine", "year"])["num_pages"]
                  .sum()
                  .reset_index(name="value")
        )
    elif hue_by == "ads":
        preds = pd.read_csv(predictions_csv, usecols=["filepath", "pred_label"])
        preds["magazine"] = preds["filepath"].str.extract(r"/images/([^/]+)/")
        preds["year"] = pd.to_numeric(
            preds["filepath"].str.extract(r"/(\d{4})")[0], errors="coerce"
        )
        preds = preds.dropna(subset=["year"])
        preds["year"] = preds["year"].astype(int)
        # pred_label 1 = full-page ad only
        return (
            preds[(preds["pred_label"] == 1) & preds["magazine"].isin(MAG_ORDER)]
            .groupby(["magazine", "year"])
            .size()
            .reset_index(name="value")
        )
    else:
        raise ValueError(f"Unknown hue_by={hue_by!r}")


def plot_final_coverage(
    issues: pd.DataFrame,
    summary: pd.DataFrame,
    predictions_csv: Path,
    output_dir: Path,
    hue_by: str = "pages",
    suffix: str = "",
) -> None:
    """Gantt-style coverage with hue intensity mapped to pages or ad count."""
    from matplotlib.colors import LinearSegmentedColormap
    import matplotlib.colorbar as mcb

    hue_data = _build_yearly_hue_data(issues, predictions_csv, hue_by)
    all_years = np.arange(
        hue_data["year"].min(),
        max(hue_data["year"].max(), summary["end_year"].max()) + 1,
    )
    mags = list(summary["magazine"])
    n_mags = len(mags)

    fig, ax = plt.subplots(figsize=(14, 4.2))
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
    ax.set_xticklabels([str(y) for y in decade_ticks], fontsize=13)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.xaxis.grid(True, linestyle="--", alpha=0.25, zorder=0)
    ax.set_axisbelow(True)

    for i, mag in enumerate(mags):
        y_bottom = i * y_step
        ax.text(label_x, y_bottom + row_h / 2,
                DISPLAY_FINAL[mag], ha="right", va="center",
                fontsize=14, fontweight="bold", color=MAGAZINE_PALETTE[mag])

    # ── Hue-intensity legend (horizontal gray gradient bar) ──
    legend_label = "Pages per year" if hue_by == "pages" else "Full-page ads per year"
    legend_max = 10_000 if hue_by == "pages" else 4_000
    cmap = LinearSegmentedColormap.from_list(
        "intensity", [(1, 1, 1), (0.25, 0.25, 0.25)], N=256
    )

    cbar_ax = fig.add_axes([0.72, 0.88, 0.18, 0.025])
    norm = plt.matplotlib.colors.Normalize(vmin=0, vmax=legend_max)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cb = fig.colorbar(sm, cax=cbar_ax, orientation="horizontal")
    cb.set_label(legend_label, fontsize=11)
    cb.ax.tick_params(labelsize=10)
    cb.set_ticks([0, legend_max])
    cb.set_ticklabels(["0", f"{legend_max:,}"])

    fname = f"fig_00_dataset_coverage{suffix}"
    _save(fig, output_dir / f"{fname}.pdf")

    # Companion CSV
    hue_data.to_csv(output_dir / f"{fname}.csv", index=False)
    print(f"  saved {output_dir / fname}.csv")


def plot_dataset_coverage_with_ads(issues: pd.DataFrame, summary: pd.DataFrame,
                                   predictions_csv: Path, output_dir: Path) -> None:
    """Composite figure: Gantt coverage (top) + ad count over time (bottom)."""
    # --- data for top panel (same as plot_dataset_coverage) ---
    ipy = (
        issues.groupby(["magazine", "year"])
              .size()
              .reset_index(name="count")
    )
    all_years = np.arange(ipy["year"].min(), ipy["year"].max() + 1)
    mags = list(summary["magazine"])
    n_mags = len(mags)

    # --- data for bottom panel: ad pages per year per magazine ---
    preds = pd.read_csv(predictions_csv, usecols=["filepath", "pred_label"])
    preds["magazine"] = preds["filepath"].str.extract(r"/images/([^/]+)/")
    preds["year"] = preds["filepath"].str.extract(r"/(\d{4})")
    preds["year"] = pd.to_numeric(preds["year"], errors="coerce")
    preds = preds[preds["magazine"].isin(MAG_ORDER)].dropna(subset=["year"])
    preds["year"] = preds["year"].astype(int)
    # ad = pred_label 1, mixed = pred_label 2
    ads_by_year = (
        preds[preds["pred_label"].isin([1, 2])]
        .groupby(["magazine", "year"])
        .size()
        .reset_index(name="ad_pages")
    )

    # --- figure with two panels ---
    fig, (ax_top, ax_bot) = plt.subplots(
        2, 1, figsize=(14, 6.5),
        gridspec_kw={"height_ratios": [3, 2], "hspace": 0.12},
        sharex=True,
    )

    # ── TOP: Gantt coverage ──
    row_h, gap = 0.65, 0.35
    y_step = row_h + gap
    max_counts = {
        m: ipy[ipy["magazine"] == m]["count"].max() for m in mags
    }

    for i, mag in enumerate(mags):
        y_bottom = i * y_step
        sub = ipy[ipy["magazine"] == mag].set_index("year")["count"]
        smeta = summary[summary["magazine"] == mag].iloc[0]
        base_col = np.array(plt.matplotlib.colors.to_rgb(MAGAZINE_PALETTE[mag]))

        ax_top.barh(y_bottom + row_h / 2,
                    smeta["end_year"] - smeta["start_year"] + 1,
                    left=smeta["start_year"],
                    height=row_h,
                    color="#eeeeee", edgecolor="none", zorder=1)

        for yr in all_years:
            if yr < smeta["start_year"] or yr > smeta["end_year"]:
                continue
            cnt = sub.get(yr, 0)
            if cnt == 0:
                continue
            alpha = 0.25 + 0.75 * (cnt / max_counts[mag])
            ax_top.bar(yr, row_h, bottom=y_bottom, width=0.85,
                       color=base_col, alpha=float(alpha), edgecolor="none", zorder=2)

    decade_ticks = np.arange((all_years[0] // 10) * 10, all_years[-1] + 1, 10)
    label_x = decade_ticks[0] - 2

    ax_top.set_xlim(label_x - 10, all_years[-1] + 2)
    ax_top.set_ylim(-gap, n_mags * y_step)
    ax_top.set_yticks([])
    ax_top.spines[["top", "right", "left"]].set_visible(False)
    ax_top.xaxis.grid(True, linestyle="--", alpha=0.25, zorder=0)
    ax_top.set_axisbelow(True)
    ax_top.tick_params(axis="x", labelbottom=False)

    for i, mag in enumerate(mags):
        y_bottom = i * y_step
        ax_top.text(label_x, y_bottom + row_h / 2,
                    DISPLAY[mag], ha="right", va="center",
                    fontsize=11, fontweight="bold", color=MAGAZINE_PALETTE[mag])

    # ── BOTTOM: stacked area of ad pages over time ──
    year_range = np.arange(all_years[0], all_years[-1] + 1)
    bottom = np.zeros(len(year_range))

    for mag in mags:
        sub = ads_by_year[ads_by_year["magazine"] == mag].set_index("year")["ad_pages"]
        vals = np.array([sub.get(yr, 0) for yr in year_range], dtype=float)
        ax_bot.fill_between(year_range, bottom, bottom + vals,
                            color=MAGAZINE_PALETTE[mag], alpha=0.7,
                            linewidth=0.5, edgecolor="white")
        bottom += vals

    ax_bot.set_ylabel("Ad pages / year", fontsize=11)
    ax_bot.set_xticks(decade_ticks)
    ax_bot.set_xticklabels([str(y) for y in decade_ticks], fontsize=10)
    ax_bot.spines[["top", "right"]].set_visible(False)
    ax_bot.yaxis.grid(True, linestyle="--", alpha=0.25, zorder=0)
    ax_bot.xaxis.grid(True, linestyle="--", alpha=0.25, zorder=0)
    ax_bot.set_axisbelow(True)
    ax_bot.set_xlim(ax_top.get_xlim())

    _save(fig, output_dir / "fig_01b_dataset_coverage_with_ads.pdf")

    # Companion CSV
    csv_rows = []
    for mag in mags:
        sub = ads_by_year[ads_by_year["magazine"] == mag]
        for _, row in sub.iterrows():
            csv_rows.append({"magazine": row["magazine"], "year": int(row["year"]),
                             "ad_pages": int(row["ad_pages"])})
    pd.DataFrame(csv_rows).to_csv(
        output_dir / "fig_01b_dataset_coverage_with_ads.csv", index=False)
    print(f"  saved {output_dir / 'fig_01b_dataset_coverage_with_ads.csv'}")


def _load_ad_counts(predictions_csv: Path) -> pd.DataFrame:
    """Return per-magazine page-type counts with columns non_ad, ad, mixed."""
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
    return ad_counts


def plot_dataset_total_pages(predictions_csv: Path, output_dir: Path) -> None:
    """Fig 2 — Total pages per magazine."""
    ad_counts = _load_ad_counts(predictions_csv)
    labels  = [DISPLAY[m] for m in MAG_ORDER]
    colours = [MAGAZINE_PALETTE[m] for m in MAG_ORDER]
    x = np.arange(len(MAG_ORDER))
    bar_w = 0.55

    fig, ax = plt.subplots(figsize=(6, 4))
    totals = ad_counts.sum(axis=1).values
    bars = ax.bar(x, totals, width=bar_w, color=colours, edgecolor="white", linewidth=0.8)
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=11)
    ax.set_ylabel("Pages", fontsize=11)
    ax.set_title("")   # no title — description lives in PLOTS.md
    ax.spines[["top", "right"]].set_visible(False)
    ax.yaxis.grid(True, linestyle="--", alpha=0.3); ax.set_axisbelow(True)
    for bar, v in zip(bars, totals):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + totals.max() * 0.01,
                f"{v:,}", ha="center", va="bottom", fontsize=8, fontfamily=NUM_FONT)

    _save(fig, output_dir / "fig_02_dataset_total_pages.pdf")

    out_df = ad_counts.copy()
    out_df.index.name = "magazine"
    out_df["total"] = out_df.sum(axis=1)
    out_df.reset_index().to_csv(output_dir / "fig_02_dataset_total_pages.csv", index=False)
    print(f"  saved {output_dir / 'fig_02_dataset_total_pages.csv'}")


def plot_dataset_composition(predictions_csv: Path, output_dir: Path) -> None:
    """Fig 3 — Page-type composition per magazine (stacked bars)."""
    ad_counts = _load_ad_counts(predictions_csv)
    labels = [DISPLAY[m] for m in MAG_ORDER]
    x = np.arange(len(MAG_ORDER))
    bar_w = 0.55

    fig, ax = plt.subplots(figsize=(6, 4))
    bottoms = np.zeros(len(MAG_ORDER))
    for cls_key, label_text, label_int in [
        ("non_ad", "Non-Ad", 0),
        ("mixed",  "Mixed",  2),
        ("ad",     "Ad",     1),
    ]:
        vals = ad_counts[cls_key].values
        ax.bar(x, vals, width=bar_w, bottom=bottoms,
               color=_CLASS_COLOR[label_int], label=label_text,
               edgecolor="white", linewidth=0.5)
        bottoms += vals
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=11)
    ax.set_ylabel("Pages", fontsize=11)
    ax.set_title("")   # no title — description lives in PLOTS.md
    ax.spines[["top", "right"]].set_visible(False)
    ax.yaxis.grid(True, linestyle="--", alpha=0.3); ax.set_axisbelow(True)
    ax.legend(fontsize=9, loc="upper right")

    _save(fig, output_dir / "fig_03_dataset_composition.pdf")

    out_df = ad_counts.copy()
    out_df.index.name = "magazine"
    out_df.reset_index().to_csv(output_dir / "fig_03_dataset_composition.csv", index=False)
    print(f"  saved {output_dir / 'fig_03_dataset_composition.csv'}")


def run_dataset_section(predictions_csv: Path, output_dir: Path) -> None:
    print("\n=== Section: dataset ===")
    print("Loading issue metadata …")
    issues  = _build_issue_table(predictions_csv)
    summary = _compute_summary(issues)

    print("Generating fig_00_dataset_coverage (pages) → finalplots/ …")
    finalplots = PROJECT_ROOT / "finalplots"
    finalplots.mkdir(exist_ok=True)
    plot_final_coverage(issues, summary, predictions_csv, finalplots,
                        hue_by="pages", suffix="_pages")
    print("Generating fig_00_dataset_coverage (ads) → finalplots/ …")
    plot_final_coverage(issues, summary, predictions_csv, finalplots,
                        hue_by="ads", suffix="_ads")

    print("Generating fig_01_dataset_coverage …")
    plot_dataset_coverage(issues, summary, output_dir)

    print("Generating fig_01b_dataset_coverage_with_ads …")
    plot_dataset_coverage_with_ads(issues, summary, predictions_csv, output_dir)

    print("Generating fig_02_dataset_total_pages …")
    plot_dataset_total_pages(predictions_csv, output_dir)

    print("Generating fig_03_dataset_composition …")
    plot_dataset_composition(predictions_csv, output_dir)


# ---------------------------------------------------------------------------
# Section 2: Page classifier evaluation
# ---------------------------------------------------------------------------

_NUM_ROUNDS = 9
_ANNOTATORS  = ["lavinia", "sravani", "dallas"]


def _normalize_label(raw) -> int | None:
    if isinstance(raw, (int, float)):
        v = int(raw)
        return 1 if v == 3 else v
    if isinstance(raw, str):
        return LABEL_MAP.get(raw.strip())
    return None


def load_annotations(annotation_dir: Path, magazines: list[str],
                     splits: list[str] = ("test",)) -> pd.DataFrame:
    """Load annotation CSVs and return a DataFrame with (filename, magazine, gt_label)."""
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
                # Legacy Life format
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
        print("ERROR: No annotation rows loaded. Check --annotation_dir.", file=sys.stderr)
        sys.exit(1)

    ann = pd.DataFrame(rows).drop_duplicates(subset=["filename", "magazine"])
    print(f"  Loaded {len(ann)} annotated images.")
    return ann


def _page_key(fname: str) -> tuple[int, ...] | None:
    """Parse (year, month, day, page) as ints from a magazine page filename.

    Handles zero-padding differences between annotation CSVs (e.g. 1951-11-1-68-9-13.jpg)
    and predictions / on-disk files (e.g. 1951-11-01-68-09-013.jpg) by comparing
    integer values rather than raw strings.  Page is always the last dash-separated
    component before the extension.
    """
    parts = Path(fname).stem.split("-")
    if len(parts) < 4:
        return None
    try:
        return (int(parts[0]), int(parts[1]), int(parts[2]), int(parts[-1]))
    except ValueError:
        return None


def _match_predictions(ann: pd.DataFrame, predictions_csv: Path) -> pd.DataFrame:
    """Join annotations with predictions via (magazine, year, month, day, page) key.

    Annotation filenames and prediction filenames share the same logical structure
    but differ in zero-padding; matching on parsed integer components is robust to
    that variation.
    """
    preds = pd.read_csv(predictions_csv, usecols=["filepath", "pred_label"])
    preds["magazine"] = preds["filepath"].str.extract(r"/images/([^/]+)/")
    preds["_key"] = preds["filepath"].apply(lambda p: _page_key(Path(p).name))
    preds = preds.dropna(subset=["_key"]).drop_duplicates(subset=["magazine", "_key"])

    ann = ann.copy()
    ann["_key"] = ann["filename"].apply(_page_key)
    ann = ann.dropna(subset=["_key"])

    merged = ann.merge(
        preds[["magazine", "_key", "pred_label"]],
        on=["magazine", "_key"],
        how="inner",
    ).drop(columns=["_key"])

    n_missing = len(ann) - len(merged)
    if n_missing:
        print(f"  Warning: {n_missing} annotated images not found in predictions CSV.")
    print(f"  Matched {len(merged)} annotated images to predictions.")
    return merged


def plot_classifier_confusion(merged: pd.DataFrame, output_dir: Path) -> None:
    """Fig 4 — Confusion matrix (3-class, raw counts, hued by count)."""
    present = sorted(set(merged["gt_label"]) | set(merged["pred_label"]))
    class_names = [CLASS_NAMES[c] for c in present]

    cm = confusion_matrix(merged["gt_label"], merged["pred_label"], labels=present)

    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    plot_confusion_matrix(cm, class_names, ax, normalize=False)
    _save(fig, output_dir / "fig_04_classifier_confusion.pdf")

    cm_df = pd.DataFrame(cm,
                         index=[f"true_{CLASS_NAMES[c]}" for c in present],
                         columns=[f"pred_{CLASS_NAMES[c]}" for c in present])
    cm_df.to_csv(output_dir / "fig_04_classifier_confusion.csv")
    print(f"  saved {output_dir / 'fig_04_classifier_confusion.csv'}")


def plot_classifier_f1(merged: pd.DataFrame, output_dir: Path) -> None:
    """Fig 5 — F1 by class (x-axis) and magazine (grouped bars, MAGAZINE_PALETTE colors)."""
    present     = sorted(set(merged["gt_label"]) | set(merged["pred_label"]))
    class_names = [CLASS_NAMES[c] for c in present]

    # Overall macro F1 (all magazines combined) for the reference line
    overall_report = classification_report(
        merged["gt_label"], merged["pred_label"],
        labels=present, target_names=class_names,
        digits=4, zero_division=0, output_dict=True,
    )
    macro_f1 = overall_report["macro avg"]["f1-score"]

    # Per-magazine F1 — only include magazines present in the matched data
    mags_present = [m for m in MAG_ORDER if m in merged["magazine"].values]
    mag_reports  = {}
    for mag in mags_present:
        sub = merged[merged["magazine"] == mag]
        sub_present = sorted(set(sub["gt_label"]) | set(sub["pred_label"]))
        sub_names   = [CLASS_NAMES[c] for c in sub_present]
        mag_reports[mag] = classification_report(
            sub["gt_label"], sub["pred_label"],
            labels=sub_present, target_names=sub_names,
            digits=4, zero_division=0, output_dict=True,
        )

    n_classes = len(class_names)
    n_mags    = len(mags_present)
    x         = np.arange(n_classes)
    bar_w     = 0.75 / n_mags

    fig, ax = plt.subplots(figsize=(max(6, n_classes * 1.8), 4))
    for i, mag in enumerate(mags_present):
        offset = (i - n_mags / 2 + 0.5) * bar_w
        f1s = [mag_reports[mag].get(cn, {}).get("f1-score", 0.0) for cn in class_names]
        ax.bar(x + offset, f1s, bar_w,
               color=MAGAZINE_PALETTE[mag], label=DISPLAY[mag],
               edgecolor="white", linewidth=0.5)

    ax.axhline(macro_f1, color="#555555", linestyle="--", linewidth=0.9, alpha=0.7)

    ax.set_xticks(x)
    ax.set_xticklabels(class_names, fontsize=12)
    ax.set_ylim(0, 1.1)
    ax.set_ylabel("F1", fontsize=11)
    ax.set_title("")   # no title — description lives in PLOTS.md
    ax.spines[["top", "right"]].set_visible(False)

    # Legend outside-right; macro label placed just above the dashed line at right edge
    leg = ax.legend(fontsize=9, loc="upper left",
                    bbox_to_anchor=(1.01, 0.5), borderaxespad=0)
    ax.text(1.01, macro_f1 / 1.1 - 0.04,
            f"macro = {macro_f1:.3f}", ha="left", va="top", fontsize=8,
            color="#555555", fontfamily=NUM_FONT, transform=ax.transAxes)

    _save(fig, output_dir / "fig_05_classifier_f1.pdf")

    # Companion CSV — per-magazine per-class F1 + overall macro
    rows = []
    for mag in mags_present:
        rpt = mag_reports[mag]
        for cn in class_names:
            rows.append({"magazine": mag, "class": cn,
                         "f1":      rpt.get(cn, {}).get("f1-score", float("nan")),
                         "support": rpt.get(cn, {}).get("support",  0)})
        rows.append({"magazine": mag, "class": "macro avg",
                     "f1": rpt["macro avg"]["f1-score"],
                     "support": rpt["macro avg"]["support"]})
    rows.append({"magazine": "ALL", "class": "macro avg",
                 "f1": macro_f1, "support": overall_report["macro avg"]["support"]})
    pd.DataFrame(rows).to_csv(output_dir / "fig_05_classifier_f1.csv", index=False)
    print(f"  saved {output_dir / 'fig_05_classifier_f1.csv'}")


def run_classifier_from_json(results_json: Path, output_dir: Path) -> None:
    """Generate classifier plots from a pre-computed results JSON.

    Accepts the JSON written by magazines/scripts/evaluate_feb12_classifier.py
    (or any script that writes the same schema).  Avoids re-running inference.
    """
    import json as _json
    with open(results_json) as f:
        r = _json.load(f)

    # ---- confusion matrix ----
    cm          = np.array(r["confusion_matrix"])
    class_names = r["confusion_matrix_labels"]

    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    plot_confusion_matrix(cm, class_names, ax, normalize=False)
    _save(fig, output_dir / "fig_04_classifier_confusion.pdf")

    cm_df = pd.DataFrame(cm,
                         index=[f"true_{c}" for c in class_names],
                         columns=[f"pred_{c}" for c in class_names])
    cm_df.to_csv(output_dir / "fig_04_classifier_confusion.csv")
    print(f"  saved {output_dir / 'fig_04_classifier_confusion.csv'}")

    # ---- F1 by class × magazine ----
    per_mag     = r["per_magazine"]       # {mag: {n, accuracy, per_class_f1: {cls: f1}}}
    macro_f1    = r["overall_macro_f1"]
    # Gather all class names present across magazines
    all_classes = sorted({cls for m in per_mag.values()
                          for cls in m.get("per_class_f1", {})})
    mags_present = [m for m in MAG_ORDER if m in per_mag]

    n_classes = len(all_classes)
    n_mags    = len(mags_present)
    x         = np.arange(n_classes)
    bar_w     = 0.75 / n_mags

    fig, ax = plt.subplots(figsize=(max(6, n_classes * 1.8), 4))
    for i, mag in enumerate(mags_present):
        offset = (i - n_mags / 2 + 0.5) * bar_w
        f1s    = [per_mag[mag]["per_class_f1"].get(cls, 0.0) for cls in all_classes]
        ax.bar(x + offset, f1s, bar_w,
               color=MAGAZINE_PALETTE[mag], label=DISPLAY[mag],
               edgecolor="white", linewidth=0.5)

    ax.axhline(macro_f1, color="#555555", linestyle="--", linewidth=0.9, alpha=0.7)
    ax.set_xticks(x)
    ax.set_xticklabels(all_classes, fontsize=12)
    ax.set_ylim(0, 1.1)
    ax.set_ylabel("F1", fontsize=11)
    ax.set_title("")
    ax.spines[["top", "right"]].set_visible(False)

    # Legend outside-right; macro label placed just above the dashed line at right edge
    ax.legend(fontsize=9, loc="upper left",
              bbox_to_anchor=(1.01, 0.5), borderaxespad=0)
    ax.text(1.01, macro_f1 / 1.1 - 0.04,
            f"macro = {macro_f1:.3f}", ha="left", va="top", fontsize=8,
            color="#555555", fontfamily=NUM_FONT, transform=ax.transAxes)

    _save(fig, output_dir / "fig_05_classifier_f1.pdf")

    # CSV
    rows = []
    for mag in mags_present:
        for cls in all_classes:
            rows.append({"magazine": mag, "class": cls,
                         "f1": per_mag[mag]["per_class_f1"].get(cls, float("nan")),
                         "support": per_mag[mag].get("n", 0)})
    rows.append({"magazine": "ALL", "class": "macro avg", "f1": macro_f1,
                 "support": r.get("n_total", 0)})
    pd.DataFrame(rows).to_csv(output_dir / "fig_05_classifier_f1.csv", index=False)
    print(f"  saved {output_dir / 'fig_05_classifier_f1.csv'}")


def run_classifier_section(predictions_csv: Path, annotation_dir: Path,
                            magazines: list[str], splits: list[str],
                            output_dir: Path) -> None:
    print("\n=== Section: classifier ===")
    print("Loading annotations …")
    ann = load_annotations(annotation_dir, magazines, splits)

    print("Matching to predictions …")
    merged = _match_predictions(ann, predictions_csv)

    print("Generating fig_04_classifier_confusion …")
    plot_classifier_confusion(merged, output_dir)

    print("Generating fig_05_classifier_f1 …")
    plot_classifier_f1(merged, output_dir)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--section", choices=["dataset", "classifier", "all"],
                   default="all",
                   help="Which section to generate (default: all)")
    p.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT_DIR,
                   help=f"Output directory (default: {DEFAULT_OUTPUT_DIR})")
    p.add_argument("--predictions_csv", type=Path, default=_PREDICTIONS_CSV,
                   help="Canonical predictions CSV (filepath, pred_label, ad_prob)")
    p.add_argument("--annotation_dir", type=Path, default=_ANNOTATION_DIR,
                   help="Directory containing annotation CSVs")
    p.add_argument("--magazines", nargs="+",
                   default=["forbes", "variety", "time", "life", "esquire"],
                   help="Magazines to include in classifier evaluation")
    p.add_argument("--splits", nargs="+", default=["test"],
                   choices=["train", "test"],
                   help="Annotation splits to use (default: test)")
    p.add_argument("--classifier_results_json", type=Path, default=None,
                   help="Pre-computed results JSON from evaluate_feb12_classifier.py. "
                        "If provided, classifier plots are generated from this JSON "
                        "instead of re-running inference via --predictions_csv.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    apply_style()

    if args.section in ("dataset", "all"):
        run_dataset_section(args.predictions_csv, args.output_dir)

    if args.section in ("classifier", "all"):
        if args.classifier_results_json is not None:
            print("\n=== Section: classifier (from JSON) ===")
            run_classifier_from_json(args.classifier_results_json, args.output_dir)
        else:
            run_classifier_section(
                args.predictions_csv,
                args.annotation_dir,
                args.magazines,
                args.splits,
                args.output_dir,
            )

    print("\nDone. Figures written to:", args.output_dir)


if __name__ == "__main__":
    main()
