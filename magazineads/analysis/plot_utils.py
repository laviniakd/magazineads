"""Shared plotting utilities for the magazines project.

Provides consistent color palettes, figure styles, and reusable plot helpers.
Import from any plotting script to keep visual language consistent across meetings.

Usage:
    from magazineads.analysis.plot_utils import (
        apply_style, fix_fonts,
        HEAD_PALETTE, MAGAZINE_PALETTE, AUX_COLORS, SPLIT_COLORS,
        plot_confusion_matrix, plot_f1_bars,
    )
"""

from __future__ import annotations

import re
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.font_manager as _fm
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

# ---------------------------------------------------------------------------
# Font registration — Canela Text Trial (from ~/fonts/Canela/)
# ---------------------------------------------------------------------------
_FONT_DIR = Path.home() / "fonts" / "Canela"
_CANELA_FAMILY = "Canela Text Trial"

def _register_canela() -> bool:
    """Register all Canela OTF files with matplotlib. Returns True if found."""
    if not _FONT_DIR.exists():
        return False
    for otf in _FONT_DIR.glob("*.otf"):
        _fm.fontManager.addfont(str(otf))
    return True

_CANELA_AVAILABLE = _register_canela()


# ---------------------------------------------------------------------------
# Palettes — use these everywhere so colors are consistent across meetings
# ---------------------------------------------------------------------------

# Spread classifier head types
HEAD_PALETTE: dict[str, str] = {
    "color":               "#2196F3",   # blue
    "concat":              "#4CAF50",   # green
    "cross_attn":          "#FF9800",   # orange
    "diff":                "#9C27B0",   # purple
    "similarity":          "#F44336",   # red
    "boundary":            "#00BCD4",   # cyan
    "boundary+color":      "#1565C0",   # dark blue
    "cross_attn+color":    "#AD1457",   # dark pink
    "boundary+cross_attn": "#E64A19",   # deep orange
}
HEAD_DISPLAY: dict[str, str] = {
    "color":               "Color",
    "concat":              "Concat",
    "cross_attn":          "Cross-attn",
    "diff":                "Diff",
    "similarity":          "Similarity",
    "boundary":            "Boundary",
    "boundary+color":      "Boundary+Color",
    "cross_attn+color":    "CrossAttn+Color",
    "boundary+cross_attn": "Boundary+CrossAttn",
}
HEAD_ORDER: list[str] = [
    "color", "boundary+color", "cross_attn+color", "boundary",
    "boundary+cross_attn", "concat", "cross_attn", "diff", "similarity",
]

# Magazine domains
AUX_COLORS: list[str] = ["#DAC0BA", "#B0BBB0"]

MAGAZINE_PALETTE: dict[str, str] = {
    "life":     "#C05C58",  # dusty coral
    "time":     "#B5A080",  # warm sand/taupe
    "esquire":  "#5C9E88",  # sage seafoam
    "forbes":   "#C4963C",  # warm mustard gold
    "variety":  "#28788C",  # dark teal
    "crisis":   "#5C9E88",
    "lhj":      "#C05C58",
    "cosmo":    "#C4963C",
    "usnews":   "#28788C",
}

MAGAZINE_DISPLAY: dict[str, str] = {
    "life":     "LIFE",
    "time":     "TIME",
    "esquire":  "Esquire",
    "forbes":   "Forbes",
    "variety":  "Variety",
    "crisis":   "The Crisis",
    "lhj":      "LHJ",
    "cosmo":    "Cosmopolitan",
    "usnews":   "U.S. News",
}


def magazine_label(key: str) -> str:
    """Return the formal display name for a magazine key."""
    return MAGAZINE_DISPLAY.get(key, key.title())

# Train / val split lines (used in loss curve plots)
SPLIT_COLORS: dict[str, str] = {
    "train": "#1565C0",   # dark blue
    "val":   "#E53935",   # red
    "test":  "#2E7D32",   # dark green
}

# Label class colors (page classifier)
CLASS_PALETTE: dict[str, str] = {
    "ad":     "#E53935",
    "mixed":  "#FB8C00",
    "non_ad": "#43A047",
    "spread": "#1E88E5",
    "not_spread": "#757575",
}

# General-purpose categorical palette — for plots where hue does NOT index a
# specific magazine.  Use MAGAZINE_PALETTE when colors encode magazine identity.
# Order: [dark, light-blue, light-green, tan, grey]
GENERAL_PALETTE: list[str] = [
    "#542344",  # dark burgundy  — primary / highlighted category
    "#bfd1e5",  # light blue     — secondary category
    "#d8bfaa",  # tan            — tertiary / neutral category
    "#ebf5ee",  # light green    — quaternary / background category
    "#808080",  # grey           — residual / other
]

# Confusion matrix colormap — white → project blue
CONFUSION_CMAP = LinearSegmentedColormap.from_list(
    "confusion", ["#FFFFFF", "#1565C0"]
)


# ---------------------------------------------------------------------------
# Global style
# ---------------------------------------------------------------------------

# Regex matching strings that are purely numeric (tick labels that should use Arial)
_NUM_PATTERN = re.compile(r"^[\s\d,.\-+eE%×]+$")

# Font family for numeric labels (overrides Canela)
NUM_FONT = "Arial"


def apply_style() -> None:
    """Apply project-wide matplotlib style. Call once at the top of a script."""
    font_family = _CANELA_FAMILY if _CANELA_AVAILABLE else "serif"
    plt.rcParams.update({
        "font.family": font_family,
        "font.size": 11,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.3,
        "grid.linestyle": "--",
        "figure.dpi": 150,
        "savefig.dpi": 150,
        "savefig.bbox": "tight",
        "legend.framealpha": 0.8,
        "legend.edgecolor": "#CCCCCC",
    })


def fix_fonts(fig: plt.Figure, num_family: str = NUM_FONT) -> None:
    """Enforce font convention: Canela for words, num_family (Arial) for numerals.

    Call this just before savefig — after tight_layout and canvas.draw so that
    auto-generated tick labels have their text populated.

    Affects:
    - All tick labels whose text matches _NUM_PATTERN → num_family
    - Colorbar tick labels (which are auto-generated numbers) → num_family
    """
    fig.canvas.draw()
    for ax in fig.axes:
        for label in ax.get_xticklabels() + ax.get_yticklabels():
            t = label.get_text().strip()
            if t and _NUM_PATTERN.fullmatch(t):
                label.set_fontfamily(num_family)


# ---------------------------------------------------------------------------
# Confusion matrix
# ---------------------------------------------------------------------------

def plot_confusion_matrix(
    cm: np.ndarray,
    class_names: list[str],
    ax: plt.Axes,
    title: str = "",
    normalize: bool = True,
) -> None:
    """Plot a confusion matrix on ax.

    Args:
        cm:           Integer confusion matrix (rows=actual, cols=predicted).
        class_names:  Row/column labels in the same order as cm.
        ax:           Matplotlib axes to draw on.
        title:        Axes title (optional).
        normalize:    If True, show row-normalized values (recall per class)
                      with raw counts as secondary annotation.
    """
    n = len(class_names)
    if normalize:
        row_sums = cm.sum(axis=1, keepdims=True).clip(min=1)
        cm_norm = cm.astype(float) / row_sums
    else:
        cm_norm = cm.astype(float)

    vmax = 1.0 if normalize else float(cm_norm.max())
    im = ax.imshow(cm_norm, interpolation="nearest", cmap=CONFUSION_CMAP,
                   vmin=0, vmax=vmax)

    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(class_names, fontsize=10)
    ax.set_yticklabels(class_names, fontsize=10)
    ax.set_xlabel("Predicted", fontsize=11)
    ax.set_ylabel("Actual", fontsize=11)
    if title:
        ax.set_title(title, fontsize=11, fontweight="bold")

    # Annotate each cell — numbers use Arial per font convention
    thresh = cm_norm.max() / 2.0
    for i in range(n):
        for j in range(n):
            raw = int(cm[i, j])
            pct = cm_norm[i, j]
            color = "white" if pct > thresh else "black"
            if normalize:
                ax.text(j, i, f"{pct:.2f}\n({raw})",
                        ha="center", va="center", fontsize=9, color=color,
                        fontfamily=NUM_FONT)
            else:
                ax.text(j, i, str(raw),
                        ha="center", va="center", fontsize=10, color=color,
                        fontfamily=NUM_FONT)

    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04).set_label(
        "Recall (row-normalized)" if normalize else "Count", fontsize=9
    )


def confusion_matrix_from_report(
    report: dict, class_names: list[str]
) -> np.ndarray | None:
    """Reconstruct integer confusion matrix from a sklearn classification_report dict.

    Works when support > 0 and precision > 0 for all classes.
    Returns None if reconstruction is not possible (e.g. zero-precision class).

    Only valid for binary classification (2 classes).
    """
    if len(class_names) != 2:
        return None
    try:
        rows = []
        for name in class_names:
            p  = report[name]["precision"]
            r  = report[name]["recall"]
            sup = int(round(report[name]["support"]))
            tp = round(r * sup)
            fn = sup - tp
            fp = round(tp / p - tp) if p > 0 else None
            rows.append((tp, fn, fp, sup))

        # For binary: FP of class 0 = FN of class 1, and vice versa
        # Verify consistency; if mismatch, fall back to None
        tp0, fn0, fp0, sup0 = rows[0]
        tp1, fn1, fp1, sup1 = rows[1]

        # cm[i,j] = actual i, predicted j
        # actual 0, pred 0 = TN for class 1 = tp0
        # actual 0, pred 1 = FP for class 0 = fp0 (= fn1)
        # actual 1, pred 0 = FN for class 1 = fn1
        # actual 1, pred 1 = TP for class 1 = tp1
        if fp0 is None:
            # zero precision — class 0 predicted for nothing → FP=0
            fp0 = 0
        if fp1 is None:
            fp1 = 0

        cm = np.array([
            [tp0, fn0],   # actual=0: TN, FP (using class-0 perspective)
            [fn1, tp1],   # actual=1: FN, TP
        ], dtype=int)
        return cm
    except (KeyError, ZeroDivisionError, ValueError):
        return None


# ---------------------------------------------------------------------------
# F1 bar chart
# ---------------------------------------------------------------------------

def plot_f1_bars(
    results: list[dict],
    ax: plt.Axes,
    head_order: list[str] | None = None,
    metric_keys: list[str] | None = None,
    metric_labels: list[str] | None = None,
) -> None:
    """Grouped bar chart of F1 metrics by head type.

    Args:
        results:       List of dicts with 'head_type', 'spread_f1', 'macro_f1'.
        ax:            Matplotlib axes.
        head_order:    Order for x-axis (default: HEAD_ORDER).
        metric_keys:   Keys to plot (default: ['spread_f1', 'macro_f1']).
        metric_labels: Display labels for the metrics.
    """
    if head_order is None:
        head_order = HEAD_ORDER
    if metric_keys is None:
        metric_keys = ["spread_f1", "macro_f1"]
    if metric_labels is None:
        metric_labels = metric_keys

    by_head = {r["head_type"]: r for r in results}
    heads = [h for h in head_order if h in by_head]
    n_metrics = len(metric_keys)
    x = np.arange(len(heads))
    width = 0.8 / n_metrics

    for i, (key, label) in enumerate(zip(metric_keys, metric_labels)):
        vals = [by_head[h].get(key, 0) for h in heads]
        offset = (i - n_metrics / 2 + 0.5) * width
        alpha = 0.9 if i == 0 else 0.5
        bars = ax.bar(
            x + offset, vals, width,
            label=label,
            color=[HEAD_PALETTE.get(h, "#999999") for h in heads],
            alpha=alpha,
            edgecolor=[HEAD_PALETTE.get(h, "#999999") for h in heads] if i > 0 else "none",
            linewidth=1.2 if i > 0 else 0,
        )
        if i == 0:  # annotate primary metric only
            for bar, v in zip(bars, vals):
                if v > 0.05:
                    ax.text(bar.get_x() + bar.get_width() / 2, v + 0.01,
                            f"{v:.3f}", ha="center", va="bottom", fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels([HEAD_DISPLAY.get(h, h) for h in heads], fontsize=11)
    ax.set_ylim(0, 1.08)
    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.2f"))
    ax.set_ylabel("F1 score", fontsize=11)
    ax.legend(fontsize=9)
