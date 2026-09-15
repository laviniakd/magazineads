"""Paper plots: share of ads containing people, over time, corrected via PPI++.

Raw YOLO detection rate is a biased estimator because YOLO has both false
positives and false negatives (the latter worse in early decades due to
illustrated content).  Correction uses Prediction-Powered Inference (PPI++;
Angelopoulos, Duchi & Zrnic 2023) with a stratified validation sample.

Inputs:
  --detections_csv      people_detections.csv from infer_people.py
  --validation_csv      validation_results_merged.csv
  --output_dir          where to write plots (default: paperplots/)

Outputs:
  fig_people_prevalence_corrected.png/pdf   all-magazine trend, raw vs PPI++
  fig_people_prevalence_by_magazine.png/pdf per-magazine PPI++ lines
  fig_people_detection_bias.png/pdf         corrected − raw per decade
  people_prevalence_summary.txt             text summary
  people_decade_prevalence.csv              machine-readable table

Usage:
  python -m magazines.scripts.plot_people_over_time \\
      --detections_csv /data/laviniad/magazines/people_detections/yolov8l/people_detections.csv \\
      --validation_csv /data/laviniad/magazines/people_validation/validation_results_merged.csv \\
      --output_dir paperplots
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from magazineads.ppi import get_ppi_estimate
from magazineads.analysis.plot_utils import (
    MAGAZINE_PALETTE,
    apply_style,
    fix_fonts,
    magazine_label,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

MAGAZINES = ["esquire", "forbes", "life", "time", "variety"]

# Minimum validation images per stratum to use per-decade correction;
# fall back to overall otherwise.
MIN_VAL_DETECTED   = 5
MIN_VAL_UNDETECTED = 5


# ---------------------------------------------------------------------------
# Wilson CI
# ---------------------------------------------------------------------------

def _wilson_ci(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (0.0, 1.0)
    p = k / n
    denom = 1 + z**2 / n
    centre = (p + z**2 / (2 * n)) / denom
    margin = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
    return (max(0.0, centre - margin), min(1.0, centre + margin))


# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------

def load_detections(csv_path: Path, min_year: int, max_year: int) -> pd.DataFrame:
    """Load people_detections.csv; returns rows with valid year in [min_year, max_year]."""
    df = pd.read_csv(csv_path)
    if "year" not in df.columns:
        df["year"] = df["image_path"].apply(
            lambda p: int(Path(p).stem.split("-")[0]) if pd.notna(p) else None
        )
    df = df.dropna(subset=["year"])
    df["year"]    = df["year"].astype(int)
    df["decade"]  = (df["year"] // 10 * 10).astype(int)
    df["magazine"] = df["magazine"].str.lower().str.strip()
    df = df[(df["year"] >= min_year) & (df["year"] <= max_year)]
    df["has_person"] = df["n_persons"].fillna(0).astype(int) > 0
    df["yhat"]       = df["has_person"].astype(int)
    log.info(
        "Loaded %d detections, years %d–%d  (%d with ≥1 person, %.1f%%)",
        len(df), df["year"].min(), df["year"].max(),
        df["has_person"].sum(), 100 * df["has_person"].mean(),
    )
    return df


# ---------------------------------------------------------------------------
# π/φ (classical plug-in) correction
# ---------------------------------------------------------------------------

def load_correction_params(validation_csv: Path) -> pd.DataFrame:
    """Derive per-decade precision (π) and FN-rate (φ) from validation_results.csv.

    Returns a DataFrame indexed by decade with columns:
        pi, pi_lo, pi_hi      precision + 95% Wilson CI
        phi, phi_lo, phi_hi   FN rate (P(has person | YOLO=0)) + 95% CI
        n_det, n_undet         sample sizes
        fallback               True if overall params were substituted
    Also adds row decade=-1 (overall) as the fallback source.
    """
    vdf = pd.read_csv(validation_csv)
    vdf["decade"] = (vdf["year"].astype(float) // 10 * 10).fillna(-1).astype(int)

    def _params(sub: pd.DataFrame) -> dict:
        sure   = sub[sub["ann_answered"].astype(bool)].copy()
        det    = sure[sure["yolo_detected"].astype(bool)]
        undet  = sure[~sure["yolo_detected"].astype(bool)]
        tp = int(sure["is_tp"].astype(int).sum())
        fp = int(sure["is_fp"].astype(int).sum())
        fn = int(sure["is_fn"].astype(int).sum())
        n_det, n_undet = len(det), len(undet)
        pi,  pi_lo,  pi_hi  = (tp / (tp + fp),) + _wilson_ci(tp, tp + fp) if (tp + fp) > 0 else (float("nan"), 0.0, 1.0)
        phi, phi_lo, phi_hi = (fn / n_undet,)   + _wilson_ci(fn, n_undet)  if n_undet > 0   else (float("nan"), 0.0, 1.0)
        return dict(pi=pi, pi_lo=pi_lo, pi_hi=pi_hi,
                    phi=phi, phi_lo=phi_lo, phi_hi=phi_hi,
                    n_det=n_det, n_undet=n_undet)

    overall = _params(vdf)
    overall["decade"] = -1
    overall["fallback"] = False
    log.info("Overall correction: π=%.3f [%.3f–%.3f]  φ=%.3f [%.3f–%.3f]",
             overall["pi"], overall["pi_lo"], overall["pi_hi"],
             overall["phi"], overall["phi_lo"], overall["phi_hi"])

    rows = [overall]
    for decade, sub in vdf.groupby("decade"):
        if decade < 0:
            continue
        p = _params(sub)
        p["decade"] = int(decade)
        use_fallback = (p["n_det"] < MIN_VAL_DETECTED or p["n_undet"] < MIN_VAL_UNDETECTED
                        or np.isnan(p["pi"]) or np.isnan(p["phi"]))
        if use_fallback:
            p.update({k: overall[k] for k in ["pi", "pi_lo", "pi_hi", "phi", "phi_lo", "phi_hi"]})
        p["fallback"] = use_fallback
        rows.append(p)
        log.info("Decade %ds: π=%.3f [%.3f–%.3f]  φ=%.3f [%.3f–%.3f]  n=%d+%d  fallback=%s",
                 decade, p["pi"], p["pi_lo"], p["pi_hi"],
                 p["phi"], p["phi_lo"], p["phi_hi"],
                 p["n_det"], p["n_undet"], use_fallback)

    return pd.DataFrame(rows).set_index("decade")


def _corrected_rate(f: float, pi: float, phi: float) -> float:
    return f * pi + (1 - f) * phi


def compute_decade_prevalence(
    det_df: pd.DataFrame,
    corr: pd.DataFrame,
    group_cols: list[str],
    min_n: int,
) -> pd.DataFrame:
    """Apply per-decade π/φ correction to each group."""
    overall = corr.loc[-1]
    rows = []
    for keys, sub in det_df.groupby(group_cols):
        if not isinstance(keys, tuple):
            keys = (keys,)
        kv = dict(zip(group_cols, keys))
        n = len(sub)
        if n < min_n:
            continue
        f = float(sub["has_person"].mean())
        decade = kv["decade"]
        c = corr.loc[decade] if decade in corr.index else overall
        pi,  pi_lo,  pi_hi  = float(c["pi"]),  float(c["pi_lo"]),  float(c["pi_hi"])
        phi, phi_lo, phi_hi = float(c["phi"]), float(c["phi_lo"]), float(c["phi_hi"])
        p_hat    = _corrected_rate(f, pi,    phi)
        p_hat_lo = _corrected_rate(f, pi_lo, phi_lo)
        p_hat_hi = _corrected_rate(f, pi_hi, phi_hi)
        raw_lo, raw_hi = _wilson_ci(int(sub["has_person"].sum()), n)
        rows.append({**kv, "n": n, "raw_rate": f,
                     "raw_lo": raw_lo, "raw_hi": raw_hi,
                     "corrected": p_hat, "corrected_lo": p_hat_lo, "corrected_hi": p_hat_hi,
                     "bias": p_hat - f,
                     "fallback": bool(c.get("fallback", False))})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# PPI prevalence
# ---------------------------------------------------------------------------

def _load_val_for_ppi(validation_csv: Path) -> pd.DataFrame:
    """Load and prep validation_results.csv for PPI computation."""
    vdf = pd.read_csv(validation_csv)
    vdf = vdf[vdf["ann_answered"].astype(bool)].copy()
    vdf["decade"] = (vdf["year"].astype(float) // 10 * 10).astype(int)
    vdf["gt"]     = (vdf["ann_n_persons"] > 0).astype(int)
    vdf["yhat"]   = vdf["yolo_detected"].astype(int)
    return vdf


def compute_decade_prevalence_ppi(
    det_df: pd.DataFrame,
    val_df: pd.DataFrame,
    group_cols: list[str],
    min_n_labeled: int = 5,
    min_n: int = 20,
    force_lambda: float | None = None,
) -> pd.DataFrame:
    """Compute prevalence per group using PPI with importance-weighted rectifier.

    force_lambda=None  → PPI++ (optimal λ per stratum)
    force_lambda=1.0   → vanilla PPI (Angelopoulos et al. 2023, original)

    Importance weights correct for the 50/50 detected/undetected stratification
    in the validation sample:
        w_det   = f_corpus / (n_det_labeled / n_labeled)
        w_undet = (1 − f_corpus) / (n_undet_labeled / n_labeled)

    Per-magazine groups use the same per-decade weights (validation is pooled
    over magazines — not enough per-cell observations).
    """
    rows = []
    for keys, det_sub in det_df.groupby(group_cols):
        if not isinstance(keys, tuple):
            keys = (keys,)
        kv = dict(zip(group_cols, keys))
        N = len(det_sub)
        if N < min_n:
            continue

        decade = kv["decade"]
        Yu = det_sub["yhat"].values.astype(float)
        f  = float(Yu.mean())

        # Always filter validation by decade (the bias parameter that matters most)
        val_sub = val_df[val_df["decade"] == decade]
        n = len(val_sub)

        raw_lo, raw_hi = _wilson_ci(int(Yu.sum()), N)

        if n < min_n_labeled:
            rows.append({**kv, "n": N, "n_labeled": n,
                         "raw_rate": f, "raw_lo": raw_lo, "raw_hi": raw_hi,
                         "corrected": float("nan"), "corrected_lo": float("nan"),
                         "corrected_hi": float("nan"), "lam": float("nan"),
                         "ppi_available": False})
            continue

        Y    = val_sub["gt"].values.astype(float)
        Yhat = val_sub["yhat"].values.astype(float)
        n_det   = int(Yhat.sum())
        n_undet = n - n_det
        if n_det > 0 and n_undet > 0:
            w = np.where(Yhat == 1, f / (n_det / n), (1 - f) / (n_undet / n))
        else:
            w = np.ones(n)

        ppi = get_ppi_estimate(Y, Yhat, Yu, weights=w, lam=force_lambda)
        rows.append({**kv, "n": N, "n_labeled": n,
                     "raw_rate": f, "raw_lo": raw_lo, "raw_hi": raw_hi,
                     "corrected": ppi.estimate,
                     "corrected_lo": ppi.ci_lo, "corrected_hi": ppi.ci_hi,
                     "lam": ppi.lam, "ppi_available": True})

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def _save(fig: plt.Figure, path: Path) -> None:
    fix_fonts(fig)
    fig.savefig(path.with_suffix(".pdf"), dpi=200, bbox_inches="tight")
    fig.savefig(path.with_suffix(".png"), dpi=200, bbox_inches="tight")
    plt.close(fig)
    log.info("Saved %s (.pdf + .png)", path.stem)


def plot_prevalence_corrected(
    prev: pd.DataFrame,
    output_dir: Path,
    suffix: str = "",
    method_label: str = "Corrected estimate",
) -> None:
    """All-magazine trend: raw (dashed) vs corrected (solid) with CI band."""
    fig, ax = plt.subplots(figsize=(9, 5))

    color_raw  = "#999999"
    color_corr = "#542344"  # project primary

    idx = prev.set_index("decade")
    decades = prev["decade"].sort_values().values
    raw    = idx["raw_rate"].reindex(decades).values * 100
    corr   = idx["corrected"].reindex(decades).values * 100
    lo     = idx["corrected_lo"].reindex(decades).values * 100
    hi     = idx["corrected_hi"].reindex(decades).values * 100
    valid  = ~np.isnan(corr)

    ax.plot(decades, raw, color=color_raw, lw=1.5, ls="--", label="Raw YOLO rate", alpha=0.7)
    if valid.any():
        ax.fill_between(decades[valid], lo[valid], hi[valid],
                        color=color_corr, alpha=0.15, label=f"{method_label} ± 95% CI")
        ax.plot(decades[valid], corr[valid], color=color_corr, lw=2.5, label=method_label)

    ax.set_xlabel("Decade")
    ax.set_ylabel("% of ads containing ≥1 person")
    ax.set_xticks(decades)
    ax.set_xticklabels([f"{d}s" for d in decades], rotation=45, ha="right")
    ax.set_ylim(0, 100)
    ax.legend(fontsize=9)
    _save(fig, output_dir / f"fig_people_prevalence_corrected{suffix}.png")


def plot_prevalence_by_magazine(
    det_df: pd.DataFrame,
    val_df: pd.DataFrame,
    output_dir: Path,
    min_n: int,
    suffix: str = "",
    force_lambda: float | None = None,
) -> None:
    """Per-magazine corrected prevalence; raw rates shown as matching dashed lines."""
    prev = compute_decade_prevalence_ppi(
        det_df, val_df, ["decade", "magazine"], min_n=min_n, force_lambda=force_lambda
    )

    if prev.empty:
        log.warning("No magazine × decade data; skipping magazine plot")
        return

    mags_present = [m for m in MAGAZINES if m in prev["magazine"].values]
    fig, ax = plt.subplots(figsize=(11, 6))

    for mag in mags_present:
        sub = prev[prev["magazine"] == mag].sort_values("decade")
        # Only plot decades with a valid corrected estimate
        sub_valid = sub[sub["corrected"].notna()]
        color = MAGAZINE_PALETTE.get(mag, "#555555")
        lw    = 3.0 if mag == "time" else 1.8
        alpha = 1.0 if mag == "time" else 0.80

        if not sub_valid.empty:
            ax.fill_between(sub_valid["decade"],
                            sub_valid["corrected_lo"] * 100,
                            sub_valid["corrected_hi"] * 100,
                            color=color, alpha=0.10)
            ax.plot(sub_valid["decade"], sub_valid["corrected"] * 100,
                    color=color, lw=lw, alpha=alpha, label=magazine_label(mag))
        ax.plot(sub["decade"], sub["raw_rate"] * 100,
                color=color, lw=lw * 0.6, ls="--", alpha=0.4)

    ax.plot([], [], color="#888888", lw=1.2, ls="--", alpha=0.6, label="Raw (dashed)")

    ax.set_xlabel("Decade")
    ax.set_ylabel("% of ads containing ≥1 person (corrected)")
    decades = sorted(prev["decade"].unique())
    ax.set_xticks(decades)
    ax.set_xticklabels([f"{d}s" for d in decades], rotation=45, ha="right")
    ax.set_ylim(0, 100)
    ax.legend(fontsize=9, ncol=2)
    _save(fig, output_dir / f"fig_people_prevalence_by_magazine{suffix}.png")


def plot_bias_by_decade(prev: pd.DataFrame, output_dir: Path, suffix: str = "") -> None:
    """Bar chart of corrected − raw rate by decade: where does the bias matter?"""
    sub = prev[prev["corrected"].notna()].sort_values("decade")
    if sub.empty:
        return
    bias_pct = (sub["corrected"] - sub["raw_rate"]) * 100
    lo_err   = np.clip(bias_pct - (sub["corrected_lo"] - sub["raw_rate"]) * 100, 0, None)
    hi_err   = np.clip((sub["corrected_hi"] - sub["raw_rate"]) * 100 - bias_pct, 0, None)

    colors = ["#E63946" if b > 0 else "#457B9D" for b in bias_pct]
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.bar(sub["decade"], bias_pct, width=8, color=colors, alpha=0.80)
    ax.errorbar(sub["decade"], bias_pct,
                yerr=[lo_err, hi_err],
                fmt="none", color="black", capsize=4, lw=1.2)
    ax.axhline(0, color="black", lw=0.8, alpha=0.5)

    for _, row in sub.iterrows():
        b = (row["corrected"] - row["raw_rate"]) * 100
        ax.text(row["decade"], b + np.sign(b) * 0.5,
                f"{b:+.1f}pp", ha="center", fontsize=8)

    ax.set_xlabel("Decade")
    ax.set_ylabel("Corrected − Raw (percentage points)")
    ax.set_xticks(sub["decade"])
    ax.set_xticklabels([f"{d}s" for d in sub["decade"]], rotation=45, ha="right")
    _save(fig, output_dir / f"fig_people_detection_bias{suffix}.png")


def write_summary(prev: pd.DataFrame, output_dir: Path, label: str = "Corrected") -> None:
    lines = [
        f"PEOPLE PREVALENCE — {label.upper()} ESTIMATES",
        "=" * 50,
        "",
        f"  {'Decade':<8}  {'N ads':>7}  {'Raw':>6}  {label:>10}  {'Bias':>6}",
        "  " + "-" * 55,
    ]
    for _, row in prev.sort_values("decade").iterrows():
        corr_str = (
            f"{100*row['corrected']:>7.1f}% [{100*row['corrected_lo']:.1f}–{100*row['corrected_hi']:.1f}%]"
            if pd.notna(row["corrected"]) else "       —"
        )
        bias_str = f"{(row['corrected'] - row['raw_rate'])*100:>+5.1f}pp" if pd.notna(row["corrected"]) else "      —"
        lines.append(
            f"  {row['decade']:<8}  {int(row['n']):>7,}  {100*row['raw_rate']:>5.1f}%  {corr_str}  {bias_str}"
        )
    out = output_dir / "people_prevalence_summary.txt"
    out.write_text("\n".join(lines) + "\n")
    log.info("Saved %s", out)
    for line in lines:
        log.info(line)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--detections_csv", type=Path, required=True,
                   help="people_detections.csv from infer_people.py")
    p.add_argument("--validation_csv", type=Path, required=True,
                   help="validation_results.csv from analyze_people_validation.py")
    p.add_argument("--output_dir", type=Path, default=Path("paperplots"))
    p.add_argument("--min_year", type=int, default=1920)
    p.add_argument("--max_year", type=int, default=2017)
    p.add_argument("--min_n", type=int, default=20,
                   help="Minimum ads per decade (or decade × magazine) to include")
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    apply_style()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    det_df = load_detections(args.detections_csv, args.min_year, args.max_year)
    val_df = _load_val_for_ppi(args.validation_csv)

    # ── π/φ classical plug-in correction ─────────────────────────────────
    corr   = load_correction_params(args.validation_csv)
    prev_phi = compute_decade_prevalence(det_df, corr, ["decade"], min_n=args.min_n)
    plot_prevalence_corrected(prev_phi, args.output_dir,
                              suffix="_phi", method_label="π/φ correction")
    plot_bias_by_decade(prev_phi, args.output_dir, suffix="_phi")
    plot_prevalence_by_magazine(det_df, val_df, args.output_dir,
                                min_n=args.min_n, suffix="_phi", force_lambda=1.0)
    prev_phi.to_csv(args.output_dir / "people_decade_prevalence_phi.csv", index=False)

    # ── PPI (λ=1, Angelopoulos et al. 2023 original) ─────────────────────
    prev_ppi = compute_decade_prevalence_ppi(
        det_df, val_df, ["decade"], min_n=args.min_n, force_lambda=1.0
    )
    plot_prevalence_corrected(prev_ppi, args.output_dir,
                              suffix="_ppi", method_label="PPI (λ=1)")
    plot_bias_by_decade(prev_ppi, args.output_dir, suffix="_ppi")
    plot_prevalence_by_magazine(det_df, val_df, args.output_dir,
                                min_n=args.min_n, suffix="_ppi", force_lambda=1.0)
    prev_ppi.to_csv(args.output_dir / "people_decade_prevalence_ppi.csv", index=False)

    # ── PPI++ (optimal λ) ─────────────────────────────────────────────────
    prev_ppipp = compute_decade_prevalence_ppi(
        det_df, val_df, ["decade"], min_n=args.min_n
    )
    plot_prevalence_corrected(prev_ppipp, args.output_dir,
                              suffix="_ppipp", method_label="PPI++")
    plot_bias_by_decade(prev_ppipp, args.output_dir, suffix="_ppipp")
    plot_prevalence_by_magazine(det_df, val_df, args.output_dir,
                                min_n=args.min_n, suffix="_ppipp")

    write_summary(prev_ppipp, args.output_dir, label="PPI++")
    prev_ppipp.to_csv(args.output_dir / "people_decade_prevalence.csv", index=False)
    log.info("Saved people_decade_prevalence.csv")


if __name__ == "__main__":
    main()
