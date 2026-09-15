"""Thin wrappers around the official ppi_py package (Angelopoulos et al.).

All computation is delegated to ppi_py (pip: ppi-python), which implements
the estimators from:

  PPI++: Angelopoulos, Duchi, Zrnic (2023). arXiv:2311.01453
  PPI:   Angelopoulos et al. (2023). arXiv:2301.09633

This module adds:
  - structured result dataclasses (PPIResult, PPICorrelationResult)
  - importance-weight normalisation for our stratified validation design
  - lambda reporting (ppi_py computes but does not return it)
  - a correlation wrapper that converts an OLS slope to Pearson r

Install ppi_py:
    pip install ppi-python
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from ppi_py import (
    ppi_mean_ci,
    ppi_mean_pointestimate,
    ppi_ols_ci,
    ppi_ols_pointestimate,
)
from scipy import stats as scipy_stats


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------

@dataclass
class PPIResult:
    """Result from a PPI++ prevalence estimate."""
    estimate:    float
    se:          float
    ci_lo:       float
    ci_hi:       float
    lam:         float
    lam_opt:     bool
    n_labeled:   int
    n_unlabeled: int
    classical:   float


@dataclass
class PPICorrelationResult:
    """PPI++ estimate of Pearson r(X, Y) where X is fully observed."""
    r:           float
    se:          float
    ci_lo:       float
    ci_hi:       float
    lam:         float
    raw_r:       float
    classical_r: float
    n_labeled:   int
    n_unlabeled: int


# ---------------------------------------------------------------------------
# Internal: optimal-lambda formula (for reporting only)
# ---------------------------------------------------------------------------

def _optimal_lambda(
    Y: np.ndarray,
    Yhat: np.ndarray,
    Yhat_unlabeled: np.ndarray,
) -> float:
    """Variance-minimising lambda for 1-D mean estimation (scalar case)."""
    n = len(Y)
    N = len(Yhat_unlabeled)
    cov = float(np.cov(Y.astype(float), Yhat.astype(float))[0, 1])
    var_pooled = float(np.var(
        np.concatenate([Yhat.astype(float), Yhat_unlabeled.astype(float)]),
        ddof=1,
    ))
    if var_pooled == 0.0:
        return 1.0
    return float(np.clip(cov / ((1.0 + n / N) * var_pooled), 0.0, 1.0))


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def get_ppi_estimate(
    Y:               np.ndarray,
    Yhat:            np.ndarray,
    Yhat_unlabeled:  np.ndarray,
    alpha:           float = 0.05,
    lam:             float | None = None,
    weights:         np.ndarray | None = None,
) -> PPIResult:
    """PPI++ prevalence estimate delegating to ppi_py.ppi_mean_ci.

    Parameters
    ----------
    Y               Ground-truth labels on the labeled set, values in {0,1}.
    Yhat            Model predictions on the labeled set.
    Yhat_unlabeled  Model predictions on the full corpus.
    alpha           Significance level (default 0.05 -> 95% CI).
    lam             lambda in [0,1].  None -> auto-optimised (PPI++).  1.0 -> PPI.
    weights         Importance weights for the labeled set (for stratified designs).
    """
    Y    = np.asarray(Y,    dtype=float)
    Yhat = np.asarray(Yhat, dtype=float)
    Yhat_unlabeled = np.asarray(Yhat_unlabeled, dtype=float)
    n, N = len(Y), len(Yhat_unlabeled)

    w = np.asarray(weights, dtype=float) if weights is not None else None

    lam_opt = lam is None
    lam_report = lam if not lam_opt else _optimal_lambda(Y, Yhat, Yhat_unlabeled)

    estimate = float(ppi_mean_pointestimate(Y, Yhat, Yhat_unlabeled, lam=lam, w=w).squeeze())
    ci_lo_arr, ci_hi_arr = ppi_mean_ci(Y, Yhat, Yhat_unlabeled, alpha=alpha, lam=lam, w=w)
    ci_lo = float(ci_lo_arr.squeeze())
    ci_hi = float(ci_hi_arr.squeeze())

    z   = scipy_stats.norm.ppf(1.0 - alpha / 2.0)
    se  = (ci_hi - ci_lo) / (2.0 * z)

    classical = float(np.average(Y, weights=w) if w is not None else Y.mean())

    return PPIResult(
        estimate=estimate, se=se, ci_lo=ci_lo, ci_hi=ci_hi,
        lam=lam_report, lam_opt=lam_opt,
        n_labeled=n, n_unlabeled=N,
        classical=classical,
    )


def get_ppi_correlation(
    Y:               np.ndarray,
    Yhat:            np.ndarray,
    Yhat_unlabeled:  np.ndarray,
    X_labeled:       np.ndarray,
    X_unlabeled:     np.ndarray,
    alpha:           float = 0.05,
    lam:             float | None = None,
    weights:         np.ndarray | None = None,
) -> PPICorrelationResult:
    """PPI++ estimate of Pearson r(X, Y) via ppi_py.ppi_ols_ci.

    X is a continuous covariate fully observed in the corpus (e.g. year).
    Y is a binary outcome observed only for the labeled set (e.g. has_person).

    Internally fits Y ~ 1 + X via PPI++ OLS, then converts the slope to
    Pearson r via r = slope * SD_X / SD_Y.
    """
    Y    = np.asarray(Y,    dtype=float)
    Yhat = np.asarray(Yhat, dtype=float)
    Yhat_unlabeled = np.asarray(Yhat_unlabeled, dtype=float)
    X_labeled   = np.asarray(X_labeled,   dtype=float)
    X_unlabeled = np.asarray(X_unlabeled, dtype=float)
    n, N = len(Y), len(Yhat_unlabeled)

    w = np.asarray(weights, dtype=float) if weights is not None else None

    X_lab_d = np.column_stack([np.ones(n), X_labeled])
    X_unl_d = np.column_stack([np.ones(N), X_unlabeled])

    beta_hat = ppi_ols_pointestimate(
        X_lab_d, Y, Yhat, X_unl_d, Yhat_unlabeled, lam=lam, w=w
    )
    ci_lo_arr, ci_hi_arr = ppi_ols_ci(
        X_lab_d, Y, Yhat, X_unl_d, Yhat_unlabeled, alpha=alpha, lam=lam, w=w
    )

    slope    = float(beta_hat[1])
    slope_lo = float(ci_lo_arr[1])
    slope_hi = float(ci_hi_arr[1])

    SD_X = float(X_unlabeled.std())

    p_hat = float(ppi_mean_pointestimate(Y, Yhat, Yhat_unlabeled, lam=lam, w=w).squeeze())
    SD_Y  = float(np.sqrt(max(p_hat * (1.0 - p_hat), 1e-12)))

    scale = SD_X / SD_Y
    r     = slope    * scale
    r_lo  = slope_lo * scale
    r_hi  = slope_hi * scale

    z  = scipy_stats.norm.ppf(1.0 - alpha / 2.0)
    se = (r_hi - r_lo) / (2.0 * z)

    lam_opt    = lam is None
    lam_report = lam if not lam_opt else _optimal_lambda(Y, Yhat, Yhat_unlabeled)

    SD_Yhat  = float(np.sqrt(max(Yhat_unlabeled.mean() * (1 - Yhat_unlabeled.mean()), 1e-12)))
    cov_raw  = float(np.cov(X_unlabeled, Yhat_unlabeled)[0, 1])
    raw_r    = cov_raw / (SD_X * SD_Yhat) if SD_X > 0 else 0.0

    w_norm  = w / w.mean() if w is not None else np.ones(n)
    E_Y_cl  = float(np.average(Y,             weights=w_norm))
    E_XY_cl = float(np.average(X_labeled * Y, weights=w_norm))
    SD_Y_cl = float(np.sqrt(max(E_Y_cl * (1 - E_Y_cl), 1e-12)))
    cov_cl  = E_XY_cl - X_unlabeled.mean() * E_Y_cl
    classical_r = cov_cl / (SD_X * SD_Y_cl) if SD_X > 0 else 0.0

    return PPICorrelationResult(
        r=r, se=se, ci_lo=float(min(r_lo, r_hi)), ci_hi=float(max(r_lo, r_hi)),
        lam=lam_report, raw_r=raw_r, classical_r=classical_r,
        n_labeled=n, n_unlabeled=N,
    )


def get_ppi_estimates_by_stratum(
    Y:              np.ndarray,
    Yhat:           np.ndarray,
    Yhat_unlabeled: np.ndarray,
    stratum_labeled:   np.ndarray,
    stratum_unlabeled: np.ndarray,
    alpha:          float = 0.05,
    weights:        np.ndarray | None = None,
    lam:            float | None = None,
    min_n_labeled:  int = 5,
    min_n_unlabeled: int = 20,
) -> dict[str, PPIResult | None]:
    """Run get_ppi_estimate independently within each stratum (e.g. decade)."""
    Y    = np.asarray(Y,    dtype=float)
    Yhat = np.asarray(Yhat, dtype=float)
    Yhat_unlabeled    = np.asarray(Yhat_unlabeled,    dtype=float)
    stratum_labeled   = np.asarray(stratum_labeled)
    stratum_unlabeled = np.asarray(stratum_unlabeled)

    strata = sorted(set(stratum_labeled) | set(stratum_unlabeled))
    results: dict = {}

    for s in strata:
        mask_l = stratum_labeled   == s
        mask_u = stratum_unlabeled == s
        if mask_l.sum() < min_n_labeled or mask_u.sum() < min_n_unlabeled:
            results[s] = None
            continue
        w_s = weights[mask_l] if weights is not None else None
        results[s] = get_ppi_estimate(
            Y[mask_l], Yhat[mask_l], Yhat_unlabeled[mask_u],
            alpha=alpha, lam=lam, weights=w_s,
        )

    return results
