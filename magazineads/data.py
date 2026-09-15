"""Corpus-level data loading with regulatory boilerplate filtering.

Provides load_ad_texts() which loads the OCR text CSV and optionally removes
FDA-mandated pharma text and tobacco Surgeon General's warnings that inject
formulaic regulatory language into text-as-data analyses.
"""

import re

import pandas as pd
from pathlib import Path

from magazineads.paths import paths as _paths


# ── Regulatory boilerplate filter ──
# Validated 2026-07-13: see docs/design/med_warning_filter.md

# --- High-precision patterns (trigger independently) ---
_MED_PATTERNS_STRONG = [
    r"\bside effects?\b",
    r"\bask your doctor\b",
    r"\btell your doctor\b",
    r"\bdosage\b",
    r"\bcontraindications?\b",
    r"\badverse reactions?\b",
    r"\bclinical\s+(trials?|studies|pharmacology)\b",
    r"\bplacebo\b",
    r"\bnursing mothers?\b",
    r"\bpregnancy\b",
]

# --- Tobacco Surgeon General's warnings ---
_TOBACCO_WARNING_PATTERN = (
    r"\bsurgeon\s+general\b"
)

# --- Weak patterns: only count if >=2 co-occur in the same ad ---
_MED_PATTERNS_WEAK = [
    r"\bprescription\b",
    r"\bprecautions?\s*:",
    r"\bwarnings?\s*:",
    r"\bindications?\s*(and|:)",
]

_STRONG_RE = re.compile("|".join(_MED_PATTERNS_STRONG), re.IGNORECASE)
_TOBACCO_RE = re.compile(_TOBACCO_WARNING_PATTERN, re.IGNORECASE)
_WEAK_RES = [re.compile(p, re.IGNORECASE) for p in _MED_PATTERNS_WEAK]


def _is_regulatory_boilerplate(text: str) -> bool:
    """Return True if text contains regulatory boilerplate (pharma or tobacco)."""
    if _STRONG_RE.search(text):
        return True
    if _TOBACCO_RE.search(text):
        return True
    weak_hits = sum(1 for pat in _WEAK_RES if pat.search(text))
    if weak_hits >= 2:
        return True
    return False


def load_ad_texts(
    csv_path: str | Path | None = None,
    min_words: int = 10,
    drop_med_warnings: bool = True,
) -> pd.DataFrame:
    """Load the corpus-wide ad texts CSV with standard preprocessing.

    Returns a DataFrame with columns: stem, magazine, text, year,
    clean_text, text_lower, n_words.  Optionally filters out
    medication-warning ads and very short texts.

    Parameters
    ----------
    csv_path : path to ad_texts.csv.  Defaults to
        ``{data_root}/ad_texts.csv``.
    min_words : drop ads with fewer than this many words (0 to keep all).
    drop_med_warnings : if True, remove ads matching pharma/tobacco
        regulatory boilerplate (~3-4 % of corpus).
    """
    if csv_path is None:
        csv_path = _paths.data_root / "ad_texts.csv"
    df = pd.read_csv(csv_path)

    df["year"] = df["stem"].str[:4].astype(int)
    df["clean_text"] = df["text"].fillna("").str.replace(r"<[^>]+>", " ", regex=True)
    df["text_lower"] = df["clean_text"].str.lower()
    df["n_words"] = df["clean_text"].str.split().str.len().fillna(0).astype(int)

    n0 = len(df)

    if drop_med_warnings:
        is_med = df["text_lower"].apply(_is_regulatory_boilerplate)
        n_med = is_med.sum()
        df = df[~is_med].copy()
        print(f"Regulatory boilerplate removed: {n_med:,} ({n_med / n0 * 100:.1f}%)")

    if min_words > 0:
        df = df[df["n_words"] >= min_words].copy()

    print(f"{len(df):,} ads loaded (of {n0:,})")
    return df
