"""
magazineads/corpus.py

Canonical data-loading layer for analysis scripts.

Provides:
  - load_predictions()    — predictions CSV → filtered DataFrame with magazine/year
  - sample_ads()          — stratified ad sampling
  - load_ocr_page()       — Surya JSON → OcrPage (with text-joining policy)
  - split_sentences()     — spaCy sentence splitting over an OcrPage
  - load_argument_cache() — argument mining cache CSV → filtered DataFrame
  - load_spacy()          — shared spaCy model loader

Text joining policy
-------------------
OcrPage.text(policy) controls how OCR lines are concatenated:
  "space"   — join with spaces (good for classification / embedding)
  "newline" — join with newlines (better for display or reading-order-sensitive tasks)
  "lines"   — return raw list via .lines_text() instead

Known limitation: Surya reading order can be wrong for multi-column layouts
(two columns + splash may linearize as col1 + splash + col2). This is a
property of the OCR source, not of the joining policy.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd

MAG_ORDER = ["time", "esquire", "life", "forbes", "variety"]

JoinPolicy = Literal["space", "newline"]


# ---------------------------------------------------------------------------
# Predictions
# ---------------------------------------------------------------------------

def load_predictions(
    csv_path: Path,
    magazines: list[str] = MAG_ORDER,
    ad_label: int = 1,
) -> pd.DataFrame:
    """Load the page classification predictions CSV.

    Returns a DataFrame with columns:
      filepath, pred_label, magazine, stem, year

    Only rows matching ad_label and the given magazines are returned.
    Rows where year cannot be parsed are dropped.
    """
    df = pd.read_csv(csv_path, usecols=["filepath", "pred_label"])
    df = df[df["pred_label"] == ad_label].copy()
    df["magazine"] = df["filepath"].str.extract(r"/images/([^/]+)/")
    df = df[df["magazine"].isin(magazines)].copy()
    df["stem"] = df["filepath"].apply(lambda fp: Path(fp).stem)
    df["year"] = df["stem"].apply(_year_from_stem)
    df = df.dropna(subset=["year"]).copy()
    df["year"] = df["year"].astype(int)
    return df.reset_index(drop=True)


def _year_from_stem(stem: str) -> int | None:
    try:
        return int(stem.split("-")[0])
    except (ValueError, IndexError):
        return None


# ---------------------------------------------------------------------------
# Sampling
# ---------------------------------------------------------------------------

def sample_ads(
    df: pd.DataFrame,
    n: int,
    seed: int = 42,
    stratify_by_magazine: bool = True,
) -> pd.DataFrame:
    """Sample n distinct ads from a predictions-style DataFrame.

    If stratify_by_magazine=True, draws roughly equal numbers from each
    magazine present in df. If a magazine has fewer ads than its quota,
    all of its ads are included and the remainder is filled from others.

    Returns a DataFrame with one row per ad (magazine, stem, year).
    """
    unique_ads = df[["magazine", "stem", "year"]].drop_duplicates()

    if not stratify_by_magazine:
        return (
            unique_ads.sample(min(n, len(unique_ads)), random_state=seed)
                      .reset_index(drop=True)
        )

    magazines = [m for m in MAG_ORDER if m in unique_ads["magazine"].values]
    n_per_mag = n // len(magazines) + 1
    sampled = []
    for mag in magazines:
        mag_ads = unique_ads[unique_ads["magazine"] == mag]
        sampled.append(mag_ads.sample(min(len(mag_ads), n_per_mag), random_state=seed))

    return (
        pd.concat(sampled)
          .sample(frac=1, random_state=seed)
          .head(n)
          .reset_index(drop=True)
    )


# ---------------------------------------------------------------------------
# OCR page loading
# ---------------------------------------------------------------------------

@dataclass
class OcrLine:
    line_idx: int
    text: str
    confidence: float
    polygon: list | None = field(default=None, repr=False)
    bbox: list | None = field(default=None, repr=False)


@dataclass
class OcrPage:
    """OCR output for a single ad page.

    Attributes
    ----------
    magazine, stem : identifiers
    lines          : all non-empty text lines from Surya, in reading order
                     (note: reading order may be unreliable for multi-column layouts)
    page_conf      : mean Surya confidence across all lines — use as a
                     page-level scan quality indicator
    """
    magazine: str
    stem: str
    lines: list[OcrLine]

    @property
    def page_conf(self) -> float:
        if not self.lines:
            return 0.0
        return float(np.mean([l.confidence for l in self.lines]))

    def lines_text(self, min_conf: float = 0.0) -> list[str]:
        """Return text of lines above min_conf, preserving order."""
        return [l.text for l in self.lines if l.confidence >= min_conf]

    def text(self, policy: JoinPolicy = "space", min_conf: float = 0.0) -> str:
        """Return page text as a single string.

        policy="space"   — join with spaces (for classification / embedding)
        policy="newline" — join with newlines (for display / reading-order tasks)
        """
        sep = " " if policy == "space" else "\n"
        return sep.join(self.lines_text(min_conf=min_conf))

    def __len__(self) -> int:
        return len(self.lines)

    def __bool__(self) -> bool:
        return bool(self.lines)


def load_ocr_page(
    surya_dir: Path,
    magazine: str,
    stem: str,
) -> OcrPage | None:
    """Load a Surya results.json and return an OcrPage.

    Returns None if the file is missing or unreadable.
    """
    json_path = surya_dir / magazine / stem / "results.json"
    try:
        with open(json_path) as f:
            d = json.load(f)
    except Exception:
        return None

    try:
        pages = next(iter(d.values()))
    except StopIteration:
        return None

    lines = []
    line_idx = 0
    for page in pages:
        for raw in page.get("text_lines", []):
            text = raw.get("text", "").strip()
            if not text:
                continue
            lines.append(OcrLine(
                line_idx=line_idx,
                text=text,
                confidence=float(raw.get("confidence", 0.0)),
                polygon=raw.get("polygon"),
                bbox=raw.get("bbox"),
            ))
            line_idx += 1

    return OcrPage(magazine=magazine, stem=stem, lines=lines)


# ---------------------------------------------------------------------------
# Sentence splitting
# ---------------------------------------------------------------------------

def load_spacy():
    """Return a spaCy pipeline with sentence boundary detection.

    Tries en_core_web_sm (best boundaries); falls back to a bare sentencizer
    if the model is not installed.
    """
    import spacy
    try:
        return spacy.load("en_core_web_sm", disable=["ner", "lemmatizer", "tagger"])
    except OSError:
        nlp = spacy.blank("en")
        nlp.add_pipe("sentencizer")
        return nlp


@dataclass
class Sentence:
    line_idx: int
    sent_idx: int
    text: str
    ocr_confidence: float  # confidence of the source OcrLine


def split_sentences(
    nlp,
    page: OcrPage,
    min_chars: int = 10,
    min_conf: float = 0.0,
) -> list[Sentence]:
    """Sentence-split an OcrPage using spaCy.

    Each sentence inherits the confidence of its source OCR line.
    Sentences shorter than min_chars or from lines below min_conf are dropped.

    Note: sentence boundaries are line-local — spaCy sees one OCR line at a
    time, not the whole page. Cross-line sentence boundaries are not detected.
    """
    eligible = [l for l in page.lines if l.confidence >= min_conf]
    if not eligible:
        return []

    result = []
    texts = [l.text for l in eligible]
    for line, doc in zip(eligible, nlp.pipe(texts)):
        for sent_idx, sent in enumerate(doc.sents):
            text = sent.text.strip()
            if len(text) >= min_chars:
                result.append(Sentence(
                    line_idx=line.line_idx,
                    sent_idx=sent_idx,
                    text=text,
                    ocr_confidence=line.confidence,
                ))
    return result


# ---------------------------------------------------------------------------
# Argument mining cache
# ---------------------------------------------------------------------------

def load_argument_cache(
    csv_path: Path,
    magazines: list[str] = MAG_ORDER,
    min_page_conf: float = 0.0,
    min_sent_conf: float = 0.0,
) -> pd.DataFrame:
    """Load the argument mining cache CSV with standard filters applied.

    Columns: magazine, stem, year, line_idx, sent_idx, text,
             ocr_confidence, page_ocr_conf, arg_label, arg_score

    Filters are advisory (for analysis/plotting). Use the inference-time
    floors in argument_mining.py to control what gets written to the cache.
    """
    df = pd.read_csv(csv_path)
    df = df[df["magazine"].isin(magazines)].copy()
    if min_page_conf > 0.0:
        df = df[df["page_ocr_conf"] >= min_page_conf].copy()
    if min_sent_conf > 0.0:
        df = df[df["ocr_confidence"] >= min_sent_conf].copy()
    return df.reset_index(drop=True)
