"""Compute whitespace fraction for full-page magazine ads.

Whitespace is defined as the fraction of an ad image not occupied by either
text or visual content.  We combine two signals:

    1. **Text regions** — bounding boxes from precomputed Surya OCR output.
       The entire bbox area counts as "text" (inter-line space within a text
       block is not meaningful whitespace in a design sense).
    2. **Visual content** — pixels with high local variance (detected via
       box-filter variance, window=51, threshold=100) that fall *outside*
       text bboxes, consolidated with morphological closing (k=31).
       This captures illustrations, photographs, logos, rules, borders, etc.
       Unlike Sauvola binarization, local variance treats uniform dark regions
       (solid black backgrounds) as background, not content.
    3. **Whitespace** = 1 − text_frac − visual_content_frac.

Output CSV columns:
    filepath, magazine, year, decade,
    whitespace_frac, text_frac, visual_frac,
    n_text_lines, content_density,
    img_width, img_height

Usage:
    python -m magazines.scripts.compute_whitespace \
        --out data/whitespace.csv \
        [--magazine time] \
        [--n-per-year 50] \
        [--workers 8]
"""

from __future__ import annotations

import argparse
import json
import re
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

Image.MAX_IMAGE_PIXELS = 160_000_000

KNOWN_MAGAZINES = {"esquire", "forbes", "life", "time", "variety",
                    "crisis", "lhj", "cosmo", "usnews"}
YEAR_RE = re.compile(r"^(\d{4})")
MAX_PIXELS = 1_000_000

try:
    from magazineads.paths import paths
    DEFAULT_PREDICTIONS_CSV = paths.predictions_dir / "predictions_all.csv"
    DEFAULT_SURYA_DIR = paths.surya_output_dir
except Exception:
    DEFAULT_PREDICTIONS_CSV = Path("/data/laviniad/magazines/page_classification_predictions/predictions_all.csv")
    DEFAULT_SURYA_DIR = Path("/data/laviniad/magazines/surya_output")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _infer_magazine(fp: str) -> str | None:
    parts = fp.split("/")
    mag = parts[-2] if len(parts) >= 2 else None
    return mag if mag in KNOWN_MAGAZINES else None


def _infer_year(fp: str) -> int | None:
    stem = Path(fp).stem
    m = YEAR_RE.match(stem)
    if m:
        y = int(m.group(1))
        return y if 1900 <= y <= 2020 else None
    return None


def _load_surya_bboxes(surya_dir: Path, magazine: str, stem: str):
    """Load text line bboxes from Surya output.

    Returns (list of [x1, y1, x2, y2] in surya coords, surya image_bbox)
    or (None, None).
    """
    results_path = surya_dir / magazine / stem / "results.json"

    # Time magazine fallback: -x-x- -> -x-
    if not results_path.exists() and magazine == "time" and "-x-x-" in stem:
        fallback_stem = stem.replace("-x-x-", "-x-", 1)
        results_path = surya_dir / magazine / fallback_stem / "results.json"

    if not results_path.exists():
        return None, None

    try:
        with open(results_path) as f:
            data = json.load(f)
    except (json.JSONDecodeError, OSError):
        return None, None

    for key in data:
        pages = data[key]
        if isinstance(pages, list) and len(pages) > 0:
            page = pages[0]
            image_bbox = page.get("image_bbox", None)
            text_lines = page.get("text_lines", [])
            bboxes = [
                line["bbox"] for line in text_lines
                if line.get("confidence", 0) >= 0.3 and "bbox" in line
            ]
            return bboxes, image_bbox

    return None, None


def _build_text_mask(bboxes, image_bbox, img_w: int, img_h: int) -> np.ndarray:
    """Create a boolean mask (True = text region) at image resolution."""
    mask = np.zeros((img_h, img_w), dtype=bool)

    surya_w = image_bbox[2] - image_bbox[0]
    surya_h = image_bbox[3] - image_bbox[1]
    sx = img_w / surya_w if surya_w > 0 else 1.0
    sy = img_h / surya_h if surya_h > 0 else 1.0

    for bbox in bboxes:
        x1 = max(0, min(int(bbox[0] * sx), img_w - 1))
        y1 = max(0, min(int(bbox[1] * sy), img_h - 1))
        x2 = max(0, min(int(bbox[2] * sx), img_w))
        y2 = max(0, min(int(bbox[3] * sy), img_h))
        mask[y1:y2, x1:x2] = True

    return mask


def _detect_content(gray: np.ndarray, var_window: int = 51,
                    var_threshold: float = 100, close_k: int = 31) -> np.ndarray:
    """Return boolean mask (True = visual content pixel).

    Uses local variance to detect visually active regions: pixels whose
    neighborhood has high variance contain edges, texture, or detail.
    Uniform regions — whether solid black, white, or any flat color — are
    treated as background (whitespace).

    Morphological closing (dilate then erode) consolidates content regions
    so that light patches within photographs are not fragmented into
    whitespace.
    """
    gray_f = gray.astype(np.float64)
    mean = cv2.blur(gray_f, (var_window, var_window))
    mean_sq = cv2.blur(gray_f ** 2, (var_window, var_window))
    variance = np.maximum(mean_sq - mean ** 2, 0)

    content = (variance > var_threshold).astype(np.uint8)

    if close_k > 0:
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (close_k, close_k)
        )
        content = cv2.morphologyEx(content, cv2.MORPH_CLOSE, kernel)

    return content.astype(bool)


# ---------------------------------------------------------------------------
# Per-image worker
# ---------------------------------------------------------------------------

def _process_image(args):
    """Compute whitespace decomposition for one image."""
    filepath, surya_dir, max_pixels = args
    try:
        magazine = _infer_magazine(filepath)
        stem = Path(filepath).stem

        img = Image.open(filepath).convert("RGB")
        orig_w, orig_h = img.size

        # Downsample if needed
        if max_pixels and orig_w * orig_h > max_pixels:
            scale = (max_pixels / (orig_w * orig_h)) ** 0.5
            img = img.resize(
                (max(1, int(orig_w * scale)), max(1, int(orig_h * scale))),
                Image.LANCZOS,
            )

        arr = np.array(img)
        gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
        img_h, img_w = gray.shape
        total_pixels = img_h * img_w

        # Detect content pixels via local variance + morphological closing
        content = _detect_content(gray)
        content_density = float(content.sum() / total_pixels)

        # Load Surya bboxes and build text mask
        bboxes, image_bbox = _load_surya_bboxes(Path(surya_dir), magazine, stem)

        if bboxes and image_bbox:
            text_mask = _build_text_mask(bboxes, image_bbox, img_w, img_h)
            text_frac = float(text_mask.sum() / total_pixels)
            n_text_lines = len(bboxes)
        else:
            text_mask = np.zeros((img_h, img_w), dtype=bool)
            text_frac = 0.0
            n_text_lines = 0

        # Visual content = content pixels outside text bboxes
        visual_mask = content & ~text_mask
        visual_frac = float(visual_mask.sum() / total_pixels)

        # Whitespace = everything else
        whitespace_frac = 1.0 - text_frac - visual_frac

        return {
            "filepath": filepath,
            "whitespace_frac": whitespace_frac,
            "text_frac": text_frac,
            "visual_frac": visual_frac,
            "n_text_lines": n_text_lines,
            "content_density": content_density,
            "img_width": orig_w,
            "img_height": orig_h,
        }
    except Exception:
        return {"filepath": filepath, "error": traceback.format_exc()}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute whitespace fraction for full-page magazine ads."
    )
    parser.add_argument("--out", required=True, help="Output CSV path")
    parser.add_argument(
        "--predictions-csv", type=Path, default=DEFAULT_PREDICTIONS_CSV,
    )
    parser.add_argument(
        "--surya-dir", type=Path, default=DEFAULT_SURYA_DIR,
    )
    parser.add_argument(
        "--magazine", type=str, default=None,
        help="Restrict to one magazine (default: all)",
    )
    parser.add_argument("--n-per-year", type=int, default=50)
    parser.add_argument("--max-pixels", type=int, default=MAX_PIXELS)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    # Load full-page ads
    print(f"Loading {args.predictions_csv}")
    df = pd.read_csv(args.predictions_csv)
    df = df[df["pred_label"] == 1].copy()
    df["magazine"] = df["filepath"].apply(_infer_magazine)
    df["year"] = df["filepath"].apply(_infer_year)
    df = df.dropna(subset=["magazine", "year"])
    df["year"] = df["year"].astype(int)
    df["decade"] = (df["year"] // 10) * 10

    if args.magazine:
        df = df[df["magazine"] == args.magazine]
        print(f"  Filtered to {args.magazine}: {len(df):,} ads")
    else:
        print(f"  {len(df):,} full-page ads across {df['magazine'].nunique()} magazines")

    # Stratified sample: n-per-year per (magazine, year)
    rng = np.random.RandomState(args.seed)
    frames = []
    for _, g in df.groupby(["magazine", "year"]):
        frames.append(g.sample(n=min(args.n_per_year, len(g)), random_state=rng))
    sample = pd.concat(frames, ignore_index=True)
    print(f"  Sampled {len(sample):,} images")

    # Compute in parallel
    work = [(fp, str(args.surya_dir), args.max_pixels) for fp in sample["filepath"]]
    results = []
    errors = []

    desc = getattr(args, "desc", None) or f"ws/{args.magazine or 'all'}"
    print(f"Computing whitespace (workers={args.workers}) ...")
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(_process_image, item): item[0] for item in work}
        for fut in tqdm(as_completed(futures), total=len(futures), desc=desc):
            r = fut.result()
            if "error" in r:
                errors.append(r)
            else:
                results.append(r)

    if errors:
        print(f"  {len(errors)} errors")
        for e in errors[:3]:
            print(f"    {e['filepath']}: {e['error'][:200]}")

    results_df = pd.DataFrame(results)
    results_df = results_df.merge(
        sample[["filepath", "magazine", "year", "decade"]],
        on="filepath", how="left",
    )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(out_path, index=False)
    print(f"\nSaved {len(results_df):,} rows -> {out_path}")

    # Summary
    print(f"  Mean whitespace fraction: {results_df['whitespace_frac'].mean():.3f}")
    print(f"  Mean text fraction:       {results_df['text_frac'].mean():.3f}")
    print(f"  Mean visual fraction:     {results_df['visual_frac'].mean():.3f}")
    has_ocr = results_df["n_text_lines"] > 0
    print(f"  Images with OCR text:     {has_ocr.sum():,} / {len(results_df):,}")


if __name__ == "__main__":
    main()
