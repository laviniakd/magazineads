"""Face detection inference using SCRFD (InsightFace) on magazine ads.

Runs the SCRFD face detector on full-page ads and outputs a CSV with
per-image face counts, bounding boxes, confidences, and landmarks.

Input sources (priority order):
  1. --image_paths PATH [PATH ...]   explicit list
  2. --image_dir PATH                all JPEGs in a directory
  3. (default) predictions CSV       full-page ads from predictions_all.csv

Outputs (in --output_dir):
  face_detections.csv    per-image row with face bounding boxes + confidences

CSV columns:
  image_path, magazine, year, n_faces,
  face_confs  (semicolon-separated floats),
  face_boxes  (JSON list of [x1,y1,x2,y2] normalised to [0,1]),
  max_conf, mean_conf

Usage:
  python -m magazines.scripts.infer_faces \\
      --output_dir {data_root}/face_detections/scrfd

  # Resume interrupted run
  python -m magazines.scripts.infer_faces \\
      --output_dir {data_root}/face_detections/scrfd --skip_existing
"""
from __future__ import annotations

import argparse
import json
import logging
import queue
import random
import threading
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

from magazineads.paths import paths as _paths

DEFAULT_PREDICTIONS_CSV = _paths.predictions_dir / "predictions_all.csv"
DEFAULT_N_SAMPLE = 0
DEFAULT_SEED = 42
FULL_PAGE_AD_LABEL = 1
MAGAZINES = ["esquire", "forbes", "life", "time", "variety"]
MAX_DIM = 1500
FLUSH_INTERVAL = 5000  # write partial CSV every N images
PREFETCH_WORKERS = 8   # threads for image loading
PREFETCH_QUEUE = 64    # max images buffered ahead


def parse_year(image_path: str) -> int | None:
    try:
        return int(Path(image_path).stem.split("-")[0])
    except (ValueError, IndexError):
        return None


def parse_magazine(image_path: str) -> str:
    return Path(image_path).parent.name


def collect_from_predictions(predictions_csv: Path, magazine: str | None) -> list[str]:
    log.info("Loading predictions from %s ...", predictions_csv)
    df = pd.read_csv(predictions_csv, usecols=["filepath", "pred_label"])
    full_page = df[df["pred_label"] == FULL_PAGE_AD_LABEL]["filepath"]
    if magazine is not None:
        full_page = full_page[full_page.apply(lambda p: Path(p).parent.name == magazine)]
        log.info("  %d full-page ads for magazine '%s'", len(full_page), magazine)
    else:
        log.info("  %d / %d images are full-page ads", len(full_page), len(df))
    return full_page.tolist()


def collect_from_dir(image_dir: Path) -> list[str]:
    exts = ("*.jpg", "*.jpeg", "*.png", "*.JPG", "*.JPEG", "*.PNG")
    found: list[Path] = []
    for ext in exts:
        found.extend(image_dir.glob(ext))
    return [str(p) for p in sorted(found)]


def _load_and_resize(img_path: str) -> tuple[str, np.ndarray | None]:
    """Read and resize one image (runs in worker thread)."""
    img = cv2.imread(img_path)
    if img is None:
        return img_path, None
    h, w = img.shape[:2]
    if max(h, w) > MAX_DIM:
        scale = MAX_DIM / max(h, w)
        img = cv2.resize(img, (int(w * scale), int(h * scale)))
    return img_path, img


def _prefetch_images(
    paths: list[str],
    out_q: queue.Queue,
    n_workers: int,
) -> None:
    """Fill out_q with (img_path, img) tuples using a thread pool."""
    from concurrent.futures import ThreadPoolExecutor

    with ThreadPoolExecutor(max_workers=n_workers) as pool:
        for result in pool.map(_load_and_resize, paths):
            out_q.put(result)  # blocks if queue is full
    out_q.put(None)  # sentinel


def run_inference(
    image_paths: list[str],
    output_dir: Path,
    det_size: int,
    skip_existing: bool,
) -> list[dict]:
    from insightface.app import FaceAnalysis

    log.info("Loading SCRFD (InsightFace buffalo_l)...")
    app = FaceAnalysis(allowed_modules=["detection"])
    app.prepare(ctx_id=0, det_size=(det_size, det_size))
    log.info("Model loaded. Running on %d images...", len(image_paths))

    rows: list[dict] = []
    csv_path = output_dir / "face_detections.csv"

    # Load existing results for skip_existing mode
    existing_stems: set[str] = set()
    if skip_existing and csv_path.exists():
        existing_df = pd.read_csv(csv_path, usecols=["image_path"])
        existing_stems = set(existing_df["image_path"])
        log.info("  skip_existing: %d images already done", len(existing_stems))
        rows = pd.read_csv(csv_path).to_dict("records")

    # Filter to only images we need to process
    todo = [p for p in image_paths if p not in existing_stems]
    log.info("  %d to process (%d skipped)", len(todo), len(existing_stems))

    if not todo:
        return rows

    # Start prefetch threads
    img_q: queue.Queue = queue.Queue(maxsize=PREFETCH_QUEUE)
    loader = threading.Thread(
        target=_prefetch_images, args=(todo, img_q, PREFETCH_WORKERS), daemon=True,
    )
    loader.start()

    pbar = tqdm(total=len(todo), desc="SCRFD inference", unit="img", dynamic_ncols=True)
    processed = 0

    while True:
        item = img_q.get()
        if item is None:
            break
        img_path, img = item

        if img is None:
            log.warning("Could not read %s", img_path)
            rows.append(_empty_row(img_path))
        else:
            try:
                h, w = img.shape[:2]
                faces = app.get(img)

                n_faces = len(faces) if faces else 0
                confs = [float(f.det_score) for f in faces] if faces else []
                boxes = []
                if faces:
                    for f in faces:
                        x1, y1, x2, y2 = f.bbox
                        boxes.append([
                            round(float(x1 / w), 6),
                            round(float(y1 / h), 6),
                            round(float(x2 / w), 6),
                            round(float(y2 / h), 6),
                        ])

                rows.append({
                    "image_path": img_path,
                    "magazine": parse_magazine(img_path),
                    "year": parse_year(img_path),
                    "n_faces": n_faces,
                    "face_confs": ";".join(f"{c:.4f}" for c in confs) if confs else "",
                    "face_boxes": json.dumps(boxes),
                    "max_conf": max(confs) if confs else None,
                    "mean_conf": float(np.mean(confs)) if confs else None,
                })
            except Exception as e:
                log.error("Error on %s: %s", img_path, e)
                rows.append(_empty_row(img_path))

        processed += 1
        pbar.update(1)
        if processed % FLUSH_INTERVAL == 0:
            _flush_csv(rows, csv_path)

    pbar.close()
    loader.join()
    return rows


def _empty_row(img_path: str) -> dict:
    return {
        "image_path": img_path,
        "magazine": parse_magazine(img_path),
        "year": parse_year(img_path),
        "n_faces": 0,
        "face_confs": "",
        "face_boxes": "[]",
        "max_conf": None,
        "mean_conf": None,
    }


def _flush_csv(rows: list[dict], csv_path: Path) -> None:
    df = pd.DataFrame(rows)
    df.to_csv(csv_path, index=False)
    log.info("  flushed %d rows to %s", len(df), csv_path)


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    src = ap.add_mutually_exclusive_group()
    src.add_argument("--image_paths", nargs="+", type=str)
    src.add_argument("--image_dir", type=Path)
    ap.add_argument("--predictions_csv", type=Path, default=DEFAULT_PREDICTIONS_CSV)
    ap.add_argument("--magazine", choices=MAGAZINES)
    ap.add_argument("--n_sample", type=int, default=DEFAULT_N_SAMPLE)
    ap.add_argument("--seed", type=int, default=DEFAULT_SEED)
    ap.add_argument("--det_size", type=int, default=640,
                    help="SCRFD input size (default: 640)")
    ap.add_argument("--output_dir", type=Path, required=True)
    ap.add_argument("--skip_existing", action="store_true",
                    help="Resume: skip images already in face_detections.csv")
    args = ap.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Collect candidates
    if args.image_paths:
        candidates = args.image_paths
    elif args.image_dir:
        candidates = collect_from_dir(args.image_dir)
    else:
        candidates = collect_from_predictions(args.predictions_csv, args.magazine)

    if args.n_sample > 0 and args.n_sample < len(candidates):
        random.seed(args.seed)
        candidates = random.sample(candidates, args.n_sample)
        log.info("Sampled %d images (seed=%d)", len(candidates), args.seed)

    log.info("Total candidates: %d", len(candidates))

    rows = run_inference(candidates, args.output_dir, args.det_size, args.skip_existing)

    # Final save
    df = pd.DataFrame(rows)
    csv_path = args.output_dir / "face_detections.csv"
    df.to_csv(csv_path, index=False)

    # Summary
    n_with_face = (df["n_faces"] > 0).sum()
    total_faces = df["n_faces"].sum()
    log.info("Done. %d images processed.", len(df))
    log.info("  %d (%.1f%%) have >= 1 face", n_with_face, 100 * n_with_face / len(df))
    log.info("  %d total face detections", total_faces)
    log.info("Saved %s", csv_path)


if __name__ == "__main__":
    main()
