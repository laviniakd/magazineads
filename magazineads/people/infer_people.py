"""Person detection inference using off-the-shelf YOLO models on magazine ads.

Designed specifically for detecting people in magazine ads — handles both
photographic and illustration representations. Filters COCO person class
(class 0) only for efficiency and per-detection confidence tracking.

The confidence threshold defaults to 0.20, lower than the general zero-shot
default, to maximise recall for illustrated people who typically produce
weaker signals than photographic subjects.

Input sources (priority order):
  1. --image_paths PATH [PATH ...]   explicit list
  2. --image_dir PATH                all JPEGs in a directory
  3. (default) predictions CSV       full-page ads from predictions_all.csv

Outputs (in --output_dir):
  people_detections.csv    per-image row with bounding boxes + confidence scores
  labels/                  YOLO .txt files (person class only, for downstream use)
  viz/                     annotated visualisations (only with --save_viz)

CSV columns:
  image_path, magazine, year, n_persons,
  person_confs  (semicolon-separated floats),
  person_boxes  (JSON list of [x1,y1,x2,y2] normalised to [0,1]),
  max_conf, mean_conf

Usage:
  # Full run on all full-page ads with YOLOv8l
  python -m magazines.scripts.infer_people \\
      --model yolov8l.pt \\
      --output_dir {data_root}/people_detections/yolov8l

  # Subset run with YOLO26m for comparison
  python -m magazines.scripts.infer_people \\
      --model yolo26m.pt --n_sample 1000 --seed 42 \\
      --output_dir {data_root}/people_detections/yolo26m_subset

  # Incremental resume
  python -m magazines.scripts.infer_people \\
      --model yolov8l.pt --skip_existing \\
      --output_dir {data_root}/people_detections/yolov8l
"""

from __future__ import annotations

import argparse
import json
import logging
import random
from pathlib import Path

import pandas as pd
import PIL.Image

# Magazine scans can be very high resolution; disable the decompression-bomb guard.
PIL.Image.MAX_IMAGE_PIXELS = None

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

from magazineads.paths import paths as _paths

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

DEFAULT_PREDICTIONS_CSV = _paths.predictions_dir / "predictions_all.csv"
DEFAULT_CONF = 0.20   # lower than general zero-shot (0.25) to maximise recall for illustrations
DEFAULT_IOU = 0.45
DEFAULT_BATCH = 16
DEFAULT_N_SAMPLE = 0  # 0 = use all candidates
DEFAULT_SEED = 42

PERSON_CLASS = 0       # COCO class 0 = person (valid for YOLOv8 and YOLO26 COCO-pretrained weights)
FULL_PAGE_AD_LABEL = 1  # pred_label value in predictions_all.csv

MAGAZINES = ["esquire", "forbes", "life", "time", "variety"]


# ---------------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------------


def parse_year(image_path: str) -> int | None:
    """Extract year from filename stem, e.g. '1945-01-29-18-5-088' -> 1945."""
    try:
        return int(Path(image_path).stem.split("-")[0])
    except (ValueError, IndexError):
        return None


def parse_magazine(image_path: str) -> str:
    """Derive magazine name from the immediate parent directory."""
    return Path(image_path).parent.name


# ---------------------------------------------------------------------------
# Candidate collection
# ---------------------------------------------------------------------------


def collect_from_predictions(
    predictions_csv: Path, magazine: str | None
) -> list[str]:
    """Return image paths for all pred_label==1 rows in the predictions CSV."""
    log.info("Loading predictions from %s …", predictions_csv)
    df = pd.read_csv(predictions_csv, usecols=["filepath", "pred_label"])
    full_page = df[df["pred_label"] == FULL_PAGE_AD_LABEL]["filepath"]
    if magazine is not None:
        full_page = full_page[
            full_page.apply(lambda p: Path(p).parent.name == magazine)
        ]
        log.info(
            "  %d full-page ads for magazine '%s'", len(full_page), magazine
        )
    else:
        log.info(
            "  %d / %d images are full-page ads",
            len(full_page), len(df),
        )
    return full_page.tolist()


def collect_from_dir(image_dir: Path) -> list[str]:
    exts = ("*.jpg", "*.jpeg", "*.png", "*.JPG", "*.JPEG", "*.PNG")
    found: list[Path] = []
    for ext in exts:
        found.extend(image_dir.glob(ext))
    return [str(p) for p in sorted(found)]


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------


def run_inference(
    image_paths: list[str],
    model_name: str,
    output_dir: Path,
    conf: float,
    iou: float,
    batch_size: int,
    save_viz: bool,
) -> list[dict]:
    """Run batched YOLO inference, return per-image summary rows.

    Only the person class (COCO class 0) is requested from the model;
    all other COCO classes are suppressed via the ``classes`` argument.
    """
    from ultralytics import YOLO

    labels_dir = output_dir / "labels"
    labels_dir.mkdir(parents=True, exist_ok=True)
    if save_viz:
        (output_dir / "viz").mkdir(parents=True, exist_ok=True)

    log.info("Loading model: %s", model_name)
    model = YOLO(model_name, task="detect")
    class_names = model.names  # {id: name}

    if class_names.get(PERSON_CLASS) != "person":
        raise ValueError(
            f"Expected class {PERSON_CLASS}='person', got '{class_names.get(PERSON_CLASS)}'. "
            "Verify that --model uses COCO-pretrained weights."
        )
    log.info(
        "Model loaded. %d total classes. Filtering to class %d (person) only.",
        len(class_names), PERSON_CLASS,
    )

    rows: list[dict] = []

    for batch_start in range(0, len(image_paths), batch_size):
        batch = image_paths[batch_start: batch_start + batch_size]
        end_idx = min(batch_start + batch_size, len(image_paths))
        log.info(
            "Inferring images %d–%d / %d …",
            batch_start + 1, end_idx, len(image_paths),
        )

        try:
            results = model.predict(
                batch,
                conf=conf,
                iou=iou,
                classes=[PERSON_CLASS],
                verbose=False,
            )
        except Exception as e:
            log.error("Batch failed starting at image %d: %s", batch_start, e)
            continue

        for result, img_path in zip(results, batch):
            stem = Path(img_path).stem
            label_path = labels_dir / f"{stem}.txt"

            boxes_norm: list[list[float]] = []
            confs_list: list[float] = []

            if result.boxes is not None and len(result.boxes) > 0:
                xyxyn = result.boxes.xyxyn.cpu().numpy()   # shape (N, 4)
                confs_arr = result.boxes.conf.cpu().numpy()  # shape (N,)

                confs_list = [float(c) for c in confs_arr]
                boxes_norm = [b.tolist() for b in xyxyn]

                with open(label_path, "w") as f:
                    for (x1, y1, x2, y2) in xyxyn:
                        cx = (x1 + x2) / 2
                        cy = (y1 + y2) / 2
                        w = x2 - x1
                        h = y2 - y1
                        # Clamp to [0,1] to handle minor float precision issues
                        cx, cy, w, h = (max(0.0, min(1.0, v)) for v in (cx, cy, w, h))
                        f.write(f"0 {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n")
            else:
                # YOLO convention: empty file means no detections
                label_path.touch()

            if save_viz:
                result.save(filename=str(output_dir / "viz" / f"{stem}.jpg"))

            mag = parse_magazine(img_path)
            year = parse_year(img_path)

            rows.append({
                "image_path": img_path,
                "magazine": mag,
                "year": year,
                "n_persons": len(confs_list),
                "person_confs": ";".join(f"{c:.4f}" for c in confs_list),
                "person_boxes": json.dumps(boxes_norm),
                "max_conf": max(confs_list) if confs_list else float("nan"),
                "mean_conf": (sum(confs_list) / len(confs_list)) if confs_list else float("nan"),
            })

    return rows


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Input source (mutually exclusive)
    src = p.add_mutually_exclusive_group()
    src.add_argument(
        "--image_paths", nargs="+", metavar="PATH",
        help="Explicit image paths (skips all sampling logic).",
    )
    src.add_argument(
        "--image_dir", type=Path, metavar="DIR",
        help="Process all JPEGs in this directory.",
    )
    p.add_argument(
        "--predictions_csv", type=Path, default=DEFAULT_PREDICTIONS_CSV,
        help="predictions_all.csv from classify_and_embed_ads (default source).",
    )
    p.add_argument(
        "--magazine", choices=MAGAZINES, default=None,
        help="Restrict predictions-CSV source to one magazine.",
    )

    # Sampling
    p.add_argument(
        "--n_sample", type=int, default=DEFAULT_N_SAMPLE,
        help="Number of images to sample randomly (0 = use all).",
    )
    p.add_argument(
        "--seed", type=int, default=DEFAULT_SEED,
        help="Random seed for reproducible sampling.",
    )

    # Model
    p.add_argument(
        "--model", default="yolov8l.pt",
        help="YOLO model name or path. Must be COCO-pretrained (class 0 = person).",
    )
    p.add_argument("--conf", type=float, default=DEFAULT_CONF)
    p.add_argument("--iou", type=float, default=DEFAULT_IOU)
    p.add_argument("--batch_size", type=int, default=DEFAULT_BATCH)

    # Output
    p.add_argument("--output_dir", type=Path, required=True)
    p.add_argument(
        "--save_viz", action="store_true",
        help="Save annotated visualisation images to output_dir/viz/.",
    )
    p.add_argument(
        "--skip_existing", action="store_true",
        help="Skip images whose label .txt already exists (for incremental runs).",
    )

    return p.parse_args()


def main() -> None:
    args = _parse_args()
    random.seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # --- Collect candidates ---
    if args.image_paths:
        candidates = list(args.image_paths)
        log.info("Using %d explicitly provided image paths.", len(candidates))
    elif args.image_dir:
        candidates = collect_from_dir(args.image_dir)
        log.info("Found %d images in %s.", len(candidates), args.image_dir)
    else:
        candidates = collect_from_predictions(args.predictions_csv, args.magazine)

    if not candidates:
        log.error("No candidate images found. Check --predictions_csv / --image_dir / --image_paths.")
        raise SystemExit(1)

    # --- Skip already-processed images ---
    if args.skip_existing:
        labels_dir = args.output_dir / "labels"
        before = len(candidates)
        candidates = [
            p for p in candidates
            if not (labels_dir / (Path(p).stem + ".txt")).exists()
        ]
        log.info(
            "skip_existing: %d already done, %d remaining.",
            before - len(candidates), len(candidates),
        )

    if not candidates:
        log.info("All images already processed. Nothing to do.")
        return

    # --- Sample ---
    if args.n_sample and args.n_sample < len(candidates):
        n_before = len(candidates)
        candidates = random.sample(candidates, args.n_sample)
        log.info("Sampled %d / %d images (seed=%d).", len(candidates), n_before, args.seed)
    else:
        log.info("Processing all %d images.", len(candidates))

    # --- Inference ---
    rows = run_inference(
        image_paths=candidates,
        model_name=args.model,
        output_dir=args.output_dir,
        conf=args.conf,
        iou=args.iou,
        batch_size=args.batch_size,
        save_viz=args.save_viz,
    )

    # --- Write CSV ---
    if rows:
        out_csv = args.output_dir / "people_detections.csv"
        pd.DataFrame(rows).to_csv(out_csv, index=False)
        log.info("Saved %s", out_csv)

    # --- Summary ---
    n_with = sum(1 for r in rows if r["n_persons"] > 0)
    total_persons = sum(r["n_persons"] for r in rows)
    if rows:
        log.info(
            "Done. %d images processed: %d (%.1f%%) with ≥1 person, %d total detections.",
            len(rows), n_with, 100 * n_with / len(rows), total_persons,
        )
    log.info("Output directory: %s", args.output_dir)


if __name__ == "__main__":
    main()
