"""Batch surya OCR processing via the Python API.

Replaces surya_ocr.sh by loading models once and processing images in batches,
avoiding the overhead of spawning 6 parallel CLI processes each loading multi-GB models.

Usage:
    python -m magazines.ocr.precompute.surya_batch \
        --image_dir /data/laviniad/magazines/images/esquire \
        --output_dir /data/laviniad/magazines/surya_output/esquire \
        --batch_size 32

    # Process multiple magazines:
    python -m magazines.ocr.precompute.surya_batch \
        --image_dir /data/laviniad/magazines/images \
        --output_dir /data/laviniad/magazines/surya_output \
        --magazines esquire forbes life time variety \
        --batch_size 32
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import List

from PIL import Image
from tqdm import tqdm


def collect_images(image_dir: str, extensions: tuple = (".jpg", ".jpeg", ".png", ".tif", ".tiff")) -> List[str]:
    """Collect all image paths from a directory (non-recursive)."""
    paths = []
    for f in sorted(os.listdir(image_dir)):
        if f.lower().endswith(extensions):
            paths.append(os.path.join(image_dir, f))
    return paths


def filter_unprocessed(image_paths: List[str], output_dir: str) -> List[str]:
    """Filter out images that already have surya output.

    For Time magazine images named YYYY-MM-DD-x-x-N, also accepts an existing
    single-x directory (YYYY-MM-DD-x-N) as sufficient coverage, since those were
    produced from identically-content images under the old naming convention.
    """
    unprocessed = []
    for path in image_paths:
        stem = Path(path).stem
        results_path = os.path.join(output_dir, stem, "results.json")
        if os.path.isfile(results_path):
            continue
        # Check single-x fallback for double-x Time stems
        if "-x-x-" in stem:
            fallback_stem = stem.replace("-x-x-", "-x-", 1)
            fallback_path = os.path.join(output_dir, fallback_stem, "results.json")
            if os.path.isfile(fallback_path):
                continue
        unprocessed.append(path)
    return unprocessed


def resize_for_ocr(img: Image.Image, max_side: int) -> Image.Image:
    """Downscale image so its longest side is at most max_side pixels."""
    w, h = img.size
    longest = max(w, h)
    if longest > max_side:
        scale = max_side / longest
        img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
    return img


def process_batch(
    image_paths: List[str],
    output_dir: str,
    recognition_predictor,
    detection_predictor,
    detection_batch_size: int = None,
    recognition_batch_size: int = None,
    max_side: int = None,
    highres_max_side: int = None,
):
    """Process a batch of images through surya and write results.

    If highres_max_side is set, images are loaded at two scales:
      - max_side (low-res) for bbox detection
      - highres_max_side (high-res) for text recognition crops
    This mirrors surya's CLI behavior and is significantly faster.
    """
    images = []
    highres_images = [] if highres_max_side else None
    valid_paths = []

    for path in image_paths:
        try:
            img = Image.open(path).convert("RGB")
            if highres_max_side:
                highres_images.append(resize_for_ocr(img, highres_max_side))
                images.append(resize_for_ocr(img, max_side) if max_side else img)
            else:
                if max_side:
                    img = resize_for_ocr(img, max_side)
                images.append(img)
            valid_paths.append(path)
        except Exception as e:
            print(f"Failed to load {path}: {e}")

    if not images:
        return

    # Run surya on the batch
    predictions = recognition_predictor(
        images,
        det_predictor=detection_predictor,
        highres_images=highres_images,
        math_mode=False,
        detection_batch_size=detection_batch_size,
        recognition_batch_size=recognition_batch_size,
    )

    # Write results in surya's expected format
    for path, pred in zip(valid_paths, predictions):
        stem = Path(path).stem
        result_dir = os.path.join(output_dir, stem)
        os.makedirs(result_dir, exist_ok=True)

        # Convert prediction to serializable format
        page_data = {
            "text_lines": [],
            "image_bbox": getattr(pred, "image_bbox", []),
            "page": 0,
        }

        for line in pred.text_lines:
            line_data = {
                "text": line.text,
                "confidence": line.confidence,
                "polygon": line.polygon,
                "bbox": line.bbox,
            }
            if hasattr(line, "chars") and line.chars:
                line_data["chars"] = [
                    {"text": c.text, "confidence": c.confidence, "bbox": c.bbox}
                    for c in line.chars
                ]
            page_data["text_lines"].append(line_data)

        result = {stem: [page_data]}
        with open(os.path.join(result_dir, "results.json"), "w") as f:
            json.dump(result, f, indent=2)


def main():
    parser = argparse.ArgumentParser(description="Batch surya OCR processing via Python API.")
    parser.add_argument("--image_dir", type=str, required=True,
                        help="Directory containing images (or parent dir if using --magazines).")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Output directory for surya results.")
    parser.add_argument("--magazines", nargs="*", default=None,
                        help="List of magazine subdirectories to process. If omitted, processes image_dir directly.")
    parser.add_argument("--batch_size", type=int, default=32,
                        help="Number of images loaded per outer batch.")
    parser.add_argument("--detection_batch_size", type=int, default=None,
                        help="Surya internal detection batch size. Defaults to surya's auto setting.")
    parser.add_argument("--recognition_batch_size", type=int, default=None,
                        help="Surya internal recognition batch size. Defaults to surya's auto setting.")
    parser.add_argument("--max_side", type=int, default=1000,
                        help="Detection scale: longest side in pixels for bbox detection (default: 1000).")
    parser.add_argument("--highres_max_side", type=int, default=2000,
                        help="Recognition scale: longest side in pixels for text recognition crops (default: 2000). "
                             "Set to 0 to disable dual-scale mode.")
    parser.add_argument("--skip_existing", action="store_true", default=True,
                        help="Skip images with existing output (default: True).")
    parser.add_argument("--no_skip_existing", action="store_false", dest="skip_existing",
                        help="Reprocess all images even if output exists.")
    args = parser.parse_args()

    # Import surya here so the script fails fast with a clear message
    try:
        from surya.recognition import RecognitionPredictor, FoundationPredictor
        from surya.detection import DetectionPredictor
    except ImportError:
        print("ERROR: surya is not installed. Install with: pip install surya-ocr")
        sys.exit(1)

    print("Loading surya models (this may take a moment)...")
    foundation_predictor = FoundationPredictor()
    recognition_predictor = RecognitionPredictor(foundation_predictor)
    detection_predictor = DetectionPredictor()
    print("Models loaded.")

    # Build list of (image_dir, output_dir) pairs
    tasks = []
    if args.magazines:
        for mag in args.magazines:
            mag_image_dir = os.path.join(args.image_dir, mag)
            mag_output_dir = os.path.join(args.output_dir, mag)
            if not os.path.isdir(mag_image_dir):
                print(f"Warning: {mag_image_dir} not found, skipping {mag}")
                continue
            tasks.append((mag, mag_image_dir, mag_output_dir))
    else:
        mag_name = os.path.basename(args.image_dir.rstrip("/"))
        tasks.append((mag_name, args.image_dir, args.output_dir))

    for mag_name, img_dir, out_dir in tasks:
        print(f"\nProcessing {mag_name}...")
        os.makedirs(out_dir, exist_ok=True)

        all_images = collect_images(img_dir)
        if args.skip_existing:
            images = filter_unprocessed(all_images, out_dir)
            print(f"  {len(all_images)} total images, {len(all_images) - len(images)} already processed, {len(images)} remaining.")
        else:
            images = all_images
            print(f"  {len(images)} images to process.")

        if not images:
            print(f"  Nothing to do for {mag_name}.")
            continue

        # Process in batches
        for i in tqdm(range(0, len(images), args.batch_size), desc=f"  {mag_name}"):
            batch = images[i : i + args.batch_size]
            process_batch(
                batch, out_dir, recognition_predictor, detection_predictor,
                detection_batch_size=args.detection_batch_size,
                recognition_batch_size=args.recognition_batch_size,
                max_side=args.max_side,
                highres_max_side=args.highres_max_side or None,
            )

        print(f"  Finished {mag_name}.")

    print("\nDone.")


if __name__ == "__main__":
    main()
