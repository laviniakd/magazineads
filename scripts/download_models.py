"""Download trained model weights from HuggingFace.

Downloads all fine-tuned models needed for the annotation pipeline (Step 3)
into the local models directory.

Usage:
    python scripts/download_models.py                    # all models
    python scripts/download_models.py --model page       # just page classifier
    python scripts/download_models.py --output_dir ./models  # custom location

Models are downloaded from: https://huggingface.co/laviniad/magazineads
"""

from __future__ import annotations

import argparse
from pathlib import Path

from huggingface_hub import hf_hub_download, snapshot_download

REPO_ID = "laviniad/magazineads"

MODELS = {
    "page": {
        "description": "Page classifier (CLIP ViT-L/14@336 + linear head, 3-class)",
        "files": ["page_classifier/model.safetensors"],
        "subdir": "page_classifier",
    },
    "yolo": {
        "description": "Ad localization (YOLOv8l, fine-tuned on magazine ads)",
        "files": ["yolo_ad_detection/best.pt"],
        "subdir": "yolo_ad_detection",
    },
    "spread": {
        "description": "Spread classifier (dual CLIP encoder + diff head)",
        "files": ["spread_classifier/spread_classifier.pt"],
        "subdir": "spread_classifier",
    },
    "product": {
        "description": "Product category (CLIP+MALLET+LogReg, 8 categories)",
        "files": [
            "product_category/model.joblib",
            "product_category/scaler.joblib",
            "product_category/label_map.json",
            "product_category/config.json",
        ],
        "subdir": "product_category",
    },
}


def download_model(name: str, output_dir: Path) -> None:
    """Download a single model's files from HuggingFace."""
    info = MODELS[name]
    print(f"\n--- {info['description']} ---")
    for filename in info["files"]:
        local_path = hf_hub_download(
            repo_id=REPO_ID,
            filename=filename,
            local_dir=str(output_dir),
        )
        print(f"  {filename} -> {local_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Download magazineads model weights from HuggingFace",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Available models: " + ", ".join(MODELS.keys()),
    )
    parser.add_argument(
        "--model", choices=list(MODELS.keys()),
        help="Download a specific model (default: all)",
    )
    parser.add_argument(
        "--output_dir", type=Path, default=Path("models"),
        help="Directory to save models (default: ./models)",
    )
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    if args.model:
        download_model(args.model, args.output_dir)
    else:
        print(f"Downloading all models from {REPO_ID}")
        for name in MODELS:
            download_model(name, args.output_dir)

    print(f"\nDone. Models saved to {args.output_dir}/")


if __name__ == "__main__":
    main()
