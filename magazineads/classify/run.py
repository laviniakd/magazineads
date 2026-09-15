"""Run page classification inference on magazine page images.

Loads a fine-tuned CLIP page classifier and classifies pages as:
  0 = not ad, 1 = full-page ad, 2 = mixed (ad + editorial)

Usage:
    python -m magazineads.classify.run \\
        --model_path models/page_classifier \\
        --data_path /path/to/images \\
        --predictions_dir /path/to/output
"""

from __future__ import annotations

import argparse
import os
import time

import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageFile
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from tqdm import tqdm
from safetensors.torch import load_file

from magazineads.classify.model import CLIPImageClassifier
from magazineads.paths import paths as _paths

ImageFile.LOAD_TRUNCATED_IMAGES = True

RESIZE_HEIGHT = 336
RESIZE_WIDTH = 336


class _PageImageDataset(Dataset):
    """Simple dataset that loads and transforms page images."""

    def __init__(self, image_paths: list[str], transform):
        self.image_paths = image_paths
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        try:
            img = Image.open(self.image_paths[idx]).convert("RGB")
        except Exception:
            img = Image.new("RGB", (RESIZE_WIDTH, RESIZE_HEIGHT), color="black")
        pixel_values = self.transform(img)
        return {"pixel_values": pixel_values}


def get_image_paths(directory: str, magazines: list[str] | None = None) -> list[str]:
    """Recursively find all JPEG images in directory."""
    image_paths = []
    for root, _, files in os.walk(directory):
        for f in files:
            if f.lower().endswith(".jpg"):
                if magazines is None or any(m in root for m in magazines):
                    image_paths.append(os.path.join(root, f))
    return sorted(image_paths)


def load_model(model_path: str, device: str = "cuda") -> CLIPImageClassifier:
    """Load a fine-tuned page classifier from a checkpoint directory.

    Expects either:
      - model_path/model.safetensors (safetensors format)
      - model_path as a HuggingFace-compatible directory
    """
    safetensors_path = os.path.join(model_path, "model.safetensors")

    model = CLIPImageClassifier(num_labels=3, load_weights=False)

    if os.path.exists(safetensors_path):
        state_dict = load_file(safetensors_path)
        # The HF Trainer checkpoint wraps CLIPImageClassifier inside a
        # VisionOnlyClassifier container, adding a "vision_model." prefix.
        # Strip it so keys match CLIPImageClassifier directly.
        prefix = "vision_model."
        if any(k.startswith(prefix) for k in state_dict):
            state_dict = {
                k[len(prefix):] if k.startswith(prefix) else k: v
                for k, v in state_dict.items()
            }
        model.load_state_dict(state_dict)
    else:
        from transformers import AutoModelForImageClassification
        model = AutoModelForImageClassification.from_pretrained(model_path)

    model = model.to(device)
    model.eval()
    return model


def classify_pages(
    image_paths: list[str],
    model: CLIPImageClassifier,
    batch_size: int = 64,
    num_workers: int = 8,
    device: str = "cuda",
) -> tuple[np.ndarray, np.ndarray]:
    """Run page classification inference.

    Returns (pred_labels, logits) arrays.
    """
    transform = transforms.Compose([
        transforms.Resize((RESIZE_HEIGHT, RESIZE_WIDTH)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),
    ])

    dataset = _PageImageDataset(image_paths, transform)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        prefetch_factor=2,
        collate_fn=lambda batch: torch.stack([b["pixel_values"] for b in batch]),
    )

    all_logits = []
    with torch.no_grad():
        for batch_pixels in tqdm(dataloader, desc="Classifying pages"):
            batch_pixels = batch_pixels.to(device, non_blocking=True)
            outputs = model(batch_pixels)
            all_logits.append(outputs.logits.cpu().numpy())

    if all_logits:
        logits = np.vstack(all_logits)
        pred_labels = np.argmax(logits, axis=1)
    else:
        logits = np.zeros((0, 3))
        pred_labels = np.zeros(0, dtype=int)

    return pred_labels, logits


def save_predictions(
    image_paths: list[str],
    pred_labels: np.ndarray,
    logits: np.ndarray | None,
    output_dir: str,
):
    """Save predictions to predictions_all.csv."""
    os.makedirs(output_dir, exist_ok=True)

    rows = []
    if logits is not None:
        exp = np.exp(logits - np.max(logits, axis=1, keepdims=True))
        probs = exp / np.sum(exp, axis=1, keepdims=True)
    else:
        probs = None

    for i, path in enumerate(image_paths):
        lbl = int(pred_labels[i])
        ad_prob = float(probs[i, 1]) if probs is not None and probs.shape[1] > 1 else 0.0
        rows.append({"filepath": path, "pred_label": lbl, "ad_prob": ad_prob})

    df = pd.DataFrame(rows)

    # Timestamped copy
    ts = time.strftime("%Y%m%d_%H%M%S")
    df.to_csv(os.path.join(output_dir, f"predictions_{ts}.csv"), index=False)

    # Append to cumulative file
    all_path = os.path.join(output_dir, "predictions_all.csv")
    header = not os.path.exists(all_path)
    df.to_csv(all_path, mode="a", header=header, index=False)

    print(f"Saved {len(df)} predictions to {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Run page classification on magazine images")
    parser.add_argument("--model_path", type=str, required=True,
                        help="Path to fine-tuned model checkpoint directory")
    parser.add_argument("--data_path", type=str, default=str(_paths.images_dir),
                        help="Path to images directory")
    parser.add_argument("--magazines", nargs="+",
                        default=["time", "variety", "forbes", "life", "esquire"])
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--predictions_dir", type=str,
                        default=str(_paths.predictions_dir))
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Loading model from {args.model_path}")
    model = load_model(args.model_path, device=device)

    print(f"Scanning images in {args.data_path}")
    image_paths = get_image_paths(args.data_path, magazines=args.magazines)
    print(f"Found {len(image_paths)} images")

    pred_labels, logits = classify_pages(
        image_paths, model,
        batch_size=args.batch_size,
        device=device,
    )

    save_predictions(image_paths, pred_labels, logits, args.predictions_dir)


if __name__ == "__main__":
    main()
