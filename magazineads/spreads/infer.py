"""Infer two-page spread status for adjacent full-page ad pairs.

Reads a page classification predictions CSV, identifies adjacent pairs where both
pages are classified as full-page ads, and scores each pair with the trained
SpreadClassifier.

Output CSV columns: left_page, right_page, spread_prob, pred_spread

Usage:
    python -m magazineads.spreads.infer \\
        --checkpoint models/spread_classifier.pt \\
        --predictions_csv data/predictions_all.csv \\
        --output_csv data/spread_predictions.csv \\
        --head_type diff --threshold 0.4
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from magazineads.spreads.model import (
    HEAD_TYPES,
    SpreadClassifier,
    build_transform,
    find_adjacent_page,
)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class _AdjacentPairDataset(Dataset):
    """Unlabeled adjacent-pair dataset for spread inference."""

    def __init__(self, pairs, image_size=336):
        self.pairs = pairs
        self.transform = build_transform(image_size)

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        pair = self.pairs[idx]
        left_img = self.transform(Image.open(pair["left_page"]).convert("RGB"))
        right_img = self.transform(Image.open(pair["right_page"]).convert("RGB"))
        return {"left_pixels": left_img, "right_pixels": right_img}


class _H5PairDataset(Dataset):
    """Inference dataset backed by precomputed embeddings HDF5."""

    def __init__(self, pairs, h5_path, image_size=336):
        import h5py
        self.pairs = pairs
        self.transform = build_transform(image_size)

        f = h5py.File(h5_path, "r")
        self._path_to_idx = {
            p.decode(): i for i, p in enumerate(f["paths"][:])
        }
        self._emb = torch.from_numpy(f["emb"][:].astype(np.float32))
        self._hist = torch.from_numpy(f["hist"][:])
        self._text_emb = (
            torch.from_numpy(f["text_emb"][:].astype(np.float32))
            if "text_emb" in f else None
        )
        f.close()

        n_in = sum(1 for p in pairs
                   if p["left_page"] in self._path_to_idx
                   and p["right_page"] in self._path_to_idx)
        print(f"H5 coverage: {n_in}/{len(pairs)} pairs")

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        pair = self.pairs[idx]
        lp, rp = pair["left_page"], pair["right_page"]
        li = self._path_to_idx.get(lp)
        ri = self._path_to_idx.get(rp)

        if li is None or ri is None:
            left_img = self.transform(Image.open(lp).convert("RGB"))
            right_img = self.transform(Image.open(rp).convert("RGB"))
            return {"left_pixels": left_img, "right_pixels": right_img}

        item = {
            "left_emb": self._emb[li], "right_emb": self._emb[ri],
            "left_hist": self._hist[li], "right_hist": self._hist[ri],
        }
        if self._text_emb is not None:
            item["left_text_emb"] = self._text_emb[li]
            item["right_text_emb"] = self._text_emb[ri]
        return item


def _build_adjacent_pairs(predictions_csv, min_prob=0.0):
    """Find adjacent pairs where both pages are predicted as full-page ads."""
    df = pd.read_csv(predictions_csv)
    ad_df = df[df["pred_label"] == 1].copy()
    if "ad_prob" in df.columns and min_prob > 0:
        ad_df = ad_df[ad_df["ad_prob"] >= min_prob]

    ad_paths = {Path(p) for p in ad_df["filepath"].tolist()}
    print(f"Full-page ads (pred_label=1): {len(ad_paths)}")

    pairs = []
    seen = set()

    for ad_path in sorted(ad_paths):
        adj = find_adjacent_page(ad_path)
        if adj is None or adj not in ad_paths:
            continue

        key = tuple(sorted([str(ad_path), str(adj)]))
        if key in seen:
            continue
        seen.add(key)

        page_str = ad_path.stem.split("-")[-1]
        try:
            page_num = int(page_str.lstrip("0") or "0")
        except ValueError:
            continue

        if page_num % 2 == 0:
            left, right = ad_path, adj
        else:
            left, right = adj, ad_path

        pairs.append({"left_page": str(left), "right_page": str(right)})

    print(f"Adjacent ad pairs found: {len(pairs)}")
    return pairs


@torch.no_grad()
def _run_inference(model, pairs, batch_size, image_size, threshold,
                   embeddings_h5=None):
    if embeddings_h5 is not None:
        dataset = _H5PairDataset(pairs, embeddings_h5, image_size=image_size)
    else:
        dataset = _AdjacentPairDataset(pairs, image_size=image_size)

    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False,
        num_workers=4, pin_memory=True,
    )

    model.eval()
    all_probs = []
    for batch in tqdm(loader, desc="Inferring spreads"):
        kwargs = {}
        for key in ("left_pixels", "right_pixels", "left_emb", "right_emb",
                     "left_hist", "right_hist", "left_text_emb", "right_text_emb"):
            if key in batch:
                kwargs[key] = batch[key].to(DEVICE)
        out = model(**kwargs)
        all_probs.extend(torch.sigmoid(out["logits"]).cpu().tolist())

    return [
        {
            "left_page": pair["left_page"],
            "right_page": pair["right_page"],
            "spread_prob": round(prob, 4),
            "pred_spread": int(prob >= threshold),
        }
        for pair, prob in zip(pairs, all_probs)
    ]


def main():
    parser = argparse.ArgumentParser(description="Spread classification inference")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--predictions_csv", type=Path, required=True)
    parser.add_argument("--output_csv", type=Path, required=True)
    parser.add_argument("--head_type", default="diff", choices=list(HEAD_TYPES))
    parser.add_argument("--clip_model", default="openai/clip-vit-large-patch14-336")
    parser.add_argument("--image_size", type=int, default=336)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--min_prob", type=float, default=0.0)
    parser.add_argument("--threshold", type=float, default=0.4)
    parser.add_argument("--text_emb_dim", type=int, default=768)
    parser.add_argument("--mlp_hidden_dim", type=int, default=256)
    parser.add_argument("--embeddings_h5", type=Path, default=None)
    args = parser.parse_args()

    print(f"Device: {DEVICE}, Head: {args.head_type}, Threshold: {args.threshold}")

    model = SpreadClassifier(
        clip_model_name=args.clip_model,
        freeze_encoder=True,
        head_type=args.head_type,
        mlp_hidden_dim=args.mlp_hidden_dim,
        text_emb_dim=args.text_emb_dim,
    )
    state = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    model = model.to(DEVICE)
    print(f"Loaded checkpoint: {args.checkpoint}")

    pairs = _build_adjacent_pairs(args.predictions_csv, min_prob=args.min_prob)
    if not pairs:
        print("No adjacent ad pairs found.")
        return

    results = _run_inference(
        model, pairs, args.batch_size, args.image_size, args.threshold,
        embeddings_h5=args.embeddings_h5,
    )

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(results).to_csv(args.output_csv, index=False)

    n_spread = sum(r["pred_spread"] for r in results)
    print(f"Total pairs: {len(results)}, Predicted spreads: {n_spread} "
          f"({100 * n_spread / len(results):.1f}%)")
    print(f"Output -> {args.output_csv}")


if __name__ == "__main__":
    main()
