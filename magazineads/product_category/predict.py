"""Product category prediction using CLIP embeddings + MALLET topic features.

Architecture: CLIP ViT-L/14@336 CLS embeddings (768-dim) concatenated with
MALLET LDA topic distributions (100-dim), fed into a LogisticRegression classifier.

The model is trained on 8 reliable product categories:
  Alcohol, Vehicles, Clothing and Accessories, Finance and Insurance,
  Media, Tobacco, Travel and Tourism, Food and Beverage Services

Usage:
    python -m magazineads.product_category.predict \\
        --model_path models/product_category/model.joblib \\
        --scaler_path models/product_category/scaler.joblib \\
        --embeddings_cache data/corpus_embeddings.npz \\
        --mallet_csv data/mallet_topics.csv \\
        --predictions_csv data/predictions_all.csv \\
        --output data/product_category_predictions.csv
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger(__name__)

# Categories that merge into broader groups
CATEGORY_MERGES = {
    "Oil and Gas": "Business and Industrial",
    "Home and Garden Services": "Business and Industrial",
    "Events and Performances": "Media",
}

RELIABLE_CATEGORIES = [
    "Alcohol", "Vehicles", "Clothing and Accessories",
    "Finance and Insurance", "Media", "Tobacco",
    "Travel and Tourism", "Food and Beverage Services",
]


def load_embeddings(cache_path: Path, emb_dir: Path | None = None):
    """Load CLIP embeddings from .npz cache or individual .npy files.

    Returns (X, ids) where X is (N, 768) and ids is list of (magazine, stem).
    """
    if cache_path.exists():
        logger.info("Loading embeddings from cache: %s", cache_path)
        data = np.load(cache_path, allow_pickle=True)
        X = data["embeddings"].astype(np.float32)
        mags = list(data["magazines"])
        stems = list(data["stems"])
        logger.info("Loaded %d embeddings, dim=%d", len(X), X.shape[1])
        return X, list(zip(mags, stems))

    if emb_dir is None or not emb_dir.exists():
        raise FileNotFoundError(
            f"No embedding cache at {cache_path} and no emb_dir provided")

    file_list = []
    for mag_dir in sorted(emb_dir.iterdir()):
        if not mag_dir.is_dir() or mag_dir.name == "faiss_index":
            continue
        mag = mag_dir.name
        for f in sorted(mag_dir.glob("*.npy")):
            stem = f.stem
            if stem.startswith(f"{mag}_"):
                stem = stem[len(f"{mag}_"):]
            file_list.append((f, mag, stem))

    logger.info("Loading %d embeddings from individual files...", len(file_list))
    dim = np.load(file_list[0][0]).shape[0]
    X = np.empty((len(file_list), dim), dtype=np.float32)
    all_ids = []
    for i, (f, mag, stem) in enumerate(file_list):
        X[i] = np.load(f, mmap_mode="r").astype(np.float32)
        all_ids.append((mag, stem))
    return X, all_ids


def load_mallet_topics(csv_path: Path) -> dict[str, np.ndarray]:
    """Load MALLET topic distributions keyed by stem.

    CSV columns: stem, topic_0, topic_1, ..., topic_N
    Returns dict mapping stem -> topic distribution array.
    """
    df = pd.read_csv(csv_path)
    topic_cols = [c for c in df.columns if c.startswith("topic_")]
    lookup = {}
    for _, row in df.iterrows():
        lookup[row["stem"]] = row[topic_cols].values.astype(np.float32)
    logger.info("Loaded %d MALLET topic distributions (%d topics)",
                len(lookup), len(topic_cols))
    return lookup


def predict_corpus(
    model,
    scaler,
    embeddings: np.ndarray,
    emb_ids: list[tuple[str, str]],
    mallet_lookup: dict[str, np.ndarray],
    predictions_csv: Path | None = None,
) -> pd.DataFrame:
    """Run product category prediction on the full corpus.

    Returns DataFrame with columns: magazine, stem, predicted_category, confidence.
    """
    # Build index: stem -> emb row
    emb_index = {stem: i for i, (_, stem) in enumerate(emb_ids)}

    # Filter to full-page ads if predictions CSV provided
    if predictions_csv is not None:
        pred_df = pd.read_csv(predictions_csv)
        ad_stems = set()
        for _, row in pred_df[pred_df["pred_label"] == 1].iterrows():
            stem = Path(row["filepath"]).stem
            ad_stems.add(stem)
        logger.info("Filtering to %d full-page ad stems", len(ad_stems))
    else:
        ad_stems = {stem for _, stem in emb_ids}

    n_topics = len(next(iter(mallet_lookup.values()))) if mallet_lookup else 0
    zero_topics = np.zeros(n_topics, dtype=np.float32)

    rows = []
    batch_X = []
    batch_meta = []

    for mag, stem in tqdm(emb_ids, desc="Preparing features"):
        if stem not in ad_stems:
            continue
        idx = emb_index.get(stem)
        if idx is None:
            continue

        emb = embeddings[idx]
        topics = mallet_lookup.get(stem, zero_topics)
        features = np.concatenate([emb, topics])
        batch_X.append(features)
        batch_meta.append((mag, stem))

    if not batch_X:
        logger.warning("No features extracted — empty predictions")
        return pd.DataFrame(columns=["magazine", "stem", "predicted_category", "confidence"])

    X = np.stack(batch_X)
    X_scaled = scaler.transform(X)

    logger.info("Predicting %d samples...", len(X))
    predictions = model.predict(X_scaled)

    # Get confidence scores if available
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X_scaled)
        confidences = proba.max(axis=1)
    elif hasattr(model, "decision_function"):
        dec = model.decision_function(X_scaled)
        if dec.ndim == 1:
            confidences = np.abs(dec)
        else:
            confidences = dec.max(axis=1)
    else:
        confidences = np.ones(len(predictions))

    for (mag, stem), pred, conf in zip(batch_meta, predictions, confidences):
        rows.append({
            "magazine": mag,
            "stem": stem,
            "predicted_category": pred,
            "confidence": round(float(conf), 4),
        })

    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(
        description="Predict product categories for magazine ads")
    parser.add_argument("--model_path", type=Path, required=True,
                        help="Path to trained sklearn model (.joblib)")
    parser.add_argument("--scaler_path", type=Path, required=True,
                        help="Path to fitted StandardScaler (.joblib)")
    parser.add_argument("--embeddings_cache", type=Path, required=True,
                        help="Path to corpus embeddings .npz cache")
    parser.add_argument("--mallet_csv", type=Path, required=True,
                        help="Path to MALLET topic distributions CSV")
    parser.add_argument("--predictions_csv", type=Path, default=None,
                        help="Page classification CSV (filter to ads only)")
    parser.add_argument("--output", type=Path, required=True,
                        help="Output predictions CSV")
    args = parser.parse_args()

    import joblib
    model = joblib.load(args.model_path)
    scaler = joblib.load(args.scaler_path)
    logger.info("Loaded model from %s", args.model_path)

    embeddings, emb_ids = load_embeddings(args.embeddings_cache)
    mallet_lookup = load_mallet_topics(args.mallet_csv)

    results = predict_corpus(
        model, scaler, embeddings, emb_ids, mallet_lookup,
        predictions_csv=args.predictions_csv,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(args.output, index=False)
    logger.info("Saved %d predictions to %s", len(results), args.output)

    # Summary
    print(f"\nPrediction summary ({len(results)} ads):")
    for cat in sorted(results["predicted_category"].unique()):
        n = (results["predicted_category"] == cat).sum()
        print(f"  {cat}: {n} ({100 * n / len(results):.1f}%)")


if __name__ == "__main__":
    main()
