"""CLIP embedding computation for magazine ad images.

Supports two CLIP variants:
  - OpenAI CLIP ViT-L/14@336px (preferred, via ``pip install git+https://github.com/openai/CLIP``)
  - HuggingFace transformers CLIPModel (fallback)

Usage:
    from magazineads.embed_ads import get_clip_model, embed_images

    model_info = get_clip_model()
    embed_images(image_paths, embedding_paths, model_info)
"""

import os
import threading
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

Image.MAX_IMAGE_PIXELS = 200_000_000

RESIZE_HEIGHT = 336
RESIZE_WIDTH = 336


def get_clip_model(device: str | None = None) -> dict:
    """Load CLIP model and return a dict with model, preprocess, and variant info.

    Returns dict with keys: model, preprocess (or processor), variant, device.
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    try:
        import clip
        model, preprocess = clip.load("ViT-L/14@336px", device=device)
        model.eval()
        return {"model": model, "preprocess": preprocess, "variant": "openai", "device": device}
    except ImportError:
        pass

    from transformers import CLIPProcessor, CLIPModel
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14-336")
    model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14-336").to(device)
    model.eval()
    return {"model": model, "processor": processor, "variant": "hf", "device": device}


class ImageDataset(Dataset):
    """Dataset that loads and resizes images for CLIP embedding."""

    def __init__(self, image_paths, embedding_paths, transform=None):
        self.image_paths = image_paths
        self.embedding_paths = embedding_paths
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        try:
            img = Image.open(self.image_paths[idx]).convert("RGB")
            img = img.resize((RESIZE_WIDTH, RESIZE_HEIGHT))
            if self.transform:
                img = self.transform(img)
            return img, self.embedding_paths[idx], idx
        except Exception as e:
            print(f"Error loading {self.image_paths[idx]}: {e}")
            black_img = Image.new("RGB", (RESIZE_WIDTH, RESIZE_HEIGHT), color="black")
            if self.transform:
                black_img = self.transform(black_img)
            return black_img, self.embedding_paths[idx], idx


def embed_images(
    image_paths: list[str],
    embedding_paths: list[str],
    model_info: dict | None = None,
    batch_size: int = 128,
    num_workers: int = 4,
    skip_existing: bool = True,
):
    """Embed images using CLIP and save as .npy files.

    Parameters
    ----------
    image_paths : paths to source images.
    embedding_paths : paths where .npy embeddings will be saved.
    model_info : dict from get_clip_model(). Loaded automatically if None.
    batch_size : batch size for GPU inference.
    num_workers : DataLoader workers.
    skip_existing : skip images that already have embeddings on disk.
    """
    if model_info is None:
        model_info = get_clip_model()

    # Filter to unembedded images
    if skip_existing:
        pairs = [(ip, ep) for ip, ep in zip(image_paths, embedding_paths)
                 if not os.path.exists(ep)]
        if not pairs:
            print("All images already embedded.")
            return
        image_paths, embedding_paths = zip(*pairs)
        image_paths = list(image_paths)
        embedding_paths = list(embedding_paths)

    device = model_info["device"]
    model = model_info["model"]

    if model_info["variant"] == "openai":
        dataset = ImageDataset(image_paths, embedding_paths, transform=model_info["preprocess"])
        collate = _collate_openai
    else:
        dataset = ImageDataset(image_paths, embedding_paths, transform=None)
        collate = _collate_hf

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        prefetch_factor=2 if num_workers > 0 else None,
        collate_fn=collate,
        persistent_workers=num_workers > 0,
    )

    with torch.no_grad():
        for images, paths, indices in tqdm(dataloader, desc="Embedding images"):
            if model_info["variant"] == "openai":
                images = images.to(device, non_blocking=True)
                embeddings = model.encode_image(images).cpu().numpy()
            else:
                inputs = model_info["processor"](images=images, return_tensors="pt")
                pixel_values = inputs["pixel_values"].to(device, non_blocking=True)
                embeddings = model.get_image_features(pixel_values=pixel_values).cpu().numpy()

            for path, embedding in zip(paths, embeddings):
                os.makedirs(os.path.dirname(path), exist_ok=True)
                np.save(path, embedding)


def load_embedding(path: str | Path) -> np.ndarray | None:
    """Load a saved .npy embedding, or return None if missing."""
    if os.path.exists(path):
        return np.load(path)
    return None


# ---------------------------------------------------------------------------
# Internal collate functions
# ---------------------------------------------------------------------------

def _collate_openai(batch):
    images = torch.stack([item[0] for item in batch])
    paths = [item[1] for item in batch]
    indices = [item[2] for item in batch]
    return images, paths, indices


def _collate_hf(batch):
    images = [item[0] for item in batch]
    paths = [item[1] for item in batch]
    indices = [item[2] for item in batch]
    return images, paths, indices
