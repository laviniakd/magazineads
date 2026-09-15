"""Two-page spread classifier model architecture (inference only).

Given two adjacent pages (both predicted as full-page ads), determines whether
they form a single two-page spread advertisement.

Architecture: shared CLIP vision encoder encodes each page independently.
Multiple head types combine the encodings:

  diff        — MLP on element-wise difference (left - right). Best performer.
  concat      — [left || right] -> MLP.
  similarity  — Geometric features (cosine sim, L2 dist, norm ratio) -> Linear.
  cross_attn  — Cross-attention over CLIP patch tokens.
  color       — Soft per-channel RGB histograms + CLIP concat features.
  boundary    — Cross-attention over gutter-adjacent patch tokens.
  text        — CLIP text encoder on OCR text + visual cross-attention.
"""

from __future__ import annotations

from pathlib import Path
from typing import Union

from PIL import Image as _Image
_Image.MAX_IMAGE_PIXELS = None

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision.transforms import v2
from transformers import CLIPModel

# OpenAI CLIP normalization constants
_CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
_CLIP_STD  = (0.26862954, 0.26130258, 0.27577711)

HEAD_TYPES = ("concat", "similarity", "diff", "cross_attn", "color", "boundary", "text")

HeadSpec = Union[str, list[str]]


def _parse_head_spec(spec: HeadSpec) -> list[str]:
    if isinstance(spec, list):
        types = spec
    elif "+" in spec:
        types = spec.split("+")
    else:
        types = [spec]
    for t in types:
        if t not in HEAD_TYPES:
            raise ValueError(f"Unknown head type {t!r}. Valid: {HEAD_TYPES}")
    return types


def find_adjacent_page(img_path: Path) -> Path | None:
    """Return the spread-partner page path, or None if not found."""
    stem = img_path.stem
    parts = stem.split("-")
    page_str = parts[-1]
    try:
        page_number = int(page_str.lstrip("0") or "0")
    except ValueError:
        return None

    adjacent_number = page_number - 1 if page_number % 2 == 1 else page_number + 1
    base = "-".join(parts[:-1])
    pad_len = len(page_str)

    candidates = [
        img_path.parent / f"{base}-{str(adjacent_number).zfill(pad_len)}.jpg",
        img_path.parent / f"{base}-{adjacent_number}.jpg",
        img_path.parent / f"{base}-{str(adjacent_number).zfill(2)}.jpg",
        img_path.parent / f"{base}-{str(adjacent_number).zfill(3)}.jpg",
    ]
    seen: set[Path] = set()
    for candidate in candidates:
        if candidate in seen:
            continue
        seen.add(candidate)
        if candidate.exists():
            return candidate
    return None


def build_transform(image_size: int = 336) -> v2.Transform:
    """Return a CLIP-compatible image transform pipeline."""
    return v2.Compose([
        v2.Resize((image_size, image_size), interpolation=v2.InterpolationMode.BICUBIC),
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=_CLIP_MEAN, std=_CLIP_STD),
    ])


def _soft_histogram(x: torch.Tensor, n_bins: int = 16) -> torch.Tensor:
    """Differentiable per-channel soft histogram."""
    mean = torch.tensor(_CLIP_MEAN, device=x.device).view(1, 3, 1, 1)
    std  = torch.tensor(_CLIP_STD,  device=x.device).view(1, 3, 1, 1)
    rgb  = (x * std + mean).clamp(0.0, 1.0)

    B, C, H, W = rgb.shape
    flat = rgb.reshape(B, C, -1)

    centers = torch.linspace(0.0, 1.0, n_bins, device=x.device)
    sigma   = 1.5 / n_bins

    dists   = (flat.unsqueeze(-1) - centers.view(1, 1, 1, -1)) ** 2
    weights = torch.exp(-dists / (2.0 * sigma ** 2))
    hist    = weights.sum(dim=2)
    hist    = hist / (hist.sum(dim=-1, keepdim=True) + 1e-8)
    return hist.reshape(B, C * n_bins)


# ---------------------------------------------------------------------------
# Head modules
# ---------------------------------------------------------------------------

class _SimilarityHead(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(3, 1)

    def forward(self, left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
        left_norm  = left.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        right_norm = right.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        norm_ratio = (torch.minimum(left_norm, right_norm) /
                      torch.maximum(left_norm, right_norm)).squeeze(-1)
        left_n  = F.normalize(left,  dim=-1)
        right_n = F.normalize(right, dim=-1)
        cos = F.cosine_similarity(left_n, right_n, dim=-1)
        l2  = torch.norm(left_n - right_n, dim=-1)
        features = torch.stack([cos, l2, norm_ratio], dim=-1)
        return self.linear(features).squeeze(-1)


class _DiffHead(nn.Module):
    def __init__(self, hidden_size: int, mlp_hidden_dim: int, dropout: float) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_size, mlp_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden_dim, 1),
        )

    def forward(self, left: torch.Tensor, right: torch.Tensor, **_) -> torch.Tensor:
        return self.net(left - right).squeeze(-1)


class _ConcatHead(nn.Module):
    def __init__(self, hidden_size: int, mlp_hidden_dim: int, dropout: float) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(2 * hidden_size, mlp_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden_dim, 1),
        )

    def forward(self, left: torch.Tensor, right: torch.Tensor, **_) -> torch.Tensor:
        return self.net(torch.cat([left, right], dim=-1)).squeeze(-1)


class _CrossAttnHead(nn.Module):
    def __init__(self, hidden_size: int, proj_dim: int = 128,
                 num_heads: int = 4, dropout: float = 0.1) -> None:
        super().__init__()
        self.proj       = nn.Linear(hidden_size, proj_dim)
        self.cross_attn = nn.MultiheadAttention(
            proj_dim, num_heads, dropout=dropout, batch_first=True)
        self.norm       = nn.LayerNorm(proj_dim)
        self.classifier = nn.Sequential(nn.Dropout(dropout), nn.Linear(proj_dim, 1))

    def forward(self, left_patches: torch.Tensor, right_patches: torch.Tensor) -> torch.Tensor:
        lp = self.proj(left_patches)
        rp = self.proj(right_patches)
        attended, _ = self.cross_attn(lp, rp, rp)
        attended     = self.norm(attended + lp)
        pooled       = attended.mean(dim=1)
        return self.classifier(pooled).squeeze(-1)


class _BoundaryHead(nn.Module):
    def __init__(self, hidden_size: int, proj_dim: int = 128, num_heads: int = 4,
                 dropout: float = 0.1, grid_size: int = 24, boundary_cols: int = 4) -> None:
        super().__init__()
        self.grid_size = grid_size
        self.boundary_cols = boundary_cols

        right_edge = [r * grid_size + c for r in range(grid_size)
                      for c in range(grid_size - boundary_cols, grid_size)]
        left_edge = [r * grid_size + c for r in range(grid_size)
                     for c in range(boundary_cols)]
        self.register_buffer("right_patch_idx", torch.tensor(right_edge, dtype=torch.long))
        self.register_buffer("left_patch_idx",  torch.tensor(left_edge,  dtype=torch.long))

        self.proj       = nn.Linear(hidden_size, proj_dim)
        self.cross_attn = nn.MultiheadAttention(
            proj_dim, num_heads, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(proj_dim)

        input_dim = 2 * hidden_size + proj_dim
        self.classifier = nn.Sequential(
            nn.Dropout(dropout), nn.Linear(input_dim, proj_dim),
            nn.GELU(), nn.Dropout(dropout), nn.Linear(proj_dim, 1))

    def forward(self, left_emb, right_emb, left_patches, right_patches):
        left_boundary  = left_patches[:,  self.right_patch_idx, :]
        right_boundary = right_patches[:, self.left_patch_idx,  :]
        lp = self.proj(left_boundary)
        rp = self.proj(right_boundary)
        attended, _ = self.cross_attn(lp, rp, rp)
        attended = self.norm(attended + lp)
        pooled   = attended.mean(dim=1)
        features = torch.cat([left_emb, right_emb, pooled], dim=-1)
        return self.classifier(features).squeeze(-1)


class _TextHead(nn.Module):
    def __init__(self, vis_hidden_size: int, txt_hidden_size: int,
                 mlp_hidden_dim: int, dropout: float,
                 proj_dim: int = 128, num_heads: int = 4) -> None:
        super().__init__()
        self.vis_proj = nn.Linear(vis_hidden_size, proj_dim)
        self.txt_proj = nn.Linear(txt_hidden_size, proj_dim)
        self.cross_attn = nn.MultiheadAttention(
            proj_dim, num_heads, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(proj_dim)
        input_dim = 6 * proj_dim
        self.net = nn.Sequential(
            nn.Dropout(dropout), nn.Linear(input_dim, mlp_hidden_dim),
            nn.GELU(), nn.Dropout(dropout), nn.Linear(mlp_hidden_dim, 1))

    def _cross(self, query, key):
        q = query.unsqueeze(1)
        k = key.unsqueeze(1)
        out, _ = self.cross_attn(q, k, k)
        return self.norm(out.squeeze(1) + query)

    def forward(self, left_vis, right_vis, left_txt, right_txt):
        lv = self.vis_proj(left_vis)
        rv = self.vis_proj(right_vis)
        lt = self.txt_proj(left_txt)
        rt = self.txt_proj(right_txt)
        cross_lt_rv = self._cross(lt, rv)
        cross_rt_lv = self._cross(rt, lv)
        features = torch.cat([lv, rv, lt, rt, cross_lt_rv, cross_rt_lv], dim=-1)
        return self.net(features).squeeze(-1)


class _ColorHead(nn.Module):
    def __init__(self, hidden_size: int, mlp_hidden_dim: int,
                 dropout: float, n_bins: int = 16) -> None:
        super().__init__()
        color_dim = 3 * n_bins
        input_dim = 3 * color_dim + 2 * hidden_size
        self.n_bins = n_bins
        self.net = nn.Sequential(
            nn.Dropout(dropout), nn.Linear(input_dim, mlp_hidden_dim),
            nn.GELU(), nn.Dropout(dropout), nn.Linear(mlp_hidden_dim, 1))

    def forward(self, left_emb, right_emb, left_pixels, right_pixels):
        left_hist  = _soft_histogram(left_pixels,  self.n_bins)
        right_hist = _soft_histogram(right_pixels, self.n_bins)
        diff_hist  = (left_hist - right_hist).abs()
        features   = torch.cat(
            [left_emb, right_emb, left_hist, right_hist, diff_hist], dim=-1)
        return self.net(features).squeeze(-1)


# ---------------------------------------------------------------------------
# Main model
# ---------------------------------------------------------------------------

class SpreadClassifier(nn.Module):
    """Dual CLIP encoder for two-page spread detection.

    Both pages are encoded by the same shared CLIP vision encoder (frozen by
    default). The head_type controls how the two encodings are combined.

    Forward output dict:
        logits: (batch,) raw un-sigmoided spread scores.
        loss:   weighted BCE loss if labels are provided, else None.
    """

    def __init__(
        self,
        clip_model_name: str = "openai/clip-vit-large-patch14-336",
        freeze_encoder: bool = True,
        unfreeze_last_n: int = 0,
        dropout: float = 0.1,
        mlp_hidden_dim: int = 256,
        head_type: HeadSpec = "concat",
        proj_dim: int = 128,
        num_heads: int = 4,
        label_smoothing: float = 0.0,
        text_emb_dim: int = 0,
    ) -> None:
        super().__init__()

        head_types = _parse_head_spec(head_type)
        self.head_types = head_types
        self.text_emb_dim = text_emb_dim

        clip = CLIPModel.from_pretrained(clip_model_name)
        self.encoder = clip.vision_model
        vis_hidden_size: int = clip.config.vision_config.hidden_size

        if freeze_encoder:
            for param in self.encoder.parameters():
                param.requires_grad = False

        if unfreeze_last_n > 0:
            layers = self.encoder.encoder.layers
            for layer in layers[-unfreeze_last_n:]:
                for param in layer.parameters():
                    param.requires_grad = True
            for param in self.encoder.post_layernorm.parameters():
                param.requires_grad = True

        self.text_encoder = None
        if "text" in head_types:
            self.text_encoder = clip.text_model
            txt_hidden_size: int = clip.config.text_config.hidden_size
            if freeze_encoder:
                for param in self.text_encoder.parameters():
                    param.requires_grad = False

        emb_input_dim = vis_hidden_size + text_emb_dim

        built_heads: list[nn.Module] = []
        for ht in head_types:
            built_heads.append(
                self._build_head(ht, vis_hidden_size, emb_input_dim, mlp_hidden_dim,
                                 dropout, proj_dim, num_heads,
                                 txt_hidden_size=(
                                     clip.config.text_config.hidden_size
                                     if ht == "text" else 0
                                 ))
            )
        self.heads = nn.ModuleList(built_heads)

        if len(head_types) > 1:
            self.combiner = nn.Linear(len(head_types), 1, bias=False)
            nn.init.constant_(self.combiner.weight, 1.0 / len(head_types))
        else:
            self.combiner = None

        self.label_smoothing = label_smoothing
        self.head_type = head_types[0] if len(head_types) == 1 else "+".join(head_types)
        self.head = self.heads[0] if len(head_types) == 1 else None

    @staticmethod
    def _build_head(head_type, vis_hidden_size, emb_input_dim, mlp_hidden_dim,
                    dropout, proj_dim, num_heads, txt_hidden_size=0):
        if head_type == "similarity":
            return _SimilarityHead()
        elif head_type == "diff":
            return _DiffHead(emb_input_dim, mlp_hidden_dim, dropout)
        elif head_type == "concat":
            return _ConcatHead(emb_input_dim, mlp_hidden_dim, dropout)
        elif head_type == "cross_attn":
            return _CrossAttnHead(vis_hidden_size, proj_dim, num_heads, dropout)
        elif head_type == "boundary":
            return _BoundaryHead(vis_hidden_size, proj_dim, num_heads, dropout)
        elif head_type == "color":
            return _ColorHead(vis_hidden_size, mlp_hidden_dim, dropout)
        elif head_type == "text":
            return _TextHead(vis_hidden_size, txt_hidden_size, mlp_hidden_dim,
                             dropout, proj_dim, num_heads)
        else:
            raise ValueError(f"Unknown head type: {head_type!r}")

    def _encode(self, pixel_values):
        out = self.encoder(pixel_values=pixel_values, output_hidden_states=False)
        return out.pooler_output, out.last_hidden_state

    def _encode_text(self, input_ids, attention_mask):
        out = self.text_encoder(input_ids=input_ids, attention_mask=attention_mask)
        return out.pooler_output

    def _apply_head(self, head_type, head, left_emb, right_emb,
                    left_vis_emb, right_vis_emb,
                    left_patches, right_patches,
                    left_pixels, right_pixels,
                    left_hist, right_hist,
                    left_txt=None, right_txt=None):
        if head_type == "cross_attn":
            return head(left_patches, right_patches)
        elif head_type == "color":
            if left_hist is not None:
                diff_hist = (left_hist - right_hist).abs()
                features  = torch.cat(
                    [left_vis_emb, right_vis_emb, left_hist, right_hist, diff_hist], dim=-1)
                return head.net(features).squeeze(-1)
            return head(left_vis_emb, right_vis_emb, left_pixels, right_pixels)
        elif head_type == "boundary":
            return head(left_vis_emb, right_vis_emb, left_patches, right_patches)
        elif head_type == "text":
            return head(left_vis_emb, right_vis_emb, left_txt, right_txt)
        else:
            return head(left_emb, right_emb)

    def forward(
        self,
        left_pixels=None, right_pixels=None,
        left_emb=None, right_emb=None,
        left_patches=None, right_patches=None,
        left_hist=None, right_hist=None,
        labels=None, sample_weights=None,
        left_input_ids=None, left_attention_mask=None,
        right_input_ids=None, right_attention_mask=None,
        left_text_emb=None, right_text_emb=None,
    ) -> dict[str, torch.Tensor | None]:
        if left_pixels is not None:
            left_emb,  left_hidden  = self._encode(left_pixels)
            right_emb, right_hidden = self._encode(right_pixels)
            left_patches  = left_hidden[:, 1:, :]
            right_patches = right_hidden[:, 1:, :]
            left_hist = right_hist = None
            _left_px  = left_pixels
            _right_px = right_pixels
        else:
            if left_emb is None:
                raise ValueError("forward() requires either left_pixels or left_emb")
            _left_px = _right_px = None

        left_txt = right_txt = None
        if self.text_encoder is not None and left_input_ids is not None:
            left_txt  = self._encode_text(left_input_ids,  left_attention_mask)
            right_txt = self._encode_text(right_input_ids, right_attention_mask)
            if left_text_emb is None and self.text_emb_dim > 0:
                left_text_emb  = left_txt
                right_text_emb = right_txt

        left_vis_emb  = left_emb
        right_vis_emb = right_emb

        if left_text_emb is not None and self.text_emb_dim > 0:
            left_emb  = torch.cat([left_vis_emb,  left_text_emb],  dim=-1)
            right_emb = torch.cat([right_vis_emb, right_text_emb], dim=-1)

        all_logits = [
            self._apply_head(ht, head, left_emb, right_emb,
                             left_vis_emb, right_vis_emb,
                             left_patches, right_patches,
                             _left_px, _right_px,
                             left_hist, right_hist,
                             left_txt, right_txt)
            for ht, head in zip(self.head_types, self.heads)
        ]

        if self.combiner is not None:
            logits = self.combiner(torch.stack(all_logits, dim=-1)).squeeze(-1)
        else:
            logits = all_logits[0]

        loss = None
        if labels is not None:
            targets = labels.float()
            if self.label_smoothing > 0.0:
                targets = targets * (1.0 - self.label_smoothing) + self.label_smoothing / 2.0
            bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
            if sample_weights is not None:
                bce = bce * sample_weights
            loss = bce.mean()

        return {"loss": loss, "logits": logits}
