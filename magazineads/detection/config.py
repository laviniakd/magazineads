"""Configuration for YOLO ad detection inference."""

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Dict, Any

import torch

from magazineads.paths import paths as _paths


@dataclass
class YOLOConfig:
    """Configuration for YOLO ad detection.

    For inference, the key parameters are:
      - model_name, confidence_threshold, iou_threshold
      - inference_batch_size, img_size
      - output_dir, inference_dir (where results are saved)
    """

    # Directory settings
    output_dir: str = field(default_factory=lambda: str(_paths.models_dir / "yolo"))
    inference_dir: str = field(default_factory=lambda: str(_paths.inference_dir))

    # Device
    device: str = "cuda:0" if torch.cuda.is_available() else "cpu"

    # Model
    model_name: str = "yolov8n.pt"
    num_classes: int = 1  # single "ad" class

    # Image settings
    img_size: int = 1024

    # Inference settings
    confidence_threshold: float = 0.5
    iou_threshold: float = 0.45
    max_detections: int = 300
    inference_batch_size: int = 8

    def to_dict(self) -> Dict[str, Any]:
        config_dict = {}
        for field_name in self.__dataclass_fields__:
            value = getattr(self, field_name)
            if isinstance(value, (str, int, float, bool, type(None), list, dict)):
                config_dict[field_name] = value
        return config_dict

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "YOLOConfig":
        valid_keys = set(cls.__dataclass_fields__.keys())
        filtered = {k: v for k, v in config_dict.items() if k in valid_keys}
        return cls(**filtered)

    @classmethod
    def from_json(cls, json_path: str) -> "YOLOConfig":
        with open(json_path) as f:
            return cls.from_dict(json.load(f))

    def save_json(self, json_path: str):
        with open(json_path, "w") as f:
            json.dump(self.to_dict(), f, indent=4)
