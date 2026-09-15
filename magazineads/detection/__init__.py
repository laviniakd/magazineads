"""YOLO-based ad bounding box detection — inference only."""

from magazineads.detection.config import YOLOConfig
from magazineads.detection.infer import YOLOInference

__all__ = ["YOLOConfig", "YOLOInference"]
