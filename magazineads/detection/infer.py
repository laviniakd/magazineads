"""YOLO ad detection inference.

Runs a fine-tuned YOLO model on magazine page images to detect ad bounding boxes.

Usage:
    python -m magazineads.detection.infer \\
        --model_path models/yolo/best.pt \\
        --image_dir /path/to/images
"""

import os
import json
import logging
import argparse
from pathlib import Path
from typing import List, Dict, Any, Optional

from PIL import Image
from ultralytics import YOLO

from magazineads.detection.config import YOLOConfig

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


class YOLOInference:
    """YOLO model wrapper for ad detection inference."""

    def __init__(self, config: YOLOConfig, model_path: Optional[str] = None):
        self.config = config
        self.model_path = model_path or self._find_best_model()

        logger.info(f"Loading model from: {self.model_path}")
        self.model = YOLO(self.model_path, task="detect")
        logger.info("Model loaded successfully")

        os.makedirs(config.inference_dir, exist_ok=True)

    def _find_best_model(self) -> str:
        search_dir = Path(self.config.output_dir)

        for ext in ["best.pt", "best.onnx"]:
            for model_path in search_dir.rglob(f"*/{ext}"):
                logger.info(f"Found model: {model_path}")
                return str(model_path)

        pt_files = list(search_dir.rglob("*.pt"))
        if pt_files:
            logger.warning(f"Could not find best model, using: {pt_files[0]}")
            return str(pt_files[0])

        raise FileNotFoundError(
            f"No model found in {search_dir}. Please specify --model_path"
        )

    @staticmethod
    def _load_and_resize(path: str, max_side: int) -> Image.Image:
        img = Image.open(path).convert("RGB")
        w, h = img.size
        long = max(w, h)
        if long > max_side:
            scale = max_side / long
            img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
        return img

    def infer_on_images(
        self,
        image_paths: List[str],
        save_visualizations: bool = True,
        save_json: bool = True,
        max_side: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """Run inference on a list of images, batched for GPU efficiency."""
        logger.info(
            f"Running inference on {len(image_paths)} images "
            f"(batch_size={self.config.inference_batch_size}"
            + (f", max_side={max_side}" if max_side else "")
            + ")..."
        )

        resolved = []
        for path in image_paths:
            if not os.path.exists(path):
                alt = path.replace("-x-", "-x-x-")
                path = alt if os.path.exists(alt) else path
            resolved.append(path)

        all_results = []
        batch_size = self.config.inference_batch_size

        for batch_start in range(0, len(resolved), batch_size):
            batch = resolved[batch_start : batch_start + batch_size]
            try:
                if max_side is not None:
                    batch_inputs = [self._load_and_resize(p, max_side) for p in batch]
                else:
                    batch_inputs = batch

                results = self.model(
                    batch_inputs,
                    conf=self.config.confidence_threshold,
                    iou=self.config.iou_threshold,
                    max_det=self.config.max_detections,
                    verbose=False,
                )

                for r, path in zip(results, batch):
                    result_dict = self._process_result(r, path)
                    all_results.append(result_dict)

                    if save_visualizations:
                        vis_path = os.path.join(
                            self.config.inference_dir,
                            f"result_{os.path.basename(path)}",
                        )
                        r.save(filename=vis_path)

                    if save_json and r.boxes is not None and len(r.boxes) > 0:
                        json_path = os.path.join(
                            self.config.inference_dir,
                            f"{os.path.splitext(os.path.basename(path))[0]}_boxes.json",
                        )
                        with open(json_path, "w") as f:
                            json.dump(result_dict, f, indent=2)

            except Exception as e:
                logger.error(f"Failed to process batch: {e}", exc_info=True)
                continue

        logger.info(f"Inference completed on {len(all_results)} images")
        return all_results

    def _process_result(self, result, image_path: str) -> Dict[str, Any]:
        result_dict = {
            "image_path": image_path,
            "image_name": os.path.basename(image_path),
            "detections": [],
        }

        if result.boxes is None or len(result.boxes) == 0:
            result_dict["num_detections"] = 0
            return result_dict

        boxes = result.boxes
        result_dict["num_detections"] = len(boxes)

        for i, box in enumerate(boxes):
            detection = {
                "box_id": i,
                "class": int(box.cls[0]) if hasattr(box, "cls") else 0,
                "confidence": float(box.conf[0]) if hasattr(box, "conf") else 0.0,
            }

            if hasattr(box, "xyxy"):
                coords = box.xyxy[0].tolist()
                detection["bbox"] = {
                    "x1": coords[0],
                    "y1": coords[1],
                    "x2": coords[2],
                    "y2": coords[3],
                    "width": coords[2] - coords[0],
                    "height": coords[3] - coords[1],
                }

            result_dict["detections"].append(detection)

        return result_dict

    def run_batch_inference(
        self,
        image_dir: str,
        max_images: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """Run inference on all images in a directory."""
        image_dir = Path(image_dir)
        image_paths = []
        for ext in ["*.jpg", "*.jpeg", "*.png", "*.JPG", "*.JPEG", "*.PNG"]:
            image_paths.extend(list(image_dir.glob(ext)))
        image_paths = [str(p) for p in image_paths]

        if max_images:
            image_paths = image_paths[:max_images]

        logger.info(f"Found {len(image_paths)} images in {image_dir}")
        return self.infer_on_images(image_paths)


def main():
    parser = argparse.ArgumentParser(description="YOLO Ad Detection Inference")
    parser.add_argument("--config_file", type=str, help="Path to JSON configuration file")
    parser.add_argument("--model_path", type=str, help="Path to YOLO model (.pt)")
    parser.add_argument("--model_name", type=str,
                        help="Base YOLO model name (e.g. yolov8m)")
    parser.add_argument("--image_dir", type=str, required=True,
                        help="Directory containing images for inference")
    parser.add_argument("--image_paths", nargs="+", help="Specific image paths")
    parser.add_argument("--max_images", type=int)
    parser.add_argument("--no_viz", action="store_true")
    parser.add_argument("--no_json", action="store_true")
    args = parser.parse_args()

    config = YOLOConfig.from_json(args.config_file) if args.config_file else YOLOConfig()

    model_path = args.model_path
    if args.model_name:
        name = args.model_name
        if not name.endswith(".pt"):
            name += ".pt"
        model_path = name

    inference = YOLOInference(config, model_path=model_path)

    if args.image_paths:
        results = inference.infer_on_images(
            args.image_paths,
            save_visualizations=not args.no_viz,
            save_json=not args.no_json,
        )
    else:
        results = inference.run_batch_inference(args.image_dir, max_images=args.max_images)

    total_det = sum(r.get("num_detections", 0) for r in results)
    logger.info(f"Processed {len(results)} images, {total_det} total detections")


if __name__ == "__main__":
    main()
