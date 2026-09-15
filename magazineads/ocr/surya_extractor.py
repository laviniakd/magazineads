"""OCR extractor that reads precomputed surya output directories.

Surya output structure:
    {surya_output_dir}/{magazine}/{image_stem}/results.json

Each results.json contains:
    {image_stem: [{text_lines: [{polygon, confidence, text, chars, ...}], image_bbox, page}]}
"""

import os
import json
from functools import lru_cache
from typing import Optional, Dict, Any, List
from PIL import Image

from magazineads.ocr.base import AbstractOCRExtractor


class SuryaOCRExtractor(AbstractOCRExtractor):
    """Extractor that reads precomputed surya OCR output.

    Resolves image paths to surya output directories, caches loaded JSON,
    and returns text concatenated from all text_lines above a confidence threshold.

    Args:
        surya_output_dir: Root directory of surya output (e.g. /data/.../surya_output).
        min_confidence: Minimum confidence threshold for including text lines.
        cache_size: Max number of results.json files to keep in LRU cache.
    """

    def __init__(
        self,
        surya_output_dir: str,
        min_confidence: float = 0.3,
        cache_size: int = 10000,
    ):
        self.surya_output_dir = surya_output_dir
        self.min_confidence = min_confidence

        # Create a bounded LRU cache for loaded results
        self._load_result = lru_cache(maxsize=cache_size)(self._load_result_uncached)

    def _resolve_surya_path(self, image_path: str) -> Optional[str]:
        """Resolve an image path to its surya results.json path.

        Strategy:
        1. Extract magazine name from the image path (parent directory name).
        2. Use the image stem (filename without extension) as the subdirectory name.
        3. Look for {surya_output_dir}/{magazine}/{stem}/results.json.
        """
        image_path = os.path.abspath(image_path)
        stem = os.path.splitext(os.path.basename(image_path))[0]

        # Walk up the path to find the magazine name.
        # Image paths look like: .../images/{magazine}/{image}.jpg
        # or: .../images/{magazine}/{split}/{class}/{image}.jpg
        parts = image_path.replace("\\", "/").split("/")

        # Find "images" in the path and take the next component as magazine
        magazine = None
        for i, part in enumerate(parts):
            if part == "images" and i + 1 < len(parts):
                magazine = parts[i + 1]
                break

        if magazine is None:
            # Fallback: try all magazine dirs in surya_output_dir
            for mag_dir in os.listdir(self.surya_output_dir):
                candidate = os.path.join(
                    self.surya_output_dir, mag_dir, stem, "results.json"
                )
                if os.path.isfile(candidate):
                    return candidate
            return None

        results_path = os.path.join(
            self.surya_output_dir, magazine, stem, "results.json"
        )
        if os.path.isfile(results_path):
            return results_path

        # Time magazine images were renamed from YYYY-MM-DD-x-N to YYYY-MM-DD-x-x-N.
        # Fall back to the single-x directory if the double-x one is missing.
        if magazine == "time" and "-x-x-" in stem:
            fallback_stem = stem.replace("-x-x-", "-x-", 1)
            fallback_path = os.path.join(
                self.surya_output_dir, magazine, fallback_stem, "results.json"
            )
            if os.path.isfile(fallback_path):
                return fallback_path

        return None

    def _load_result_uncached(self, results_path: str) -> Optional[Dict[str, Any]]:
        """Load and parse a results.json file."""
        try:
            with open(results_path, "r") as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError):
            return None

    def _get_text_lines(self, image_path: str) -> Optional[List[Dict[str, Any]]]:
        """Get text_lines for an image from surya output."""
        results_path = self._resolve_surya_path(image_path)
        if results_path is None:
            return None

        data = self._load_result(results_path)
        if data is None:
            return None

        stem = os.path.splitext(os.path.basename(image_path))[0]

        # results.json key is the image stem
        if stem in data:
            pages = data[stem]
            if isinstance(pages, list) and len(pages) > 0:
                return pages[0].get("text_lines", [])

        # Fallback: try the first key in the dict
        for key in data:
            pages = data[key]
            if isinstance(pages, list) and len(pages) > 0:
                return pages[0].get("text_lines", [])

        return None

    def extract_text(self, image_path: str, image: Optional[Image.Image] = None) -> str:
        """Return concatenated text from surya output, filtered by confidence.

        Uses column-aware reading order: lines are grouped into columns by
        x-coordinate, then read top-to-bottom within each column, columns
        ordered left-to-right.
        """
        text_lines = self._get_text_lines(image_path)
        if text_lines is None:
            return ""

        filtered = []
        for line in text_lines:
            confidence = line.get("confidence", 0.0)
            if confidence >= self.min_confidence:
                text = line.get("text", "").strip()
                if text:
                    filtered.append(line)

        if not filtered:
            return ""

        return " ".join(
            line.get("text", "").strip()
            for line in _column_sort(filtered)
        )


def _column_sort(lines: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Sort text lines in column-aware reading order.

    1. Separate full-width lines (headings) from column-width lines.
    2. Detect columns by clustering x-left positions.
    3. Within each column, sort top-to-bottom.
    4. Read columns left-to-right, inserting full-width lines at their
       y-position.

    Args:
        lines: List of surya text_line dicts with 'bbox' [x1, y1, x2, y2].
    """
    if len(lines) <= 1:
        return lines

    # Extract bboxes; fall back to y-sort if no bbox data
    bboxed = []
    for line in lines:
        bbox = line.get("bbox")
        if bbox and len(bbox) == 4:
            bboxed.append((bbox, line))
    if not bboxed:
        return sorted(lines, key=lambda l: l.get("bbox", [0, 0])[1] if l.get("bbox") else 0)

    # Page dimensions
    page_x1 = min(b[0] for b, _ in bboxed)
    page_x2 = max(b[2] for b, _ in bboxed)
    page_width = page_x2 - page_x1
    if page_width <= 0:
        return lines

    # Separate full-width lines (spanning > 60% of page) from column lines
    full_width = []
    col_lines = []
    for bbox, line in bboxed:
        x1, y1, x2, y2 = bbox
        width = x2 - x1
        if width > 0.6 * page_width:
            full_width.append((y1, line))
        else:
            col_lines.append((x1, y1, line))

    if not col_lines:
        full_width.sort(key=lambda t: t[0])
        return [line for _, line in full_width]

    # Detect columns by clustering x-left positions
    columns = _cluster_into_columns(col_lines, page_width)

    # Sort lines within each column by y, then order columns left-to-right
    sorted_columns = []
    for col in columns:
        col.sort(key=lambda t: t[1])  # sort by y within column
        col_x = min(t[0] for t in col)  # leftmost x in this column
        sorted_columns.append((col_x, col))
    sorted_columns.sort(key=lambda t: t[0])  # left-to-right

    # Merge columns with full-width lines inserted at their y-position
    result = []
    full_width.sort(key=lambda t: t[0])
    fw_idx = 0

    for _col_x, col in sorted_columns:
        col_y_start = col[0][1] if col else float("inf")
        while fw_idx < len(full_width) and full_width[fw_idx][0] <= col_y_start:
            result.append(full_width[fw_idx][1])
            fw_idx += 1
        for _, _, line in col:
            result.append(line)

    while fw_idx < len(full_width):
        result.append(full_width[fw_idx][1])
        fw_idx += 1

    return result


def _cluster_into_columns(
    col_lines: List[tuple],
    page_width: float,
) -> List[List[tuple]]:
    """Cluster lines into columns by x-left position.

    Uses a simple 1D clustering: sort by x-left, then split whenever the
    gap between consecutive x-left values exceeds a threshold based on
    the median line width.

    Args:
        col_lines: List of (x_left, y_top, line_dict) tuples.
        page_width: Total page width for fallback threshold.

    Returns:
        List of columns, each a list of (x_left, y_top, line_dict).
    """
    if len(col_lines) <= 1:
        return [col_lines]

    # Compute median line width for adaptive gap threshold
    widths = []
    for x1, y1, line in col_lines:
        bbox = line.get("bbox", [0, 0, 0, 0])
        widths.append(bbox[2] - bbox[0])
    widths.sort()
    median_width = widths[len(widths) // 2] if widths else page_width * 0.3

    # Gap threshold: half the median line width, but at least 5% of page
    gap_threshold = max(median_width * 0.5, page_width * 0.05)

    # Sort by x-left
    sorted_lines = sorted(col_lines, key=lambda t: t[0])

    columns: list[list[tuple]] = [[sorted_lines[0]]]
    # Track the median x-left of each column for assignment
    col_centers = [sorted_lines[0][0]]

    for i in range(1, len(sorted_lines)):
        x_left = sorted_lines[i][0]
        # Find the closest existing column center
        best_col = min(range(len(col_centers)), key=lambda c: abs(col_centers[c] - x_left))
        if abs(col_centers[best_col] - x_left) <= gap_threshold:
            columns[best_col].append(sorted_lines[i])
            # Update column center as running mean of x-left values
            n = len(columns[best_col])
            col_centers[best_col] = (col_centers[best_col] * (n - 1) + x_left) / n
        else:
            # New column
            columns.append([sorted_lines[i]])
            col_centers.append(x_left)

    return columns

    def extract_metadata(self, image_path: str) -> Optional[Dict[str, Any]]:
        """Return full surya metadata including bounding boxes, confidence, chars."""
        text_lines = self._get_text_lines(image_path)
        if text_lines is None:
            return None

        return {
            "text_lines": [
                {
                    "text": line.get("text", ""),
                    "confidence": line.get("confidence", 0.0),
                    "polygon": line.get("polygon", []),
                    "bbox": line.get("bbox", []),
                    "chars": line.get("chars", []),
                }
                for line in text_lines
                if line.get("confidence", 0.0) >= self.min_confidence
            ],
            "source": "surya",
        }
