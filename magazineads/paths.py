"""Centralized path configuration for the magazineads project.

All project paths are resolved here and exposed as a frozen dataclass singleton.
Other modules should import from here instead of hardcoding paths.

Resolution order for DATA_ROOT:
    1. magazineads.toml [paths] data_root
    2. MAGAZINEADS_DATA_ROOT environment variable
    3. Default: ./data (relative to project root)

PROJECT_ROOT is always auto-detected from this file's location.

Usage:
    from magazineads.paths import paths

    image_dir = paths.images_dir / "esquire"
    model_file = paths.models_dir / "page_classifier"
"""

import os
import warnings
from dataclasses import dataclass
from pathlib import Path


# Auto-detect project root: magazineads/paths.py -> PROJECT_ROOT/magazineads/paths.py
PROJECT_ROOT = Path(__file__).resolve().parent.parent

_CONFIG_FILE = PROJECT_ROOT / "magazineads.toml"


def _load_toml_config() -> dict:
    """Load magazineads.toml if it exists, return empty dict otherwise."""
    if not _CONFIG_FILE.exists():
        return {}
    import tomllib
    with open(_CONFIG_FILE, "rb") as f:
        return tomllib.load(f)


_config = _load_toml_config()
_path_config = _config.get("paths", {})


def _resolve_data_root() -> Path:
    """Resolve DATA_ROOT from config file, env var, or default."""
    # 1. Config file
    if "data_root" in _path_config:
        return Path(_path_config["data_root"])
    # 2. Environment variable
    env_val = os.environ.get("MAGAZINEADS_DATA_ROOT")
    if env_val:
        return Path(env_val)
    # 3. Default
    return PROJECT_ROOT / "data"


@dataclass(frozen=True)
class ProjectPaths:
    """All project paths as a single frozen object.

    Project-relative paths are under project_root.
    Data-relative paths are under data_root.
    """

    # Roots
    project_root: Path
    data_root: Path

    # Project-relative
    models_dir: Path

    # Data-relative
    images_dir: Path
    surya_output_dir: Path
    embeddings_dir: Path
    inference_dir: Path
    predictions_dir: Path
    zero_shot_labels_dir: Path
    text_embeddings_dir: Path
    tables_dir: Path
    classification_results_dir: Path
    plots_dir: Path

    # Project-relative: classifier train/test images
    classifier_images_dir: Path
    # Project-relative: loose files (CSVs, JSONs)
    local_data_dir: Path


def _build_paths() -> ProjectPaths:
    """Build the paths object, allowing per-field overrides from config."""
    data_root = _resolve_data_root()

    def _data_path(key: str, default_subpath: str) -> Path:
        if key in _path_config:
            return Path(_path_config[key])
        return data_root / default_subpath

    def _proj_path(key: str, default_subpath: str) -> Path:
        if key in _path_config:
            return Path(_path_config[key])
        return PROJECT_ROOT / default_subpath

    return ProjectPaths(
        project_root=PROJECT_ROOT,
        data_root=data_root,
        models_dir=_proj_path("models_dir", "models"),
        images_dir=_data_path("images_dir", "images"),
        surya_output_dir=_data_path("surya_output_dir", "surya_output"),
        embeddings_dir=_data_path("embeddings_dir", "embeddings"),
        inference_dir=_data_path("inference_dir", "bounding_boxes/inferences"),
        predictions_dir=_data_path("predictions_dir", "page_classification_predictions"),
        zero_shot_labels_dir=_data_path("zero_shot_labels_dir", "zero_shot_yolo_labels"),
        text_embeddings_dir=_data_path("text_embeddings_dir", "text_embeddings"),
        tables_dir=_data_path("tables_dir", "tables"),
        classification_results_dir=_data_path("classification_results_dir", "classification_results"),
        plots_dir=_data_path("plots_dir", "plots"),
        classifier_images_dir=_proj_path("classifier_images_dir", "data/images"),
        local_data_dir=_proj_path("local_data_dir", "data"),
    )


# Module-level singleton -- import this
paths = _build_paths()
