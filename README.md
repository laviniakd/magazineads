# "Better Things for Better Living": A Pipeline for Downloading, Processing, and Analyzing Magazine Advertisements

Computational pipeline for extracting and analyzing advertisements from 20th-century American magazine scans. This repository accompanies the dataset paper describing a corpus of 752,326 pages from five magazines (TIME, Esquire, LIFE, Forbes, Variety), spanning 1924--2017. TIME is not available on the Internet Archive; the downloading code in this repo only retrieves the other magazines.

## Pipeline overview

The pipeline has four stages, each with a corresponding script in `scripts/`:

| Step | Script | What it does |
|------|--------|-------------|
| 1 | `scripts/01_download.sh` | Download raw magazine scans from the Internet Archive |
| 2 | `scripts/02_convert.sh` | Convert PDF/JP2 archives to per-page JPEG images |
| 3 | `scripts/03_annotate.sh` | Run classification, detection, OCR, and labeling models |
| 4 | `scripts/04_analyze.sh` | Generate publication-quality figures and tables |

## Quick start

```bash
# Install
pip install -e .

# Configure data paths (or set MAGAZINEADS_DATA_ROOT env var)
cp magazineads.toml.example magazineads.toml
# Edit magazineads.toml to set data_root

# Download trained model weights from HuggingFace
python scripts/download_models.py

# Run the full pipeline
bash scripts/01_download.sh
bash scripts/02_convert.sh
bash scripts/03_annotate.sh
bash scripts/04_analyze.sh
```

## Requirements

- Python >= 3.11
- GPU recommended for Steps 2--3
- `pdftoppm` (from [Poppler](https://poppler.freedesktop.org/)) for PDF conversion
- Model weights from HuggingFace (see [Models](#models) below)

Install Python dependencies:

```bash
pip install -e .
```

## Models

Step 3 requires trained model weights. Download them all at once:

```bash
python scripts/download_models.py
```

Or download individual models:

```bash
python scripts/download_models.py --model page     # page classifier only
python scripts/download_models.py --model yolo     # YOLO ad localizer only
python scripts/download_models.py --model spread   # spread classifier only
python scripts/download_models.py --model product  # product category only
```

All models are hosted at [`laviniad/magazineads`](https://huggingface.co/laviniad/magazineads) on HuggingFace.

| Model | Task | Architecture | Size |
|-------|------|-------------|------|
| Page classifier | 3-class page labeling (ad / mixed / non-ad) | CLIP ViT-L/14@336px + linear head | 1.2 GB |
| Ad localizer | Bounding boxes on mixed pages | YOLOv8l (fine-tuned) | 84 MB |
| Spread classifier | Two-page ad spread detection | Dual CLIP encoder + diff head | 1.2 GB |
| Product category | 8-class product taxonomy | CLIP embeddings + MALLET topics + LogReg | 65 KB |

Off-the-shelf models (no download needed):
- **OCR**: [Surya](https://github.com/VikParuchuri/surya) (installed via pip)
- **Person detection**: YOLOv8l (COCO class 0, via `ultralytics`)
- **Face detection**: SCRFD (via `insightface`)

## Project structure

```
magazineads/
    paths.py              # Centralized path config (TOML / env var)
    corpus.py             # Data loading (predictions CSV, ad sampling, OCR)
    data.py               # Regulatory text filter, ad text loading
    ppi.py                # PPI++ wrappers for bias-corrected prevalence
    embed_ads.py          # CLIP embedding extraction
    utils.py              # Shared utilities

    scraping/
        ia_scraper.py     # Download via Internet Archive search API (recommended)
        legacy/           # Original template-based scrapers (preserved for provenance)

    convert/
        to_jpeg.py        # PDF/JP2 zip -> per-page JPEG conversion

    classify/
        model.py          # CLIPImageClassifier (3-class page classifier)
        run.py            # Batch inference CLI

    detection/
        config.py         # YOLOConfig (inference-only fields)
        infer.py          # YOLOInference batch runner

    spreads/
        model.py          # SpreadClassifier (7 head architectures)
        infer.py          # Spread inference on adjacent page pairs

    ocr/
        base.py           # AbstractOCRExtractor
        surya_extractor.py
        precompute/
            surya_batch.py  # Batch OCR precomputation

    people/
        infer_people.py   # YOLOv8l person detection
        infer_faces.py    # SCRFD face detection

    product_category/
        predict.py        # CLIP+topics -> LogReg prediction

    analysis/
        plot_utils.py     # Shared palettes, styles, confusion matrix helper
        paper_plots.py    # Unified paper figure generator
        make_paper_plots.py
        descriptive_stats.py
        compute_whitespace.py
        plot_*.py         # Individual analysis plots

scripts/
    01_download.sh        # Step 1: download from IA
    02_convert.sh         # Step 2: archive -> JPEG
    03_annotate.sh        # Step 3: model inference
    04_analyze.sh         # Step 4: plots and tables
```

## Configuration

Paths are configured via `magazineads.toml` (copy from `magazineads.toml.example`):

```toml
[paths]
data_root = "/data/magazines"
```

Or set the `MAGAZINEADS_DATA_ROOT` environment variable. See [magazineads.toml.example](magazineads.toml.example) for all options.

## Scraping

The recommended scraper uses the Internet Archive's search API to discover and download magazine issues:

```bash
# Download all Variety issues
python -m magazineads.scraping.ia_scraper --magazine variety --output_dir data/originals/variety

# Dry run (list items without downloading)
python -m magazineads.scraping.ia_scraper --magazine esquire --dry-run

# Custom search query with date range
python -m magazineads.scraping.ia_scraper \
    --query "identifier:sim_life_* AND date:[1945-01-01 TO 1960-12-31]" \
    --output_dir data/originals/life_postwar
```

Available magazine presets: `esquire`, `life`, `forbes`, `variety`. (Time's IA collection is not public.)

The `legacy/` subdirectory preserves the original template-based scrapers that were used to build the corpus. These construct download URLs by predicting exact IA identifiers from date/volume/number patterns and are retained for reproducibility.

## License

MIT. See [LICENSE](LICENSE).
