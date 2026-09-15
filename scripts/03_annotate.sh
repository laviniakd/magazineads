#!/usr/bin/env bash
# ═══════════════════════════════════════════════════════════════════
# Step 3: Add metadata with models (classification, detection, OCR)
# ═══════════════════════════════════════════════════════════════════
#
# Runs the full annotation pipeline on converted page images:
#
#   3a. Page classification  — label each page as ad / mixed / non-ad
#   3b. Ad localization      — draw bounding boxes on mixed pages (YOLO)
#   3c. Spread detection     — identify two-page ad spreads
#   3d. OCR                  — extract text from all ad regions (Surya)
#   3e. Product category     — classify ads by product type
#   3f. Person detection     — detect people in ads (YOLOv8l)
#   3g. Face detection       — detect faces in ads (SCRFD / InsightFace)
#
# Each stage writes its outputs to a subdirectory under $DATA_ROOT.
# Stages are independent after 3a (classification), so you can
# comment out stages you don't need.
#
# Usage:
#   bash scripts/03_annotate.sh              # full pipeline
#   bash scripts/03_annotate.sh variety       # single magazine
#
# Prerequisites:
#   - GPU recommended for all stages
#   - Model weights downloaded from HuggingFace (see README.md)
# ═══════════════════════════════════════════════════════════════════
set -euo pipefail

DATA_ROOT="${MAGAZINEADS_DATA_ROOT:-./data}"
MAGAZINES="${1:-esquire life forbes variety}"
BATCH_SIZE="${BATCH_SIZE:-32}"

# Model paths — override these or set in magazineads.toml
PAGE_CLASSIFIER_PATH="${PAGE_CLASSIFIER_PATH:-$DATA_ROOT/models/page_classifier}"
YOLO_MODEL_PATH="${YOLO_MODEL_PATH:-$DATA_ROOT/models/yolo_ad_detection/best.pt}"
SPREAD_MODEL_PATH="${SPREAD_MODEL_PATH:-$DATA_ROOT/models/spread_classifier_v5/diff}"
PRODUCT_MODEL_PATH="${PRODUCT_MODEL_PATH:-$DATA_ROOT/models/product_category/model.joblib}"

echo "=== Step 3: Annotate pages with models ==="
echo "Data root:  $DATA_ROOT"
echo "Magazines:  $MAGAZINES"
echo "Batch size: $BATCH_SIZE"
echo ""

# ── 3a. Page classification ──────────────────────────────────────
echo "--- 3a. Page classification (CLIP ViT-L/14@336) ---"
for mag in $MAGAZINES; do
    python -m magazineads.classify.run \
        --model_path "$PAGE_CLASSIFIER_PATH" \
        --data_path "$DATA_ROOT/images" \
        --magazines "$mag" \
        --batch_size "$BATCH_SIZE" \
        --predictions_dir "$DATA_ROOT/page_classification_predictions"
done
echo ""

# ── 3b. Ad localization (YOLO) ───────────────────────────────────
echo "--- 3b. Ad localization (YOLO) ---"
for mag in $MAGAZINES; do
    python -m magazineads.detection.infer \
        --model "$YOLO_MODEL_PATH" \
        --source "$DATA_ROOT/images/$mag" \
        --output_dir "$DATA_ROOT/yolo_predictions/$mag"
done
echo ""

# ── 3c. Spread detection ─────────────────────────────────────────
echo "--- 3c. Spread detection (dual CLIP + diff head) ---"
for mag in $MAGAZINES; do
    python -m magazineads.spreads.infer \
        --model_dir "$SPREAD_MODEL_PATH" \
        --image_dir "$DATA_ROOT/images/$mag" \
        --output_csv "$DATA_ROOT/spread_predictions/$mag.csv" \
        --head_type diff \
        --threshold 0.4 \
        --batch_size "$BATCH_SIZE"
done
echo ""

# ── 3d. OCR (Surya) ──────────────────────────────────────────────
echo "--- 3d. OCR (Surya) ---"
for mag in $MAGAZINES; do
    python -m magazineads.ocr.precompute.surya_batch \
        --image_dir "$DATA_ROOT/images" \
        --magazines "$mag" \
        --output_dir "$DATA_ROOT/surya_output/$mag" \
        --batch_size "$BATCH_SIZE"
done
echo ""

# ── 3e. Product category ─────────────────────────────────────────
echo "--- 3e. Product category classification ---"
python -m magazineads.product_category.predict \
    --model_path "$PRODUCT_MODEL_PATH" \
    --predictions_csv "$DATA_ROOT/page_classification_predictions/predictions_all.csv" \
    --output_csv "$DATA_ROOT/product_category_predictions.csv"
echo ""

# ── 3f. Person detection ─────────────────────────────────────────
echo "--- 3f. Person detection (YOLOv8l) ---"
for mag in $MAGAZINES; do
    python -m magazineads.people.infer_people \
        --image_dir "$DATA_ROOT/images/$mag" \
        --output_csv "$DATA_ROOT/people_predictions/$mag.csv"
done
echo ""

# ── 3g. Face detection ───────────────────────────────────────────
echo "--- 3g. Face detection (SCRFD / InsightFace) ---"
for mag in $MAGAZINES; do
    python -m magazineads.people.infer_faces \
        --image_dir "$DATA_ROOT/images/$mag" \
        --output_csv "$DATA_ROOT/face_predictions/$mag.csv"
done
echo ""

echo "=== Annotation complete ==="
echo "Predictions are in: $DATA_ROOT/{page_classification_predictions,yolo_predictions,...}/"
echo "Next step: bash scripts/04_analyze.sh"
