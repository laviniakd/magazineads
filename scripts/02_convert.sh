#!/usr/bin/env bash
# ═══════════════════════════════════════════════════════════════════
# Step 2: Convert downloaded archives to per-page JPEGs
# ═══════════════════════════════════════════════════════════════════
#
# Converts JP2 zip archives (or PDFs) into individual JPEG images,
# one per page. Two-page spread scans are split vertically when
# --split_pages is used (required for magazines whose scans contain
# two pages per image, e.g., Esquire).
#
# Output naming convention: {issue_stem}-{page:03d}.jpg
#
# Usage:
#   bash scripts/02_convert.sh                # all magazines
#   bash scripts/02_convert.sh variety         # single magazine
#
# Prerequisites:
#   - pdftoppm (from Poppler) for PDF conversion
#   - Pillow for JP2 conversion
# ═══════════════════════════════════════════════════════════════════
set -euo pipefail

DATA_ROOT="${MAGAZINEADS_DATA_ROOT:-./data}"
MAGAZINES="${1:-esquire life forbes variety}"
NUM_PROCESSES="${NUM_PROCESSES:-8}"

# Per-magazine conversion settings
# Format: "magazine:format:split"
# Life scans are two-page spreads that need splitting; others are single pages.
declare -A MAG_SETTINGS=(
    [esquire]="--jp2"
    [life]="--jp2 --split_pages"
    [forbes]="--jp2"
    [variety]="--jp2"
)

echo "=== Step 2: Convert archives to per-page JPEGs ==="
echo "Data root: $DATA_ROOT"
echo "Magazines: $MAGAZINES"
echo "Workers:   $NUM_PROCESSES"
echo ""

for mag in $MAGAZINES; do
    echo "--- Converting: $mag ---"
    settings="${MAG_SETTINGS[$mag]:-}"
    python -m magazineads.convert.to_jpeg \
        --magazine "$mag" \
        --data_path "$DATA_ROOT" \
        --num_processes "$NUM_PROCESSES" \
        $settings
    echo ""
done

echo "=== Conversion complete ==="
echo "Page JPEGs are in: $DATA_ROOT/images/{magazine}/"
echo "Next step: bash scripts/03_annotate.sh"
