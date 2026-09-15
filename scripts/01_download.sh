#!/usr/bin/env bash
# ═══════════════════════════════════════════════════════════════════
# Step 1: Download raw magazine scans from the Internet Archive
# ═══════════════════════════════════════════════════════════════════
#
# Uses the ``internetarchive`` Python library to search for and download
# magazine issues.  Each magazine is downloaded as JP2 zip archives
# (preferred) or PDFs into its own subdirectory.
#
# NOTE: Time magazine's IA collection is not public and cannot be
# downloaded with this script.
#
# Usage:
#   bash scripts/01_download.sh              # all 4 public magazines
#   bash scripts/01_download.sh variety       # single magazine
#   bash scripts/01_download.sh --dry-run     # list items only
#
# Prerequisites:
#   pip install internetarchive
# ═══════════════════════════════════════════════════════════════════
set -euo pipefail

DATA_ROOT="${MAGAZINEADS_DATA_ROOT:-./data}"
MAGAZINES="${1:-esquire life forbes variety}"
DRY_RUN=""

# Check for --dry-run flag
for arg in "$@"; do
    if [ "$arg" = "--dry-run" ]; then
        DRY_RUN="--dry-run"
        MAGAZINES="${MAGAZINES/--dry-run/}"
    fi
done

echo "=== Step 1: Download raw magazine scans ==="
echo "Data root: $DATA_ROOT"
echo "Magazines: $MAGAZINES"
echo ""

for mag in $MAGAZINES; do
    echo "--- Downloading: $mag ---"
    python -m magazineads.scraping.ia_scraper \
        --magazine "$mag" \
        --output_dir "$DATA_ROOT/originals/$mag" \
        $DRY_RUN
    echo ""
done

echo "=== Download complete ==="
echo "Raw archives are in: $DATA_ROOT/originals/"
echo "Next step: bash scripts/02_convert.sh"
