#!/usr/bin/env bash
# ═══════════════════════════════════════════════════════════════════
# Step 4: Generate analysis plots and tables for the paper
# ═══════════════════════════════════════════════════════════════════
#
# Runs the plotting and analysis scripts that produce publication-
# quality figures.  All output goes to paperplots/ and finalplots/
# at the project root.
#
# This step requires the model predictions from Step 3 to already
# exist on disk.
#
# Usage:
#   bash scripts/04_analyze.sh               # all sections
#   bash scripts/04_analyze.sh coverage       # just dataset coverage
#   bash scripts/04_analyze.sh classifiers    # just classifier evals
#   bash scripts/04_analyze.sh people         # just people/faces
#
# No GPU required for this step.
# ═══════════════════════════════════════════════════════════════════
set -euo pipefail

DATA_ROOT="${MAGAZINEADS_DATA_ROOT:-./data}"
SECTION="${1:-all}"

echo "=== Step 4: Analysis and plotting ==="
echo "Section: $SECTION"
echo ""

# ── Paper plots (unified figure generator) ────────────────────────
echo "--- Running paper_plots.py (section: $SECTION) ---"
python -m magazineads.analysis.paper_plots --section "$SECTION"
echo ""

# ── Dataset coverage and classifier plots ─────────────────────────
if [ "$SECTION" = "all" ] || [ "$SECTION" = "classifiers" ]; then
    echo "--- Running make_paper_plots.py ---"
    python -m magazineads.analysis.make_paper_plots --section all \
        --predictions_csv "$DATA_ROOT/page_classification_predictions/predictions_all.csv"
    echo ""
fi

# ── Descriptive statistics ────────────────────────────────────────
if [ "$SECTION" = "all" ] || [ "$SECTION" = "coverage" ]; then
    echo "--- Running descriptive_stats.py ---"
    python -m magazineads.analysis.descriptive_stats
    echo ""
fi

echo "=== Analysis complete ==="
echo "Figures are in: paperplots/ and finalplots/"
