#!/usr/bin/env bash
# ────────────────────────────────────────────────────────────────
# bundle.sh — Package the fstt-priors SDK + model into a tarball
#
# Usage:
#   ./bundle.sh <model_dir> [output_name]
#
# Example:
#   ./bundle.sh ../models/retrieval_minilm_l3_user_only_20260211_001736
#   # produces: fstt-priors-1.0.0.tar.gz
#
#   ./bundle.sh ../models/retrieval_minilm_l3_user_only_20260211_001736 fstt-priors-shokudo
#   # produces: fstt-priors-shokudo.tar.gz
# ────────────────────────────────────────────────────────────────
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
MODEL_DIR="${1:?Usage: $0 <model_dir> [output_name]}"
OUTPUT_NAME="${2:-fstt-priors-1.0.0}"

# Validate model directory
if [ ! -d "$MODEL_DIR/best_model" ]; then
    echo "ERROR: $MODEL_DIR/best_model not found" >&2
    exit 1
fi
if [ ! -d "$MODEL_DIR/shared_index" ]; then
    echo "ERROR: $MODEL_DIR/shared_index not found" >&2
    exit 1
fi
if [ ! -f "$MODEL_DIR/shared_index/candidates.json" ]; then
    echo "ERROR: $MODEL_DIR/shared_index/candidates.json not found" >&2
    exit 1
fi

# Create staging directory
STAGING=$(mktemp -d)
BUNDLE_DIR="$STAGING/$OUTPUT_NAME"
mkdir -p "$BUNDLE_DIR"

echo "Bundling into $OUTPUT_NAME.tar.gz ..."

# 1. Copy the Python package
cp -r "$SCRIPT_DIR/pyproject.toml" "$BUNDLE_DIR/"
cp -r "$SCRIPT_DIR/fstt_priors" "$BUNDLE_DIR/"
cp -r "$SCRIPT_DIR/README.md" "$BUNDLE_DIR/" 2>/dev/null || true

# 2. Copy model artifacts
MODEL_DEST="$BUNDLE_DIR/model"
mkdir -p "$MODEL_DEST"
cp -r "$MODEL_DIR/best_model" "$MODEL_DEST/"
mkdir -p "$MODEL_DEST/shared_index"
cp "$MODEL_DIR/shared_index/candidates.json" "$MODEL_DEST/shared_index/"
# Copy FAISS index if it exists
cp "$MODEL_DIR/shared_index/index.faiss" "$MODEL_DEST/shared_index/" 2>/dev/null || true
# Copy sklearn index if it exists
cp "$MODEL_DIR/shared_index/index.pkl" "$MODEL_DEST/shared_index/" 2>/dev/null || true
# Copy meta.json if it exists
cp "$MODEL_DIR/shared_index/meta.json" "$MODEL_DEST/shared_index/" 2>/dev/null || true

# 3. Create the tarball
tar -czf "$SCRIPT_DIR/$OUTPUT_NAME.tar.gz" -C "$STAGING" "$OUTPUT_NAME"

# 4. Cleanup
rm -rf "$STAGING"

SIZE=$(du -h "$SCRIPT_DIR/$OUTPUT_NAME.tar.gz" | cut -f1)
echo ""
echo "Done! Created: $SCRIPT_DIR/$OUTPUT_NAME.tar.gz ($SIZE)"
echo ""
echo "To install on the production server:"
echo "  tar xzf $OUTPUT_NAME.tar.gz"
echo "  cd $OUTPUT_NAME"
echo "  pip install ."
echo ""
echo "Then in Python:"
echo '  from fstt_priors import PriorPredictor'
echo '  predictor = PriorPredictor("model")'
echo '  terms = predictor.predict("SYSTEM: Hi\nUSER: I want ramen")'
