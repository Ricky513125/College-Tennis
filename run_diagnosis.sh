#!/bin/bash
# Quick script to run diagnosis from any location

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
OUTPUT_DIR="${1:-md_fed_outputs/stage1}"

echo "Script directory: $SCRIPT_DIR"
echo "Output directory: $OUTPUT_DIR"
echo ""

cd "$SCRIPT_DIR"

if [ ! -f "diagnose_training.py" ]; then
    echo "Error: diagnose_training.py not found in $SCRIPT_DIR"
    echo "Current directory contents:"
    ls -la | head -20
    exit 1
fi

echo "Running diagnosis..."
python diagnose_training.py "$OUTPUT_DIR"

echo ""
echo "Running prediction check..."
if [ -f "check_predictions.py" ]; then
    python check_predictions.py "$OUTPUT_DIR"
else
    echo "Warning: check_predictions.py not found"
fi
