#!/usr/bin/env bash

set -euo pipefail

echo "Setting up Tesseract OCR as reliable offline fallback"
echo "====================================================="
echo

# Check Tesseract is installed (from Nix)
if ! command -v tesseract >/dev/null 2>&1; then
    echo "❌ Tesseract not found. Enter nix shell first:"
    echo "   nix develop"
    exit 1
fi

echo "✅ Tesseract installed: $(tesseract --version 2>&1 | head -1)"

# Test Tesseract on the image
IMAGE="${1:-data/images/handwritten.jpeg}"

if [ -f "$IMAGE" ]; then
    echo
    echo "Testing Tesseract OCR on: $IMAGE"
    echo "---------------------------------"
    
    # Run OCR
    tesseract "$IMAGE" stdout 2>/dev/null | tee /tmp/tesseract_output.txt
    
    echo
    echo "---------------------------------"
    echo "✅ Tesseract extracted $(wc -w < /tmp/tesseract_output.txt) words"
else
    echo "⚠️  No test image found at: $IMAGE"
fi

echo
echo "Tesseract + Gemma 3 Pipeline:"
echo "=============================="
echo "1. OCR with Tesseract (fast, reliable):"
echo "   tesseract image.jpg stdout > words.txt"
echo
echo "2. Enrich with Gemma 3 (definitions, examples):"
echo "   python src/vocabulary_enricher.py words.txt"
echo
echo "3. Create Anki cards:"
echo "   python src/anki_exporter.py enriched.json"
echo
echo "This pipeline is 100% offline after model download!"
