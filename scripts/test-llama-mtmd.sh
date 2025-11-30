#!/usr/bin/env bash

set -euo pipefail

echo "Testing llama-mtmd-cli setup"
echo "============================"
echo

MODEL_DIR="$HOME/.cache/llama.cpp/models"
MODEL="$MODEL_DIR/gemma-3-4b-it-q4_0.gguf"
MMPROJ="$MODEL_DIR/mmproj-model-f16-4B.gguf"

# Check binary
echo "1. Checking binary..."
if command -v llama-mtmd-cli >/dev/null 2>&1; then
    echo "✅ llama-mtmd-cli found at: $(which llama-mtmd-cli)"
    llama-mtmd-cli --version 2>&1 | head -5 || echo "  (version check failed)"
else
    echo "❌ llama-mtmd-cli not found"
    echo "   Build with: ./scripts/build-llama-gemma3-cli.sh"
    exit 1
fi

# Check model files
echo
echo "2. Checking model files..."
if [ -f "$MODEL" ]; then
    echo "✅ Model: $MODEL ($(du -h "$MODEL" | cut -f1))"
else
    echo "❌ Model not found: $MODEL"
    exit 1
fi

if [ -f "$MMPROJ" ]; then
    echo "✅ Vision: $MMPROJ ($(du -h "$MMPROJ" | cut -f1))"
else
    echo "❌ Vision projector not found: $MMPROJ"
    exit 1
fi

# Test text generation - llama-mtmd-cli REQUIRES mmproj even for text
echo
echo "3. Testing text generation (with required mmproj)..."
echo "   Command: llama-mtmd-cli -m MODEL --mmproj MMPROJ -p 'Hi' -n 5"

llama-mtmd-cli \
    -m "$MODEL" \
    --mmproj "$MMPROJ" \
    -p "Hi" \
    -n 5 \
    2>&1 | head -30 || {
    echo
    echo "⚠️  llama-mtmd-cli failed!"
    echo
    echo "For text-only, use standard llama-cli instead:"
    echo "  llama-cli -m \"$MODEL\" -p 'Hi' -n 5"
    exit 1
}

echo
echo "✅ Text generation works!"

# Test vision 
echo
echo "4. Testing vision..."

TEST_IMG="data/images/handwritten.jpeg"
if [ -f "$TEST_IMG" ]; then
    echo "   Using image: $TEST_IMG"
    echo "   Command: llama-mtmd-cli -m MODEL --mmproj MMPROJ --image IMG -p 'What text do you see?' -n 50"
    
    llama-mtmd-cli \
        -m "$MODEL" \
        --mmproj "$MMPROJ" \
        --image "$TEST_IMG" \
        -p "What text do you see?" \
        -n 50 \
        2>&1 | head -50 || echo "  Vision test failed"
else
    echo "   No test image found at: $TEST_IMG"
fi

echo
echo "✅ Done!"
