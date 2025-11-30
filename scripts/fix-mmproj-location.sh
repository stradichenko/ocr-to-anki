#!/usr/bin/env bash

set -euo pipefail

echo "Fixing mmproj Location for llama-server Auto-Detection"
echo "======================================================="
echo

MODEL_DIR="$HOME/.cache/llama.cpp/models"
LLAMA_CACHE="$HOME/.cache/llama.cpp"

# The mmproj we downloaded
SRC_MMPROJ="$MODEL_DIR/mmproj-model-f16-4B.gguf"

# Where llama-server expects to find it (next to the model)
# llama-server looks for mmproj files in the same directory as the model
# with specific naming patterns

echo "Current setup:"
ls -lh "$MODEL_DIR"/*.gguf 2>/dev/null || echo "  No GGUF files found"
echo

if [ ! -f "$SRC_MMPROJ" ]; then
    echo "❌ mmproj not found at: $SRC_MMPROJ"
    echo
    echo "Downloading mmproj from HuggingFace..."
    echo "(Requires authentication)"
    echo
    
    # Try to download with huggingface-cli
    if command -v huggingface-cli >/dev/null 2>&1; then
        huggingface-cli download \
            google/gemma-3-4b-it-qat-q4_0-gguf \
            mmproj-model-f16-4B.gguf \
            --local-dir "$MODEL_DIR" \
            --local-dir-use-symlinks False || {
            echo "❌ Download failed"
            echo "Please log in: huggingface-cli login"
            exit 1
        }
    else
        echo "❌ huggingface-cli not found"
        echo "Make sure you're in the Nix environment: nix develop"
        exit 1
    fi
fi

# Create symlink with name llama-server expects
# llama-server auto-detects mmproj if it has the same base name
MODEL_BASE=$(basename "$MODEL_DIR/gemma-3-4b-it-q4_0.gguf" .gguf)
DEST_MMPROJ="$MODEL_DIR/${MODEL_BASE}-mmproj-f16.gguf"

if [ -L "$DEST_MMPROJ" ] || [ -f "$DEST_MMPROJ" ]; then
    echo "Removing existing mmproj link/file..."
    rm -f "$DEST_MMPROJ"
fi

echo "Creating mmproj symlink for auto-detection..."
ln -s "$SRC_MMPROJ" "$DEST_MMPROJ"

echo
echo "✅ mmproj configured!"
echo
echo "Files:"
ls -lh "$MODEL_DIR"/*.gguf
echo
echo "llama-server will now auto-detect the vision projector."
echo
echo "Test it:"
echo "  python src/llama_cpp_server.py --verbose"
