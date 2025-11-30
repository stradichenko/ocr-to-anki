#!/usr/bin/env bash

set -euo pipefail

echo "Downloading Gemma 3 Vision Projector Only"
echo "=========================================="
echo

MODELS_DIR="${LLAMA_CPP_MODELS:-$HOME/.cache/llama.cpp/models}"
MMPROJ_PATH="$MODELS_DIR/mmproj-model-f16-4B.gguf"

# Check if already downloaded
if [ -f "$MMPROJ_PATH" ]; then
    SIZE=$(du -h "$MMPROJ_PATH" | cut -f1)
    echo "✅ Vision projector already exists ($SIZE)"
    echo "   Location: $MMPROJ_PATH"
    exit 0
fi

# Check for huggingface-cli
if ! command -v huggingface-cli >/dev/null 2>&1; then
    echo "❌ huggingface-cli not found"
    echo
    echo "You need to reload the Nix environment:"
    echo "  exit"
    echo "  nix develop"
    exit 1
fi

# Check login
if ! huggingface-cli whoami >/dev/null 2>&1; then
    echo "⚠️  Not logged in to HuggingFace"
    echo
    echo "Logging in..."
    huggingface-cli login || {
        echo "❌ Login failed"
        exit 1
    }
fi

echo "✅ Logged in as: $(huggingface-cli whoami)"
echo

# Download mmproj
echo "Downloading vision projector (851 MB)..."
echo "Repository: google/gemma-3-4b-it-qat-q4_0-gguf"
echo "File: mmproj-model-f16-4B.gguf"
echo

huggingface-cli download \
    google/gemma-3-4b-it-qat-q4_0-gguf \
    mmproj-model-f16-4B.gguf \
    --local-dir "$MODELS_DIR" \
    --local-dir-use-symlinks False || {
    echo
    echo "❌ Download failed"
    echo
    echo "Make sure you've accepted Gemma terms:"
    echo "  https://huggingface.co/google/gemma-3-4b-it-qat-q4_0-gguf"
    exit 1
}

echo
echo "✅ Vision projector downloaded!"
ls -lh "$MMPROJ_PATH"
echo
echo "Total llama.cpp models:"
ls -lh "$MODELS_DIR"/*.gguf
echo
echo "Next: Restart server to load vision support"
echo "  pkill llama-server"
echo "  python src/llama_cpp_server.py --verbose"
