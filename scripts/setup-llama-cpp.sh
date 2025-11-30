#!/usr/bin/env bash
set -euo pipefail

# Setup llama.cpp with Google Gemma 3 4B + AUTO-DOWNLOAD vision projector
# Uses llama-server's built-in HuggingFace integration

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Model configuration
HF_REPO="google/gemma-3-4b-it-qat-q4_0-gguf"
MODEL_FILE="gemma-3-4b-it-q4_0.gguf"
MODELS_DIR="${LLAMA_CPP_MODELS:-$HOME/.cache/llama.cpp/models}"
MODEL_PATH="${MODELS_DIR}/${MODEL_FILE}"

echo "=== llama.cpp Gemma 3 4B Setup (With Auto Vision) ==="
echo ""
echo "Repository: ${HF_REPO}"
echo "Model: ${MODEL_FILE} (~3.16 GB)"
echo "Vision: Auto-downloaded by llama-server when needed"
echo ""
echo "⚠️  IMPORTANT: HuggingFace Authentication Required"
echo "   1. Log in: huggingface-cli login"
echo "   2. Accept terms: https://huggingface.co/${HF_REPO}"
echo ""

# Create models directory
mkdir -p "$MODELS_DIR"

# Check for huggingface-cli
if ! command -v huggingface-cli >/dev/null 2>&1; then
    echo "❌ Error: huggingface-cli not found"
    echo "   Should be available in Nix environment"
    exit 1
fi

# Check if logged in
if ! huggingface-cli whoami >/dev/null 2>&1; then
    echo "❌ Not logged in to HuggingFace"
    echo ""
    echo "Please log in first:"
    echo "  huggingface-cli login"
    echo ""
    echo "Then accept Gemma terms at:"
    echo "  https://huggingface.co/${HF_REPO}"
    exit 1
fi

echo "✓ Logged in as: $(huggingface-cli whoami)"
echo ""

# Download ONLY the main model file
if [ -f "$MODEL_PATH" ]; then
    MODEL_SIZE=$(du -h "$MODEL_PATH" | cut -f1)
    echo "✓ Model already exists ($MODEL_SIZE)"
else
    echo "Downloading model: ${MODEL_FILE}..."
    huggingface-cli download \
        "${HF_REPO}" \
        "${MODEL_FILE}" \
        --local-dir "${MODELS_DIR}" \
        --local-dir-use-symlinks False || {
        echo ""
        echo "❌ Download failed. Ensure you:"
        echo "  1. Are logged in: huggingface-cli login"
        echo "  2. Accepted terms: https://huggingface.co/${HF_REPO}"
        exit 1
    }
fi

# Download vision projector for local use
MMPROJ_FILE="mmproj-model-f16-4B.gguf"
MMPROJ_PATH="${MODELS_DIR}/${MMPROJ_FILE}"

if [ -f "$MMPROJ_PATH" ]; then
    MMPROJ_SIZE=$(du -h "$MMPROJ_PATH" | cut -f1)
    echo "✓ Vision projector already exists ($MMPROJ_SIZE)"
else
    echo ""
    echo "Downloading vision projector: ${MMPROJ_FILE}..."
    huggingface-cli download \
        "${HF_REPO}" \
        "${MMPROJ_FILE}" \
        --local-dir "${MODELS_DIR}" \
        --local-dir-use-symlinks False || {
        echo ""
        echo "⚠️  Vision projector download failed"
        echo "   Vision OCR will not be available"
        echo "   Text-only mode will still work"
    }
fi

# Verify downloads
if [ -f "$MODEL_PATH" ]; then
    MODEL_SIZE=$(du -h "$MODEL_PATH" | cut -f1)
    
    echo ""
    echo "✅ Setup complete!"
    echo ""
    echo "Model files:"
    echo "  📦 $MODEL_PATH ($MODEL_SIZE)"
    
    if [ -f "$MMPROJ_PATH" ]; then
        MMPROJ_SIZE=$(du -h "$MMPROJ_PATH" | cut -f1)
        echo "  📦 $MMPROJ_PATH ($MMPROJ_SIZE)"
        echo ""
        echo "✅ Vision support enabled!"
    else
        echo ""
        echo "⚠️  Vision projector not available"
        echo "   Text-only mode will work"
    fi
else
    echo "❌ Download failed"
    exit 1
fi

echo ""
echo "=== Next Steps ==="
echo ""
echo "1. Build vision CLI:"
echo "   ./scripts/build-llama-gemma3-cli.sh"
echo ""
echo "2. Test vision OCR:"
echo "   python src/vision_ocr_gemma3.py data/images/handwritten.jpeg"
echo ""
