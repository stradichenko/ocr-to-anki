#!/usr/bin/env bash
set -euo pipefail

# Setup llama.cpp with Google Gemma 4 E2B -- fully offline, no HuggingFace CLI needed.
# Downloads model + vision projector via direct GGUF URLs (wget/curl).

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# ------------------------------------------------------------------
# Model configuration -- direct download URLs (public, no auth)
# ------------------------------------------------------------------
# Gemma 4 E2B, the same model the shipped registry offers as its default, so
# desktop and Android run one model rather than two. Gemma 4 is Apache-2.0,
# unlike Gemma 3's custom terms. Keep these values in step with the first entry
# of app/assets/models.json: the backend takes its filenames from the client,
# and a mismatch here means the model it is told to load is never downloaded.
MODEL_FILE="gemma-4-e2b-it-q4_0.gguf"
MMPROJ_FILE="mmproj-gemma-4-e2b-f16.gguf"
MODELS_DIR="${LLAMA_CPP_MODELS:-$HOME/.cache/llama.cpp/models}"
MODEL_PATH="${MODELS_DIR}/${MODEL_FILE}"
MMPROJ_PATH="${MODELS_DIR}/${MMPROJ_FILE}"

# Public direct-download URLs -- no authentication required.
# Note the mmproj is saved under a name different from its URL path, matching
# what the registry records, so the two stay interchangeable.
MODEL_URL="https://huggingface.co/unsloth/gemma-4-E2B-it-GGUF/resolve/0314792d7f1f7e229411f620751375812bb9faf2/gemma-4-E2B-it-Q4_0.gguf"
MMPROJ_URL="https://huggingface.co/unsloth/gemma-4-E2B-it-GGUF/resolve/0314792d7f1f7e229411f620751375812bb9faf2/mmproj-F16.gguf"

echo "=== llama.cpp Gemma 4 E2B Setup (Fully Offline) ==="
echo ""
echo "Model:    ${MODEL_FILE}  (~3.0 GB)"
echo "Vision:   ${MMPROJ_FILE} (~986 MB)"
echo "Location: ${MODELS_DIR}"
echo ""

# ------------------------------------------------------------------
# Pick a download tool (wget preferred, curl fallback)
# ------------------------------------------------------------------
download() {
    local url="$1"
    local dest="$2"
    local tmp="${dest}.part"

    if command -v wget >/dev/null 2>&1; then
        wget --continue --progress=bar:force -O "$tmp" "$url" && mv "$tmp" "$dest"
    elif command -v curl >/dev/null 2>&1; then
        curl -L -C - --progress-bar -o "$tmp" "$url" && mv "$tmp" "$dest"
    else
        echo "[ERR] Neither wget nor curl found. Install one and retry."
        exit 1
    fi
}

# ------------------------------------------------------------------
# Create models directory
# ------------------------------------------------------------------
mkdir -p "$MODELS_DIR"

# ------------------------------------------------------------------
# Download main model
# ------------------------------------------------------------------
if [ -f "$MODEL_PATH" ]; then
    MODEL_SIZE=$(du -h "$MODEL_PATH" | cut -f1)
    echo "[OK] Model already exists ($MODEL_SIZE)"
else
    echo "Downloading model: ${MODEL_FILE}..."
    echo "  From: ${MODEL_URL}"
    echo ""
    download "$MODEL_URL" "$MODEL_PATH" || {
        echo ""
        echo "[ERR] Download failed."
        echo "   If the URL is gated, download the file manually and place it at:"
        echo "   ${MODEL_PATH}"
        rm -f "${MODEL_PATH}.part"
        exit 1
    }
    MODEL_SIZE=$(du -h "$MODEL_PATH" | cut -f1)
    echo "[OK] Model downloaded ($MODEL_SIZE)"
fi

# ------------------------------------------------------------------
# Download vision projector
# ------------------------------------------------------------------
echo ""
if [ -f "$MMPROJ_PATH" ]; then
    MMPROJ_SIZE=$(du -h "$MMPROJ_PATH" | cut -f1)
    echo "[OK] Vision projector already exists ($MMPROJ_SIZE)"
else
    echo "Downloading vision projector: ${MMPROJ_FILE}..."
    echo "  From: ${MMPROJ_URL}"
    echo ""
    download "$MMPROJ_URL" "$MMPROJ_PATH" || {
        echo ""
        echo "[WARN] Vision projector download failed"
        echo "   Vision OCR will not be available."
        echo "   Text-only mode will still work."
        echo "   You can download manually and place at:"
        echo "   ${MMPROJ_PATH}"
        rm -f "${MMPROJ_PATH}.part"
    }
fi

# ------------------------------------------------------------------
# Summary
# ------------------------------------------------------------------
echo ""
echo "═══════════════════════════════════════════════════"
echo "  Setup Summary"
echo "═══════════════════════════════════════════════════"

if [ -f "$MODEL_PATH" ]; then
    MODEL_SIZE=$(du -h "$MODEL_PATH" | cut -f1)
    echo "  [OK] Model:   $MODEL_PATH ($MODEL_SIZE)"
else
    echo "  [ERR] Model:   NOT FOUND"
fi

if [ -f "$MMPROJ_PATH" ]; then
    MMPROJ_SIZE=$(du -h "$MMPROJ_PATH" | cut -f1)
    echo "  [OK] Vision:  $MMPROJ_PATH ($MMPROJ_SIZE)"
    echo ""
    echo "  Vision OCR:  ENABLED"
else
    echo "  [WARN] Vision:  NOT FOUND"
    echo ""
    echo "  Vision OCR:  DISABLED (text-only mode)"
fi

echo ""
echo "  No HuggingFace account needed."
echo "  No internet needed after this setup."
echo ""
echo "═══════════════════════════════════════════════════"
echo ""
echo "Next steps:"
echo "  # Start the text server"
echo "  python src/backends/llama_cpp_server.py"
echo ""
echo "  # Run vision OCR on an image"
echo "  python -m backends.mtmd_cli data/images/handwritten.jpeg"
echo ""
echo "  # Start the FastAPI server"
echo "  uvicorn api.app:app --host 0.0.0.0 --port 8000"
echo ""
