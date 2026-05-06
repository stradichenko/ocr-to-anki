#!/usr/bin/env bash
set -euo pipefail

# Generate checksums.json for the AI model files.
# Usage: ./scripts/generate-checksums.sh
#
# Reads model files from ~/.cache/llama.cpp/models/ and writes
# app/assets/checksums.json with SHA-256 hashes and sizes.

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
MODELS_DIR="${LLAMA_CPP_MODELS:-$HOME/.cache/llama.cpp/models}"
OUTPUT="$PROJECT_DIR/app/assets/checksums.json"

if [[ ! -d "$MODELS_DIR" ]]; then
    echo "[ERR] Model directory not found: $MODELS_DIR"
    echo "   Run ./scripts/setup-llama-cpp.sh first to download models."
    exit 1
fi

echo ":: Computing checksums from $MODELS_DIR..."

declare -A files=(
    ["gemma-3-4b-it-q4_0_s.gguf"]=""
    ["mmproj-model-f16-4B.gguf"]=""
)

json='{'
first=true
for name in "${!files[@]}"; do
    path="$MODELS_DIR/$name"
    if [[ ! -f "$path" ]]; then
        echo "[WARN] Missing model file: $path"
        continue
    fi

    size=$(stat -c %s "$path" 2>/dev/null || stat -f %z "$path" 2>/dev/null)
    hash=$(sha256sum "$path" 2>/dev/null | awk '{print $1}' || \
           shasum -a 256 "$path" 2>/dev/null | awk '{print $1}')

    if [[ "$first" == true ]]; then
        first=false
    else
        json+=","
    fi
    json+="\n  \"$name\": {\n    \"size\": $size,\n    \"sha256\": \"$hash\"\n  }"
    echo "   $name  $hash  ($size bytes)"
done
json+="\n}\n"

mkdir -p "$(dirname "$OUTPUT")"
printf "%b" "$json" > "$OUTPUT"
echo ""
echo "[OK] Written to $OUTPUT"
