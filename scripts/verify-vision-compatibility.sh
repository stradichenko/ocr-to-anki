#!/usr/bin/env bash

set -euo pipefail

echo "Verifying Vision Model Compatibility"
echo "===================================="
echo

# 1. Check llama-server version
echo "1. llama-server version:"
llama-server --version 2>&1 | head -5 || echo "  ⚠️  Version info not available"
echo

# 2. Check model files
echo "2. Model files:"
MODEL_DIR="$HOME/.cache/llama.cpp/models"

if [ -f "$MODEL_DIR/gemma-3-4b-it-q4_0.gguf" ]; then
    SIZE=$(du -h "$MODEL_DIR/gemma-3-4b-it-q4_0.gguf" | cut -f1)
    echo "  ✅ Main model: $SIZE"
else
    echo "  ❌ Main model missing"
fi

if [ -f "$MODEL_DIR/mmproj-model-f16-4B.gguf" ]; then
    SIZE=$(du -h "$MODEL_DIR/mmproj-model-f16-4B.gguf" | cut -f1)
    echo "  ✅ Vision projector: $SIZE"
else
    echo "  ❌ Vision projector missing"
fi
echo

# 3. Test llama-server with mmproj directly
echo "3. Testing llama-server with vision projector:"
echo "   Starting server with --mmproj..."

# Kill any existing server
pkill -9 llama-server 2>/dev/null || true
sleep 1

# Start server and capture first 100 lines
timeout 30 llama-server \
  --model "$MODEL_DIR/gemma-3-4b-it-q4_0.gguf" \
  --mmproj "$MODEL_DIR/mmproj-model-f16-4B.gguf" \
  --host 127.0.0.1 \
  --port 8080 \
  --ctx-size 2048 \
  --verbose 2>&1 | head -100 > /tmp/llama_server_startup.log &

SERVER_PID=$!
sleep 5

# Check if CLIP was loaded
echo
if grep -q "clip_model_load" /tmp/llama_server_startup.log; then
    echo "  ✅ CLIP vision encoder loaded successfully!"
    grep "clip_model_load" /tmp/llama_server_startup.log | head -3
else
    echo "  ❌ CLIP vision encoder NOT loaded"
    echo
    echo "  Server output (first 20 lines):"
    head -20 /tmp/llama_server_startup.log | sed 's/^/    /'
fi

# Kill test server
kill $SERVER_PID 2>/dev/null || true
pkill -9 llama-server 2>/dev/null || true

echo
echo "Full startup log saved to: /tmp/llama_server_startup.log"
echo

# 4. Diagnosis
echo "===================================="
echo "DIAGNOSIS:"
echo

if grep -q "clip_model_load" /tmp/llama_server_startup.log; then
    echo "✅ Vision support is working correctly"
    echo "   The hallucination issue is likely due to:"
    echo "   1. Incorrect prompt format"
    echo "   2. Image encoding issues"
    echo "   3. Model limitations with handwritten text"
else
    echo "❌ Vision projector is NOT being loaded"
    echo
    echo "Possible causes:"
    echo "  1. Incompatible mmproj file format"
    echo "  2. llama-server version doesn't support this projector"
    echo "  3. Model architecture mismatch"
    echo
    echo "Solutions:"
    echo "  1. Try downloading a different mmproj:"
    echo "     wget -O $MODEL_DIR/mmproj.gguf \\"
    echo "       https://huggingface.co/.../mmproj-gemma-3-4b.gguf"
    echo
    echo "  2. Use text-only mode with Tesseract for OCR:"
    echo "     tesseract image.jpg stdout | python src/vocabulary_enricher.py"
    echo
    echo "  3. Check llama.cpp GitHub for Gemma 3 vision examples"
fi

echo
echo "View full log:"
echo "  cat /tmp/llama_server_startup.log"
