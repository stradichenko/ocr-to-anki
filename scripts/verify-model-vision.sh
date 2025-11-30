#!/usr/bin/env bash

echo "Verifying Gemma 3 Vision Model Setup"
echo "====================================="
echo

MODEL_DIR="$HOME/.cache/llama.cpp/models"

# Check model files
echo "1. Model files present:"
echo "----------------------"
if [ -f "$MODEL_DIR/gemma-3-4b-it-q4_0.gguf" ]; then
    echo "✅ Main model: gemma-3-4b-it-q4_0.gguf"
    ls -lh "$MODEL_DIR/gemma-3-4b-it-q4_0.gguf" | awk '{print "   Size: " $5}'
else
    echo "❌ Main model not found"
fi

if [ -f "$MODEL_DIR/mmproj-model-f16-4B.gguf" ]; then
    echo "✅ Vision projector: mmproj-model-f16-4B.gguf"  
    ls -lh "$MODEL_DIR/mmproj-model-f16-4B.gguf" | awk '{print "   Size: " $5}'
else
    echo "❌ Vision projector not found"
fi

echo
echo "2. Testing llama-cli with vision:"
echo "---------------------------------"
# Create a simple test
echo "Testing direct llama-cli vision support..."

# Try using llama-cli with the vision flag
if command -v llama-cli >/dev/null 2>&1; then
    echo "Found llama-cli, checking version..."
    llama-cli --version 2>/dev/null || echo "Version not available"
    
    # Check if it has vision support
    llama-cli --help 2>/dev/null | grep -i "vision\|image\|mmproj" | head -5 || echo "No vision options found in llama-cli"
fi

echo
echo "3. Alternative: Use llama-gemma3-cli if available:"
echo "--------------------------------------------------"
if command -v llama-gemma3-cli >/dev/null 2>&1; then
    echo "✅ llama-gemma3-cli found (specialized for Gemma 3 vision)"
    llama-gemma3-cli --help 2>/dev/null | grep -i "image" | head -3
else
    echo "❌ llama-gemma3-cli not found"
    echo "   This might be required for Gemma 3 vision support"
fi

echo
echo "RECOMMENDATION:"
echo "=============="
echo "The issue is that the standard llama-server doesn't recognize"
echo "the image data even with --mmproj loaded."
echo
echo "Solutions:"
echo "1. Use Tesseract for OCR instead (reliable, fast):"
echo "   tesseract data/images/handwritten.jpeg stdout"
echo
echo "2. Use the model for text enrichment only:"
echo "   - OCR with Tesseract"
echo "   - Enrich vocabulary with Gemma 3 text model"
echo
echo "3. Try Ollama with vision support (requires internet):"
echo "   ollama pull llava"
