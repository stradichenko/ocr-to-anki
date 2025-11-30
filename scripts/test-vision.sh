#!/usr/bin/env bash

set -euo pipefail

echo "Testing llama.cpp vision capabilities with Gemma 3 4B"
echo "======================================================"
echo

# Check if server is running
if ! curl -s http://127.0.0.1:8080/health >/dev/null 2>&1; then
    echo "❌ Server not running. Start it first with:"
    echo "   python src/llama_cpp_server.py"
    exit 1
fi

echo "✅ Server is running"
echo

# Check if we have an image
IMAGE_PATH="${1:-data/images/handwritten.jpeg}"
if [ ! -f "$IMAGE_PATH" ]; then
    echo "❌ Image not found: $IMAGE_PATH"
    echo "Usage: $0 [image_path]"
    exit 1
fi

echo "📷 Testing with image: $IMAGE_PATH"

# Get image dimensions for debugging
if command -v identify >/dev/null 2>&1; then
    IMAGE_INFO=$(identify -format "%wx%h %B" "$IMAGE_PATH" 2>/dev/null || echo "unknown")
    echo "   Dimensions: $IMAGE_INFO"
fi
echo

# Encode image to base64
IMAGE_BASE64=$(base64 -w 0 "$IMAGE_PATH")

# Test 1: Using [img-1] token (llama.cpp standard)
echo "Test 1: Using [img-1] token format"
echo "-----------------------------------"
cat > /tmp/vision_request1.json <<EOF
{
  "prompt": "<bos><start_of_turn>user\n[img-1]\nExtract all visible text from this image. List each word or phrase on a separate line.<end_of_turn>\n<start_of_turn>model\n",
  "image_data": [{"data": "${IMAGE_BASE64}"}],
  "n_predict": 512,
  "temperature": 0.1,
  "stop": ["<end_of_turn>", "<eos>"],
  "cache_prompt": false
}
EOF

echo "🚀 Sending vision request..."
echo "   (This may take 30-60 seconds on CPU)"
echo

RESPONSE=$(curl -X POST http://127.0.0.1:8080/completion \
  -H "Content-Type: application/json" \
  -d @/tmp/vision_request1.json \
  --max-time 300 \
  -s)

if [ -n "$RESPONSE" ]; then
    echo "✅ Response received!"
    echo
    echo "OCR Result:"
    echo "==========="
    echo "$RESPONSE" | python -c "import sys, json; print(json.load(sys.stdin).get('content', 'No content'))"
else
    echo "❌ No response from server"
fi

echo
echo "-----------------------------------"
echo

# Test 2: Alternative format with explicit image instruction
echo "Test 2: Explicit image instruction"
echo "-----------------------------------"
cat > /tmp/vision_request2.json <<EOF
{
  "prompt": "<bos><start_of_turn>user\n[img-1]\nLook at this handwritten image carefully. What text do you see written? Please extract every word exactly as written.<end_of_turn>\n<start_of_turn>model\n",
  "image_data": [{"data": "${IMAGE_BASE64}"}],
  "n_predict": 512,
  "temperature": 0.1,
  "stop": ["<end_of_turn>", "<eos>"],
  "cache_prompt": false
}
EOF

RESPONSE2=$(curl -X POST http://127.0.0.1:8080/completion \
  -H "Content-Type: application/json" \
  -d @/tmp/vision_request2.json \
  --max-time 300 \
  -s)

if [ -n "$RESPONSE2" ]; then
    echo "✅ Response received!"
    echo
    echo "OCR Result:"
    echo "==========="
    echo "$RESPONSE2" | python -c "import sys, json; print(json.load(sys.stdin).get('content', 'No content'))"
else
    echo "❌ No response from server"
fi

# Clean up
rm -f /tmp/vision_request1.json /tmp/vision_request2.json

echo
echo "-----------------------------------"
echo "Note: If the model is hallucinating instead of reading the actual image,"
echo "it may mean the vision projector isn't being used correctly."
echo "Check the server logs for any warnings about image processing."
