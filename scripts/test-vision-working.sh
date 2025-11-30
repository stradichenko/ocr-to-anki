#!/usr/bin/env bash

set -euo pipefail

echo "Testing Gemma 3 Vision Model OCR"
echo "================================="
echo

# Check server is ready
echo "Checking server health..."
curl -s http://127.0.0.1:8080/health | python -c "import sys, json; print('✅ Server status:', json.load(sys.stdin).get('status', 'unknown'))"

# Check server properties to confirm multimodal
echo
echo "Server capabilities:"
curl -s http://127.0.0.1:8080/props | python -m json.tool | grep -E "(multimodal|clip|vision|image)" || true

IMAGE_PATH="${1:-data/images/handwritten.jpeg}"
if [ ! -f "$IMAGE_PATH" ]; then
    echo "❌ Image not found: $IMAGE_PATH"
    exit 1
fi

echo
echo "📷 Testing vision with: $IMAGE_PATH"
echo

# Encode image
IMAGE_BASE64=$(base64 -w 0 "$IMAGE_PATH")

# Test with different conversational prompts
echo "Test 1: Simple OCR request"
echo "---------------------------"
cat > /tmp/vision_test1.json << EOF
{
  "prompt": "<bos><start_of_turn>user\n<start_of_image>\nCan you do OCR of this?<end_of_turn>\n<start_of_turn>model\n",
  "image_data": [{"data": "${IMAGE_BASE64}"}],
  "n_predict": 512,
  "temperature": 0.1,
  "stop": ["<end_of_turn>"],
  "cache_prompt": false
}
EOF

RESPONSE1=$(curl -X POST http://127.0.0.1:8080/completion \
  -H "Content-Type: application/json" \
  -d @/tmp/vision_test1.json \
  --max-time 300 \
  -s)

echo "Response:"
echo "$RESPONSE1" | python -c "import sys, json; print(json.load(sys.stdin).get('content', 'No content'))"

echo
echo "Test 2: More specific request"
echo "-----------------------------"
cat > /tmp/vision_test2.json << EOF
{
  "prompt": "<bos><start_of_turn>user\n<start_of_image>\nWhat words are written in this handwritten note?<end_of_turn>\n<start_of_turn>model\n",
  "image_data": [{"data": "${IMAGE_BASE64}"}],
  "n_predict": 512,
  "temperature": 0.1,
  "stop": ["<end_of_turn>"],
  "cache_prompt": false
}
EOF

RESPONSE2=$(curl -X POST http://127.0.0.1:8080/completion \
  -H "Content-Type: application/json" \
  -d @/tmp/vision_test2.json \
  --max-time 300 \
  -s)

echo "Response:"
echo "$RESPONSE2" | python -c "import sys, json; print(json.load(sys.stdin).get('content', 'No content'))"

echo
echo "Test 3: Direct transcription request"
echo "------------------------------------"
cat > /tmp/vision_test3.json << EOF
{
  "prompt": "<bos><start_of_turn>user\n<start_of_image>\nPlease transcribe the handwritten text in this image word by word.<end_of_turn>\n<start_of_turn>model\n",
  "image_data": [{"data": "${IMAGE_BASE64}"}],
  "n_predict": 512,
  "temperature": 0.1,
  "stop": ["<end_of_turn>"],
  "cache_prompt": false
}
EOF

RESPONSE3=$(curl -X POST http://127.0.0.1:8080/completion \
  -H "Content-Type: application/json" \
  -d @/tmp/vision_test3.json \
  --max-time 300 \
  -s)

echo "Response:"
echo "$RESPONSE3" | python -c "import sys, json; print(json.load(sys.stdin).get('content', 'No content'))"

# Clean up
rm -f /tmp/vision_test*.json

echo
echo "-----------------------------------"
echo "Note: If the model is still hallucinating, the image might not be"
echo "properly encoded or the model might need fine-tuning for OCR tasks."
echo
echo "Alternative: Use Tesseract for pure OCR:"
echo "  tesseract $IMAGE_PATH stdout"
