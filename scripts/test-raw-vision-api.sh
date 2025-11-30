#!/usr/bin/env bash

set -euo pipefail

echo "Testing llama-server vision API directly"
echo "========================================"
echo

# Check server
if ! curl -s http://127.0.0.1:8080/health >/dev/null 2>&1; then
    echo "❌ Server not running"
    exit 1
fi

echo "✅ Server is running"
echo

# Create a tiny 64x64 test image with clear text
echo "Creating minimal test image..."
convert -size 64x64 xc:white \
        -pointsize 20 -fill black \
        -gravity center -annotate +0+0 "Hi" \
        /tmp/tiny_test.png 2>/dev/null || {
    echo "⚠️  ImageMagick not available, using existing image"
    cp data/images/handwritten.jpeg /tmp/tiny_test.png
}

IMAGE_BASE64=$(base64 -w 0 /tmp/tiny_test.png)
echo "Image size: $((${#IMAGE_BASE64} / 1024)) KB (base64)"
echo

# Test 1: Try WITHOUT image_data field (text only baseline)
echo "Test 1: Text-only (baseline)"
echo "----------------------------"
cat > /tmp/test_text.json << EOF
{
  "prompt": "<bos><start_of_turn>user\nHello<end_of_turn>\n<start_of_turn>model\n",
  "n_predict": 10,
  "temperature": 0.1
}
EOF

echo "Sending text-only request..."
START=$(date +%s)
RESPONSE=$(timeout 10 curl -X POST http://127.0.0.1:8080/completion \
  -H "Content-Type: application/json" \
  -d @/tmp/test_text.json \
  -s || echo "TIMEOUT")
END=$(date +%s)

if [ "$RESPONSE" != "TIMEOUT" ]; then
    echo "✅ Text response in $((END - START))s"
    echo "$RESPONSE" | python -c "import sys, json; print(json.load(sys.stdin).get('content', 'No content')[:50])"
else
    echo "❌ Text request timed out - server issue!"
    exit 1
fi

echo
echo "Test 2: With <start_of_image> token but NO image_data"
echo "-----------------------------------------------------"
cat > /tmp/test_token_only.json << EOF
{
  "prompt": "<bos><start_of_turn>user\n<start_of_image>\nHello<end_of_turn>\n<start_of_turn>model\n",
  "n_predict": 10,
  "temperature": 0.1
}
EOF

echo "Sending request with image token but no data..."
START=$(date +%s)
RESPONSE=$(timeout 10 curl -X POST http://127.0.0.1:8080/completion \
  -H "Content-Type: application/json" \
  -d @/tmp/test_token_only.json \
  -s || echo "TIMEOUT")
END=$(date +%s)

if [ "$RESPONSE" != "TIMEOUT" ]; then
    echo "✅ Response in $((END - START))s"
    echo "$RESPONSE" | python -c "import sys, json; print(json.load(sys.stdin).get('content', 'No content')[:50])"
else
    echo "⏱️ Timed out - image token causes hang"
fi

echo
echo "Test 3: With image_data field (full vision)"
echo "-------------------------------------------"
cat > /tmp/test_with_image.json << EOF
{
  "prompt": "<bos><start_of_turn>user\n<start_of_image>\nWhat?<end_of_turn>\n<start_of_turn>model\n",
  "image_data": [{"data": "${IMAGE_BASE64}"}],
  "n_predict": 5,
  "temperature": 0.1,
  "cache_prompt": false
}
EOF

echo "Sending request with actual image data..."
echo "⚠️  This is where it might hang..."
START=$(date +%s)
RESPONSE=$(timeout 60 curl -X POST http://127.0.0.1:8080/completion \
  -H "Content-Type: application/json" \
  -d @/tmp/test_with_image.json \
  -s -v 2>&1 || echo "TIMEOUT")
END=$(date +%s)

if echo "$RESPONSE" | grep -q "TIMEOUT"; then
    echo "❌ Timed out after 60s"
    echo
    echo "DIAGNOSIS:"
    echo "=========="
    echo "The image_data field causes the request to hang."
    echo "This suggests:"
    echo "1. The vision projector isn't being invoked correctly"
    echo "2. Or the GGUF model doesn't support the image_data API"
    echo "3. Or we need a different endpoint/format"
else
    echo "✅ Got response in $((END - START))s!"
    echo "$RESPONSE" | grep -o '"content":"[^"]*"' | head -1
fi

# Clean up
rm -f /tmp/test_*.json /tmp/tiny_test.png

echo
echo "CONCLUSION:"
echo "==========="
echo "If Test 1 works but Test 3 hangs, the issue is with how"
echo "llama-server processes the image_data field with this model."
echo
echo "Next steps:"
echo "1. Check llama-server source code for image_data handling"
echo "2. Try llama-cli instead of llama-server for vision"
echo "3. Or use Tesseract + Gemma 3 text-only pipeline"
