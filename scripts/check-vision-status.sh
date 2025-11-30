#!/usr/bin/env bash

echo "Checking llama-server vision configuration"
echo "=========================================="
echo

# Check if server is running
if ! curl -s http://127.0.0.1:8080/health >/dev/null 2>&1; then
    echo "❌ Server not running"
    exit 1
fi

echo "1. Server properties:"
echo "--------------------"
curl -s http://127.0.0.1:8080/props | python -m json.tool

echo
echo "2. Model info:"
echo "-------------"
curl -s http://127.0.0.1:8080/v1/models | python -m json.tool 2>/dev/null || echo "Endpoint not available"

echo
echo "3. Slots info (shows multimodal status):"
echo "----------------------------------------"
curl -s http://127.0.0.1:8080/slots | python -m json.tool | grep -E "(multimodal|vision|clip|image)" || echo "No vision fields found"

echo
echo "4. Testing if server accepts image_data field:"
echo "----------------------------------------------"
# Create a tiny test image (1x1 pixel)
TINY_IMAGE="iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNkYPhfDwAChwGA60e6kgAAAABJRU5ErkJggg=="

cat > /tmp/test_image_field.json << EOF
{
  "prompt": "test",
  "image_data": [{"data": "${TINY_IMAGE}"}],
  "n_predict": 1
}
EOF

echo "Sending request with image_data..."
RESPONSE=$(curl -X POST http://127.0.0.1:8080/completion \
  -H "Content-Type: application/json" \
  -d @/tmp/test_image_field.json \
  --max-time 10 -s -w "\nHTTP Status: %{http_code}" 2>&1)

echo "$RESPONSE" | head -5
rm -f /tmp/test_image_field.json

echo
echo "5. Process info:"
echo "---------------"
ps aux | grep llama-server | grep -v grep | head -1

echo
echo "DIAGNOSIS:"
echo "=========="
if curl -s http://127.0.0.1:8080/props | grep -q '"vision":true'; then
    echo "✅ Vision is enabled"
else
    echo "❌ Vision is NOT enabled in server properties"
    echo
    echo "Possible issues:"
    echo "1. The --mmproj flag isn't being recognized"
    echo "2. The vision projector isn't compatible with this model"
    echo "3. llama-server needs different configuration"
fi
