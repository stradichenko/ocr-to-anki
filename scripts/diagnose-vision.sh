#!/usr/bin/env bash

set -euo pipefail

LOGFILE="logs/vision_diagnosis_$(date +%Y%m%d_%H%M%S).log"
mkdir -p logs

echo "Vision Model Diagnostic Tool"
echo "============================"
echo "Logging to: $LOGFILE"
echo

# Function to extract key info from logs
extract_key_info() {
    echo "=== Key Information ===" >> "$LOGFILE"
    echo >> "$LOGFILE"
    
    # Check for vision/multimodal keywords
    echo "Vision/Multimodal References:" >> "$LOGFILE"
    grep -i "vision\|multimodal\|clip\|image\|mmproj" "$LOGFILE" | head -20 >> "${LOGFILE}.summary" 2>/dev/null || echo "None found" >> "${LOGFILE}.summary"
    
    # Check for image processing
    echo -e "\nImage Processing:" >> "${LOGFILE}.summary"
    grep -i "image_data\|img-1\|start_of_image" "$LOGFILE" | head -10 >> "${LOGFILE}.summary" 2>/dev/null || echo "None found" >> "${LOGFILE}.summary"
    
    # Check for errors
    echo -e "\nErrors/Warnings:" >> "${LOGFILE}.summary"
    grep -i "error\|warning\|failed" "$LOGFILE" | head -10 >> "${LOGFILE}.summary" 2>/dev/null || echo "None found" >> "${LOGFILE}.summary"
    
    # Check token processing
    echo -e "\nToken Processing:" >> "${LOGFILE}.summary"
    grep -i "tokens_evaluated\|tokens_predicted\|n_tokens" "$LOGFILE" | head -5 >> "${LOGFILE}.summary" 2>/dev/null || echo "None found" >> "${LOGFILE}.summary"
}

# 1. Kill existing server
echo "Stopping any existing server..."
pkill -f llama-server 2>/dev/null || true
sleep 2

# 2. Start server with full logging
echo "Starting llama-server with full logging..."
echo "=== SERVER START ===" >> "$LOGFILE"
date >> "$LOGFILE"

llama-server \
  --model ~/.cache/llama.cpp/models/gemma-3-4b-it-q4_0.gguf \
  --mmproj ~/.cache/llama.cpp/models/mmproj-model-f16-4B.gguf \
  --host 127.0.0.1 \
  --port 8080 \
  --ctx-size 2048 \
  --verbose \
  --log-disable false \
  2>&1 | tee -a "$LOGFILE" &

SERVER_PID=$!
echo "Server PID: $SERVER_PID"

# 3. Wait for server
echo "Waiting for server to be ready..."
for i in {1..30}; do
  if curl -s http://127.0.0.1:8080/health >/dev/null 2>&1; then
    echo "✅ Server ready!"
    break
  fi
  sleep 1
done

# 4. Check server properties
echo -e "\n=== SERVER PROPERTIES ===" >> "$LOGFILE"
curl -s http://127.0.0.1:8080/props 2>&1 | tee -a "$LOGFILE" | python -m json.tool | grep -E "(vision|multimodal|clip|image)" || true

# 5. Prepare test image
IMAGE_PATH="${1:-data/images/handwritten.jpeg}"
if [ ! -f "$IMAGE_PATH" ]; then
    echo "Creating test image with clear text..."
    # Create a simple test image with ImageMagick if available
    if command -v convert >/dev/null 2>&1; then
        convert -size 400x100 xc:white \
                -pointsize 30 -fill black \
                -gravity center -annotate +0+0 "Hello World Test" \
                /tmp/test_image.jpg
        IMAGE_PATH="/tmp/test_image.jpg"
        echo "Created test image: $IMAGE_PATH"
    else
        echo "❌ No image found and ImageMagick not available"
        kill $SERVER_PID
        exit 1
    fi
fi

IMAGE_BASE64=$(base64 -w 0 "$IMAGE_PATH")

# 6. Test different prompt formats
echo -e "\n=== VISION TESTS ===" >> "$LOGFILE"

# Test 1: Standard format
echo "Test 1: Standard <start_of_image> format" | tee -a "$LOGFILE"
cat > /tmp/test1.json << EOF
{
  "prompt": "<bos><start_of_turn>user\n<start_of_image>\nWhat text do you see?\n<end_of_turn>\n<start_of_turn>model\n",
  "image_data": [{"data": "${IMAGE_BASE64}"}],
  "n_predict": 100,
  "temperature": 0.1
}
EOF

echo -e "\n--- Request 1 ---" >> "$LOGFILE"
RESPONSE1=$(curl -X POST http://127.0.0.1:8080/completion \
  -H "Content-Type: application/json" \
  -d @/tmp/test1.json \
  --max-time 60 -v 2>&1 | tee -a "$LOGFILE")

echo -e "\n--- Response 1 Content ---" >> "$LOGFILE"
echo "$RESPONSE1" | grep -o '"content":"[^"]*"' >> "$LOGFILE" || echo "No content found" >> "$LOGFILE"

# Test 2: Check if image is being processed
echo -e "\nTest 2: Checking image processing" | tee -a "$LOGFILE"

# Look for image-related processing in the last 100 lines of the log
echo -e "\n--- Image Processing Check ---" >> "$LOGFILE"
tail -100 "$LOGFILE" | grep -i "image\|clip\|vision" >> "${LOGFILE}.imagecheck" 2>/dev/null || echo "No image processing found" >> "${LOGFILE}.imagecheck"

# 7. Kill server
echo -e "\nStopping server..."
kill $SERVER_PID 2>/dev/null || true

# 8. Analyze logs
echo -e "\n=== ANALYSIS ===" | tee -a "$LOGFILE"
extract_key_info

# 9. Generate summary
echo
echo "Diagnostic Summary"
echo "=================="
echo

if grep -q "clip_model_load\|mmproj.*loaded" "$LOGFILE"; then
    echo "✅ Vision projector loaded successfully"
else
    echo "❌ Vision projector may not be loaded"
fi

if grep -q "multimodal.*true\|vision.*true" "$LOGFILE"; then
    echo "✅ Server reports multimodal/vision support"
else
    echo "❌ Server doesn't report vision support"
fi

if grep -q "image_data.*processed\|clip.*processed" "$LOGFILE"; then
    echo "✅ Image data appears to be processed"
else
    echo "⚠️  Image data might not be processed"
fi

echo
echo "Key findings saved to: ${LOGFILE}.summary"
echo "Full log saved to: $LOGFILE"
echo
echo "To view key information:"
echo "  cat ${LOGFILE}.summary"
echo
echo "To search for specific patterns:"
echo "  grep -i 'pattern' $LOGFILE"
