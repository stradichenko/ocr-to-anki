#!/usr/bin/env bash

echo "Testing llama-server vision support directly"
echo "============================================="
echo

# Kill any existing server
pkill -f llama-server
sleep 2

echo "Starting server with vision projector..."
llama-server \
  --model ~/.cache/llama.cpp/models/gemma-3-4b-it-q4_0.gguf \
  --mmproj ~/.cache/llama.cpp/models/mmproj-model-f16-4B.gguf \
  --host 127.0.0.1 \
  --port 8080 \
  --ctx-size 2048 \
  --verbose \
  --parallel 1 &

SERVER_PID=$!
echo "Server PID: $SERVER_PID"

# Wait for server to start
echo "Waiting for server to be ready..."
for i in {1..30}; do
  if curl -s http://127.0.0.1:8080/health >/dev/null 2>&1; then
    echo "✅ Server ready!"
    break
  fi
  sleep 1
done

# Check server properties
echo
echo "Server properties:"
curl -s http://127.0.0.1:8080/props | python -m json.tool | grep -E "(multimodal|clip|vision|image)" || echo "No vision properties found"

echo
echo "Testing simple text generation first..."
curl -X POST http://127.0.0.1:8080/completion \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Hello, how are you?", "n_predict": 10}' \
  -s | python -c "import sys, json; print(json.load(sys.stdin).get('content', 'ERROR'))"

echo
echo "Press Enter to stop server..."
read

kill $SERVER_PID
