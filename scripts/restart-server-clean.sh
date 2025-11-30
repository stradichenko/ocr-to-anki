#!/usr/bin/env bash

set -euo pipefail

echo "Clean Server Restart Diagnostic"
echo "================================"
echo

# 1. Kill ALL llama-server processes
echo "1. Killing all llama-server processes..."
pkill -9 -f llama-server || true
sleep 2

# Verify they're dead
if pgrep -f llama-server >/dev/null; then
    echo "❌ llama-server still running! Force killing..."
    killall -9 llama-server || true
    sleep 2
fi
echo "✅ All llama-server processes killed"
echo

# 2. Check if port 8080 is available
echo "2. Checking port 8080..."
if lsof -i :8080 >/dev/null 2>&1; then
    echo "❌ Port 8080 is in use!"
    echo "Process using port 8080:"
    lsof -i :8080
    exit 1
fi
echo "✅ Port 8080 is available"
echo

# 3. Start fresh server WITHOUT vision (simpler test)
echo "3. Starting llama-server WITHOUT vision projector (simpler)..."
llama-server \
  --model ~/.cache/llama.cpp/models/gemma-3-4b-it-q4_0.gguf \
  --host 127.0.0.1 \
  --port 8080 \
  --ctx-size 2048 \
  --threads 4 \
  --parallel 1 \
  --verbose &

SERVER_PID=$!
echo "Server PID: $SERVER_PID"
echo

# 4. Wait for server with timeout
echo "4. Waiting for server to respond..."
for i in {1..30}; do
  if curl -s --max-time 2 http://127.0.0.1:8080/health >/dev/null 2>&1; then
    echo "✅ Server responded to health check!"
    break
  fi
  if [ $i -eq 30 ]; then
    echo "❌ Server never responded to health check"
    echo
    echo "Server logs (last 50 lines):"
    jobs -p | xargs -I {} tail -50 /proc/{}/fd/1 2>/dev/null || echo "No logs available"
    kill $SERVER_PID 2>/dev/null
    exit 1
  fi
  echo -n "."
  sleep 1
done
echo

# 5. Test simple completion
echo "5. Testing simple text completion..."
cat > /tmp/test_simple.json << 'EOF'
{
  "prompt": "Hello",
  "n_predict": 5,
  "temperature": 0.1
}
EOF

echo "Sending simple request with 10s timeout..."
RESPONSE=$(timeout 10 curl -X POST http://127.0.0.1:8080/completion \
  -H "Content-Type: application/json" \
  -d @/tmp/test_simple.json \
  -s 2>&1 || echo "TIMEOUT")

if echo "$RESPONSE" | grep -q "TIMEOUT"; then
    echo "❌ Simple text completion timed out!"
    echo
    echo "DIAGNOSIS: The model is stuck during generation."
    echo "Possible causes:"
    echo "  1. Model file is corrupted"
    echo "  2. llama-server has a bug"
    echo "  3. System resources exhausted"
    echo
    echo "Try:"
    echo "  1. Re-download the model: rm ~/.cache/llama.cpp/models/*.gguf && ./scripts/setup-llama-cpp.sh"
    echo "  2. Check system resources: free -h && top"
    echo "  3. Try a different model size"
else
    echo "✅ Got response!"
    echo "$RESPONSE" | python -c "import sys, json; print(json.load(sys.stdin).get('content', 'ERROR')[:50])"
    echo
    echo "SUCCESS: Text completion works!"
    echo "The issue was with vision requests specifically."
fi

# Cleanup
rm -f /tmp/test_simple.json
kill $SERVER_PID 2>/dev/null || true
