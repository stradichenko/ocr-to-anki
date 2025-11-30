#!/usr/bin/env bash

echo "Quick llama-server Diagnostic"
echo "=============================="
echo

# 1. Check if server process exists
echo "1. Server process check:"
if pgrep -f llama-server >/dev/null; then
    echo "   ✓ llama-server is running"
    ps aux | grep llama-server | grep -v grep | head -1
else
    echo "   ✗ llama-server not running"
fi
echo

# 2. Check if port is available
echo "2. Port 8080 status:"
if lsof -i :8080 >/dev/null 2>&1; then
    echo "   Port 8080 in use by:"
    lsof -i :8080 | tail -1
else
    echo "   ✓ Port 8080 available"
fi
echo

# 3. Quick health check
echo "3. Health endpoint test:"
HEALTH=$(curl -s --max-time 2 http://127.0.0.1:8080/health 2>&1 || echo "FAILED")
if echo "$HEALTH" | grep -q "ok"; then
    echo "   ✓ Server responding"
else
    echo "   ✗ Server not responding"
    echo "   Error: ${HEALTH:0:100}"
fi
echo

# 4. Test simple completion (10s timeout)
echo "4. Simple text completion test:"
TEST_RESPONSE=$(timeout 10 curl -X POST http://127.0.0.1:8080/completion \
  -H "Content-Type: application/json" \
  -d '{"prompt":"Hi","n_predict":5}' \
  -s 2>&1 || echo "TIMEOUT")

if echo "$TEST_RESPONSE" | grep -q "content"; then
    echo "   ✓ Text completion works"
    CONTENT=$(echo "$TEST_RESPONSE" | grep -o '"content":"[^"]*"' | head -1)
    echo "   Response: $CONTENT"
else
    echo "   ✗ Text completion failed"
    echo "   Error: ${TEST_RESPONSE:0:100}"
fi
echo

# 5. Model file check
echo "5. Model file check:"
MODEL_PATH="$HOME/.cache/llama.cpp/models/gemma-3-4b-it-q4_0.gguf"
if [ -f "$MODEL_PATH" ]; then
    SIZE=$(du -h "$MODEL_PATH" | cut -f1)
    echo "   ✓ Model exists ($SIZE)"
    
    # Quick corruption check (file should be ~2.2-2.4 GB)
    SIZE_BYTES=$(stat -f%z "$MODEL_PATH" 2>/dev/null || stat -c%s "$MODEL_PATH" 2>/dev/null)
    if [ "$SIZE_BYTES" -lt 2000000000 ]; then
        echo "   ⚠️  Model file seems too small (possible corruption)"
    fi
else
    echo "   ✗ Model not found at: $MODEL_PATH"
fi
echo

# Summary
echo "=========================="
echo "SUMMARY:"
if [ "$HEALTH" = '{"status":"ok"}' ] && echo "$TEST_RESPONSE" | grep -q "content"; then
    echo "✅ Server is WORKING"
    echo "   Next: Run python tests/test_llama_cpp.py"
else
    echo "❌ Server has ISSUES"
    echo
    echo "Recommendations:"
    if ! pgrep -f llama-server >/dev/null; then
        echo "  1. Start server: python src/llama_cpp_server.py"
    elif ! echo "$HEALTH" | grep -q "ok"; then
        echo "  1. Kill server: pkill -9 llama-server"
        echo "  2. Check model: ls -lh $MODEL_PATH"
        echo "  3. Re-download if corrupted: rm $MODEL_PATH && ./scripts/setup-llama-cpp.sh"
    elif echo "$TEST_RESPONSE" | grep -q "TIMEOUT"; then
        echo "  1. Model is stuck during inference"
        echo "  2. Possible causes:"
        echo "     - Model file corrupted"
        echo "     - Insufficient RAM (need ~8GB free)"
        echo "     - CPU too slow (needs AVX2 support)"
        echo "  3. Try: rm $MODEL_PATH && ./scripts/setup-llama-cpp.sh"
    fi
fi
