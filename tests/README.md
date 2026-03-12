# Tests

## Test Scripts Overview

### 1. Server Vision Test (`test_server_vision.py`)

Tests consecutive vision requests to llama-server via the OpenAI-compatible `/v1/chat/completions` endpoint. Diagnoses the 2nd-request-empty-output bug.

```bash
python tests/test_server_vision.py
```

**Prerequisites:**
- llama-server running on port 8090
- Model: `gemma-3-4b-it-q4_0_s.gguf` + vision projector

**What it tests:**
- Sends multiple vision requests in sequence
- Verifies each response contains extracted text
- Measures timing per request

### 2. Consecutive Request Test (`test_consecutive.py`)

Tests consecutive vision requests using the `/completion` endpoint with unique request IDs (`[rid:...]` prefix) and `cache_prompt: false` to work around the LCP cache bug.

```bash
python tests/test_consecutive.py
```

**What it tests:**
- Multiple sequential vision requests with unique IDs
- Token counts, prompt eval time, generation speed
- Reliability of consecutive requests without stale cache

### 3. Benchmark Pipeline (`benchmark_pipeline.py`)

End-to-end benchmark of the OCR-to-Anki pipeline, focused on GPU vision via OpenCL on Intel iGPU.

```bash
python tests/benchmark_pipeline.py
```

**What it benchmarks:**
- GPU vision via OpenCL (llama-mtmd-cli)
- Image encoding and generation timing
- Multiple images (full pages + cropped highlights)

**Prerequisites:**
```bash
ls -lh ~/.cache/llama.cpp/models/gemma-3-4b-it-q4_0_s.gguf
ls -lh ~/.cache/llama.cpp/models/mmproj-model-f16-4B.gguf
```

**Output:**
- Per-image timing breakdown (encode, prompt eval, generation)
- Results saved to `tests/benchmark_results.json`
