# Ollama Model Metadata Documentation

## Overview

This document provides a detailed computer science perspective on the metadata returned by Ollama's API when generating responses. Understanding these metrics is crucial for optimizing model performance, debugging issues, and making informed decisions about resource allocation.

## Metadata Fields

### `model`
```json
"model": "qwen3-vl:2b"
```

**Type:** String  
**Description:** Identifies the specific model and variant used for inference.

**Components:**
- `qwen3-vl`: Model family (Qwen 3 Vision-Language)
- `2b`: Parameter count (2 billion parameters)

**Computer Science Context:**
- Parameters are the learnable weights in the neural network
- A 2B parameter model requires approximately 4-8GB of VRAM (depending on quantization)
- Larger models generally have better performance but slower inference times

---

### `created_at`
```json
"created_at": "2025-11-23T21:53:02.307638969Z"
```

**Type:** ISO 8601 Timestamp  
**Description:** UTC timestamp marking when the response generation completed.

**Computer Science Context:**
- Follows RFC 3339 format with nanosecond precision
- The 'Z' suffix indicates Zulu time (UTC+0)
- Useful for logging, debugging, and temporal analysis of requests

---

### `done`
```json
"done": true
```

**Type:** Boolean  
**Description:** Indicates whether the generation process completed successfully.

**Possible Values:**
- `true`: Generation completed (may have hit stop conditions or max tokens)
- `false`: Generation still in progress (when streaming is enabled)

**Computer Science Context:**
- In synchronous mode (stream=false), this should always be `true`
- In streaming mode, intermediate responses have `done: false`
- Essential for state management in async/streaming implementations

---

### `total_duration`
```json
"total_duration": 562566733831
```

**Type:** Integer (nanoseconds)  
**Description:** Total wall-clock time from request receipt to response completion.

**Value:** 562,566,733,831 ns = **562.57 seconds ≈ 9.4 minutes**

**Formula:**


### [model-info.md](file:///home/gespitia/projects/ocr-to-anki/docs/model-info.md)

Create comprehensive documentation explaining Ollama model metadata from a computer science perspective.

```markdown
# Ollama Model Metadata Documentation

## Overview

This document provides a detailed computer science perspective on the metadata returned by Ollama's API when generating responses. Understanding these metrics is crucial for optimizing model performance, debugging issues, and making informed decisions about resource allocation.

## Metadata Fields

### `model`
```json
"model": "qwen3-vl:2b"
```

**Type:** String  
**Description:** Identifies the specific model and variant used for inference.

**Components:**
- `qwen3-vl`: Model family (Qwen 3 Vision-Language)
- `2b`: Parameter count (2 billion parameters)

**Computer Science Context:**
- Parameters are the learnable weights in the neural network
- A 2B parameter model requires approximately 4-8GB of VRAM (depending on quantization)
- Larger models generally have better performance but slower inference times

---

### `created_at`
```json
"created_at": "2025-11-23T21:53:02.307638969Z"
```

**Type:** ISO 8601 Timestamp  
**Description:** UTC timestamp marking when the response generation completed.

**Computer Science Context:**
- Follows RFC 3339 format with nanosecond precision
- The 'Z' suffix indicates Zulu time (UTC+0)
- Useful for logging, debugging, and temporal analysis of requests

---

### `done`
```json
"done": true
```

**Type:** Boolean  
**Description:** Indicates whether the generation process completed successfully.

**Possible Values:**
- `true`: Generation completed (may have hit stop conditions or max tokens)
- `false`: Generation still in progress (when streaming is enabled)

**Computer Science Context:**
- In synchronous mode (stream=false), this should always be `true`
- In streaming mode, intermediate responses have `done: false`
- Essential for state management in async/streaming implementations

---

### `total_duration`
```json
"total_duration": 562566733831
```

**Type:** Integer (nanoseconds)  
**Description:** Total wall-clock time from request receipt to response completion.

**Value:** 562,566,733,831 ns = **562.57 seconds ≈ 9.4 minutes**

**Formula:**
```
total_duration = load_duration + prompt_eval_duration + eval_duration + overhead
```

**Components Include:**
- Model loading (if not cached)
- Prompt encoding and evaluation
- Token generation
- Network overhead
- Serialization/deserialization

**Computer Science Context:**
- Measured in nanoseconds (10⁻⁹ seconds) for precision
- Wall-clock time includes all system activities (I/O, context switches)
- Does not reflect pure computation time (see `eval_duration`)

**Optimization Considerations:**
- High total_duration suggests I/O bottlenecks or resource contention
- Keep models loaded in memory to reduce load_duration
- Monitor for context window size impact on prompt_eval_duration

---

### `load_duration`
```json
"load_duration": 140498742
```

**Type:** Integer (nanoseconds)  
**Description:** Time spent loading the model into memory.

**Value:** 140,498,742 ns = **140.5 ms = 0.14 seconds**

**Computer Science Context:**

**Cold Start vs. Warm Start:**
- **Cold Start:** Model loaded from disk → VRAM/RAM (slow)
- **Warm Start:** Model already in memory (fast, ~0ms)

**What Happens During Load:**
1. **Disk I/O:** Reading model weights from storage (SSD/HDD)
2. **Memory Allocation:** Reserving GPU/CPU memory
3. **Weight Loading:** Transferring parameters to compute device
4. **Initialization:** Setting up inference pipeline

**Performance Factors:**
- Storage speed (NVMe SSD >> SATA SSD >> HDD)
- Available VRAM/RAM
- Model size and quantization
- Memory bandwidth

**Example Load Times:**
- 2B model (quantized): 100-500ms
- 7B model (quantized): 500-2000ms
- 13B model (quantized): 1000-5000ms

**Optimization:**
```python
# Keep model loaded between requests
# Ollama automatically caches loaded models
# Configure keep_alive to prevent unloading
payload["keep_alive"] = "10m"  # Keep in memory for 10 minutes
```

---

### `prompt_eval_count`
```json
"prompt_eval_count": 1234
```

**Type:** Integer  
**Description:** Number of tokens in the input prompt (including image tokens).

**Value:** 1,234 tokens

**Computer Science Context:**

**Token Definition:**
A token is the fundamental unit of text processing in LLMs:
- Can be a word, subword, or character
- Average English word ≈ 1.3-1.5 tokens
- Special tokens (BOS, EOS, padding) also counted

**For Vision-Language Models:**
```
total_tokens = text_tokens + image_tokens + special_tokens
```

**Image Tokenization:**
- Images are converted to token sequences using a vision encoder
- For an 800×1422 image: ~600-1000 image tokens (model-dependent)
- Higher resolution = more tokens = slower processing

**Example Breakdown:**
```
Text prompt: "Extract all visible text..." ≈ 100 tokens
Image tokens: 800×1422 image ≈ 1,100 tokens
Special tokens: <BOS>, <IMG>, <EOS> ≈ 34 tokens
Total: ~1,234 tokens
```

**Context Window:**
- Models have maximum context length (e.g., 4096, 8192 tokens)
- `prompt_eval_count + eval_count ≤ context_window`
- Exceeding causes truncation or errors

---

### `prompt_eval_duration`
```json
"prompt_eval_duration": 63853846700
```

**Type:** Integer (nanoseconds)  
**Description:** Time spent processing and encoding the input prompt.

**Value:** 63,853,846,700 ns = **63.85 seconds ≈ 1.06 minutes**

**Computer Science Context:**

**What Happens During Prompt Evaluation:**

1. **Tokenization:** Converting text/image to tokens (negligible time)
2. **Embedding Lookup:** Mapping tokens to vector representations
3. **Forward Pass Through Encoder:** 
   - Multi-head attention across prompt tokens
   - Feed-forward neural network layers
   - Layer normalization
4. **Key-Value Cache Generation:** Storing attention states for efficient generation
5. **Vision Encoding (if image present):** Processing image through vision transformer

**Computational Complexity:**
```
O(n²) for self-attention, where n = prompt_eval_count
```

**Time Analysis:**
```
63.85s / 1,234 tokens = 51.7ms per token
```

This is relatively slow, suggesting:
- Large image requiring extensive vision processing
- CPU inference (GPU would be 5-10× faster)
- Resource contention or thermal throttling

**Performance Characteristics:**
- Prompt eval is **memory-bandwidth bound** (reading weights from VRAM)
- Scales quadratically with prompt length due to attention
- Vision encoding is computationally expensive
- Can be parallelized across prompt tokens

**Optimization Strategies:**
```python
# 1. Reduce image resolution
max_width = 600  # vs 800

# 2. Use shorter prompts
prompt = "Extract text: "  # vs verbose instructions

# 3. Enable GPU acceleration
# Ensure CUDA/ROCm is properly configured

# 4. Batch processing
# Process multiple images in parallel if model supports
```

---

### `eval_count`
```json
"eval_count": 2000
```

**Type:** Integer  
**Description:** Number of tokens generated in the response.

**Value:** 2,000 tokens

**Computer Science Context:**

**Generation Process:**
Autoregressive token generation - each token depends on all previous tokens:

```
for i in 1 to num_tokens:
    logits = model(context + generated_tokens[:i])
    token[i] = sample(logits)
    generated_tokens.append(token[i])
```

**Token Limits:**
```python
"num_predict": 2000  # Maximum tokens to generate
```

**Why 2000 Tokens Exactly?**
The model hit the `num_predict` limit, which means:
- Generation was truncated (didn't naturally complete)
- The response was cut off mid-sentence
- Likely why no valid JSON was extracted

**Expected Token Usage for JSON Array:**
```json
["word1", "word2", "word3", ..., "word50"]
≈ 150-300 tokens for 50 words
```

**Problem Diagnosis:**
Generating 2000 tokens for a simple JSON array suggests:
1. Model is not following instructions properly
2. Model is adding explanatory text
3. Model is repeating content
4. Wrong output format (not JSON)

**Solution:**
```python
# Reduce to reasonable limit
"num_predict": 500  # Enough for ~100 words + JSON formatting

# Add stop sequences
"stop": ["\n\n\n", "</s>"]  # Stop at triple newline or EOS
```

---

### `eval_duration`
```json
"eval_duration": 383017470288
```

**Type:** Integer (nanoseconds)  
**Description:** Time spent generating output tokens.

**Value:** 383,017,470,288 ns = **383.02 seconds ≈ 6.38 minutes**

**Computer Science Context:**

**Generation Speed:**
```
383.02s / 2000 tokens = 191.5ms per token
≈ 5.22 tokens/second
```

**Performance Analysis:**

| Metric | Value | Expected (GPU) | Status |
|--------|-------|----------------|---------|
| Tokens/sec | 5.22 | 20-50 | ⚠️ Slow |
| ms/token | 191.5 | 20-50 | ⚠️ Slow |

**Why So Slow?**

1. **CPU Inference:** 
   - CPUs process tokens sequentially
   - GPUs parallelize matrix operations (~10× faster)

2. **Model Size:**
   - 2B parameters require significant computation per token
   - Each token generation: billions of floating-point operations

3. **Memory Bandwidth:**
   - Bottlenecked by reading weights from RAM/VRAM
   - Token generation is **memory-bound**, not compute-bound

4. **Autoregressive Nature:**
   - Cannot parallelize across tokens (each depends on previous)
   - Can only parallelize within token (matrix operations)

**Computational Complexity:**

Per-token generation:
```
FLOPs ≈ 2 × num_parameters × num_layers
≈ 2 × 2B × ~24 layers = ~96 billion FLOPs per token
```

Total computation:
```
2000 tokens × 96B FLOPs = 192 trillion FLOPs
383s → 501 GFLOPs/sec throughput
```

**Optimization Techniques:**

1. **Hardware Acceleration:**
   ```bash
   # Use GPU if available
   nvidia-smi  # Check GPU availability
   ollama run qwen3-vl:2b --gpu  # Force GPU usage
   ```

2. **Quantization:**
   ```bash
   # Use 4-bit quantized model (2-3× faster)
   ollama pull qwen3-vl:2b-q4
   ```

3. **Reduce Generation Length:**
   ```python
   "num_predict": 500,  # Generate fewer tokens
   "stop": ["</s>", "\n\n"]  # Stop early
   ```

4. **KV Cache Optimization:**
   ```python
   "num_ctx": 4096,  # Larger context for better caching
   ```

5. **Temperature Tuning:**
   ```python
   "temperature": 0.1,  # Lower = faster sampling
   "top_k": 40,  # Limit consideration set
   ```

---

## Performance Metrics Summary

### Time Breakdown
```
Total Duration:       562.57s (100.0%)
├─ Load:               0.14s (  0.0%)  ✓ Good
├─ Prompt Eval:       63.85s ( 11.4%)  ⚠️ Slow (image processing)
├─ Generation:       383.02s ( 68.1%)  ⚠️ Very Slow (CPU bottleneck)
└─ Overhead:         115.56s ( 20.5%)  ⚠️ High (investigate)
```

### Throughput Analysis
```
Prompt Processing:  19.3 tokens/sec (1,234 tokens in 63.85s)
Token Generation:    5.2 tokens/sec (2,000 tokens in 383.02s)
Overall:             5.7 tokens/sec (3,234 tokens in 562.57s)
```

### Recommendations

**Immediate Optimizations:**
1. ✅ Reduce `num_predict` from 2000 to 500
2. ✅ Lower image resolution from 800px to 600px
3. ✅ Simplify prompt to reduce token count
4. ✅ Add explicit stop sequences

**Hardware Optimizations:**
1. 🔥 Use GPU instead of CPU (10-20× speedup)
2. 💾 Use quantized model (2-3× speedup)
3. 🚀 Increase RAM/VRAM allocation
4. 📊 Monitor thermal throttling

**Expected Improvements:**
```
Current:   5.2 tokens/sec
With GPU:  50-100 tokens/sec (10-20× faster)
With Q4:   80-150 tokens/sec (15-30× faster)
```

---

## Diagnostic Checklist

When analyzing slow performance:

- [ ] Check `load_duration` > 1s → Model loading from disk (cache model)
- [ ] Check `prompt_eval_duration` > 10s → Large image or long prompt
- [ ] Check `eval_count` == `num_predict` → Hit token limit (incomplete response)
- [ ] Check `eval_duration` / `eval_count` > 100ms → CPU bottleneck (use GPU)
- [ ] Check `total_duration` > sum of parts → High overhead (investigate I/O)

---

## References

- [Ollama API Documentation](https://github.com/ollama/ollama/blob/main/docs/api.md)
- [Attention Is All You Need](https://arxiv.org/abs/1706.03762) - Transformer architecture
- [LLaMA Paper](https://arxiv.org/abs/2302.13971) - Efficient inference techniques
- [Flash Attention](https://arxiv.org/abs/2205.14135) - Optimized attention computation

---

## Ollama API Configuration Flags

### Overview

The `ollama_ocr.py` script uses various Ollama API parameters to control model behavior and performance. Understanding these flags is crucial for optimizing OCR quality and speed.

### Configuration Location

All flags are configured in `config/settings.yaml`:

```yaml
ollama_ocr:
  url: "http://localhost:11434"
  model: "qwen3-vl:2b"
  timeout: 300
  max_image_width: 600
  verbose_logging: false
  save_response: true
  save_raw_request: false
  save_model_info: false
```

### Request Payload Options

These are passed in the `options` field of the API request:

```python
payload = {
    "model": "qwen3-vl:2b",
    "prompt": "...",
    "images": ["base64_encoded_image"],
    "stream": false,
    "options": {
        "num_ctx": 4096,
        "num_predict": 2000,
        "temperature": 0.1,
        "top_k": 40,
        "top_p": 0.9,
        "stop": ["\n\n\n"]
    }
}
```

---

### Core Model Parameters

#### `num_ctx` - Context Window Size

**Type:** Integer  
**Default:** 2048  
**Recommended:** 4096  
**Range:** 512 - 128000 (model-dependent)

**Description:** Maximum number of tokens the model can process (prompt + response).

**Computer Science Context:**
- Controls the size of the attention matrix (O(n²) memory)
- Larger contexts allow longer prompts and responses
- Directly impacts VRAM usage: ~4 bytes per token × context size

**Memory Usage:**



### [model-info.md](file:///home/gespitia/projects/ocr-to-anki/docs/model-info.md)

Add section on Ollama API configuration flags and their impact on performance.

```markdown
# ...existing code...

## References

- [Ollama API Documentation](https://github.com/ollama/ollama/blob/main/docs/api.md)
- [Attention Is All You Need](https://arxiv.org/abs/1706.03762) - Transformer architecture
- [LLaMA Paper](https://arxiv.org/abs/2302.13971) - Efficient inference techniques
- [Flash Attention](https://arxiv.org/abs/2205.14135) - Optimized attention computation

---

## Ollama API Configuration Flags

### Overview

The `ollama_ocr.py` script uses various Ollama API parameters to control model behavior and performance. Understanding these flags is crucial for optimizing OCR quality and speed.

### Configuration Location

All flags are configured in `config/settings.yaml`:

```yaml
ollama_ocr:
  url: "http://localhost:11434"
  model: "qwen3-vl:2b"
  timeout: 300
  max_image_width: 600
  verbose_logging: false
  save_response: true
  save_raw_request: false
  save_model_info: false
```

### Request Payload Options

These are passed in the `options` field of the API request:

```python
payload = {
    "model": "qwen3-vl:2b",
    "prompt": "...",
    "images": ["base64_encoded_image"],
    "stream": false,
    "options": {
        "num_ctx": 4096,
        "num_predict": 2000,
        "temperature": 0.1,
        "top_k": 40,
        "top_p": 0.9,
        "stop": ["\n\n\n"]
    }
}
```

---

### Core Model Parameters

#### `num_ctx` - Context Window Size

**Type:** Integer  
**Default:** 2048  
**Recommended:** 4096  
**Range:** 512 - 128000 (model-dependent)

**Description:** Maximum number of tokens the model can process (prompt + response).

**Computer Science Context:**
- Controls the size of the attention matrix (O(n²) memory)
- Larger contexts allow longer prompts and responses
- Directly impacts VRAM usage: ~4 bytes per token × context size

**Memory Usage:**
```
VRAM = model_size + (num_ctx × hidden_dim × num_layers × 4 bytes)

For 2B model:
- num_ctx=2048: ~2.5GB VRAM
- num_ctx=4096: ~3.0GB VRAM
- num_ctx=8192: ~4.0GB VRAM
```

**When to Adjust:**
- **Increase** if you get "context length exceeded" errors
- **Increase** for high-resolution images (more image tokens)
- **Decrease** if running out of VRAM
- **Decrease** for faster inference on small images

**Example:**
```python
# For high-resolution OCR
"num_ctx": 8192  # Support larger images

# For low-VRAM systems
"num_ctx": 2048  # Minimal context
```

---

#### `num_predict` - Max Generation Length

**Type:** Integer  
**Default:** 128  
**Recommended for OCR:** 500-2000  
**Range:** -1 (unlimited), 1 - 128000

**Description:** Maximum number of tokens to generate in the response.

**Computer Science Context:**
- Acts as a hard limit on generation
- Generation stops when:
  1. `num_predict` tokens generated, OR
  2. Model generates stop token (`</s>`), OR
  3. Stop sequence matched
- Does not affect prompt processing time

**Performance Impact:**
```
Generation time ≈ num_predict × time_per_token

For 5 tokens/sec:
- num_predict=500: ~100 seconds
- num_predict=2000: ~400 seconds
- num_predict=-1: potentially infinite
```

**OCR-Specific Considerations:**

**Too Low:**
```python
"num_predict": 100  # Only ~50 words max
# Result: Truncated JSON array, missing words
```

**Optimal:**
```python
"num_predict": 500  # ~250 words - sufficient for most images
# Result: Complete response, faster processing
```

**Too High:**
```python
"num_predict": 5000  # Model may ramble
# Result: Explanatory text, repeated content, wasted time
```

**Recommendations by Image Type:**
```python
# Simple text (1-50 words)
"num_predict": 300

# Standard document (50-200 words)
"num_predict": 800

# Dense page (200+ words)
"num_predict": 2000

# Testing/debugging
"num_predict": 5000  # See full model behavior
```

---

#### `temperature` - Sampling Randomness

**Type:** Float  
**Default:** 0.8  
**Recommended for OCR:** 0.1-0.3  
**Range:** 0.0 - 2.0

**Description:** Controls randomness in token selection. Lower = more deterministic.

**Mathematical Definition:**
```
Temperature scaling applied to logits before softmax:

P(token_i) = exp(logit_i / temperature) / Σ exp(logit_j / temperature)

temperature=0.1: Peaked distribution (deterministic)
temperature=1.0: Original distribution
temperature=2.0: Flattened distribution (random)
```

**Visual Representation:**
```
temperature=0.1          temperature=1.0          temperature=2.0
(Deterministic)          (Balanced)               (Random)

    ▁                        ▄                         ▃
   ▁▆█                      █▆▃                       ▅▆▅▆
  ▁▂▃▅██                   ▅███▅▂                    ▆████▅
 ▁▁▁▁▂▃▅███              ▃▅█████▅▃▁                ▅███████▆
────────────            ──────────────            ──────────────
 Most   Least            Balanced                   Random
Likely  Likely           Selection                  Selection
```

**For OCR Tasks:**

**Too Low (temperature=0.0):**
- ✅ Consistent output
- ✅ Follows format strictly
- ❌ May get stuck in repetitive patterns
- ❌ Less creative word interpretation

**Optimal (temperature=0.1-0.3):**
- ✅ Deterministic enough for JSON format
- ✅ Slight variation for ambiguous characters
- ✅ Consistent between runs
- ✅ Good for production

**Too High (temperature=1.0+):**
- ❌ Inconsistent output format
- ❌ May hallucinate words
- ❌ JSON parsing errors
- ❌ Unreliable results

**Example:**
```python
# Production OCR
"temperature": 0.1

# Handwriting (ambiguous characters)
"temperature": 0.3

# Creative text extraction
"temperature": 0.5

# Never for OCR
"temperature": 1.5  # Too random
```

---

#### `top_k` - Top-K Sampling

**Type:** Integer  
**Default:** 40  
**Recommended:** 20-40  
**Range:** 1 - 100

**Description:** Limits token selection to top K most probable tokens.

**How It Works:**
```
Step 1: Model produces probability distribution over vocabulary
Step 2: Keep only top K tokens by probability
Step 3: Renormalize probabilities among top K
Step 4: Sample from this restricted distribution

Example with top_k=3:
All tokens:    ["the": 0.5, "a": 0.3, "an": 0.15, "to": 0.05]
After top_k=3: ["the": 0.526, "a": 0.316, "an": 0.158]
                 (renormalized to sum to 1.0)
```

**Interaction with Temperature:**
```python
# Very restrictive (deterministic)
"temperature": 0.1,
"top_k": 10  # Only consider top 10 tokens

# Balanced (recommended for OCR)
"temperature": 0.2,
"top_k": 40  # Consider top 40 tokens

# More creative
"temperature": 0.5,
"top_k": 100  # Consider top 100 tokens
```

**Performance Impact:**
- Lower `top_k` → Faster sampling (fewer tokens to consider)
- Higher `top_k` → Slower sampling, more diversity

**OCR Recommendations:**
```python
# Printed text (high confidence)
"top_k": 20

# Handwritten text (more ambiguity)
"top_k": 40

# Low quality images
"top_k": 60
```

---

#### `top_p` - Nucleus Sampling

**Type:** Float  
**Default:** 0.9  
**Recommended:** 0.9-0.95  
**Range:** 0.0 - 1.0

**Description:** Keeps smallest set of tokens whose cumulative probability ≥ top_p.

**How It Works:**
```
Step 1: Sort tokens by probability (descending)
Step 2: Include tokens until cumulative probability ≥ top_p
Step 3: Sample from this nucleus

Example with top_p=0.9:
Tokens:        ["the": 0.5, "a": 0.3, "an": 0.15, "to": 0.04, "in": 0.01]
Cumulative:    [       0.5,      0.8,       0.95,      0.99,       1.0]
Selected:      ["the": 0.5, "a": 0.3, "an": 0.15]  (sum=0.95 ≥ 0.9)
```

**Top-K vs Top-P:**

| Scenario | top_k=40 | top_p=0.9 |
|----------|----------|-----------|
| Peaked distribution | Selects 40 tokens (wasteful) | Selects ~5 tokens (efficient) |
| Flat distribution | Selects 40 tokens (limiting) | Selects ~80 tokens (adaptive) |

**Best Practice:** Use both together
```python
"top_k": 40,   # Hard limit on tokens
"top_p": 0.9   # Adaptive limit based on confidence
```

**OCR Settings:**
```python
# High confidence needed
"top_p": 0.85

# Balanced (recommended)
"top_p": 0.9

# Handle ambiguous text
"top_p": 0.95
```

---

#### `stop` - Stop Sequences

**Type:** List of strings  
**Default:** None  
**Recommended:** `["\n\n\n", "</s>"]`

**Description:** Generation stops when any stop sequence is encountered.

**Why Use Stop Sequences for OCR:**

1. **Prevent Rambling:**
```python
Without stop sequences:
["word1", "word2", "word3"]

Explanation: The image contains three words visible on a white background...
(wastes time and tokens)

With stop=["\n\n\n"]:
["word1", "word2", "word3"]
(stops immediately after JSON)
```

2. **Save Computation:**
```python
# Without stop sequences
"num_predict": 2000  # Model uses all 2000 tokens

# With stop sequences
"num_predict": 2000  # Model stops at 300 tokens
                     # 85% time saved!
```

**Recommended Stop Sequences:**

```python
# Basic
"stop": ["</s>"]  # Model's end-of-sequence token

# For JSON
"stop": ["\n\n\n", "</s>", "```"]  # Stop after JSON block

# For strict JSON
"stop": ["]\n\n", "</s>"]  # Stop after closing bracket

# Debug mode
"stop": []  # No stops - see full model output
```

**Example Impact:**
```python
# No stop sequences
Time: 400s for 2000 tokens
Response: ["word1", "word2"] followed by 1800 tokens of explanation

# With stop=["\n\n\n"]
Time: 60s for 300 tokens
Response: ["word1", "word2"]  (stops cleanly)
```

---

### Script-Specific Configuration Flags

These are set in settings.yaml and control script behavior:

#### `timeout` - Request Timeout

**Type:** Integer (seconds)  
**Default:** 300  
**Recommended:** 600 for testing, 300 for production

```yaml
ollama_ocr:
  timeout: 600  # 10 minutes
```

**When to Adjust:**
- High-resolution images: Increase to 600-900s
- Low-resolution images: Decrease to 60-180s
- Production systems: Set conservatively (300s)
- Testing: Set generously (900s+)

**Failure Modes:**
```python
# Timeout too low
timeout: 60  # seconds
Error: "ReadTimeout: Read timed out after 60s"

# Timeout too high
timeout: 3600  # 1 hour
Risk: Hung requests, resource exhaustion
```

---

#### `max_image_width` - Image Resizing

**Type:** Integer (pixels)  
**Default:** 600  
**Recommended:** 600-800

```yaml
ollama_ocr:
  max_image_width: 600
```

**Impact on Performance:**

| Width | Tokens | Prompt Time | Quality |
|-------|--------|-------------|---------|
| 400px | ~400 | 20s | Low |
| 600px | ~900 | 40s | Good |
| 800px | ~1400 | 60s | Better |
| 1200px | ~3000 | 120s | Best |

**Trade-offs:**
```python
# Fast but lower quality
max_image_width: 400

# Balanced (recommended)
max_image_width: 600

# High quality but slow
max_image_width: 800

# Maximum detail (very slow)
max_image_width: 1200
```

---

#### `verbose_logging` - Debug Output

**Type:** Boolean  
**Default:** false

```yaml
ollama_ocr:
  verbose_logging: true
```

**When Enabled:**
- Shows prompt length
- Shows image size
- Shows token counts
- Shows timing breakdowns
- Shows model metadata
- Shows response previews

**Use Cases:**
```python
# Development
verbose_logging: true  # See everything

# Production
verbose_logging: false  # Clean logs

# Debugging
verbose_logging: true  # Diagnose issues
```

---

#### `save_model_info` - Performance Metrics

**Type:** Boolean  
**Default:** false

```yaml
ollama_ocr:
  save_model_info: true
```

**Saved Metrics:**
```json
{
  "model_info": {
    "model": "qwen3-vl:2b",
    "total_duration_ms": 562566,
    "load_duration_ms": 140,
    "prompt_eval_count": 1234,
    "prompt_eval_duration_ms": 63853,
    "eval_count": 2000,
    "eval_duration_ms": 383017
  }
}
```

**Use Cases:**
- Performance monitoring
- Optimization experiments
- Debugging slow requests
- Resource planning

---

## Complete Configuration Examples

### Fast OCR (Production)

Optimize for speed on simple documents:

```yaml
ollama_ocr:
  timeout: 180
  max_image_width: 400
  verbose_logging: false
  save_model_info: false
```

```python
payload["options"] = {
    "num_ctx": 2048,
    "num_predict": 300,
    "temperature": 0.1,
    "top_k": 20,
    "top_p": 0.85,
    "stop": ["]\n\n", "</s>"]
}
```

**Expected Performance:**
- ~30-60 seconds per image
- ~20-50 words extracted
- Minimal VRAM usage (2-3GB)

---

### Balanced OCR (Recommended)

Good quality and reasonable speed:

```yaml
ollama_ocr:
  timeout: 300
  max_image_width: 600
  verbose_logging: false
  save_model_info: false
```

```python
payload["options"] = {
    "num_ctx": 4096,
    "num_predict": 500,
    "temperature": 0.2,
    "top_k": 40,
    "top_p": 0.9,
    "stop": ["\n\n\n", "</s>"]
}
```

**Expected Performance:**
- ~60-120 seconds per image
- ~50-200 words extracted
- Moderate VRAM usage (3-4GB)

---

### High-Quality OCR (Research)

Maximum accuracy for complex documents:

```yaml
ollama_ocr:
  timeout: 600
  max_image_width: 800
  verbose_logging: true
  save_model_info: true
```

```python
payload["options"] = {
    "num_ctx": 8192,
    "num_predict": 2000,
    "temperature": 0.3,
    "top_k": 60,
    "top_p": 0.95,
    "stop": ["</s>"]
}
```

**Expected Performance:**
- ~120-400 seconds per image
- ~200-500 words extracted
- High VRAM usage (4-6GB)

---

### Debug Mode

Maximum visibility for troubleshooting:

```yaml
ollama_ocr:
  timeout: 900
  max_image_width: 600
  verbose_logging: true
  save_model_info: true
  save_raw_request: true
```

```python
payload["options"] = {
    "num_ctx": 4096,
    "num_predict": 5000,  # Let model complete fully
    "temperature": 0.2,
    "top_k": 40,
    "top_p": 0.9,
    "stop": []  # No stops - see everything
}
```

---

## Troubleshooting Guide

### Problem: Empty Responses

**Symptoms:**
```
Response length: 0 characters
Words extracted: 0
Tokens generated: 2000
```

**Diagnosis:**
```python
# Enable debugging
verbose_logging: true
save_raw_request: true

# Check response preview
Response preview: [blank or garbled]
```

**Solutions:**
1. Check if model supports vision (Qwen-VL, LLaVA)
2. Verify image encoding (base64 valid)
3. Simplify prompt
4. Reduce image size
5. Increase `num_predict`

---

### Problem: Slow Performance

**Symptoms:**
```
Processing time: 400s+
Tokens/sec: <10
```

**Diagnosis:**
```python
prompt_eval_duration: 60s+  → Image processing bottleneck
eval_duration: 300s+        → CPU inference bottleneck
```

**Solutions:**
1. Use GPU acceleration
2. Reduce `max_image_width`
3. Reduce `num_predict`
4. Use quantized model
5. Decrease `num_ctx`

---

### Problem: Truncated Responses

**Symptoms:**
```
eval_count: 2000 (equals num_predict)
words: Incomplete JSON array
```

**Solutions:**
```python
# Increase generation limit
"num_predict": 3000

# Add stop sequences to end cleanly
"stop": ["]\n", "</s>"]

# Reduce image complexity
max_image_width: 400
```

---

### Problem: Inconsistent Results

**Symptoms:**
```
Run 1: ["word1", "word2"]
Run 2: ["word1", "word3"]
Run 3: ["different", "words"]
```

**Solutions:**
```python
# Decrease randomness
"temperature": 0.05  # Very deterministic

# Restrict sampling
"top_k": 10
"top_p": 0.8

# Use seed (if supported)
"seed": 42
```

---

## Performance Optimization Checklist

- [ ] Use GPU if available (`nvidia-smi` to check)
- [ ] Set `temperature` to 0.1-0.2 for consistency
- [ ] Set `num_predict` to match expected output length
- [ ] Add stop sequences to prevent over-generation
- [ ] Reduce `max_image_width` for speed
- [ ] Increase `num_ctx` if getting truncation errors
- [ ] Enable `verbose_logging` during development
- [ ] Enable `save_model_info` for performance monitoring
- [ ] Use quantized models (Q4, Q8) for faster inference
- [ ] Keep model loaded between requests (`keep_alive`)
