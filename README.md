# OCR to Anki — Fully Offline with llama.cpp

Everything runs locally. No HuggingFace login. No Tesseract. No Ollama. No cloud.

## Quick Start

```bash
# 1. Enter dev environment
nix develop

# 2. Download model + vision projector (~3 GB total, one-time)
./scripts/setup-llama-cpp.sh

# 3. Start the API server
PYTHONPATH=src uvicorn api.app:app --host 0.0.0.0 --port 8000
```

## Architecture

| Component | Tool | Purpose |
|-----------|------|---------|
| **Vision OCR** | `llama-mtmd-cli` | Extract text from images (GPU-accelerated via Vulkan) |
| **Text tasks** | `llama-server` | Definitions, examples, vocabulary enrichment |
| **API** | FastAPI | REST endpoints for all operations |
| **Model** | Gemma 3 4B Q4_0 | Single model for both vision and text |

### Model Files

| File | Size | Purpose |
|------|------|---------|
| `gemma-3-4b-it-q4_0.gguf` | ~2.3 GB | Main language model |
| `mmproj-model-f16-4B.gguf` | ~812 MB | Vision projector (CLIP encoder) |

Both are downloaded by `./scripts/setup-llama-cpp.sh` via direct URL — no authentication required.

## API Endpoints

```
GET  /health                  # Backend status
GET  /backends                # Detected GPU hardware
POST /ocr/vision              # Vision OCR (base64 image)
POST /ocr/vision/upload       # Vision OCR (file upload)
POST /generate                # Raw text generation
POST /enrich                  # Vocabulary enrichment (definitions + examples)
POST /pipeline/image-to-cards # Full pipeline: image → OCR → enrich → Anki cards
```

## Building llama-mtmd-cli (Vision)

The vision backend requires `llama-mtmd-cli` built with GPU support:

```bash
# Build with Vulkan (recommended for Intel/AMD GPUs)
./scripts/build-llama-mtmd-vulkan.sh

# The binary is installed to ~/.local/bin/llama-mtmd-cli
```

Auto-detection picks the best available backend: CUDA > Metal > Vulkan > CPU.

## Configuration

**config/settings.yaml:**
```yaml
ai_backend:
  type: 'llama_cpp'

llama_cpp:
  host: '127.0.0.1'
  port: 8080
  context_size: 4096
  n_gpu_layers: -1
```

## Project Structure

```
src/
├── api/                    # FastAPI application
│   ├── app.py              # Endpoints and lifespan
│   └── models.py           # Pydantic request/response models
├── backends/               # AI inference backends
│   ├── auto_detect.py      # GPU/backend auto-detection
│   ├── mtmd_cli.py         # llama-mtmd-cli wrapper (vision)
│   └── llama_cpp_server.py # llama-server wrapper (text)
├── preprocessing/          # Image preprocessing
├── workflows/              # End-to-end pipelines
└── output/                 # Anki export
```

## Development Shells

```bash
nix develop           # Default (Vulkan + CPU)
nix develop .#cuda    # With CUDA toolkit
nix develop .#sycl    # With Intel OneAPI/SYCL
```
