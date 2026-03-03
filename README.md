# OCR to Anki -- Fully Offline with llama.cpp

Cross-platform app to extract vocabulary from images and create Anki flashcards.
Everything runs locally. No cloud dependencies.

## Components

| Layer | Technology | Purpose |
|-------|-----------|---------|
| **Flutter GUI** | Dart / Material 3 | Cross-platform UI (Linux, macOS, Windows, Android) |
| **Python API** | FastAPI + llama.cpp | Vision OCR and text enrichment backend |
| **Vision OCR** | `llama-mtmd-cli` | Extract text from images (GPU-accelerated) |
| **Text tasks** | `llama-server` | Definitions, examples, vocabulary enrichment |
| **Model** | Gemma 3 4B QAT Q4_0 | Single model for both vision and text |

## Quick Start

### 1. Backend (Python + llama.cpp)

```bash
# Enter dev environment
nix develop

# Download model + vision projector (~3 GB total, one-time)
./scripts/setup-llama-cpp.sh

# Start the API server
PYTHONPATH=src uvicorn api.app:app --host 0.0.0.0 --port 8000
```

### 2. Flutter App

```bash
cd app
flutter pub get
flutter run -d linux    # or: -d macos, -d windows, -d android
```

The Flutter app connects to the FastAPI backend (default `http://localhost:8000`).
Configure the server URL in **Settings > Inference > Server URL**.

## Workflow

1. **Select context** -- handwritten/printed text or highlighted words (pick colour)
2. **Add image** -- file picker or camera (mobile)
3. **Vision OCR** -- Gemma 3 4B extracts words from the image
4. **Enrich** -- LLM generates definitions and example sentences
5. **Review** -- edit cards before export
6. **Export** -- send to Anki via AnkiConnect, or save as JSON

### Model Files

| File | Size | Source | Purpose |
|------|------|--------|---------|
| `gemma-3-4b-it-qat-q4_0_s.gguf` | ~2.4 GB | [stduhpf/google-gemma-3-4b-it-qat-q4_0-gguf-small](https://huggingface.co/stduhpf/google-gemma-3-4b-it-qat-q4_0-gguf-small) | Main language model (QAT -- better quality than standard Q4_0) |
| `mmproj-model-f16-4B.gguf` | ~812 MB | [google/gemma-3-4b-it-qat-q4_0-gguf](https://huggingface.co/google/gemma-3-4b-it-qat-q4_0-gguf) | Vision projector (SigLIP encoder) |

Both are downloaded by `./scripts/setup-llama-cpp.sh` via direct URL -- no authentication required.

**Why QAT?** Quantization-Aware Training (QAT) produces ~15% better perplexity
than standard post-training Q4_0 quantization at the same size. The stduhpf
repack also fixes broken control token metadata (`special_eos_id` warning).

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
# Build with OpenCL (RECOMMENDED for Intel iGPUs -- correct vision + fast)
nix develop .#sycl
./scripts/build-llama-mtmd-opencl.sh

# Build with Vulkan (fallback -- vision encoder corrupted on Intel iGPUs)
./scripts/build-llama-mtmd-vulkan.sh
```

Auto-detection picks the best available backend: CUDA > Metal > OpenCL > Vulkan > CPU.

### Intel iGPU: OpenCL vs Vulkan

| Backend | Vision encoder | Image encode time | Text gen | Binary |
|---------|---------------|-------------------|----------|--------|
| **OpenCL** [OK] | Correct | **~2 min** (GPU) | 4.1 tok/s | `llama-mtmd-cli-opencl` |
| Vulkan [ERR] | Corrupted | 0.4s (garbage) | 3.6 tok/s | `llama-mtmd-cli` |
| CPU fallback | Correct | ~43 min | 0.7 tok/s | any binary + `--no-mmproj-offload` |

The OpenCL backend is **20× faster** than CPU vision and produces correct output.
It requires a one-line patch for Intel work group sizes (applied automatically
by the build script): [patches/opencl-intel-workgroup-fix.patch](patches/opencl-intel-workgroup-fix.patch).

<details>
<summary>Vulkan corruption details</summary>

On Intel integrated GPUs (e.g. UHD Graphics CML GT2), the Vulkan compute
backend produces **corrupted output from the SigLIP vision encoder**. Text
generation works fine on Vulkan -- only the vision projector (mmproj) is
affected.

**Root cause:** Intel's Vulkan compute shaders produce f16 underflow/overflow in
the CLIP/SigLIP transformer. Debug embeddings show 75%+ of values saturate to
exactly -1.0 (clamped NaN/inf). This is a [known class of bug on integrated
GPUs](https://github.com/ggml-org/llama.cpp/issues/15034) -- a CUDA fix exists
(PR #16308) but no equivalent Vulkan fix.

**What was tested (all produced garbage):**
| Flag | Result |
|------|--------|
| Default (mmproj on GPU) | Multilingual gibberish |
| `--flash-attn off` | Different corruption pattern, early stop |
| `--no-op-offload` | Same multilingual gibberish |
| Different model (QAT vs standard Q4_0) | Same corruption |

</details>

If you have a discrete NVIDIA GPU, Vulkan/CUDA both work fine -- set
`mmproj_offload: true` in `config/settings.yaml`.

## Configuration

**config/settings.yaml:**
```yaml
ai_backend:
  type: 'llama_cpp'

llama_cpp:
  host: '127.0.0.1'
  port: 8090
  context_size: 4096
  n_gpu_layers: -1
  mmproj_offload: false  # Set true when using OpenCL backend
```

## Project Structure

```
app/                        # Flutter GUI application
├── lib/
│   ├── main.dart           # Entry point + routing
│   ├── models/             # Data models (AnkiNote, AppSettings, HighlightColor, etc.)
│   ├── services/           # Business logic
│   │   ├── inference_service.dart     # LLM inference (remote FastAPI / embedded)
│   │   ├── highlight_detector.dart    # HSV highlight colour detection
│   │   └── anki_export_service.dart   # AnkiConnect + JSON export
│   ├── database/           # Drift (SQLite) local storage
│   ├── providers/          # Riverpod state management
│   └── screens/            # UI screens (Home, Processing, Review, Settings, History)
src/                        # Python backend
├── api/                    # FastAPI application
│   ├── app.py              # Endpoints and lifespan
│   └── models.py           # Pydantic request/response models
├── backends/               # AI inference backends
│   ├── auto_detect.py      # GPU/backend auto-detection
│   ├── mtmd_cli.py         # llama-mtmd-cli wrapper (vision, subprocess)
│   └── llama_cpp_server.py # llama-server wrapper (vision + text, ~28x faster)
├── preprocessing/          # Image preprocessing
│   └── highlight_cropper.py # HSV-based highlight detection (Python reference)
├── workflows/              # End-to-end pipelines
└── output/                 # Anki export
config/
└── settings.yaml           # All configuration
scripts/                    # Build and setup scripts
```

## Development Shells

```bash
nix develop           # Default (Vulkan + CPU)
nix develop .#cuda    # With CUDA toolkit
nix develop .#sycl    # With Intel OneAPI/SYCL
```
