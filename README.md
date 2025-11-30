# OCR to Anki - Fully Offline with llama.cpp

## AI Backend: llama.cpp with Gemma 3 4B Vision

**The ONLY backend this project uses.**

### Quick Start

1. Enter dev environment: `nix develop`
2. Download model: `./scripts/setup-llama-cpp.sh`
3. Start server: `python src/llama_cpp_server.py`
4. Test: `python tests/test_llama_cpp.py`

### Vision OCR Workflow

```bash
# Extract text from image
python src/vision_ocr.py data/images/handwritten.jpeg

# Results saved to data/ocr_results/
```

No internet required after initial setup.

# ocr-to-anki


```
ocr-to-anki/
├── flake.nix                  # Main Nix flake: devShell, packages, apps
├── flake.lock                 # Auto-generated lock file for Nix reproducibility
│
├── nix/                       # All Nix-related modules (keeps flake.nix clean)
│   ├── devshell.nix           # Defines development environment (Python, GTK, tools)
│   ├── python-env.nix         # Python environment (if using Nix for Python deps)
│   ├── overlays.nix           # Optional overlays or custom derivations
│   └── packages.nix           # Definitions for building the app as a Nix package
│
├── src/                       # ***ALL application code lives here***
│   ├── app_core/              # Core logic (independent from UI)
│   │   ├── __init__.py
│   │   ├── engine.py          # Example: processing logic
│   │   └── utils.py           # Shared utilities
│   │
│   ├── app_ui/                # GUI code (GTK)
│   │   ├── __init__.py
│   │   ├── main_window.py     # Main GTK window
│   │   ├── widgets/           # Custom widgets
│   │   │   ├── __init__.py
│   │   │   └── progress_panel.py
│   │   ├── dialogs/           # Popups, configuration dialogs
│   │   │   ├── __init__.py
│   │   │   └── settings_dialog.py
│   │   └── ui/                # GTK Builder XML/UI definitions
│   │       ├── main_window.ui
│   │       └── styles.css
│   │
│   ├── cli/                   # CLI commands the user can run
│   │   ├── __init__.py
│   │   └── main.py            # Defines commands: myapp analyze / export / etc.
│   │
│   ├── __main__.py            # Entry point for `python -m src` or `nix run`
│   └── config.py              # Centralized Python config loader
│
├── resources/                 # Files shipped with the app
│   ├── icons/                 # PNG/SVG icons
│   └── sample_data/           # Optional bundled data
│
├── logs/                      # ***Dev-only logs*** (ignored by git)
│   └── .keep                  # Empty file to keep folder in repo
│
├── tests/                     # Automated tests
│   ├── unit/                  # Tests pure logic (app_core)
│   │   └── test_engine.py
│   ├── integration/           # UI + core interaction tests
│   └── ui/                    # Optional: automated GTK tests
│
├── scripts/                   # Dev/maintainer scripts (NOT shipped to users)
│   ├── format.sh              # Code formatting helper
│   ├── update.sh              # Update Nix flake inputs
│   ├── run-dev.sh             # Run app with dev paths enabled
│   └── generate-docs.sh       # Build documentation
│
├── bash/                      # Runtime bash scripts (if part of the app)
│   ├── __init__.txt           # Documentation for scripts
│   ├── helper.sh              # Script used by the Python app via subprocess
│   └── collect_info.sh        # Example: gather system information
│
├── docker/                    # All Docker-related files
│   ├── Dockerfile             # Build the app image
│   └── compose.yml            # Optional docker-compose setup
│
├── docs/                      # Documentation for users/devs
│   ├── installation.md
│   ├── architecture.md
│   └── ui-design.md           # Describe GTK layout & structure
│
├── .gitignore
└── README.md                  # Project overview
```

## AI Backend Options

This project supports three AI backends for vocabulary enrichment:

### 1. **llama.cpp (Fully Offline - Recommended)**

**Pros:**
- ✅ **Completely offline** - no internet needed after setup
- ✅ **Fast inference** with Q4 quantization
- ✅ **No external services** - runs locally
- ✅ **Privacy-focused** - data never leaves your machine
- ✅ **Works on CPU and GPU**

**Setup:**

```bash
# Enter Nix development shell
nix develop

# Download Gemma 3 4B model (one-time, ~2.36 GB)
./scripts/setup-llama-cpp.sh

# Start llama.cpp server
python src/llama_cpp_server.py

# In another terminal, test it works
python tests/test_llama_cpp.py
```

**Configuration:**

In `config/settings.yaml`:
```yaml
ai_backend:
  type: 'llama_cpp'

llama_cpp:
  host: '127.0.0.1'
  port: 8080
  context_size: 4096
  n_gpu_layers: -1  # Use all GPU layers (or 0 for CPU only)
```

### 2. **Ollama (Local with Internet)**

**Pros:**
- ✅ Easy setup
- ✅ Multiple model support
- ✅ Good documentation

**Cons:**
- ❌ Requires internet for initial model download
- ❌ Larger disk usage (unquantized models)

**Setup:**

```bash
# Install Ollama
curl https://ollama.ai/install.sh | sh

# Pull model
ollama pull gemma3:4b

# Start server
ollama serve
```

**Configuration:**

```yaml
ai_backend:
  type: 'ollama'
```

### 3. **vLLM (GPU Only - Highest Performance)**

**Pros:**
- ✅ Fastest inference
- ✅ Batch processing

**Cons:**
- ❌ Requires NVIDIA GPU
- ❌ More complex setup

**Setup:**

```bash
pip install vllm
# See docs/general-use.md for details
```

## Quick Start with llama.cpp

```bash
# 1. Enter development environment
nix develop

# 2. Download model (one-time setup)
./scripts/setup-llama-cpp.sh

# 3. Start llama.cpp server
python src/llama_cpp_server.py &

# 4. Process images with OCR
python src/ollama_ocr.py  # Will auto-detect llama.cpp backend

# 5. Enrich vocabulary
python tests/test_gemma_vocabulary_enricher.py
```

## Comparison: AI Backends

| Feature | llama.cpp | Ollama | vLLM |
|---------|-----------|--------|------|
| **Offline** | ✅ Yes | ❌ No (downloads models) | ❌ No |
| **Setup** | Easy | Easy | Complex |
| **Speed** | Fast | Fast | Fastest |
| **GPU** | Optional | Optional | Required |
| **CPU** | ✅ Yes | ✅ Yes | ❌ No |
| **Disk Usage** | ~2.4 GB | ~4-8 GB | ~8-16 GB |
| **Privacy** | Maximum | High | High |

**Recommendation:** Use llama.cpp for production and Ollama for development/testing

## AI Backend: llama.cpp with Gemma 3 4B (Recommended)

**Features:**
- ✅ **Fully offline** after initial download
- ✅ **Vision + Text support** (with mmproj)
- ✅ **Fast inference** with Q4_0 quantization
- ✅ **Privacy-focused** - data never leaves your machine
- ✅ **Works on CPU and GPU**

**Setup:**

```bash
# 1. Enter Nix development shell
nix develop

# 2. Login to HuggingFace (one-time)
huggingface-cli login
# Or use: hf auth login

# 3. Accept Gemma terms (one-time)
# Visit: https://huggingface.co/google/gemma-3-4b-it-qat-q4_0-gguf
# Click "Agree and access repository"

# 4. Download model + vision projector (~4 GB total)
./scripts/setup-llama-cpp.sh

# 5. Start server
python src/llama_cpp_server.py

# 6. Test it
python tests/test_llama_cpp.py
```

**What gets downloaded:**
- `gemma-3-4b-it-q4_0.gguf` (~3.16 GB) - Main text model
- `mmproj-model-f16-4B.gguf` (~851 MB) - Vision projector

**Capabilities:**
```python
from llama_cpp_server import LlamaCppServer

# Text generation
with LlamaCppServer() as server:
    result = server.generate("Define: bonjour")
    print(result['content'])

# Image understanding (with vision projector)
with LlamaCppServer() as server:
    image_base64 = encode_image("vocab.jpg")
    result = server.generate(
        "Extract all visible text",
        image_data=image_base64
    )
    print(result['content'])
```

## Alternative: Tesseract OCR + Gemma 3 Text

If you prefer traditional OCR for images:

```bash
# OCR with Tesseract
tesseract handwritten.jpg stdout > words.txt

# Enrich with Gemma 3
python src/vocabulary_enricher.py words.txt

# Create Anki flashcards
python src/anki_exporter.py enriched.json
```

## Configuration

**config/settings.yaml:**
```yaml
ai_backend:
  type: 'llama_cpp'

llama_cpp:
  host: '127.0.0.1'
  port: 8080
  context_size: 8192
  n_gpu_layers: -1  # -1 = use all GPU layers, 0 = CPU only
```

## Recommended Workflow

### Option A: Vision-based OCR (Gemma 3 with mmproj)
