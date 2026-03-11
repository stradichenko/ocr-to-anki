<h1 align="center">
  OCR to Anki
</h1>

<h3 align="center">

![Build Status](https://github.com/stradichenko/ocr-to-anki/actions/workflows/build.yml/badge.svg)
![GitHub License](https://img.shields.io/github/license/stradichenko/ocr-to-anki)
![GitHub Release](https://img.shields.io/github/v/release/stradichenko/ocr-to-anki?include_prereleases)

</h3>

<h4 align="center">
  <a href="https://github.com/stradichenko/ocr-to-anki/releases">Download</a>
</h4>

<h4 align="center">
  Consider supporting:<br><br>
  <a href="https://www.patreon.com/8153512/join">
    <img src="https://img.shields.io/badge/Patreon-F96854?style=for-the-badge&logo=patreon&logoColor=white" alt="Patreon">
  </a>
  <a href="https://github.com/sponsors/stradichenko">
    <img src="https://img.shields.io/badge/sponsor-30363D?style=for-the-badge&logo=GitHub-Sponsors&logoColor=#EA4AAA" alt="GitHub Sponsors">
  </a>
</h4>

<h4 align="center">

[![Share on X](https://img.shields.io/badge/-Share%20on%20X-gray?style=flat&logo=x)](https://x.com/intent/tweet?text=OCR%20to%20Anki!%20Extract%20vocabulary%20from%20images%20and%20create%20flashcards%20offline%20with%20local%20AI.&url=https://github.com/stradichenko/ocr-to-anki&hashtags=Anki,OCR,LLM,llama)

</h4>

## About

Cross-platform desktop application for extracting vocabulary from images and
creating [Anki](https://apps.ankiweb.net/) flashcards. Everything runs locally
using [llama.cpp](https://github.com/ggerganov/llama.cpp) and the
[Gemma 3 4B](https://ai.google.dev/gemma/docs/gemma3) model. No cloud
dependencies, no API keys, fully offline.

The application is composed of two layers: a Flutter GUI that provides the user
interface and a Python FastAPI backend that handles vision OCR and vocabulary
enrichment through llama.cpp.

| Layer | Technology | Purpose |
|-------|-----------|---------|
| Flutter GUI | Dart, Material 3 | Desktop interface (Linux, macOS, Windows) |
| Python API | FastAPI, llama.cpp | Vision OCR and text enrichment backend |
| Vision OCR | llama-mtmd-cli | Extract text from images (GPU accelerated) |
| Text tasks | llama-server | Definitions, examples, vocabulary enrichment |
| Model | Gemma 3 4B QAT Q4_0 | Single model for both vision and text |

## Installation

### Prerequisites

Have [Nix](https://zero-to-nix.com/start/install) installed with flakes enabled
(`experimental-features = nix-command flakes` in your Nix configuration).

### Download a release (Linux)

Grab the latest tarball from the
[releases page](https://github.com/stradichenko/ocr-to-anki/releases),
extract, and run:

```bash
tar xzf ocr-to-anki-v0.1.0-linux-x86_64.tar.gz
cd ocr-to-anki-v0.1.0-linux-x86_64

# GTK3 is required at runtime.
# On Ubuntu/Debian: sudo apt install libgtk-3-0
# On Fedora:        sudo dnf install gtk3
# On NixOS:         already available

./run.sh
```

### Build from source

```bash
git clone https://github.com/stradichenko/ocr-to-anki.git
cd ocr-to-anki

# 1. Download the model and vision projector (~3 GB total, one time)
nix develop
./scripts/setup-llama-cpp.sh

# 2. Build the Flutter app
nix develop .#flutter
cd app
flutter pub get
flutter build linux --release

# The binary is at: app/build/linux/x64/release/bundle/ocr_to_anki
```

For a distributable tarball that bundles the backend source:

```bash
nix develop .#flutter --command ./scripts/build-flutter.sh linux
# Output: output/release/ocr-to-anki-v0.1.0-linux-x86_64.tar.gz
```

Or as a pure Nix derivation:

```bash
nix build .#flutter-app
./result/bin/ocr-to-anki
```

See [docs/building.md](docs/building.md) for macOS, Windows, and advanced build
options.

### Model files

| File | Size | Source |
|------|------|--------|
| gemma-3-4b-it-qat-q4_0_s.gguf | ~2.4 GB | [stduhpf/google-gemma-3-4b-it-qat-q4_0-gguf-small](https://huggingface.co/stduhpf/google-gemma-3-4b-it-qat-q4_0-gguf-small) |
| mmproj-model-f16-4B.gguf | ~812 MB | [google/gemma-3-4b-it-qat-q4_0-gguf](https://huggingface.co/google/gemma-3-4b-it-qat-q4_0-gguf) |

Both are downloaded by `./scripts/setup-llama-cpp.sh` via direct URL. No
authentication required.

Quantization-Aware Training (QAT) produces roughly 15% better perplexity than
standard post-training Q4_0 quantization at the same size. The stduhpf repack
also fixes broken control token metadata.

## Getting Started

### Workflow

1. Select context: handwritten or printed text, or highlighted words (pick
   colour)
2. Add images through the file picker or drag and drop
3. Vision OCR: Gemma 3 extracts words from the image
4. Enrich: the LLM generates definitions and example sentences
5. Review: edit the generated cards before export
6. Export: send to Anki via AnkiConnect, or save as TSV/JSON

### Starting the backend

The Flutter app manages the backend process automatically. When you launch the
app, it spawns the FastAPI server and waits until it reports healthy. No manual
server management is needed.

If you prefer to run the backend separately:

```bash
nix develop
PYTHONPATH=src uvicorn src.api.app:app --host 0.0.0.0 --port 8000
```

### Configuration

Edit `config/settings.yaml` to customize the backend:

```yaml
ai_backend:
  type: 'llama_cpp'

llama_cpp:
  host: '127.0.0.1'
  port: 8090
  context_size: 4096
  n_gpu_layers: -1
  mmproj_offload: false  # set true when using OpenCL backend
```

Most settings are also available through the in-app Settings screen.

## Building for other platforms

Flutter desktop does not support cross-compilation. Each platform must be built
on its native OS. The CI/CD workflow at `.github/workflows/build.yml` handles
this using platform-specific runners.

| Build host | Linux | macOS | Windows |
|------------|-------|-------|---------|
| Linux | yes | no | no |
| macOS | no | yes | no |
| Windows | no | no | yes |

### macOS

Requires a Mac with Xcode installed:

```bash
nix develop .#flutter
cd app && flutter pub get && flutter build macos --release
```

### Windows

Requires Visual Studio 2022 with the "Desktop development with C++" workload:

```powershell
cd app
flutter pub get
flutter build windows --release
```

### CI/CD

Push a version tag to trigger builds for all three platforms:

```bash
git tag v0.2.0
git push origin v0.2.0
```

This creates a draft GitHub Release with Linux, macOS, and Windows artifacts.
See [docs/building.md](docs/building.md) for the full reference.

## Building llama-mtmd-cli (vision)

The vision backend requires `llama-mtmd-cli` built with GPU support:

```bash
# OpenCL (recommended for Intel integrated GPUs)
nix develop .#sycl
./scripts/build-llama-mtmd-opencl.sh

# Vulkan (fallback, see note below)
./scripts/build-llama-mtmd-vulkan.sh
```

Auto-detection picks the best available backend: CUDA, Metal, OpenCL, Vulkan,
then CPU.

### Intel iGPU: OpenCL vs Vulkan

| Backend | Vision encoder | Encode time | Text gen | Binary |
|---------|---------------|-------------|----------|--------|
| OpenCL | correct | ~2 min (GPU) | 4.1 tok/s | llama-mtmd-cli-opencl |
| Vulkan | corrupted | 0.4s (garbage) | 3.6 tok/s | llama-mtmd-cli |
| CPU | correct | ~43 min | 0.7 tok/s | any binary with --no-mmproj-offload |

OpenCL is roughly 20x faster than CPU vision and produces correct output. It
requires a one-line patch for Intel work group sizes, applied automatically by
the build script. See
[patches/opencl-intel-workgroup-fix.patch](patches/opencl-intel-workgroup-fix.patch).

<details>
<summary>Vulkan corruption details</summary>

On Intel integrated GPUs (for example UHD Graphics CML GT2), the Vulkan compute
backend produces corrupted output from the SigLIP vision encoder. Text
generation works fine on Vulkan; only the vision projector is affected.

Root cause: Intel Vulkan compute shaders produce f16 underflow and overflow in
the CLIP/SigLIP transformer. Debug embeddings show 75%+ of values saturate to
exactly -1.0 (clamped NaN/inf). This is a
[known class of bug on integrated GPUs](https://github.com/ggml-org/llama.cpp/issues/15034).

If you have a discrete NVIDIA GPU, Vulkan and CUDA both work fine. Set
`mmproj_offload: true` in `config/settings.yaml`.

</details>

## API Endpoints

```
GET  /health                  Backend status
GET  /backends                Detected GPU hardware
POST /ocr/vision              Vision OCR (base64 image)
POST /ocr/vision/upload       Vision OCR (file upload)
POST /generate                Raw text generation
POST /enrich                  Vocabulary enrichment (definitions + examples)
POST /pipeline/image-to-cards Full pipeline: image to OCR to enrich to Anki cards
```

## Project Structure

```
app/                        Flutter GUI application
  lib/
    main.dart               Entry point and routing
    models/                 Data models (AnkiNote, AppSettings, HighlightColor)
    services/               Business logic
      inference_service.dart      LLM inference (talks to FastAPI)
      highlight_detector.dart     HSV highlight colour detection
      anki_export_service.dart    AnkiConnect and JSON export
      backend_server_service.dart Backend process lifecycle
    database/               Drift (SQLite) local storage
    providers/              Riverpod state management
    screens/                Home, Processing, Review, Settings, History
src/                        Python backend
  api/
    app.py                  FastAPI endpoints and lifespan hooks
    models.py               Pydantic request/response models
  backends/
    auto_detect.py          GPU and backend auto detection
    mtmd_cli.py             llama-mtmd-cli wrapper (vision, subprocess)
    llama_cpp_server.py     llama-server wrapper (text, persistent HTTP)
  preprocessing/
    highlight_cropper.py    HSV highlight detection (Python reference)
  workflows/                End to end pipelines
  output/                   Anki export and JSON output
config/
  settings.yaml             All configuration
scripts/                    Build and setup scripts
  build-flutter.sh          Build Flutter for Linux/macOS/Windows
  bundle-backend.sh         Bundle Python backend with PyInstaller
  setup-llama-cpp.sh        Download model and vision projector
  build-llama-mtmd-*.sh     Build llama-mtmd-cli with various GPU backends
docs/
  building.md               Full build and release documentation
```

## Nix Flake Outputs

### Development shells

```bash
nix develop             # Default: Python backend development
nix develop .#flutter   # Flutter app build and development
nix develop .#cuda      # With CUDA toolkit
nix develop .#sycl      # With Intel OneAPI/SYCL and OpenCL
```

### Packages

```bash
nix build .#flutter-app   # Flutter Linux desktop binary
nix build .#backend       # Nix-wrapped Python backend
nix build .#bundle        # Complete distribution (GUI + backend + launcher)
nix build .#dockerImage   # Docker image for server deployment
```

## License

[MIT](LICENSE)
