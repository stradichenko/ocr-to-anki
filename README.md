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
````

# OCR to Anki Pipeline

Convert images with handwritten or printed text into Anki flashcards using local Ollama vision models.

## Quick Start

### 1. Single Image OCR → Anki

```bash
# Extract words from image and create Anki notes in one pipeline
python src/test_model_ocr.py -w gemma3:4b data/images/handwritten.jpeg | \
  python src/ocr_to_json.py -o output/anki_notes.json --pretty

# Or save words first, then convert
python src/test_model_ocr.py -w gemma3:4b image.jpg > words.txt
python src/ocr_to_json.py -i words.txt -o notes.json
```

### 2. Batch Processing Multiple Images

```bash
# Process all images in input directory
python src/ollama_ocr.py

# This creates: output/all_words_YYYYMMDD_HHMMSS.txt
# Then convert to Anki format:
python src/ocr_to_json.py -i output/all_words_*.txt -o output/anki_notes.json
```

## Usage Examples

### Test Different Models

```bash
# Fast: Small model (gemma2:2b)
python src/test_model_ocr.py gemma2:2b image.jpg

# Balanced: Medium model (gemma3:4b)
python src/test_model_ocr.py gemma3:4b image.jpg

# Accurate: OCR-specialized model
python src/test_model_ocr.py deepseek-ocr:latest image.jpg
python src/test_model_ocr.py qwen2-vl:2b image.jpg
```

### Try Different Prompts

```bash
# Conversational (default, most natural)
python src/test_model_ocr.py gemma3:4b image.jpg conversational

# Simple (direct instruction)
python src/test_model_ocr.py gemma3:4b image.jpg simple

# Detailed (emphasis on accuracy)
python src/test_model_ocr.py gemma3:4b image.jpg detailed

# JSON (structured output)
python src/test_model_ocr.py gemma3:4b image.jpg json
```

### Complete Pipeline

```bash
# Extract → Filter → Convert → Import
python src/test_model_ocr.py --words-only gemma3:4b handwritten.jpg | \
  python src/ocr_to_json.py --min-length 3 --tag "french" --pretty | \
  python src/import_to_anki.py
```

## Configuration

Edit `config/settings.yaml`:

```yaml
ollama_ocr:
  model: "gemma3:4b"
  text_type: "handwritten"  # or "printed", "detect"
  language: "French"         # or "detect"
  analysis_scope: "all"      # or "highlighted"
  max_image_width: 800

import_defaults:
  deck: "French Vocabulary"
  model: "Basic"
  allow_duplicates: false
```

## Tips

1. **Model Selection**:
   - `gemma3:4b` - Best balance of speed and accuracy
   - `qwen2-vl:2b` - Specialized for vision tasks
   - `deepseek-ocr:latest` - Optimized for OCR

2. **Prompt Style**:
   - `conversational` - Works best with gemma3
   - `detailed` - Use for difficult handwriting
   - `json` - Use with OCR-specialized models

3. **Performance**:
   - Model stays loaded between requests
   - First request is slower (loads model)
   - Resize images to max 800px width

## Troubleshooting

```bash
# List available models
ollama list

# Pull a new model
ollama pull gemma3:4b

# Test model directly
ollama run gemma3:4b
>>> /image path/to/image.jpg
>>> Can you do OCR of this image?
```
