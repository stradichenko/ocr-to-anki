# Linters and Formatters

## What is a Linter?
A linter is a static code analysis tool that examines your code for potential errors, bugs, stylistic issues, and suspicious constructs without executing it. It helps identify problems like:

- Syntax errors
- Undefined variables
- Unused imports
- Code style violations
- Potential bugs

### Why Use a Linter If the Program Shows Errors Anyway?

While your program will display errors when you run it, a linter provides several advantages:

**Earlier Detection**: Catches issues while you're writing code, not after running it. This saves time in the development cycle.

**Broader Scope**: Detects problems beyond runtime errors, including style inconsistencies, code smells, and potential bugs that might not crash your program but could cause issues later.

**Better Feedback**: Provides specific, actionable suggestions with context about why something is problematic.

**Prevents Silent Failures**: Identifies issues that might not cause immediate errors but lead to bugs in production (e.g., unused variables, unreachable code).

**Development Speed**: You can fix issues as you type rather than going through a compile-run-debug cycle.

In short: Yes, using a linter is better because it acts as a first line of defense, catching issues before they become runtime problems.

## What is a Formatter?

A formatter is a tool that automatically restructures your code to follow a consistent style and layout. It handles:

- Indentation
- Line breaks
- Spacing
- Quote styles
- Line length

## What is an LSP?

LSP (Language Server Protocol) is a standardized protocol that enables communication between code editors and language servers. It provides intelligent code features like:

- Autocompletion
- Go to definition
- Find references
- Hover documentation
- Real-time error highlighting
- Code refactoring

The LSP separates the language-specific logic (language server) from the editor, allowing any editor that supports LSP to work with any language server. This means you get consistent intelligent features across different editors without each editor needing custom implementations for every language.

## Why Are They Important?

**Code Quality**: Linters catch bugs and problematic patterns before runtime.

**Consistency**: Formatters ensure uniform code style across teams and projects.

**Readability**: Well-formatted, clean code is easier to understand and maintain.

**Efficiency**: Automated formatting saves time in code reviews by eliminating style debates.

**Best Practices**: Linters enforce coding standards and help developers learn better practices.

**Collaboration**: Consistent code style reduces friction when multiple developers work on the same codebase.


## Documentation Comment

This code appears to be related to a Nix flake project setup. Here's the documentation:

---

**Regarding direnv and nix-direnv for flake.nix projects:**

Yes, **direnv** and **nix-direnv** are highly useful for projects using `flake.nix`:

### What's the Difference?

**direnv**: A general-purpose environment switcher that loads/unloads environment variables based on directory. Works with any shell and any project type (not just Nix).

**nix-direnv**: An extension for direnv that adds Nix-specific optimizations. It caches the development environment created by `nix develop` or `nix-shell`, making subsequent loads much faster.

### Should They Be Used Together?

**Yes, they work best together**:
- direnv provides the core functionality (automatic environment switching)
- nix-direnv adds Nix performance improvements on top

You don't replace one with the other; nix-direnv enhances direnv for Nix projects.

### Benefits:

1. **Automatic Environment Loading**: Automatically loads the development environment when you `cd` into the project directory
2. **Faster Shell Loading**: `nix-direnv` caches the development environment, making subsequent shell loads nearly instantaneous (compared to `nix develop` which can take seconds)
3. **Editor Integration**: Your editor/IDE automatically picks up the development environment without manual activation
4. **Reduced Memory Usage**: Shares the nix store efficiently across terminal sessions
5. **Flake-Specific Support**: `nix-direnv` has excellent support for flakes with the `use flake` directive

### Setup:

1. Install direnv and nix-direnv
2. Create a `.envrc` file in your project root:
    ```bash
    use flake
    ```
    The `.envrc` file tells direnv which environment to load when you enter this directory. The `use flake` command instructs nix-direnv to load the development environment defined in your `flake.nix`.
    
3. Run `direnv allow` to activate
    
    This command grants direnv permission to load the `.envrc` file. This is a security feature that prevents arbitrary code execution from untrusted directories. After running this, direnv will:
    - Build and cache your flake's development environment
    - Automatically load it whenever you `cd` into the directory
    - Unload it when you leave the directory

### Result:
- No need to manually run `nix develop` every time
- Environment variables, tools, and dependencies are automatically available
- Faster iteration and better developer experience

**Recommendation**: Strongly recommended for flake-based projects with active development. Use both direnv and nix-direnv together for optimal results.

---

## Ollama OCR Timeout Troubleshooting

If you're experiencing timeout errors with Ollama OCR:

### Common Causes

1. **Large Images**: Images over 2MB or with dimensions > 2000px take longer to process
2. **Model Loading**: First request after starting Ollama loads the model into memory (can take 30-60s)
3. **System Resources**: Insufficient RAM or CPU can slow down inference
4. **Model Size**: Larger models (7B, 13B) take significantly longer than 2B models

### Solutions

**Resize Images**: The test script automatically resizes images to 800px width max. For production:
```python
# Resize before encoding
from PIL import Image
img = Image.open('large.jpg')
img.thumbnail((1024, 1024))
img.save('resized.jpg')
```

**Increase Timeout**: In `config/settings.yaml`:
```yaml
ollama_ocr:
  timeout: 300  # 5 minutes instead of 60 seconds
```

**Warm Up the Model**: Run a test request first:
```bash
curl http://localhost:11434/api/generate -d '{
  "model": "qwen3-vl:2b",
  "prompt": "Hello",
  "stream": false
}'
```

**Check Ollama Status**:
```bash
# View running models
ollama ps

# Check available models
ollama list

# Monitor logs
ollama serve
```

**Reduce Response Length**: Add to request payload:
```python
payload = {
    "model": "qwen3-vl:2b",
    "prompt": prompt,
    "images": [image_base64],
    "options": {
        "num_predict": 500,    # Limit tokens
        "temperature": 0.1     # More focused output
    }
}
```

**Use GPU if Available**: Ollama automatically uses GPU when available. Check with:
```bash
nvidia-smi  # For NVIDIA GPUs
```

**Try Smaller Model**: If timeouts persist, test with a smaller model first:
```bash
ollama pull llava:7b
# Then update config to use llava:7b
```

## llama.cpp Offline Inference

### Why llama.cpp Instead of Ollama?

**Privacy & Offline:**
- llama.cpp: Fully offline after initial setup, no telemetry
- Ollama: Requires internet for model downloads, may phone home

**Performance:**
- llama.cpp: Direct GGUF inference, optimized for CPU/GPU
- Ollama: Additional abstraction layer

**Control:**
- llama.cpp: Full control over model parameters and memory
- Ollama: Managed service with less configurability

### Setup Issues

**Model download fails:**
```bash
# Try with curl instead of wget
curl -L -o ~/.cache/llama.cpp/models/gemma-3-4b-it-q4_0.gguf \
  https://huggingface.co/fernandoruiz/gemma-3-4b-it-Q4_0-GGUF/resolve/main/gemma-3-4b-it-q4_0.gguf
```

**Server won't start:**
```bash
# Check if llama-server is available
which llama-server

# If not, ensure you're in Nix shell
nix develop

# Check model exists
ls -lh ~/.cache/llama.cpp/models/gemma-3-4b-it-q4_0.gguf
```

**Slow generation (CPU only):**
```yaml
# In config/settings.yaml, reduce context size
llama_cpp:
  context_size: 2048  # Down from 4096
  max_tokens: 256     # Limit response length
```

**Out of memory errors:**
```yaml
# Reduce GPU layers for lower VRAM usage
llama_cpp:
  n_gpu_layers: 20  # Instead of -1 (all layers)
```

### Performance Tuning

**CPU Optimization:**
```bash
# Use BLAS acceleration (if available)
llama-server --model model.gguf --threads $(nproc)
```

**GPU Optimization:**
```yaml
# Maximize GPU usage
llama_cpp:
  n_gpu_layers: -1      # All layers on GPU
  context_size: 8192    # Larger context
```

**Memory vs Speed Trade-off:**
- Q4_0: Balanced (recommended)
- Q8_0: Higher quality, 2x larger
- Q2_K: Smaller, faster, lower quality