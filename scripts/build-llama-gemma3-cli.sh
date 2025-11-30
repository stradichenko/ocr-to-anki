#!/usr/bin/env bash

set -euo pipefail

echo "Building llama-mtmd-cli (Multi-Modal CLI for Gemma 3 Vision)"
echo "============================================================="
echo

BUILD_DIR="/tmp/llama.cpp-mtmd"
INSTALL_DIR="$HOME/.local/bin"

# Clone llama.cpp
echo "1. Cloning llama.cpp..."
rm -rf "$BUILD_DIR"
git clone --depth=1 https://github.com/ggerganov/llama.cpp "$BUILD_DIR"
cd "$BUILD_DIR"

echo "✓ Cloned llama.cpp"
echo

# Build llama-mtmd-cli (the new name for multimodal CLI)
echo "2. Building llama-mtmd-cli..."
echo "   Note: llama-gemma3-cli is now deprecated, using llama-mtmd-cli"
echo "   (This will take 2-5 minutes)"
echo

# Disable CURL since we don't need it for vision CLI
cmake -B build \
  -DCMAKE_BUILD_TYPE=Release \
  -DGGML_LLAMAFILE=ON \
  -DGGML_OPENMP=ON \
  -DLLAMA_CURL=OFF

# Build the llama-mtmd-cli target
cmake --build build --config Release --target llama-mtmd-cli -j$(nproc)

if [ ! -f "build/bin/llama-mtmd-cli" ]; then
    echo "❌ Build failed! llama-mtmd-cli not found"
    echo
    echo "Checking what was built:"
    ls -la build/bin/ || echo "No binaries found"
    exit 1
fi

echo "✓ Built successfully"
echo

# Install to user bin
echo "3. Installing to $INSTALL_DIR..."
mkdir -p "$INSTALL_DIR"
cp build/bin/llama-mtmd-cli "$INSTALL_DIR/"
chmod +x "$INSTALL_DIR/llama-mtmd-cli"

# Create a symlink with the old name for backwards compatibility
ln -sf "$INSTALL_DIR/llama-mtmd-cli" "$INSTALL_DIR/llama-gemma3-cli"

echo "✓ Installed"
echo

# Verify
echo "4. Verifying installation..."
if [ -f "$INSTALL_DIR/llama-mtmd-cli" ]; then
    echo "✓ llama-mtmd-cli installed at: $INSTALL_DIR/llama-mtmd-cli"
    echo "✓ Symlink created: llama-gemma3-cli -> llama-mtmd-cli"
    
    # Test help
    "$INSTALL_DIR/llama-mtmd-cli" --help 2>&1 | head -10 || true
fi

echo
echo "✅ Setup complete!"
echo
echo "llama-mtmd-cli installed at: $INSTALL_DIR/llama-mtmd-cli"
echo "  (llama-gemma3-cli symlink also created for compatibility)"
echo
echo "Add to PATH if not already:"
echo "  export PATH=\"$INSTALL_DIR:\$PATH\""
echo
echo "Next steps:"
echo "  1. Test with: python src/vision_ocr_gemma3.py data/images/handwritten.jpeg"
echo
