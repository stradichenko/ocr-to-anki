#!/usr/bin/env bash

set -euo pipefail

echo "Rebuilding llama.cpp with Vision Support"
echo "========================================"
echo

# Clone llama.cpp
BUILD_DIR="/tmp/llama.cpp-build"
rm -rf "$BUILD_DIR"
git clone --depth=1 https://github.com/ggerganov/llama.cpp "$BUILD_DIR"
cd "$BUILD_DIR"

echo "Building with LLAMA_CUBLAS=OFF and vision support..."
cmake -B build \
  -DLLAMA_CUBLAS=OFF \
  -DGGML_BACKEND_DL=ON \
  -DBUILD_SHARED_LIBS=ON \
  -DCMAKE_BUILD_TYPE=Release

cmake --build build --config Release --target llama-server llama-cli -j$(nproc)

# Install to user bin
mkdir -p "$HOME/.local/bin"
cp build/bin/llama-server "$HOME/.local/bin/"
cp build/bin/llama-cli "$HOME/.local/bin/"

echo
echo "✅ Built llama.cpp with vision support"
echo "   Installed to: $HOME/.local/bin/"
echo
echo "Add to PATH:"
echo "  export PATH=\"\$HOME/.local/bin:\$PATH\""
echo
echo "Test vision support:"
echo "  llama-server --help | grep mmproj"
