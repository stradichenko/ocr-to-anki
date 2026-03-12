#!/usr/bin/env bash
set -euo pipefail

# Build llama-mtmd-cli with Vulkan GPU backend
# Usage: ./scripts/build-llama-mtmd-vulkan.sh [--clean]
#
# Produces a statically-linked binary at ~/.local/bin/llama-mtmd-cli

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
BUILD_DIR="$PROJECT_DIR/.build/llama-cpp-vulkan"
INSTALL_DIR="$HOME/.local/bin"
LLAMA_CPP_REPO="https://github.com/ggerganov/llama.cpp.git"

# Use a known-good tag/commit
LLAMA_CPP_REF="${LLAMA_CPP_REF:-master}"

CLEAN=false
if [[ "${1:-}" == "--clean" ]]; then
    CLEAN=true
fi

echo "╔══════════════════════════════════════════════════╗"
echo "║  Build llama-mtmd-cli with Vulkan GPU backend    ║"
echo "╚══════════════════════════════════════════════════╝"
echo ""

# ------------------------------------------------------------------
# 1. Check required tools
# ------------------------------------------------------------------
check_tool() {
    if ! command -v "$1" &>/dev/null; then
        echo "[ERR] Missing: $1"
        echo "   Install via: nix-shell -p $2"
        exit 1
    fi
}

check_tool cmake cmake
check_tool gcc gcc
check_tool git git
check_tool pkg-config pkg-config

# Check Vulkan
if ! pkg-config --exists vulkan 2>/dev/null; then
    echo "[ERR] Vulkan not found via pkg-config"
    echo "   Make sure vulkan-headers and vulkan-loader are available"
    echo "   Try: nix-shell -p vulkan-headers vulkan-loader vulkan-tools shaderc"
    exit 1
fi

# Check glslc (shader compiler, from shaderc)
check_tool glslc shaderc

echo "[OK] All build dependencies found"
echo "   cmake:   $(cmake --version | head -1)"
echo "   gcc:     $(gcc --version | head -1)"
echo "   vulkan:  $(pkg-config --modversion vulkan)"
echo "   glslc:   $(glslc --version 2>&1 | head -1)"
echo ""

# ------------------------------------------------------------------
# 2. Clone / update llama.cpp
# ------------------------------------------------------------------
SRC_DIR="$BUILD_DIR/llama.cpp"

if [[ "$CLEAN" == true ]] && [[ -d "$BUILD_DIR" ]]; then
    echo ":: Cleaning previous build..."
    rm -rf "$BUILD_DIR"
fi

mkdir -p "$BUILD_DIR"

if [[ -d "$SRC_DIR/.git" ]]; then
    echo ":: Updating llama.cpp..."
    cd "$SRC_DIR"
    git fetch --depth 1 origin "$LLAMA_CPP_REF"
    git checkout FETCH_HEAD
else
    echo ":: Cloning llama.cpp ($LLAMA_CPP_REF)..."
    git clone --depth 1 --branch "$LLAMA_CPP_REF" "$LLAMA_CPP_REPO" "$SRC_DIR" 2>/dev/null \
        || git clone --depth 1 "$LLAMA_CPP_REPO" "$SRC_DIR"
fi

cd "$SRC_DIR"
COMMIT=$(git rev-parse --short HEAD)
echo "   Commit: $COMMIT"
echo ""

# ------------------------------------------------------------------
# 3. Configure with CMake (Vulkan + static)
# ------------------------------------------------------------------
CMAKE_BUILD="$BUILD_DIR/build"
mkdir -p "$CMAKE_BUILD"

echo ":: Configuring CMake with Vulkan backend..."

cmake -B "$CMAKE_BUILD" -S "$SRC_DIR" \
    -DCMAKE_BUILD_TYPE=Release \
    -DGGML_VULKAN=ON \
    -DGGML_LLAMAFILE=ON \
    -DGGML_OPENMP=ON \
    -DGGML_NATIVE=ON \
    -DLLAMA_CURL=OFF \
    -DBUILD_SHARED_LIBS=OFF \
    -DGGML_STATIC=ON \
    -DCMAKE_INSTALL_PREFIX="$BUILD_DIR/install" \
    2>&1 | tee "$BUILD_DIR/cmake-configure.log"

echo ""

# Verify Vulkan was actually found
if grep -q "GGML_VULKAN" "$CMAKE_BUILD/CMakeCache.txt" 2>/dev/null; then
    VULKAN_STATUS=$(grep "GGML_VULKAN:BOOL" "$CMAKE_BUILD/CMakeCache.txt" || echo "UNKNOWN")
    echo "   Vulkan config: $VULKAN_STATUS"
fi
echo ""

# ------------------------------------------------------------------
# 4. Build
# ------------------------------------------------------------------
NPROC=$(nproc 2>/dev/null || echo 4)
echo ":: Building llama-mtmd-cli (using $NPROC cores)..."
echo "   This may take a few minutes..."
echo ""

cmake --build "$CMAKE_BUILD" --config Release --target llama-mtmd-cli -j"$NPROC" 2>&1 \
    | tee "$BUILD_DIR/build.log" \
    | tail -5

echo ""

# ------------------------------------------------------------------
# 5. Find and install the binary
# ------------------------------------------------------------------
BINARY=$(find "$CMAKE_BUILD" -name "llama-mtmd-cli" -type f -executable 2>/dev/null | head -1)

if [[ -z "$BINARY" ]]; then
    echo "[ERR] Build failed - binary not found"
    echo "   Check logs: $BUILD_DIR/build.log"
    exit 1
fi

echo "[OK] Binary built: $BINARY"
echo "   Size: $(du -h "$BINARY" | cut -f1)"

# Verify it works
if "$BINARY" --version 2>&1 | head -1; then
    echo "   [OK] Binary executes correctly"
else
    echo "   [WARN] Binary may have runtime issues"
fi

# Install
mkdir -p "$INSTALL_DIR"
cp "$BINARY" "$INSTALL_DIR/llama-mtmd-cli"
chmod +x "$INSTALL_DIR/llama-mtmd-cli"

echo ""
echo "╔══════════════════════════════════════════════════╗"
echo "║  [OK] llama-mtmd-cli installed to:                 ║"
echo "║     $INSTALL_DIR/llama-mtmd-cli"
echo "╚══════════════════════════════════════════════════╝"
echo ""
echo "Test with:"
echo "  llama-mtmd-cli \\"
echo "    -m ~/.cache/llama.cpp/models/gemma-3-4b-it-q4_0_s.gguf \\"
echo "    --mmproj ~/.cache/llama.cpp/models/mmproj-model-f16-4B.gguf \\"
echo "    --image data/images/test.jpg \\"
echo "    -p 'What text do you see in this image?'"
echo ""
