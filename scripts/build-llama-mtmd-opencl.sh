#!/usr/bin/env bash
set -euo pipefail

# Build llama-mtmd-cli with OpenCL GPU backend
# Usage: ./scripts/build-llama-mtmd-opencl.sh [--clean]
#
# Prerequisites:
#   nix develop --impure .#sycl   (provides OpenCL headers, ICD loader, Level Zero)
#
# This is the RECOMMENDED backend for Intel GPUs older than 11th gen (Gen9, Gen11).
# Uses standard gcc -- no Intel oneAPI compiler needed.
#
# Produces a binary at ~/.local/bin/llama-mtmd-cli-opencl

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
BUILD_DIR="$PROJECT_DIR/.build/llama-cpp-opencl"
INSTALL_DIR="$HOME/.local/bin"
LLAMA_CPP_REPO="https://github.com/ggerganov/llama.cpp.git"
LLAMA_CPP_REF="${LLAMA_CPP_REF:-master}"
BINARY_NAME="llama-mtmd-cli-opencl"

CLEAN=false
if [[ "${1:-}" == "--clean" ]]; then
    CLEAN=true
fi

echo "╔══════════════════════════════════════════════════╗"
echo "║  Build llama-mtmd-cli with OpenCL GPU backend    ║"
echo "╚══════════════════════════════════════════════════╝"
echo ""

# ------------------------------------------------------------------
# 1. Check required tools
# ------------------------------------------------------------------
check_tool() {
    if ! command -v "$1" &>/dev/null; then
        echo "[ERR] Missing: $1"
        echo "   Install via: $2"
        exit 1
    fi
}

check_tool cmake    "nix develop .#sycl"
check_tool gcc      "nix develop .#sycl"
check_tool git      "nix develop .#sycl"
check_tool pkg-config "nix develop .#sycl"
check_tool python3  "nix develop .#sycl"

# Check OpenCL
HAVE_OPENCL=false
if pkg-config --exists OpenCL 2>/dev/null; then
    HAVE_OPENCL=true
    echo "[OK] OpenCL: $(pkg-config --modversion OpenCL)"
elif [[ -f /etc/OpenCL/vendors/*.icd ]] 2>/dev/null; then
    echo "[OK] OpenCL: ICD files found"
    HAVE_OPENCL=true
fi

if [[ "$HAVE_OPENCL" != true ]]; then
    echo "[ERR] OpenCL not found"
    echo "   Make sure you're in the sycl dev shell: nix develop --impure .#sycl"
    echo "   And that intel-compute-runtime is available"
    exit 1
fi

# Check Intel compute runtime
echo ""
echo "GPU Detection:"
if command -v clinfo &>/dev/null; then
    PLATFORMS=$(clinfo -l 2>/dev/null | head -10)
    if [[ -n "$PLATFORMS" ]]; then
        echo "  [OK] OpenCL platforms/devices:"
        echo "$PLATFORMS" | sed 's/^/     /'
    else
        echo "  [WARN] No OpenCL platforms detected by clinfo"
        echo "     Intel compute-runtime may need to be installed system-wide"
    fi
else
    echo "  [INFO] clinfo not available -- will detect at runtime"
fi

echo ""
echo "[OK] All build dependencies found"
echo "   cmake:   $(cmake --version | head -1)"
echo "   gcc:     $(gcc --version | head -1)"
echo "   python3: $(python3 --version)"
echo ""

# ------------------------------------------------------------------
# 2. Clone / update llama.cpp
# ------------------------------------------------------------------
SRC_DIR="$BUILD_DIR/llama.cpp"

if [[ "$CLEAN" == true ]] && [[ -d "$BUILD_DIR" ]]; then
    echo ":: Cleaning previous OpenCL build..."
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
# 2b. Apply patches
# ------------------------------------------------------------------
PATCH_DIR="$PROJECT_DIR/patches"
if [[ -f "$PATCH_DIR/opencl-intel-workgroup-fix.patch" ]]; then
    echo ":: Applying Intel work group size fix..."
    # Check if patch is already applied
    if grep -q "max_workgroup_size" "$SRC_DIR/ggml/src/ggml-opencl/ggml-opencl.cpp" 2>/dev/null \
       && grep -A2 "nth = 512" "$SRC_DIR/ggml/src/ggml-opencl/ggml-opencl.cpp" | grep -q "max_workgroup_size"; then
        echo "   Already applied, skipping"
    else
        cd "$SRC_DIR"
        git apply "$PATCH_DIR/opencl-intel-workgroup-fix.patch" 2>/dev/null \
            || patch -p1 < "$PATCH_DIR/opencl-intel-workgroup-fix.patch" 2>/dev/null \
            || {
                echo "   [WARN] Patch failed to apply cleanly -- applying manually..."
                # Manual fix: clamp GLU kernel work group size to device max
                sed -i '/const size_t nrows = ggml_nrows(src0);/{
                    N
                    s/\(size_t nth = 512;\)/\1\n    if (nth > backend_ctx->max_workgroup_size) {\n        nth = backend_ctx->max_workgroup_size;\n    }/
                }' "$SRC_DIR/ggml/src/ggml-opencl/ggml-opencl.cpp"
                echo "   Applied via sed"
            }
    fi
    echo ""
fi

# ------------------------------------------------------------------
# 3. Configure with CMake (OpenCL)
# ------------------------------------------------------------------
CMAKE_BUILD="$BUILD_DIR/build"
mkdir -p "$CMAKE_BUILD"

echo ":: Configuring CMake with OpenCL backend..."

cmake -B "$CMAKE_BUILD" -S "$SRC_DIR" \
    -DCMAKE_BUILD_TYPE=Release \
    -DGGML_OPENCL=ON \
    -DGGML_OPENCL_EMBED_KERNELS=ON \
    -DGGML_OPENCL_USE_ADRENO_KERNELS=OFF \
    -DGGML_LLAMAFILE=ON \
    -DGGML_OPENMP=ON \
    -DGGML_NATIVE=ON \
    -DLLAMA_CURL=OFF \
    -DBUILD_SHARED_LIBS=OFF \
    -DGGML_STATIC=ON \
    -DCMAKE_INSTALL_PREFIX="$BUILD_DIR/install" \
    2>&1 | tee "$BUILD_DIR/cmake-configure.log"

echo ""

# ------------------------------------------------------------------
# 4. Build
# ------------------------------------------------------------------
NPROC=$(nproc 2>/dev/null || echo 4)
echo ":: Building llama-mtmd-cli with OpenCL (using $NPROC cores)..."
echo "   This may take a few minutes..."
echo ""

cmake --build "$CMAKE_BUILD" --config Release --target llama-mtmd-cli -j"$NPROC" 2>&1 \
    | tee "$BUILD_DIR/build.log" \
    | tail -10

echo ""

# ------------------------------------------------------------------
# 5. Find and install the binary
# ------------------------------------------------------------------
BINARY=$(find "$CMAKE_BUILD" -name "llama-mtmd-cli" -type f -executable 2>/dev/null | head -1)

if [[ -z "$BINARY" ]]; then
    echo "[ERR] Build failed -- binary not found"
    echo "   Check logs: $BUILD_DIR/build.log"
    exit 1
fi

echo "[OK] Binary built: $BINARY"
echo "   Size: $(du -h "$BINARY" | cut -f1)"

# Install
mkdir -p "$INSTALL_DIR"
cp "$BINARY" "$INSTALL_DIR/$BINARY_NAME"
chmod +x "$INSTALL_DIR/$BINARY_NAME"

echo ""
echo "╔══════════════════════════════════════════════════╗"
echo "║  [OK] OpenCL binary installed:                      ║"
echo "║     $INSTALL_DIR/$BINARY_NAME"
echo "╚══════════════════════════════════════════════════╝"
echo ""
echo "Test with:"
echo "  $BINARY_NAME \\"
echo "    -m ~/.cache/llama.cpp/models/gemma-3-4b-it-qat-q4_0_s.gguf \\"
echo "    --mmproj ~/.cache/llama.cpp/models/mmproj-model-f16-4B.gguf \\"
echo "    --image data/images/handwritten.jpeg \\"
echo "    -p 'Read the handwritten French words. List each word, one per line.' \\"
echo "    --jinja -ngl 99 -n 256"
echo ""
echo "To replace Vulkan binary: cp $INSTALL_DIR/$BINARY_NAME $INSTALL_DIR/llama-mtmd-cli"
echo ""
