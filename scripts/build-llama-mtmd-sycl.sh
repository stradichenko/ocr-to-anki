#!/usr/bin/env bash
set -euo pipefail

# Build llama-mtmd-cli with Intel SYCL GPU backend
# Usage: ./scripts/build-llama-mtmd-sycl.sh [--clean]
#
# Prerequisites:
#   1. Enter the SYCL dev shell:  nix develop --impure .#sycl
#   2. Install Intel oneAPI Base Toolkit (if not already):
#        ./scripts/install-oneapi.sh
#      Or manually:
#        wget https://registrationcenter-download.intel.com/akdlm/IRC_NAS/...</
#        sudo bash l_BaseKit_p_*.sh -a --silent --eula accept
#   3. Run this script
#
# Produces a dynamically-linked binary at ~/.local/bin/llama-mtmd-cli-sycl
#
# Intel GPU compatibility:
#   - Gen9 / Gen9.5 (CML GT2, 10th gen): borderline — uses intel-compute-runtime-legacy1
#   - Gen11+ (11th gen+): fully supported
#   - Set GGML_SYCL_DISABLE_OPT=1 at runtime for pre-Gen12 devices

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
BUILD_DIR="$PROJECT_DIR/.build/llama-cpp-sycl"
INSTALL_DIR="$HOME/.local/bin"
LLAMA_CPP_REPO="https://github.com/ggerganov/llama.cpp.git"
LLAMA_CPP_REF="${LLAMA_CPP_REF:-master}"
BINARY_NAME="llama-mtmd-cli-sycl"

CLEAN=false
if [[ "${1:-}" == "--clean" ]]; then
    CLEAN=true
fi

echo "╔══════════════════════════════════════════════════╗"
echo "║  Build llama-mtmd-cli with Intel SYCL backend    ║"
echo "╚══════════════════════════════════════════════════╝"
echo ""

# ------------------------------------------------------------------
# 1. Locate and source Intel oneAPI
# ------------------------------------------------------------------
ONEAPI_ROOT="${ONEAPI_ROOT:-/opt/intel/oneapi}"

if [[ ! -d "$ONEAPI_ROOT" ]]; then
    echo "❌ Intel oneAPI not found at: $ONEAPI_ROOT"
    echo ""
    echo "Install oneAPI Base Toolkit:"
    echo "  1. Download from:"
    echo "     https://www.intel.com/content/www/us/en/developer/tools/oneapi/base-toolkit-download.html"
    echo "     (Choose: Linux → Offline installer)"
    echo ""
    echo "  2. On NixOS, install inside an FHS sandbox:"
    echo "     steam-run bash -c 'sudo bash ./l_BaseKit_p_*.sh -a --silent --eula accept'"
    echo ""
    echo "  3. Or set ONEAPI_ROOT to a custom location."
    exit 1
fi

echo "📦 Sourcing Intel oneAPI from: $ONEAPI_ROOT"
if [[ -f "$ONEAPI_ROOT/setvars.sh" ]]; then
    # shellcheck disable=SC1091
    source "$ONEAPI_ROOT/setvars.sh" --force 2>/dev/null || true
else
    echo "❌ setvars.sh not found in $ONEAPI_ROOT"
    exit 1
fi

# ------------------------------------------------------------------
# 2. Check required tools
# ------------------------------------------------------------------
check_tool() {
    if ! command -v "$1" &>/dev/null; then
        echo "❌ Missing: $1 ($2)"
        exit 1
    fi
}

echo ""
echo "Checking SYCL build dependencies..."

check_tool cmake       "cmake from nix"
check_tool git         "git from nix"
check_tool pkg-config  "pkg-config from nix"

# SYCL compiler
if command -v icpx &>/dev/null; then
    CXX_COMPILER=icpx
    C_COMPILER=icx
    SYCL_TYPE="oneAPI DPC++"
    echo "  ✅ icpx:  $(icpx --version 2>&1 | head -1)"
    echo "  ✅ icx:   $(icx --version 2>&1 | head -1)"
else
    echo "❌ Intel DPC++ compiler (icpx) not found."
    echo "   Make sure oneAPI was sourced: source $ONEAPI_ROOT/setvars.sh"
    echo ""
    echo "   Alternatively, install only the compiler component:"
    echo "   sudo $ONEAPI_ROOT/installer/installer --list-components"
    exit 1
fi

# Level Zero
if pkg-config --exists level-zero 2>/dev/null; then
    echo "  ✅ Level Zero: $(pkg-config --modversion level-zero)"
elif [[ -f "$ONEAPI_ROOT/compiler/latest/lib/libze_loader.so" ]]; then
    echo "  ✅ Level Zero: provided by oneAPI"
else
    echo "  ⚠️  Level Zero not found via pkg-config (may be provided by oneAPI)"
fi

# MKL
if pkg-config --exists mkl-sycl-blas 2>/dev/null; then
    echo "  ✅ MKL:   $(pkg-config --modversion mkl-sycl-blas 2>/dev/null || echo 'found')"
elif [[ -d "$ONEAPI_ROOT/mkl" ]] || [[ -d "$MKLROOT" ]]; then
    echo "  ✅ MKL:   provided by oneAPI (MKLROOT=${MKLROOT:-$ONEAPI_ROOT/mkl/latest})"
else
    echo "  ⚠️  MKL not detected (required by SYCL backend)"
fi

# Intel GPU driver
echo ""
echo "GPU Detection:"
if command -v sycl-ls &>/dev/null; then
    SYCL_DEVICES=$(sycl-ls 2>/dev/null | grep -i "Intel" | head -5)
    if [[ -n "$SYCL_DEVICES" ]]; then
        echo "  ✅ SYCL devices:"
        echo "$SYCL_DEVICES" | sed 's/^/     /'
    else
        echo "  ⚠️  No Intel SYCL devices found by sycl-ls"
    fi
elif command -v ze_info &>/dev/null; then
    echo "  (Using ze_info for device detection)"
    ze_info 2>/dev/null | grep -i "name\|driver\|vendor" | head -5 | sed 's/^/     /'
else
    echo "  ℹ️  No sycl-ls or ze_info — will detect at runtime"
fi
echo ""

# ------------------------------------------------------------------
# 3. Clone / update llama.cpp
# ------------------------------------------------------------------
SRC_DIR="$BUILD_DIR/llama.cpp"

if [[ "$CLEAN" == true ]] && [[ -d "$BUILD_DIR" ]]; then
    echo "🧹 Cleaning previous SYCL build..."
    rm -rf "$BUILD_DIR"
fi

mkdir -p "$BUILD_DIR"

if [[ -d "$SRC_DIR/.git" ]]; then
    echo "📦 Updating llama.cpp..."
    cd "$SRC_DIR"
    git fetch --depth 1 origin "$LLAMA_CPP_REF"
    git checkout FETCH_HEAD
else
    echo "📦 Cloning llama.cpp ($LLAMA_CPP_REF)..."
    git clone --depth 1 --branch "$LLAMA_CPP_REF" "$LLAMA_CPP_REPO" "$SRC_DIR" 2>/dev/null \
        || git clone --depth 1 "$LLAMA_CPP_REPO" "$SRC_DIR"
fi

cd "$SRC_DIR"
COMMIT=$(git rev-parse --short HEAD)
echo "   Commit: $COMMIT"
echo ""

# ------------------------------------------------------------------
# 4. Configure with CMake (SYCL)
# ------------------------------------------------------------------
CMAKE_BUILD="$BUILD_DIR/build"
mkdir -p "$CMAKE_BUILD"

echo "⚙️  Configuring CMake with SYCL backend..."

# SYCL build flags
# - GGML_SYCL=ON           → Enable SYCL backend
# - GGML_SYCL_F16=OFF      → Use float32 (safer on older hardware)
# - GGML_SYCL_DNN=ON       → Enable oneDNN if available
# - GGML_SYCL_GRAPH=OFF    → Experimental, disabled
# - GGML_NATIVE=OFF        → Don't use -march=native (safer for SYCL)
cmake -B "$CMAKE_BUILD" -S "$SRC_DIR" \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_C_COMPILER="$C_COMPILER" \
    -DCMAKE_CXX_COMPILER="$CXX_COMPILER" \
    -DGGML_SYCL=ON \
    -DGGML_SYCL_F16=OFF \
    -DGGML_SYCL_GRAPH=OFF \
    -DGGML_SYCL_DNN=ON \
    -DGGML_OPENMP=ON \
    -DGGML_NATIVE=OFF \
    -DLLAMA_CURL=OFF \
    -DCMAKE_INSTALL_PREFIX="$BUILD_DIR/install" \
    2>&1 | tee "$BUILD_DIR/cmake-configure.log"

echo ""

# Verify SYCL was found
if grep -q "SYCL found" "$BUILD_DIR/cmake-configure.log" 2>/dev/null; then
    echo "   ✅ SYCL detected by CMake"
else
    echo "   ⚠️  Check cmake output above for SYCL detection"
fi
echo ""

# ------------------------------------------------------------------
# 5. Build
# ------------------------------------------------------------------
NPROC=$(nproc 2>/dev/null || echo 4)
echo "🔨 Building llama-mtmd-cli with SYCL (using $NPROC cores)..."
echo "   This may take several minutes..."
echo ""

cmake --build "$CMAKE_BUILD" --config Release --target llama-mtmd-cli -j"$NPROC" 2>&1 \
    | tee "$BUILD_DIR/build.log" \
    | tail -10

echo ""

# ------------------------------------------------------------------
# 6. Find and install the binary
# ------------------------------------------------------------------
BINARY=$(find "$CMAKE_BUILD" -name "llama-mtmd-cli" -type f -executable 2>/dev/null | head -1)

if [[ -z "$BINARY" ]]; then
    echo "❌ Build failed — binary not found"
    echo "   Check logs: $BUILD_DIR/build.log"
    exit 1
fi

echo "✅ Binary built: $BINARY"
echo "   Size: $(du -h "$BINARY" | cut -f1)"

# Install as separate binary name to coexist with Vulkan build
mkdir -p "$INSTALL_DIR"
cp "$BINARY" "$INSTALL_DIR/$BINARY_NAME"
chmod +x "$INSTALL_DIR/$BINARY_NAME"

# Also create/update the generic name
cp "$BINARY" "$INSTALL_DIR/llama-mtmd-cli"
chmod +x "$INSTALL_DIR/llama-mtmd-cli"

echo ""
echo "╔══════════════════════════════════════════════════╗"
echo "║  ✅ SYCL binaries installed:                     ║"
echo "║     $INSTALL_DIR/$BINARY_NAME"
echo "║     $INSTALL_DIR/llama-mtmd-cli"
echo "╚══════════════════════════════════════════════════╝"
echo ""
echo "Required at runtime:"
echo "  source $ONEAPI_ROOT/setvars.sh"
echo ""
echo "Test with:"
echo "  GGML_SYCL_DISABLE_OPT=1 $BINARY_NAME \\"
echo "    -m ~/.cache/llama.cpp/models/gemma-3-4b-it-qat-q4_0_s.gguf \\"
echo "    --mmproj ~/.cache/llama.cpp/models/mmproj-model-f16-4B.gguf \\"
echo "    --image data/images/handwritten.jpeg \\"
echo "    -p 'Read the handwritten French words. List each word, one per line.' \\"
echo "    --jinja -ngl 99 -n 256"
echo ""
echo "Note: Set GGML_SYCL_DISABLE_OPT=1 for Intel GPUs older than Gen12 (12th gen)."
echo ""
