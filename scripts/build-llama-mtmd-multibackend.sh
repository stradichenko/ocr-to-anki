#!/usr/bin/env bash
# Build llama-mtmd-cli with multiple backend support
# Supports: CUDA, Vulkan, SYCL (Intel), Metal (macOS), NNAPI (Android)

set -euo pipefail

# Configuration
BUILD_DIR="/tmp/llama.cpp-multibackend"
INSTALL_DIR="$HOME/.local/bin"
ENABLE_CUDA=false
ENABLE_VULKAN=false
ENABLE_SYCL=false
ENABLE_METAL=false
ENABLE_OPENCL=false

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --cuda) ENABLE_CUDA=true; shift ;;
        --vulkan) ENABLE_VULKAN=true; shift ;;
        --sycl) ENABLE_SYCL=true; shift ;;
        --metal) ENABLE_METAL=true; shift ;;
        --opencl) ENABLE_OPENCL=true; shift ;;
        --all) 
            ENABLE_CUDA=true
            ENABLE_VULKAN=true
            # SYCL requires manual setup
            shift ;;
        *)
            echo "Usage: $0 [--cuda] [--vulkan] [--sycl] [--metal] [--opencl] [--all]"
            exit 1
            ;;
    esac
done

echo "=== llama-mtmd-cli Multi-Backend Build ==="
echo "==========================================="
echo
echo "Build configuration:"
echo "  CUDA:    $ENABLE_CUDA"
echo "  Vulkan:  $ENABLE_VULKAN"
echo "  SYCL:    $ENABLE_SYCL"
echo "  Metal:   $ENABLE_METAL"
echo "  OpenCL:  $ENABLE_OPENCL"
echo

# Clone llama.cpp
echo "1. Cloning llama.cpp..."
rm -rf "$BUILD_DIR"
git clone --depth=1 https://github.com/ggerganov/llama.cpp "$BUILD_DIR"
cd "$BUILD_DIR"

echo "✓ Cloned llama.cpp"
echo

# Build CMake arguments
CMAKE_ARGS=(
    "-DCMAKE_BUILD_TYPE=Release"
    "-DGGML_OPENMP=ON"
    "-DGGML_NATIVE=OFF"
    "-DLLAMA_CURL=OFF"
    "-DBUILD_SHARED_LIBS=OFF"
)

# CUDA backend
if $ENABLE_CUDA; then
    echo "Enabling CUDA backend..."
    if command -v nvcc >/dev/null 2>&1; then
        CMAKE_ARGS+=("-DGGML_CUDA=ON")
        CMAKE_ARGS+=("-DCMAKE_CUDA_ARCHITECTURES=all")
        echo "✓ CUDA found: $(nvcc --version | head -1)"
    else
        echo "⚠️  CUDA requested but nvcc not found"
        ENABLE_CUDA=false
    fi
fi

# Vulkan backend
if $ENABLE_VULKAN; then
    echo "Enabling Vulkan backend..."
    
    # Check for Vulkan SDK components
    if command -v glslc >/dev/null 2>&1; then
        CMAKE_ARGS+=("-DGGML_VULKAN=ON")
        
        # Set Vulkan SDK path if available (optional in Nix)
        if [ -n "${VULKAN_SDK:-}" ]; then
            CMAKE_ARGS+=("-DVulkan_INCLUDE_DIR=$VULKAN_SDK/include")
            echo "  Using VULKAN_SDK: $VULKAN_SDK"
        fi
        
        echo "✓ Vulkan found"
        echo "  glslc: $(which glslc)"
        glslc --version | head -1
    else
        echo "⚠️  Vulkan requested but glslc not found"
        echo "   In Nix: Already provided by flake"
        echo "   Outside Nix: Install vulkan-sdk"
        ENABLE_VULKAN=false
    fi
fi

# SYCL backend (Intel OneAPI)
if $ENABLE_SYCL; then
    echo "Enabling SYCL backend..."
    
    # Check for Intel OneAPI
    ONEAPI_ROOT="${ONEAPI_ROOT:-/opt/intel/oneapi}"
    if [ -d "$ONEAPI_ROOT" ]; then
        CMAKE_ARGS+=("-DGGML_SYCL=ON")
        CMAKE_ARGS+=("-DCMAKE_C_COMPILER=icx")
        CMAKE_ARGS+=("-DCMAKE_CXX_COMPILER=icpx")
        echo "✓ Intel OneAPI found at: $ONEAPI_ROOT"
    else
        echo "⚠️  SYCL requested but Intel OneAPI not found at $ONEAPI_ROOT"
        echo "   Install from: https://www.intel.com/content/www/us/en/developer/tools/oneapi/base-toolkit-download.html"
        ENABLE_SYCL=false
    fi
fi

# Metal backend (macOS only)
if $ENABLE_METAL; then
    if [[ "$OSTYPE" == "darwin"* ]]; then
        echo "Enabling Metal backend..."
        CMAKE_ARGS+=("-DGGML_METAL=ON")
    else
        echo "⚠️  Metal only available on macOS"
        ENABLE_METAL=false
    fi
fi

# OpenCL backend
if $ENABLE_OPENCL; then
    echo "Enabling OpenCL backend..."
    CMAKE_ARGS+=("-DGGML_OPENCL=ON")
fi

echo
echo "2. Configuring build with backends..."
echo "   CMake args: ${CMAKE_ARGS[@]}"
echo

cmake -B build "${CMAKE_ARGS[@]}"

echo
echo "3. Building llama-mtmd-cli..."
echo "   (This will take 5-15 minutes depending on backends)"
echo

cmake --build build --config Release --target llama-mtmd-cli -j$(nproc)

if [ ! -f "build/bin/llama-mtmd-cli" ]; then
    echo "❌ Build failed! llama-mtmd-cli not found"
    exit 1
fi

echo "✓ Built successfully"
echo

# Install
echo "4. Installing to $INSTALL_DIR..."
mkdir -p "$INSTALL_DIR"

# Create versioned binary name
BACKEND_SUFFIX=""
$ENABLE_CUDA && BACKEND_SUFFIX="${BACKEND_SUFFIX}-cuda"
$ENABLE_VULKAN && BACKEND_SUFFIX="${BACKEND_SUFFIX}-vulkan"
$ENABLE_SYCL && BACKEND_SUFFIX="${BACKEND_SUFFIX}-sycl"
$ENABLE_METAL && BACKEND_SUFFIX="${BACKEND_SUFFIX}-metal"
$ENABLE_OPENCL && BACKEND_SUFFIX="${BACKEND_SUFFIX}-opencl"

if [ -z "$BACKEND_SUFFIX" ]; then
    BACKEND_SUFFIX="-cpu"
fi

BINARY_NAME="llama-mtmd-cli${BACKEND_SUFFIX}"

cp build/bin/llama-mtmd-cli "$INSTALL_DIR/$BINARY_NAME"
chmod +x "$INSTALL_DIR/$BINARY_NAME"

# Create/update default symlink
ln -sf "$INSTALL_DIR/$BINARY_NAME" "$INSTALL_DIR/llama-mtmd-cli"

echo "✓ Installed as: $BINARY_NAME"
echo "✓ Symlink created: llama-mtmd-cli -> $BINARY_NAME"
echo

# Verify
echo "5. Verification..."
"$INSTALL_DIR/$BINARY_NAME" --version 2>&1 | head -5 || true

echo
echo "✅ Build complete!"
echo
echo "Binaries:"
echo "  • $INSTALL_DIR/$BINARY_NAME"
echo "  • $INSTALL_DIR/llama-mtmd-cli (symlink)"
echo
echo "Enabled backends:"
$ENABLE_CUDA && echo "  ✅ CUDA"
$ENABLE_VULKAN && echo "  ✅ Vulkan"
$ENABLE_SYCL && echo "  ✅ SYCL (Intel)"
$ENABLE_METAL && echo "  ✅ Metal (macOS)"
$ENABLE_OPENCL && echo "  ✅ OpenCL"
[ -z "$BACKEND_SUFFIX" ] || echo "  ✅ CPU fallback"
echo
echo "Test with:"
echo "  llama-mtmd-cli -m MODEL --mmproj MMPROJ --image IMG -p 'What text?'"
