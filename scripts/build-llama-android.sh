#!/usr/bin/env bash
set -euo pipefail

# Build llama.cpp binaries for Android (ARM64)
# Usage: ./scripts/build-llama-android.sh [--clean] [--gpu]
#
# Options:
#   --clean   Remove previous build and start fresh
#   --gpu     Enable Vulkan GPU backend (recommended for modern devices)
#
# Requires: Android NDK (set ANDROID_NDK env var)
# Produces: llama-server and llama-mtmd-cli binaries for bundling in the Flutter app

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
BUILD_DIR="$PROJECT_DIR/.build/llama-cpp-android"
ASSETS_DIR="$PROJECT_DIR/app/assets/llama-binaries/arm64-v8a"
LLAMA_CPP_REPO="https://github.com/ggerganov/llama.cpp.git"

# Use a known-good commit matching the desktop builds
LLAMA_CPP_REF="${LLAMA_CPP_REF:-647b960cf5ec5497c7d3e2c3d4eb3b7ce5be34d2}"

CLEAN=false
GPU=false
for arg in "${@:-}"; do
    case "$arg" in
        --clean) CLEAN=true ;;
        --gpu)   GPU=true ;;
    esac
done

echo "╔══════════════════════════════════════════════════╗"
echo "║  Build llama.cpp for Android (ARM64)             ║"
if [[ "$GPU" == true ]]; then
    echo "║  Backend: Vulkan GPU                             ║"
else
    echo "║  Backend: CPU only                               ║"
fi
echo "╚══════════════════════════════════════════════════╝"
echo ""

# ------------------------------------------------------------------
# 1. Check Android NDK
# ------------------------------------------------------------------
if [[ -z "${ANDROID_NDK:-}" ]]; then
    # Try common locations
    for candidate in \
        "$HOME/Android/Sdk/ndk"/* \
        "$HOME/Library/Android/sdk/ndk"/* \
        /opt/android-ndk \
        /usr/lib/android-ndk; do
        if [[ -d "$candidate" ]]; then
            ANDROID_NDK="$candidate"
            break
        fi
    done
fi

if [[ -z "${ANDROID_NDK:-}" ]] || [[ ! -d "$ANDROID_NDK" ]]; then
    echo "[ERR] Android NDK not found"
    echo "   Set ANDROID_NDK to your NDK path, e.g.:"
    echo "   export ANDROID_NDK=\$HOME/Android/Sdk/ndk/27.0.11718014"
    echo ""
    echo "   Install via Android Studio: Tools → SDK Manager → SDK Tools → NDK"
    exit 1
fi

echo "[OK] Android NDK: $ANDROID_NDK"

# Verify toolchain exists
TOOLCHAIN="$ANDROID_NDK/build/cmake/android.toolchain.cmake"
if [[ ! -f "$TOOLCHAIN" ]]; then
    echo "[ERR] CMake toolchain not found: $TOOLCHAIN"
    exit 1
fi

# ------------------------------------------------------------------
# 2. Check required tools
# ------------------------------------------------------------------
check_tool() {
    if ! command -v "$1" &>/dev/null; then
        echo "[ERR] Missing: $1"
        exit 1
    fi
}

check_tool cmake cmake
check_tool git git

echo "[OK] Build dependencies found"
echo "   cmake: $(cmake --version | head -1)"
echo ""

# ------------------------------------------------------------------
# 3. Clone / update llama.cpp
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
    git clone --depth 1 "$LLAMA_CPP_REPO" "$SRC_DIR"
    cd "$SRC_DIR"
    git fetch --depth 1 origin "$LLAMA_CPP_REF"
    git checkout FETCH_HEAD
fi

cd "$SRC_DIR"
COMMIT=$(git rev-parse --short HEAD)
echo "   Commit: $COMMIT"
echo ""

# ------------------------------------------------------------------
# 4. Configure with CMake (Android NDK cross-compile)
# ------------------------------------------------------------------
CMAKE_BUILD="$BUILD_DIR/build"
mkdir -p "$CMAKE_BUILD"

echo ":: Configuring CMake for Android ARM64..."

CMAKE_ARGS=(
    -B "$CMAKE_BUILD"
    -S "$SRC_DIR"
    -DCMAKE_TOOLCHAIN_FILE="$TOOLCHAIN"
    -DANDROID_ABI=arm64-v8a
    -DANDROID_PLATFORM=android-28
    -DCMAKE_BUILD_TYPE=Release
    -DGGML_OPENMP=OFF
    -DGGML_LLAMAFILE=OFF
    -DLLAMA_BUILD_SERVER=ON
    -DLLAMA_CURL=OFF
    -DBUILD_SHARED_LIBS=OFF
)

if [[ "$GPU" == true ]]; then
    # Check if Vulkan is available in the NDK
    VULKAN_HEADER="$ANDROID_NDK/toolchains/llvm/prebuilt/linux-x86_64/sysroot/usr/include/vulkan/vulkan.h"
    if [[ ! -f "$VULKAN_HEADER" ]]; then
        echo "[WARN] Vulkan header not found in NDK sysroot"
        echo "   Vulkan GPU backend requires NDK r25+ with Vulkan support"
        echo "   Falling back to CPU backend..."
        GPU=false
    else
        CMAKE_ARGS+=(-DGGML_VULKAN=ON)
        echo "   Vulkan: enabled"
    fi
else
    echo "   Vulkan: disabled (use --gpu for GPU acceleration)"
fi

cmake "${CMAKE_ARGS[@]}" 2>&1 | tee "$BUILD_DIR/cmake-configure.log"

echo ""

# Verify Vulkan was actually found
if [[ "$GPU" == true ]]; then
    if grep -q "GGML_VULKAN:BOOL=ON" "$CMAKE_BUILD/CMakeCache.txt" 2>/dev/null; then
        echo "   [OK] Vulkan confirmed in CMake cache"
    elif grep -q "GGML_VULKAN:BOOL=" "$CMAKE_BUILD/CMakeCache.txt" 2>/dev/null; then
        VULKAN_STATUS=$(grep "GGML_VULKAN:BOOL=" "$CMAKE_BUILD/CMakeCache.txt" | head -1)
        echo "   [WARN] Vulkan status: $VULKAN_STATUS"
    fi
fi
echo ""

# ------------------------------------------------------------------
# 5. Build
# ------------------------------------------------------------------
NPROC=$(nproc 2>/dev/null || echo 4)
echo ":: Building llama-server and llama-mtmd-cli (using $NPROC cores)..."
echo "   This may take a few minutes..."
echo ""

# Build llama-server first, then llama-mtmd-cli
# Some CMake versions don't support multiple --target args well
cmake --build "$CMAKE_BUILD" --config Release --target llama-server -j"$NPROC" 2>&1 \
    | tee "$BUILD_DIR/build-server.log" \
    | tail -5

cmake --build "$CMAKE_BUILD" --config Release --target llama-mtmd-cli -j"$NPROC" 2>&1 \
    | tee "$BUILD_DIR/build-mtmd.log" \
    | tail -5

echo ""

# ------------------------------------------------------------------
# 6. Find and verify binaries
# ------------------------------------------------------------------
SERVER_BIN=$(find "$CMAKE_BUILD" -name "llama-server" -type f 2>/dev/null | head -1)
MTMD_BIN=$(find "$CMAKE_BUILD" -name "llama-mtmd-cli" -type f 2>/dev/null | head -1)

if [[ -z "$SERVER_BIN" ]]; then
    echo "[ERR] llama-server binary not found"
    echo "   Check logs: $BUILD_DIR/build-server.log"
    exit 1
fi

if [[ -z "$MTMD_BIN" ]]; then
    echo "[ERR] llama-mtmd-cli binary not found"
    echo "   Check logs: $BUILD_DIR/build-mtmd.log"
    exit 1
fi

# Verify binaries are actually ARM64
if command -v file &>/dev/null; then
    SERVER_ARCH=$(file "$SERVER_BIN" | grep -o "ARM aarch64" || echo "unknown")
    MTMD_ARCH=$(file "$MTMD_BIN" | grep -o "ARM aarch64" || echo "unknown")
    if [[ "$SERVER_ARCH" != "ARM aarch64" ]]; then
        echo "[WARN] llama-server architecture: $SERVER_ARCH (expected ARM aarch64)"
    fi
    if [[ "$MTMD_ARCH" != "ARM aarch64" ]]; then
        echo "[WARN] llama-mtmd-cli architecture: $MTMD_ARCH (expected ARM aarch64)"
    fi
fi

echo "[OK] Binaries built:"
echo "   llama-server:     $(du -h "$SERVER_BIN" | cut -f1)"
echo "   llama-mtmd-cli:   $(du -h "$MTMD_BIN" | cut -f1)"

# ------------------------------------------------------------------
# 7. Copy binaries and libraries to assets
# ------------------------------------------------------------------
mkdir -p "$ASSETS_DIR"
cp "$SERVER_BIN" "$ASSETS_DIR/llama-server"
cp "$MTMD_BIN" "$ASSETS_DIR/llama-mtmd-cli"

# Copy libc++_shared.so if needed (for c++_shared STL)
# The NDK build uses c++_shared by default
STL_LIB=$(find "$ANDROID_NDK" -path "*/libc++_shared.so" 2>/dev/null | grep "aarch64-linux-android" | head -1)
if [[ -n "$STL_LIB" ]]; then
    cp "$STL_LIB" "$ASSETS_DIR/libc++_shared.so"
    echo "   libc++_shared.so: $(du -h "$STL_LIB" | cut -f1)"
fi

# If Vulkan is enabled, we don't need to bundle libvulkan.so
# because it's part of the Android system on devices that support it.
# However, for devices without Vulkan, the app will fall back to CPU.

echo ""
echo "╔══════════════════════════════════════════════════╗"
echo "║  [OK] Binaries copied to app assets:             ║"
echo "║     $ASSETS_DIR"
echo "╚══════════════════════════════════════════════════╝"
echo ""
echo "Next steps:"
echo "  1. Ensure pubspec.yaml includes: assets/llama-binaries/"
echo "  2. Build Flutter app: cd app && flutter build apk"
echo ""
if [[ "$GPU" == true ]]; then
    echo "GPU backend: Vulkan is enabled. The app will use GPU acceleration"
    echo "on devices that support Vulkan. Falls back to CPU otherwise."
else
    echo "GPU backend: CPU only. For GPU acceleration, rebuild with --gpu:"
    echo "  ./scripts/build-llama-android.sh --gpu"
fi
echo ""
