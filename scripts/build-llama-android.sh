#!/usr/bin/env bash
set -euo pipefail

# Build llama.cpp binaries for Android (ARM64)
# Usage: ./scripts/build-llama-android.sh [--clean] [--all]
#
# Options:
#   --clean   Remove previous build and start fresh
#   --all     Build CPU, Vulkan, and OpenCL variants (default)
#
# Requires: Android NDK (set ANDROID_NDK env var)
# Produces: llama-server and llama-mtmd-cli binaries for bundling in the Flutter app

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
BUILD_DIR="$PROJECT_DIR/.build/llama-cpp-android"
# Binaries are placed under jniLibs so Android extracts them to nativeLibraryDir
# at install time. App-private storage is mounted noexec on modern Android, so
# files copied to /data/user/0/<pkg>/ cannot be execve'd. nativeLibraryDir
# (/data/app/.../lib/arm64/) is the only place execve works from.
JNILIBS_DIR="$PROJECT_DIR/app/android/app/src/main/jniLibs/arm64-v8a"
LLAMA_CPP_REPO="https://github.com/ggerganov/llama.cpp.git"
OPENCL_HEADERS_REPO="https://github.com/KhronosGroup/OpenCL-Headers.git"

# Track master; override via LLAMA_CPP_REF env var to pin a commit/tag.
LLAMA_CPP_REF="${LLAMA_CPP_REF:-master}"

CLEAN=false
BUILD_ALL=true
for arg in "${@:-}"; do
    case "$arg" in
        --clean) CLEAN=true ;;
        --all)   BUILD_ALL=true ;;
    esac
done

echo "╔══════════════════════════════════════════════════╗"
echo "║  Build llama.cpp for Android (ARM64)             ║"
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

# Detect host prebuilt directory (linux-x86_64, darwin-x86_64, etc.)
HOST_PREBUILT=""
for d in "$ANDROID_NDK/toolchains/llvm/prebuilt/"*; do
    if [[ -d "$d" ]]; then
        HOST_PREBUILT="$d"
        break
    fi
done

if [[ -z "$HOST_PREBUILT" ]]; then
    echo "[ERR] Could not find LLVM prebuilt directory in NDK"
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
    git fetch origin
    git checkout "$LLAMA_CPP_REF"
else
    echo ":: Cloning llama.cpp ($LLAMA_CPP_REF)..."
    git clone "$LLAMA_CPP_REPO" "$SRC_DIR"
    cd "$SRC_DIR"
    git checkout "$LLAMA_CPP_REF"
fi

cd "$SRC_DIR"
COMMIT=$(git rev-parse --short HEAD)
echo "   Commit: $COMMIT"
echo ""

# ------------------------------------------------------------------
# Helper: find Vulkan headers in NDK
# ------------------------------------------------------------------
find_vulkan_header() {
    local paths=(
        "$HOST_PREBUILT/sysroot/usr/include/vulkan/vulkan.h"
        "$ANDROID_NDK/sources/third_party/vulkan/src/include/vulkan/vulkan.h"
        "$ANDROID_NDK/sysroot/usr/include/vulkan/vulkan.h"
    )
    for p in "${paths[@]}"; do
        if [[ -f "$p" ]]; then
            dirname "$(dirname "$p")"
            return 0
        fi
    done
    return 1
}

# ------------------------------------------------------------------
# Helper: download OpenCL headers
# ------------------------------------------------------------------
ensure_opencl_headers() {
    local headers_dir="$BUILD_DIR/opencl-headers"
    if [[ -d "$headers_dir/.git" ]]; then
        cd "$headers_dir"
        git fetch origin
        git checkout main
    else
        echo ":: Downloading Khronos OpenCL headers..."
        git clone --depth 1 "$OPENCL_HEADERS_REPO" "$headers_dir"
    fi
    echo "$headers_dir"
}

# ------------------------------------------------------------------
# Helper: build one variant
# ------------------------------------------------------------------
build_variant() {
    local variant="$1"
    local cmake_build="$BUILD_DIR/build-$variant"
    mkdir -p "$cmake_build"

    echo ":: Configuring CMake for $variant..."

    local cmake_args=(
        -B "$cmake_build"
        -S "$SRC_DIR"
        -DCMAKE_TOOLCHAIN_FILE="$TOOLCHAIN"
        -DANDROID_ABI=arm64-v8a
        -DANDROID_PLATFORM=android-28
        -DCMAKE_BUILD_TYPE=Release
        -DGGML_OPENMP=OFF
        -DGGML_LLAMAFILE=OFF
        -DLLAMA_BUILD_COMMON=ON
        -DLLAMA_BUILD_TOOLS=ON
        -DLLAMA_BUILD_SERVER=ON
        -DLLAMA_CURL=OFF
        -DBUILD_SHARED_LIBS=OFF
    )

    case "$variant" in
        vulkan)
            local vulkan_include
            vulkan_include=$(find_vulkan_header) || {
                echo "[WARN] Vulkan header not found — skipping $variant build"
                return 1
            }
            cmake_args+=(-DGGML_VULKAN=ON)
            echo "   Vulkan: enabled (headers at $vulkan_include)"
            ;;
        opencl)
            local opencl_headers
            opencl_headers=$(ensure_opencl_headers)
            # On Android we only have headers; the device provides libOpenCL.so
            # at runtime.  We must not link against a host libOpenCL.  Tell
            # CMake where the headers are and provide an empty library so the
            # linker doesn't complain about missing -lOpenCL.
            cmake_args+=(-DGGML_OPENCL=ON)
            cmake_args+=(-DGGML_OPENCL_EMBED_KERNELS=ON)
            cmake_args+=(-DOpenCL_INCLUDE_DIR="$opencl_headers")
            # Prevent FindOpenCL from searching the host system.
            cmake_args+=(-DOpenCL_LIBRARY="")
            echo "   OpenCL: enabled (headers at $opencl_headers)"
            ;;
        cpu)
            echo "   Vulkan: disabled"
            echo "   OpenCL: disabled"
            ;;
    esac

    cmake "${cmake_args[@]}" 2>&1 | tee "$BUILD_DIR/cmake-configure-$variant.log"

    NPROC=$(nproc 2>/dev/null || echo 4)
    echo ":: Building $variant (using $NPROC cores)..."

    cmake --build "$cmake_build" --config Release --target llama-server -j"$NPROC" 2>&1 \
        | tee "$BUILD_DIR/build-server-$variant.log" \
        | tail -5

    cmake --build "$cmake_build" --config Release --target llama-mtmd-cli -j"$NPROC" 2>&1 \
        | tee "$BUILD_DIR/build-mtmd-$variant.log" \
        | tail -5

    local server_bin=$(find "$cmake_build" -name "llama-server" -type f 2>/dev/null | head -1)
    local mtmd_bin=$(find "$cmake_build" -name "llama-mtmd-cli" -type f 2>/dev/null | head -1)

    if [[ -z "$server_bin" ]]; then
        echo "[ERR] llama-server binary not found for $variant"
        return 1
    fi
    if [[ -z "$mtmd_bin" ]]; then
        echo "[ERR] llama-mtmd-cli binary not found for $variant"
        return 1
    fi

    echo "[OK] $variant built:"
    echo "   llama-server:   $(du -h "$server_bin" | cut -f1)"
    echo "   llama-mtmd-cli: $(du -h "$mtmd_bin" | cut -f1)"

    # Copy with variant suffix
    mkdir -p "$JNILIBS_DIR"
    cp "$server_bin" "$JNILIBS_DIR/libllama-server-$variant.so"
    cp "$mtmd_bin" "$JNILIBS_DIR/libllama-mtmd-cli-$variant.so"

    return 0
}

# ------------------------------------------------------------------
# 4. Build variants
# ------------------------------------------------------------------
mkdir -p "$JNILIBS_DIR"

# Always build CPU variant
build_variant "cpu" || exit 1

# Build GPU variants if --all or requested
if [[ "$BUILD_ALL" == true ]]; then
    # Vulkan
    if build_variant "vulkan"; then
        echo ""
        echo "[OK] Vulkan variant built successfully"
    else
        echo ""
        echo "[WARN] Vulkan variant failed"
    fi

    # OpenCL
    if build_variant "opencl"; then
        echo ""
        echo "[OK] OpenCL variant built successfully"
    else
        echo ""
        echo "[WARN] OpenCL variant failed"
    fi
fi

# Copy libc++_shared.so if needed (for c++_shared STL)
STL_LIB=$(find "$ANDROID_NDK" -path "*/libc++_shared.so" 2>/dev/null | grep "aarch64-linux-android" | head -1)
if [[ -n "$STL_LIB" ]]; then
    cp "$STL_LIB" "$JNILIBS_DIR/libc++_shared.so"
    echo "   libc++_shared.so: $(du -h "$STL_LIB" | cut -f1)"
fi

echo ""
echo "╔══════════════════════════════════════════════════╗"
echo "║  [OK] Binaries copied to jniLibs:                ║"
echo "║     $JNILIBS_DIR"
echo "╚══════════════════════════════════════════════════╝"
echo ""
echo "Next steps:"
echo "  1. Ensure app/android/app/build.gradle.kts has:"
echo "     packaging { jniLibs { useLegacyPackaging = true } }"
echo "  2. Build Flutter app: cd app && flutter build apk"
echo ""
