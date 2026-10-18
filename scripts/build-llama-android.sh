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

VULKAN_HEADERS_REPO="https://github.com/KhronosGroup/Vulkan-Headers.git"
SPIRV_HEADERS_REPO="https://github.com/KhronosGroup/SPIRV-Headers.git"

# ------------------------------------------------------------------
# Helper: download Vulkan C++ headers (vulkan.hpp)
# ------------------------------------------------------------------
ensure_vulkan_hpp_headers() {
    local headers_dir="$BUILD_DIR/vulkan-headers"
    if [[ -d "$headers_dir/.git" ]]; then
        cd "$headers_dir"
        git fetch origin >/dev/null 2>&1
        git checkout main >/dev/null 2>&1
    else
        echo ":: Downloading Khronos Vulkan headers (for vulkan.hpp)..." >&2
        git clone --depth 1 "$VULKAN_HEADERS_REPO" "$headers_dir" >&2
    fi
    echo "$headers_dir/include"
}

# ------------------------------------------------------------------
# Helper: download SPIRV headers (spirv.hpp)
# ------------------------------------------------------------------
ensure_spirv_headers() {
    local headers_dir="$BUILD_DIR/spirv-headers"
    if [[ -d "$headers_dir/.git" ]]; then
        cd "$headers_dir"
        git fetch origin >/dev/null 2>&1
        git checkout main >/dev/null 2>&1
    else
        echo ":: Downloading Khronos SPIRV headers..." >&2
        git clone --depth 1 "$SPIRV_HEADERS_REPO" "$headers_dir" >&2
    fi
    echo "$headers_dir/include"
}

# ------------------------------------------------------------------
# Helper: download OpenCL headers
# ------------------------------------------------------------------
ensure_opencl_headers() {
    local headers_dir="$BUILD_DIR/opencl-headers"
    if [[ -d "$headers_dir/.git" ]]; then
        cd "$headers_dir"
        git fetch origin >/dev/null 2>&1
        git checkout main >/dev/null 2>&1
    else
        echo ":: Downloading Khronos OpenCL headers..." >&2
        git clone --depth 1 "$OPENCL_HEADERS_REPO" "$headers_dir" >&2
    fi
    echo "$headers_dir"
}

# ------------------------------------------------------------------
# Helper: build a stub OpenCL static library for linking
# ------------------------------------------------------------------
build_opencl_stub() {
    # Args: directory holding the Khronos OpenCL headers (CL/cl.h and friends).
    #
    # The stub exists so CMake can link at build time; the device supplies the
    # real libOpenCL.so at runtime.
    #
    # It used to be a hand-written list of function names, which meant that
    # every time llama.cpp called a new OpenCL function the Android build
    # failed to link — and because a GPU variant failing is only a warning
    # while a CPU failure is fatal, the result shipped as a CPU-only APK with
    # no indication that anything was wrong. The list is now generated from the
    # headers, so it cannot fall behind.
    local headers_dir="${1:-}"
    local stub_dir="$BUILD_DIR/opencl-stub"
    local stub_c="$stub_dir/stub.c"
    local stub_lib="$stub_dir/libOpenCL.a"
    mkdir -p "$stub_dir"

    if [[ -z "$headers_dir" || ! -f "$headers_dir/CL/cl.h" ]]; then
        echo "[WARN] OpenCL headers not found in '${headers_dir}' — cannot generate the stub" >&2
        return 1
    fi

    echo ":: Generating OpenCL stub from $headers_dir ..." >&2

    {
        echo "// Generated from the Khronos OpenCL headers. Do not edit."
        echo "// One no-op definition per declared entry point, so linking succeeds"
        echo "// for whatever llama.cpp currently calls."
        # Entry points are declared as clFoo(...) and exported from libOpenCL.
        grep -rhoE '\bcl[A-Z][A-Za-z0-9_]*[[:space:]]*\(' "$headers_dir/CL" 2>/dev/null \
            | sed 's/[[:space:]]*($//' \
            | sort -u \
            | while read -r fn; do
                echo "void* ${fn}(void) { return 0; }"
              done
    } > "$stub_c"

    local count
    count=$(grep -c '^void\* ' "$stub_c" || echo 0)
    if [[ "$count" -lt 20 ]]; then
        echo "[WARN] Only $count OpenCL entry points found — the headers may not be intact" >&2
        return 1
    fi
    echo ":: Stub covers $count OpenCL entry points" >&2

    local clang="$HOST_PREBUILT/bin/aarch64-linux-android28-clang"
    local ar="$HOST_PREBUILT/bin/llvm-ar"

    if [[ ! -f "$clang" ]]; then
        echo "[WARN] Android cross-compiler not found at $clang — OpenCL stub build skipped" >&2
        return 1
    fi

    rm -f "$stub_lib"
    "$clang" -c -o "$stub_dir/stub.o" "$stub_c" >&2
    "$ar" rcs "$stub_lib" "$stub_dir/stub.o" >&2

    if [[ -f "$stub_lib" ]]; then
        echo "$stub_lib"
        return 0
    fi
    return 1
}

# ------------------------------------------------------------------
# Helper: install SPIRV-Headers so CMake can find the package
# ------------------------------------------------------------------
install_spirv_headers() {
    # Args: the cloned SPIRV-Headers source directory.
    #
    # Header-only, so nothing is compiled; tests are off so the install stays a
    # file copy. Prints the install prefix on success.
    local src="$1"
    local prefix="$BUILD_DIR/spirv-headers-install"

    if [[ -f "$prefix/share/cmake/SPIRV-Headers/SPIRV-HeadersConfig.cmake" ]]; then
        echo "$prefix"
        return 0
    fi

    command -v cmake >/dev/null 2>&1 || {
        echo "[WARN] cmake not available to install SPIRV-Headers" >&2
        return 1
    }

    echo ":: Installing SPIRV-Headers into $prefix ..." >&2
    cmake -S "$src" -B "$BUILD_DIR/spirv-headers-build" \
        -DCMAKE_INSTALL_PREFIX="$prefix" \
        -DSPIRV_HEADERS_ENABLE_TESTS=OFF \
        -DSPIRV_HEADERS_ENABLE_INSTALL=ON \
        >&2 || return 1
    cmake --install "$BUILD_DIR/spirv-headers-build" >&2 || return 1

    if [[ -f "$prefix/share/cmake/SPIRV-Headers/SPIRV-HeadersConfig.cmake" ]]; then
        echo "$prefix"
        return 0
    fi
    # Some layouts put the config under lib/cmake instead.
    if [[ -f "$prefix/lib/cmake/SPIRV-Headers/SPIRV-HeadersConfig.cmake" ]]; then
        echo "$prefix"
        return 0
    fi
    echo "[WARN] SPIRV-Headers installed but no CMake config was produced" >&2
    return 1
}

# ------------------------------------------------------------------
# Helper: ensure glslc is available for Vulkan builds
# ------------------------------------------------------------------
ensure_glslc() {
    if command -v glslc &>/dev/null; then
        echo "[OK] glslc found: $(command -v glslc)" >&2
        return 0
    fi

    # Try to install via apt on Debian/Ubuntu systems
    if command -v apt-get &>/dev/null; then
        echo ":: Installing glslc (glslang-tools) via apt..." >&2
        sudo apt-get update -qq >&2
        sudo apt-get install -y -qq glslang-tools >&2
        if command -v glslc &>/dev/null; then
            echo "[OK] glslc installed: $(command -v glslc)" >&2
            return 0
        fi
    fi

    # Try to find it in the NDK
    local ndk_glslc
    ndk_glslc=$(find "$ANDROID_NDK" -name "glslc" -type f 2>/dev/null | head -1)
    if [[ -n "$ndk_glslc" ]]; then
        echo "[OK] glslc found in NDK: $ndk_glslc" >&2
        # Add to PATH for the current session
        export PATH="$(dirname "$ndk_glslc"):$PATH"
        return 0
    fi

    echo "[WARN] glslc not found — Vulkan build will likely fail" >&2
    echo "   Install with: sudo apt-get install glslang-tools" >&2
    return 1
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
            # llama.cpp's ggml-vulkan.cpp includes <vulkan/vulkan.hpp> which is
            # NOT shipped with the Android NDK (only vulkan.h is). It also needs
            # <spirv/unified1/spirv.hpp>. Download both Khronos repos.
            local vulkan_hpp_include
            vulkan_hpp_include=$(ensure_vulkan_hpp_headers)
            local spirv_include
            spirv_include=$(ensure_spirv_headers)
            # Ensure glslc is available (CMake's FindVulkan requires it)
            ensure_glslc || true
            cmake_args+=(-DGGML_VULKAN=ON)
            # Prepend the downloaded headers so vulkan.hpp and spirv.hpp are found.
            cmake_args+=(-DCMAKE_CXX_FLAGS="-I$vulkan_hpp_include -I$spirv_include -I$vulkan_include")

            # ggml-vulkan asks for the SPIRV-Headers *package*, not just its
            # include path, so the headers have to be installed for CMake to
            # find a config file. Cloning and adding -I was enough until
            # upstream changed to find_package, at which point the Vulkan build
            # started failing — silently, because a GPU variant failing is only
            # a warning, so the release shipped CPU-only.
            local spirv_prefix
            spirv_prefix=$(install_spirv_headers "$spirv_include") || {
                echo "[WARN] SPIRV-Headers could not be installed — skipping $variant build"
                return 1
            }
            cmake_args+=(-DCMAKE_PREFIX_PATH="$spirv_prefix")
            echo "   Vulkan: enabled (NDK headers at $vulkan_include, HPP at $vulkan_hpp_include, SPIRV package at $spirv_prefix)"
            ;;
        opencl)
            local opencl_headers
            opencl_headers=$(ensure_opencl_headers)
            local opencl_stub
            opencl_stub=$(build_opencl_stub "$opencl_headers") || {
                echo "[WARN] OpenCL stub library build failed — skipping $variant build"
                return 1
            }
            # On Android we only have headers; the device provides libOpenCL.so
            # at runtime.  We compile a stub static library so CMake can link
            # at build time without pulling in a host OpenCL library.
            cmake_args+=(-DGGML_OPENCL=ON)
            cmake_args+=(-DGGML_OPENCL_EMBED_KERNELS=ON)
            cmake_args+=(-DOpenCL_INCLUDE_DIR="$opencl_headers")
            cmake_args+=(-DOpenCL_LIBRARY="$opencl_stub")
            echo "   OpenCL: enabled (headers at $opencl_headers, stub at $opencl_stub)"
            ;;
        cpu)
            echo "   Vulkan: disabled"
            echo "   OpenCL: disabled"
            ;;
    esac

    cmake "${cmake_args[@]}" 2>&1 | tee "$BUILD_DIR/cmake-configure-$variant.log"

    NPROC=$(nproc 2>/dev/null || echo 4)
    echo ":: Building $variant (using $NPROC cores)..."

    local server_ok=true
    local mtmd_ok=true

    if ! cmake --build "$cmake_build" --config Release --target llama-server -j"$NPROC" 2>&1 \
            | tee "$BUILD_DIR/build-server-$variant.log"; then
        server_ok=false
    fi

    if ! cmake --build "$cmake_build" --config Release --target llama-mtmd-cli -j"$NPROC" 2>&1 \
            | tee "$BUILD_DIR/build-mtmd-$variant.log"; then
        mtmd_ok=false
    fi

    local server_bin=$(find "$cmake_build" -name "llama-server" -type f 2>/dev/null | head -1)
    local mtmd_bin=$(find "$cmake_build" -name "llama-mtmd-cli" -type f 2>/dev/null | head -1)

    if [[ -z "$server_bin" ]] || [[ "$server_ok" == false ]]; then
        echo "[ERR] llama-server build failed for $variant"
        if [[ -f "$BUILD_DIR/build-server-$variant.log" ]]; then
            echo ":: Last 80 lines of build log:"
            tail -80 "$BUILD_DIR/build-server-$variant.log"
        fi
        return 1
    fi
    if [[ -z "$mtmd_bin" ]] || [[ "$mtmd_ok" == false ]]; then
        echo "[ERR] llama-mtmd-cli build failed for $variant"
        if [[ -f "$BUILD_DIR/build-mtmd-$variant.log" ]]; then
            echo ":: Last 80 lines of build log:"
            tail -80 "$BUILD_DIR/build-mtmd-$variant.log"
        fi
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
#
# A failed GPU variant is recorded rather than shrugged off. It used to be a
# bare warning, and the script still ended by announcing that the binaries had
# been copied, so a release could ship a CPU-only APK with nothing in the log
# saying the GPU backends were missing. That is what happened: the app offered
# Vulkan and OpenCL backend detection on Android while neither binary existed.
FAILED_VARIANTS=()

if [[ "$BUILD_ALL" == true ]]; then
    # Vulkan
    if build_variant "vulkan"; then
        echo ""
        echo "[OK] Vulkan variant built successfully"
    else
        echo ""
        echo "[WARN] Vulkan variant failed"
        FAILED_VARIANTS+=("vulkan")
    fi

    # OpenCL
    if build_variant "opencl"; then
        echo ""
        echo "[OK] OpenCL variant built successfully"
    else
        echo ""
        echo "[WARN] OpenCL variant failed"
        FAILED_VARIANTS+=("opencl")
    fi
fi

# Copy libc++_shared.so if needed (for c++_shared STL)
STL_LIB=$(find "$ANDROID_NDK" -path "*/libc++_shared.so" 2>/dev/null | grep "aarch64-linux-android" | head -1)
if [[ -n "$STL_LIB" ]]; then
    cp "$STL_LIB" "$JNILIBS_DIR/libc++_shared.so"
    echo "   libc++_shared.so: $(du -h "$STL_LIB" | cut -f1)"
fi

echo ""
echo "Binaries in $JNILIBS_DIR:"
for f in "$JNILIBS_DIR"/libllama-*.so; do
    [[ -e "$f" ]] && echo "  $(basename "$f")"
done
echo ""

if [[ ${#FAILED_VARIANTS[@]} -gt 0 ]]; then
    echo "╔══════════════════════════════════════════════════╗"
    echo "║  [FAIL] These GPU variants did not build:        ║"
    for v in "${FAILED_VARIANTS[@]}"; do
        printf "║     %-44s ║\n" "$v"
    done
    echo "╚══════════════════════════════════════════════════╝"
    echo ""
    echo "The APK would ship without them, so Android would run on CPU only."
    echo "Set ALLOW_CPU_ONLY=1 to accept that deliberately."
    echo ""
    if [[ "${ALLOW_CPU_ONLY:-0}" != "1" ]]; then
        exit 1
    fi
fi

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
