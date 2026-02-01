# Building llama-mtmd-cli with Multi-Backend Support

## Overview

llama-mtmd-cli can be built with multiple compute backends for optimal performance across different hardware:

- **CUDA**: NVIDIA GPUs (GeForce, RTX, Tesla)
- **Vulkan**: Cross-platform GPU (NVIDIA, AMD, Intel)
- **SYCL**: Intel Arc GPUs, Intel CPUs with AVX512
- **Metal**: Apple Silicon (M1/M2/M3)
- **OpenCL**: Legacy GPU support
- **NNAPI**: Android devices

## Prerequisites by Backend

### CUDA (NVIDIA)
```bash
# Option 1: Use system CUDA (recommended for building)
# Make sure nvidia-driver and cuda-toolkit are installed on your system
nvcc --version

# Option 2: Use Nix CUDA environment (large download ~4GB)
nix develop --impure .#cuda

# In Nix CUDA shell
nvcc --version
```

**Note:** CUDA builds work best **outside** the Nix environment using your system's CUDA installation. The Nix CUDA packages are large and may have compatibility issues with your GPU driver.

### Vulkan (All platforms)
```bash
# Check Vulkan support
vulkaninfo --summary

# In Nix (already configured)
nix develop  # Vulkan tools included

# Outside Nix
# Ubuntu/Debian
sudo apt install vulkan-tools libvulkan-dev glslc

# Arch
sudo pacman -S vulkan-headers vulkan-icd-loader shaderc

# Verify
glslc --version
vulkaninfo | grep deviceName
```

### SYCL (Intel) with Nix

Intel OneAPI **cannot be fully packaged in Nix** due to licensing, but we provide a hybrid approach:

```bash
# 1. Install Intel OneAPI system-wide (one-time)
wget https://registrationcenter-download.intel.com/akdlm/IRC_NAS/ac92f2bb-4818-4e53-a432-f8b34d502f23/intel-oneapi-base-toolkit_2025.0.0_49274_offline.sh
sudo sh intel-oneapi-base-toolkit_*.sh

# 2. Enter Nix SYCL environment (provides build tools + Level Zero)
nix develop --impure .#sycl

# 3. Verify SYCL environment
icpx --version
check-oneapi

# 4. Build with SYCL
./scripts/build-llama-mtmd-multibackend.sh --sycl
```

**What the Nix SYCL shell provides:**
- ✅ Build tools (cmake, pkg-config, git)
- ✅ Level Zero runtime (Intel GPU interface)
- ✅ Vulkan support (can use both SYCL + Vulkan)
- ✅ Wrapper scripts for Intel compilers
- ❌ Intel OneAPI itself (must install system-wide)

### Metal (macOS only)
```bash
# Xcode Command Line Tools (includes Metal)
xcode-select --install
```

## Building

### Quick Start: Vulkan in Nix (Easiest)

```bash
nix develop
./scripts/build-llama-mtmd-multibackend.sh --vulkan
```

### SYCL Build in Nix (Intel)

```bash
# 1. Install OneAPI system-wide first (one-time)
# See "SYCL (Intel) with Nix" section above

# 2. Enter SYCL Nix environment
nix develop --impure .#sycl

# 3. Build
./scripts/build-llama-mtmd-multibackend.sh --sycl

# 4. Or combine with Vulkan
./scripts/build-llama-mtmd-multibackend.sh --sycl --vulkan
```

### Advanced: CUDA Build (Outside Nix)

CUDA builds work best using your **system CUDA** installation:

```bash
# 1. Exit Nix environment
exit

# 2. Ensure system CUDA is installed
nvcc --version  # Should show CUDA 11.8+ or 12.x

# 3. Clone llama.cpp manually
cd /tmp
git clone https://github.com/ggerganov/llama.cpp
cd llama.cpp

# 4. Build with CUDA
cmake -B build \
  -DCMAKE_BUILD_TYPE=Release \
  -DGGML_CUDA=ON \
  -DCMAKE_CUDA_ARCHITECTURES=native \
  -DLLAMA_CURL=OFF \
  -DGGML_OPENMP=ON

cmake --build build --target llama-mtmd-cli -j8

# 5. Install
mkdir -p ~/.local/bin
cp build/bin/llama-mtmd-cli ~/.local/bin/llama-mtmd-cli-cuda
chmod +x ~/.local/bin/llama-mtmd-cli-cuda

# 6. Test
~/.local/bin/llama-mtmd-cli-cuda --version
```

### Alternative: Try Nix CUDA Environment (Experimental)

```bash
# Use Nix's CUDA environment (large download)
nix develop --impure .#cuda

# Verify CUDA
nvcc --version

# Build with CUDA
./scripts/build-llama-mtmd-multibackend.sh --cuda

# May fail due to driver/CUDA version mismatch
```

### Option 1: Auto-detect Available Backends
```bash
# Build with all detected backends
./scripts/build-llama-mtmd-multibackend.sh --all
```

### Option 2: Specific Backends
```bash
# CUDA + Vulkan
./scripts/build-llama-mtmd-multibackend.sh --cuda --vulkan

# Intel SYCL only
source /opt/intel/oneapi/setvars.sh
./scripts/build-llama-mtmd-multibackend.sh --sycl

# macOS Metal
./scripts/build-llama-mtmd-multibackend.sh --metal
```

### Option 3: Manual CMake
```bash
cd /tmp/llama.cpp
git clone https://github.com/ggerganov/llama.cpp

# CUDA build
cmake -B build \
  -DCMAKE_BUILD_TYPE=Release \
  -DGGML_CUDA=ON \
  -DCMAKE_CUDA_ARCHITECTURES=all \
  -DLLAMA_CURL=OFF

cmake --build build --target llama-mtmd-cli -j8
```

## Performance Comparison

| Backend | Hardware | Build in Nix? | Speed (t/s) |
|---------|----------|---------------|-------------|
| **CPU** | Any | ✅ Yes | ~15 t/s |
| **Vulkan** | NVIDIA/AMD/Intel GPU | ✅ Yes | ~70 t/s |
| **SYCL** | Intel Arc/iGPU | ⚠️ Hybrid | ~50 t/s |
| **CUDA** | NVIDIA GPU | ❌ No (system) | ~150 t/s |
| **Metal** | Apple Silicon | ✅ Yes (macOS) | ~120 t/s |

**Nix Support Legend:**
- ✅ **Yes**: Fully supported in Nix environment
- ⚠️ **Hybrid**: Nix provides tools, but requires system component (OneAPI)
- ❌ **No**: Best built outside Nix with system tools

## Usage

```bash
# Auto-selects best backend
llama-mtmd-cli \
  -m model.gguf \
  --mmproj mmproj.gguf \
  --image image.jpg \
  -p "What text do you see?"

# Force specific backend (if multiple built)
llama-mtmd-cli-cuda ...    # CUDA version
llama-mtmd-cli-vulkan ...  # Vulkan version
llama-mtmd-cli-sycl ...    # SYCL version
```

## Troubleshooting

### SYCL: OneAPI Not Found

```bash
# Check if OneAPI is installed
ls -la /opt/intel/oneapi

# If not installed:
# Download from: https://www.intel.com/content/www/us/en/developer/tools/oneapi/base-toolkit-download.html

# Install to default location
sudo sh intel-oneapi-base-toolkit_*.sh

# Verify in Nix SYCL shell
nix develop --impure .#sycl
check-oneapi
icpx --version
```

### SYCL: Level Zero Errors

```bash
# The Nix SYCL shell includes Level Zero
# Verify drivers are loaded
lsmod | grep i915  # Intel integrated graphics
lsmod | grep xe    # Intel discrete graphics (Arc)

# Check devices
sycl-ls  # List SYCL devices (from OneAPI)
```

### CUDA: nvcc Not Found in Nix

```bash
# CUDA builds don't work well in Nix due to driver compatibility
# Solution: Build outside Nix with system CUDA

# 1. Check system CUDA
exit  # Exit Nix
nvcc --version
nvidia-smi

# 2. Build manually (see "Advanced: CUDA Build" above)
```

### Vulkan: Works Great! (Recommended)

```bash
# Vulkan is the easiest GPU backend in Nix
nix develop
./scripts/build-llama-mtmd-multibackend.sh --vulkan

# Test
llama-mtmd-cli-vulkan \
  -m ~/.cache/llama.cpp/models/gemma-3-4b-it-q4_0.gguf \
  --mmproj ~/.cache/llama.cpp/models/mmproj-model-f16-4B.gguf \
  --image test.jpg \
  -p "What text?" \
  -n 50
```

### SYCL: Compiler Not Found
```bash
# Ensure OneAPI environment is loaded
source /opt/intel/oneapi/setvars.sh

# Verify compilers
which icx icpx
```

### Vulkan: CMake Can't Find SDK
```bash
# In Nix environment
nix develop
echo $VULKAN_SDK  # Should be set

# Verify glslc is available
which glslc

# List Vulkan devices
vulkaninfo | grep -A 5 "Device Name"
```

### Vulkan: No Device Found
```bash
# List available devices
vulkaninfo | grep deviceName

# Install drivers
# NVIDIA: nvidia-driver-xxx
# AMD: mesa-vulkan-drivers
# Intel: intel-media-driver

# Verify drivers are loaded
lsmod | grep -E 'nvidia|amdgpu|i915'

# Test Vulkan
vulkaninfo --summary
```

## Integration with Nix

### Summary of Nix Support

| Backend | Nix Command | Status |
|---------|-------------|--------|
| **Vulkan** | `nix develop` | ✅ Fully supported |
| **CUDA** | `nix develop .#cuda` | ⚠️ Works but slow, better outside Nix |
| **SYCL** | `nix develop --impure .#sycl` | ⚠️ Hybrid (needs system OneAPI) |

### Why `--impure` for SYCL?

The `--impure` flag allows Nix to access system files like `/opt/intel/oneapi`:

```bash
# Pure mode (default) - isolated from system
nix develop           # ✅ Works for Vulkan

# Impure mode - can access system files
nix develop --impure .#sycl  # ✅ Needed for SYCL with system OneAPI
```
