# Vulkan Vision Encoder Corruption on Intel iGPU - Research Findings

## Problem Statement
Vulkan backend in llama.cpp produces corrupted CLIP/vision encoder (SigLIP) embeddings on Intel UHD Graphics CML GT2 (Gen9.5) iGPU, while the text LLM runs fine on Vulkan. Symptoms:
- Fast encode (~22s) but **garbage LLM output** (`****1****1111********`)
- Greedy sampling still garbage → embeddings are corrupted, not a sampling issue
- `--no-mmproj-offload` (CLIP on CPU) **works correctly** → confirms bug is GPU-side
- GPU info: `uma: 1 | fp16: 1 | bf16: 0 | warp size: 32 | shared memory: 32768 | int dot: 0 | matrix cores: none`

---

## 1. SigLIP/Gemma3 Vision Graph — Ops Used

From `tools/mtmd/models/siglip.cpp`, the Gemma3 projector builds this graph:

```
build_vit()                        → 27-layer ViT (GELU, NORM, MUL_MAT, FLASH_ATTN_EXT, etc.)
  ↓
ggml_transpose()                   → TRANSPOSE
  ↓
ggml_cont_4d()                     → CONT
  ↓
ggml_pool_2d(AVG, kernel, kernel)  → POOL_2D  ← SUSPECT #1
  ↓
ggml_reshape_3d()                  → RESHAPE
  ↓
ggml_cont()                        → CONT
  ↓
ggml_transpose()                   → TRANSPOSE
  ↓
ggml_rms_norm()                    → RMS_NORM
  ↓
ggml_mul(soft_emb_norm_w)          → MUL
  ↓
ggml_mul_mat(input_proj_w^T)       → MUL_MAT
```

**Within the 27-layer ViT (`build_vit`)**, these ops are used:
- `MUL_MAT` (attention QKV, projections, FFN)
- `FLASH_ATTN_EXT` (self-attention)
- `ADD` (residuals, biases)
- `NORM` (LayerNorm)
- `GELU` (FFN activation)
- `MUL` (scaling)
- `RESHAPE`, `PERMUTE`, `CONT`, `VIEW`

**For GLM4V-style projectors** (different model, same bug class), **IM2COL** is also used in the patch merger stage.

---

## 2. GitHub Issues — Known Vulkan + Vision Bugs

### Issue #18164: GLM4.6V Flash incoherent on Vulkan (CLOSED - FIXED)
- **Exact same symptoms**: Vision model produces garbage on Vulkan, `--no-mmproj-offload` fixes it
- **Hardware**: RTX 4090 + Intel RaptorLake iGPU (Vulkan used on the RTX 4090)
- **Root Cause**: **IM2COL shader overflowing `maxComputeWorkGroupCount` limits**
- **Debug method**: jeffbolznv ran with `GGML_VULKAN_CHECK_RESULTS` enabled:
  ```
  ERROR: Invalid value in IM2COL i3=3 i2=0 i1=0 i0=4170 result=-nan correct=-0.0597229
  First error: result=-1.70801 correct=0.416504 i3=3 i2=0 i1=0 i0=4097
  ```
  NaN values and wildly wrong results in the IM2COL op, propagating to all subsequent ops.
- **Fix**: PR #18180 — `vulkan: fix im2col overflowing maxworkgroupcount`
- **Key quote from jeffbolznv**: *"It's one of those bugs where we overflow the max workgroup count limits."*

### Issue #15846 / #15875: Vulkan gibberish on AMD RDNA1/RDNA2 (CLOSED - FIXED)
- **Root Cause**: **`subgroup_arithmetic`** operations broken on MoltenVK
- **Fix**: Disable `device->subgroup_arithmetic = false` for AMD GPUs on macOS (PR #15886)
- **Key code** (ggml-vulkan.cpp line ~3733):
  ```cpp
  device->subgroup_arithmetic = (vk11_props.subgroupSupportedStages & vk::ShaderStageFlagBits::eCompute) &&
                                (vk11_props.subgroupSupportedOperations & vk::SubgroupFeatureFlagBits::eArithmetic);
  ```
- **Key insight**: `const bool use_subgroups = device->subgroup_arithmetic && device->architecture != vk_device_architecture::AMD_GCN;`
  - GCN cards were excluded from subgroups → no bug. RDNA1/2 were not excluded → bug.
  - **Intel Gen9.5 could have similar subgroup issues** (reports `warp size: 32`, supports subgroup arithmetic)

### Issue #18616: maxComputeWorkGroupCount assertion failure (Intel A770)
- Vulkan workgroup count limit exceeded on Intel discrete GPU
- Similar class of bug to #18164

### Issue #19420: Qwen3-Coder crash on Intel Arrow Lake Vulkan (OPEN)
- **Intel Arrow Lake 130T/140T GPU** crashes with `ErrorOutOfHostMemory`
- Root cause: command buffer exhaustion (>16383 command buffers during async tensor loading)
- **@0cc4m quote**: *"Especially on Intel, which is rather temperamental."*
- Fix in progress: PR #20059

### Issue #13778: Mistral Small Multimodal fails with Vulkan (CLOSED)
- Another vision model failing on Vulkan backend

---

## 3. Vulkan pool2d Shader Analysis

### Shader exists: `pool2d.comp`
From `vulkan-shaders-gen.cpp` (~line 979):
```cpp
string_to_spv("pool2d_f32", "pool2d.comp", merge_maps(base_dict, {{"A_TYPE", "float"}, {"D_TYPE", "float"}}));
```

**Only f32 variant exists** — no f16 variant of pool2d. Compare with conv2d which has both:
```cpp
string_to_spv("conv2d_f16_f32", "conv2d.comp", merge_maps(..., {{"A_TYPE", "float16_t"}, {"D_TYPE", "float"}}));
string_to_spv("conv2d_f32",     "conv2d.comp", merge_maps(..., {{"A_TYPE", "float"},      {"D_TYPE", "float"}}));
```

### Implications for POOL_2D on Intel:
- The shader operates entirely in f32 → **precision is not the issue for pool2d itself**
- However, **upstream ops may feed corrupted data** into pool2d
- The IM2COL workgroup overflow bug (#18164) is the primary known cause of vision corruption on Vulkan

---

## 4. Potential Root Causes for Intel CML GT2

### Most Likely: IM2COL / Workgroup Count Overflow
- **Same symptoms** as #18164 (garbage output, `--no-mmproj-offload` fixes it)
- Intel CML GT2 has **smaller `maxComputeWorkGroupCount` limits** than discrete GPUs
- The fix (PR #18180) was merged Dec 2025 — **check if your llama.cpp build includes this fix**
- The SigLIP vision encoder uses operations that internally use IM2COL

### Possible: Subgroup Arithmetic Issues
- Intel Gen9.5 reports `subgroup_arithmetic = true` (via warp size 32 + compute stage support)
- MoltenVK had broken subgroup arithmetic on AMD → Intel Mesa/ANV drivers could have similar issues
- The `use_subgroups` flag controls shader selection — wrong results from subgroup ops would corrupt everything

### Possible: Workgroup Size / Shared Memory Limits
- Intel CML GT2: `shared memory: 32768` (32KB) — this is **half** of what discrete GPUs typically have
- Some Vulkan compute shaders may assume more shared memory is available
- `maxComputeWorkGroupCount[2]` assertion failure has been seen on Intel A770 (#18616)

### Less Likely: f16 Precision
- pool2d shader is f32-only, so pool2d itself won't have f16 precision issues
- However, other vision ops (MUL_MAT with f16 weights, FLASH_ATTN_EXT) could have precision issues on Intel
- Intel CML GT2 reports `fp16: 1` but Gen9.5 f16 support may have quirks

---

## 5. Debug Environment Variables & Build Options

### Runtime Environment Variables
| Variable | Purpose |
|---|---|
| `GGML_VK_MEMORY_LOGGER=1` | Log all Vulkan memory allocations/frees |
| `GGML_VK_DISABLE_ASYNC=1` | Disable async tensor loading (fixes #19420 on Intel) |
| `GGML_VK_DISABLE_HOST_VISIBLE_VIDMEM=1` | Disable host-visible video memory |
| `GGML_VK_DISABLE_MMVQ=1` | Disable matrix-matrix/vector-quantized kernels |

### Compile-Time CMake Options (must rebuild)
| CMake Option | Purpose |
|---|---|
| `GGML_VULKAN_CHECK_RESULTS=ON` | **KEY**: Compares GPU results against CPU reference for every op. Shows first divergent op. |
| `GGML_VULKAN_DEBUG=ON` | Verbose Vulkan debug output |
| `GGML_VULKAN_MEMORY_DEBUG=ON` | Memory allocation debug output |
| `GGML_VULKAN_SHADER_DEBUG_INFO=ON` | Adds `-g` flag to glslc shader compilation |
| `GGML_VULKAN_VALIDATE=ON` | Enable Vulkan validation layers |

### How jeffbolznv diagnosed #18164:
```bash
cmake -B build -DGGML_VULKAN=ON -DGGML_VULKAN_CHECK_RESULTS=ON
cmake --build build
# Run the vision model → shows first op with incorrect results
```

---

## 6. Intel-Specific Workarounds in ggml-vulkan.cpp

### Known Architecture Detection
From the GitHub issues, ggml-vulkan.cpp has:
- `device->vendor_id` — PCI vendor ID (Intel = `0x8086`)
- `device->architecture` — enumerated architecture (e.g., `vk_device_architecture::AMD_GCN`)
- `device->subgroup_arithmetic` — whether subgroup arithmetic is available/working
- `device->subgroup_shuffle` — whether subgroup shuffle is available
- `device->subgroup_ballot` — whether subgroup ballot is available

### Known Intel Handling
- Intel UMA detection: `uma: 1` flag means unified memory architecture
- Intel is known to have smaller limits than discrete GPUs
- No specific Intel Gen9.5 workarounds found (unlike AMD/MoltenVK workarounds)
- @0cc4m confirmed Intel is "temperamental" with Vulkan

---

## 7. Recommended Diagnostic Steps

### Step 1: Check llama.cpp version
```bash
# Ensure your build includes PR #18180 (im2col workgroup fix from Dec 2025)
# Check build number — must be > b7446
```

### Step 2: Build with GGML_VULKAN_CHECK_RESULTS
```bash
cmake -B build_debug \
  -DGGML_VULKAN=ON \
  -DGGML_VULKAN_CHECK_RESULTS=ON \
  -DGGML_VULKAN_VALIDATE=ON

cmake --build build_debug -j$(nproc)

# Run your vision model — will print first op with error
./build_debug/bin/llama-mtmd-cli \
  -m gemma-3-4b-it-Q4_K_M.gguf \
  --mmproj mmproj-gemma-3-4b-it-Q4_K_M-f16.gguf \
  --image test.jpg -p "describe this image"
```

Expected output will show something like:
```
node_XXX op=SOME_OP avg_err=0.00XXXX    ← normal ops
ERROR: Invalid value in <OP_NAME> ...    ← FIRST BAD OP
```

### Step 3: Test with subgroup arithmetic disabled
If Step 2 doesn't reveal the issue, try forcing off subgroup arithmetic by editing ggml-vulkan.cpp:
```cpp
// Around line 3733, after subgroup_arithmetic is set:
device->subgroup_arithmetic = false;  // Force disable for testing
```

### Step 4: Run test-backend-ops
```bash
./build_debug/bin/test-backend-ops -b Vulkan0 | grep FAIL
```
This will show which specific ops fail on your Vulkan device.

### Step 5: Try runtime env vars
```bash
GGML_VK_DISABLE_ASYNC=1 ./build/bin/llama-mtmd-cli ...
```

---

## 8. Root Cause Hypothesis

**Primary hypothesis**: The IM2COL workgroup count overflow bug (fixed in PR #18180) is the most likely cause. Intel CML GT2 has **smaller `maxComputeWorkGroupCount` limits** than the NVIDIA GPUs where this was originally found, making it even more susceptible to this overflow.

**Secondary hypothesis**: Subgroup arithmetic operations may produce incorrect results on Intel Gen9.5 Vulkan drivers (similar to the MoltenVK/AMD issue in #15846), corrupting intermediate computations in the vision encoder.

**Action**: Build with `GGML_VULKAN_CHECK_RESULTS=ON` to identify the exact failing op. This is the definitive diagnostic tool used by the llama.cpp Vulkan maintainers.
