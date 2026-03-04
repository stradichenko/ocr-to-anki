# GPU mmproj Offloading on Intel iGPU

## Status: ❌ NOT WORKING on Gen9.5

GPU mmproj offloading on Intel Gen9.5 (CML GT2) produces **corrupted
CLIP embeddings** even with all known fixes applied.  The corruption is
a Vulkan shader computation bug, likely related to subgroup arithmetic.

| Mode | Encode Time | Total OCR | Quality |
|------|------------|-----------|---------|
| CPU mmproj (**current**) | ~20 min | ~21 min | ✅ Correct |
| GPU mmproj (Vulkan) | ~22 s | ~188 s | ❌ Corrupted |

## What Was Tried

### 1. IM2COL workgroup overflow fix (PR #18180)
- **Status:** ✅ Applied (build b8182, commit 05728db)
- Prevents `maxComputeWorkGroupCount` overflow in IM2COL shader
- Required but **not sufficient** — corruption persists

### 2. i915 preemption timeout disabled
- **Status:** ✅ Applied (`preempt_timeout_ms=0`)
- Prevents GPU hangs (`vk::DeviceLostError`) and engine resets
- Required but **not sufficient** — prevents crashes, corruption persists
- The kernel still logs `Fence expiration time out` warnings

### 3. `GGML_VK_DISABLE_INTEGER_DOT_PRODUCT=1`
- **Status:** ✅ Tested — no effect (int_dot already reports 0)

### Test Results

With **both** fixes applied simultaneously:
- ✅ No GPU hangs or resets in kernel log
- ✅ Process completes (~188s)
- ❌ Output is multilingual garbage (Chinese, Japanese, Korean, Greek,
  Hindi, Arabic, Thai mixed with random English words)
- ❌ CLIP embeddings are corrupted at the shader level

## Root Cause Analysis

The corruption occurs inside the Vulkan compute shaders during the
27-layer SigLIP ViT forward pass.  The most likely candidates:

1. **Subgroup arithmetic bug** — Intel Gen9.5 reports
   `subgroup_arithmetic=true` but the Mesa ANV driver may produce
   incorrect results for subgroup reduction operations.  This is the
   same class of bug that affected AMD RDNA1/2 on MoltenVK (#15846).

2. **Shared memory or workgroup size issue** — Gen9.5 has smaller
   limits (65536 bytes shared memory) than discrete GPUs.

## Next Steps to Diagnose

To identify the exact failing Vulkan op, rebuild llama.cpp with:

```bash
cmake -B build_debug \\
  -DGGML_VULKAN=ON \\
  -DGGML_VULKAN_CHECK_RESULTS=ON \\
  -DGGML_VULKAN_VALIDATE=ON
cmake --build build_debug -j$(nproc)

# Run — will show first op with incorrect results:
./build_debug/bin/llama-mtmd-cli \\
  -m model.gguf --mmproj mmproj.gguf \\
  --image test.jpg -p "describe this image"
```

To test if subgroup arithmetic is the culprit, edit `ggml-vulkan.cpp`:
```cpp
// ~line 3733, force disable:
device->subgroup_arithmetic = false;
```

## Preemption Timeout Script

The `scripts/setup-intel-gpu-timeout.sh` helper is still useful to
prevent GPU hangs during debugging:

```bash
sudo ./scripts/setup-intel-gpu-timeout.sh          # disable timeout
sudo ./scripts/setup-intel-gpu-timeout.sh restore   # restore defaults
sudo ./scripts/setup-intel-gpu-timeout.sh status    # show current
```

## Related Issues

- [llama.cpp #18164](https://github.com/ggml-org/llama.cpp/issues/18164) — IM2COL corruption (fixed by PR #18180)
- [llama.cpp #15846](https://github.com/ggml-org/llama.cpp/issues/15846) — subgroup_arithmetic on AMD (analogous bug)
- [llama.cpp #17389](https://github.com/ggml-org/llama.cpp/issues/17389) — DeviceLostError on Intel
- [llama.cpp #19327](https://github.com/ggml-org/llama.cpp/issues/19327) — Intel xe driver timeout

## Hardware Tested

- Intel Core i3-10110U (CML GT2, Gen 9.5 iGPU)
- Mesa ANV Vulkan driver
- NixOS 25.11
- llama.cpp build b8182 (commit 05728db)
