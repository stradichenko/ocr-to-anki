# GPU mmproj Offloading on Intel iGPU

## Status: ❌ NOT WORKING on Gen9.5 — Upstream Bug

GPU mmproj offloading on Intel Gen9.5 (CML GT2) produces **corrupted
vision embeddings** due to a **graph-level Vulkan scheduling bug** in
ggml-vulkan.  Individual compute operations pass all tests, but the
27-layer SigLIP ViT graph produces wrong results when executed on GPU.

**This is an upstream bug that cannot be fixed locally.**

| Mode | Encode | Prompt Eval | Gen | Total | Quality |
|------|--------|-------------|-----|-------|---------|
| CPU mmproj + GPU LLM (**current**) | ~20 min | ~40 s | ~27 s | ~21 min | ✅ Correct |
| GPU mmproj + GPU LLM | 22 s | 40 s | 15 s | ~76 s | ❌ Corrupted |
| GPU mmproj + CPU LLM (`-ngl 0`) | 20 s | 29 s | 70 s | ~138 s | ❌ Corrupted |

## Root Cause: Graph-Level Vulkan Scheduling Bug

### Evidence

1. **Individual ops ALL pass** — `test-backend-ops` passes 140/140
   MUL_MAT tests on Vulkan0, including the exact model dimensions
   (`q4_0 m=2048 n=2 k=2560`).

2. **Graph execution corrupts** — `GGML_VULKAN_CHECK_RESULTS=ON` shows
   the first error at check 3 (`Qcur-0`, LLM layer 0 Q-projection):
   ```
   ERROR: avg_err=0.013222 in MUL_MAT (check 3)
   tensor Qcur-0: src0 q4_0 [2560,2048], src1 f32 [2560,2], result f32 [2048,2]
   ```
   Error is systematic precision drift (e.g., -4.95 vs -4.92), not
   random garbage.  Checks 1–2 pass (`inp_scaled` SCALE err=0,
   `attn_norm-0` MUL err=9.47e-08).

3. **`-ngl 0` isolation test** — With LLM on CPU and only mmproj on
   GPU, output is still garbage (`<unk><unused12><unused25>...`).
   **This proves corruption originates in the GPU vision encoder.**

4. **Vision encoder graph:** 863 nodes, 1 split (runs entirely on
   Vulkan0 in all configurations).

### Likely Mechanism

The ggml-vulkan backend uses:
- **Prealloc scratch buffers** (`prealloc_x`, `prealloc_y`,
  `prealloc_split_k`) aggressively reused across dispatches
- **Lazy barrier insertion** — barriers added only on detected
  buffer overlap, with flags `prealloc_x_need_sync` /
  `prealloc_y_need_sync`
- **Submit batching** — ~100 nodes or ~100 MB matmul accumulated
  before GPU queue submit

On Intel Gen9.5, the overlap detection or barrier insertion likely
misses a write-after-read (WAR) or read-after-write (RAW) hazard in
the 863-node vision encoder graph, causing accumulated precision
drift that corrupts the final embeddings.

## Comprehensive Test Matrix

All tests use: Gemma 3 4B QAT Q4_0, mmproj SigLIP f16, 896×896 image,
64 max tokens, `preempt_timeout_ms=0`, build b8182.

### Env Var / Flag Tests (GPU mmproj + GPU LLM)

| Test | Encode | Prompt Eval | Gen | Total | Tokens | Output |
|------|--------|-------------|-----|-------|--------|--------|
| Baseline (no env vars) | 22 s | 40 s | 15 s | 76 s | 64 | ❌ Devanagari garbage |
| `GGML_VK_DISABLE_F16=1` | 20 s | ~40 s | — | ~128 s | varies | ❌ "TheTheThe..." + pad tokens |
| `subgroup_arithmetic=false` (patched) | 20 s | 40 s | 0 ms | 54 s | 0 | ❌ Empty output |
| `GGML_VK_DISABLE_FUSION=1` | 23 s | >277 s | — | >300 s | — | ❌ Timed out |
| `GGML_VK_PREFER_HOST_MEMORY=1` | 20 s | 38 s | 4 s | 65 s | 15 | ❌ Near-empty (`**"**`) |
| `GGML_VK_DISABLE_ASYNC=1` | — | — | — | hangs | — | ❌ Hangs on tensor load |
| `GGML_VK_DISABLE_GRAPH_OPTIMIZE=1` | — | — | — | hangs | — | ❌ Hangs on tensor load |

### Architecture Isolation Tests

| Test | Encode | Prompt Eval | Gen | Total | Output |
|------|--------|-------------|-----|-------|--------|
| GPU mmproj + GPU LLM (all GPU) | 22 s | 40 s | 15 s | 76 s | ❌ Corrupted |
| GPU mmproj + CPU LLM (`-ngl 0`) | 20 s | 29 s | 70 s | 138 s | ❌ Corrupted |
| CPU mmproj + GPU LLM (`--no-mmproj-offload`) | ~20 min | 40 s | 27 s | ~21 min | ✅ Correct |

### Diagnostic Builds

| Test | Result |
|------|--------|
| `GGML_VULKAN_CHECK_RESULTS=ON` | First error at check 3 (MUL_MAT, avg_err=0.013) |
| `test-backend-ops` (140 MUL_MAT tests) | ALL PASS individually |
| `GGML_VK_DISABLE_INTEGER_DOT_PRODUCT=1` | No effect (int_dot=0 already) |

## Prerequisites Applied

### 1. IM2COL workgroup overflow fix (PR #18180)
- **Status:** ✅ Applied (build b8182, commit 05728db)
- Prevents `maxComputeWorkGroupCount` overflow in IM2COL shader
- Required but **not sufficient** — corruption persists

### 2. i915 preemption timeout disabled
- **Status:** ✅ Applied (`preempt_timeout_ms=0`)
- Prevents GPU hangs (`vk::DeviceLostError`) and engine resets
- Required but **not sufficient** — prevents crashes, corruption persists

## Preemption Timeout Script

The `scripts/setup-intel-gpu-timeout.sh` helper is still useful to
prevent GPU hangs during debugging:

```bash
sudo ./scripts/setup-intel-gpu-timeout.sh          # disable timeout
sudo ./scripts/setup-intel-gpu-timeout.sh restore   # restore defaults
sudo ./scripts/setup-intel-gpu-timeout.sh status    # show current
```

## Potential Upstream Fixes

These would need to be implemented in ggml-vulkan.cpp upstream:

1. **Full barrier after every dispatch** — would eliminate all hazards
   but likely destroy performance (similar to `DISABLE_FUSION` result).

2. **Intel Gen9.5-specific workaround** — disable prealloc buffer
   reuse for vision encoder graphs on Gen9.5. Similar to existing
   workarounds for Intel DG1 (`disable_async`) and various Intel GPUs
   (`disable_add_rms_fusion`, `disable_collectives`).

3. **Per-device submit batch size tuning** — reduce the ~100-node
   submit batch threshold for Gen9.5 to force more frequent queue
   submits with implicit barriers.

4. **Explicit synchronization for vision encoder graphs** — add
   `vkQueueWaitIdle()` or pipeline barriers between vision encoder
   graph sections.

## Related Issues

- [llama.cpp #18164](https://github.com/ggml-org/llama.cpp/issues/18164) — IM2COL corruption (fixed by PR #18180)
- [llama.cpp #15846](https://github.com/ggml-org/llama.cpp/issues/15846) — subgroup_arithmetic on AMD (analogous bug class)
- [llama.cpp #17389](https://github.com/ggml-org/llama.cpp/issues/17389) — DeviceLostError on Intel
- [llama.cpp #19327](https://github.com/ggml-org/llama.cpp/issues/19327) — Intel xe driver timeout

## Hardware Tested

- Intel Core i3-10110U (CML GT2, Gen 9.5 iGPU, UMA)
- `fp16=1, bf16=0, warp_size=32, shared_mem=65536, int_dot=0, matrix_cores=none`
- Mesa ANV Vulkan driver (NixOS 25.11)
- llama.cpp build b8182 (commit 05728db)
- Model: Gemma 3 4B QAT Q4_0 (2.3 GB), mmproj SigLIP f16 (812 MB)
