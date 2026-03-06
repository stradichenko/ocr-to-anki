# Intel Gen9/Gen9.5 Vulkan GPU Vision Fix

## Problem

On Intel Gen9.5 iGPUs (e.g. UHD Graphics CML GT2, i3-10110U), the SigLIP
vision encoder (mmproj) produces **corrupted CLIP embeddings** when running
on the Vulkan GPU backend. The LLM then generates multilingual garbage text
(Devanagari, Arabic, random tokens) instead of correct OCR output.

## Root Cause

The **Intel i915 Mesa Vulkan driver** has broken `vkCmdPipelineBarrier`
implementation. Pipeline barriers within a batched command buffer don't
properly flush shader caches between compute dispatches. This causes
read-after-write hazards where one shader reads stale data written by a
previous shader.

### Evidence

1. **Individual ops are correct**: `GGML_VULKAN_CHECK_RESULTS=ON` (which
   serializes execution via per-op fence waits) showed all 406/917 vision
   encoder ops passing with max error of 0.053%.

2. **Heisenbug**: The diagnostic itself masks the bug — CHECK_RESULTS forces
   `last_node = true` for every op, causing per-node submit + fence wait,
   which triggers full GPU cache flushes.

3. **Barriers alone fail**: Enabling `coarse_sync` (per-node pipeline barrier)
   causes `vk::DeviceLostError` — the barrier implementation is broken.

4. **Serialization works**: Forcing per-node submit + fence wait (like
   CHECK_RESULTS but without CPU reference computation) produces correct
   output consistently.

## Fix (in `ggml-vulkan.cpp`)

Three workarounds are applied when `INTEL_GEN9` architecture is detected
(subgroup size: min=8, max=32):

### 1. `serialize_graph = true`
Forces per-node command buffer submit + fence wait. Each compute operation
is fully completed before the next begins. This is slower than batched
execution but works around the broken barrier implementation.

**Location**: `ggml_backend_vk_graph_compute()` — sets `nodes_per_submit = 0`
and calls `ggml_vk_synchronize()` after each submit.

### 2. `force_f32_matmul = true`
Forces f32 accumulation in matmul compute shaders instead of fp16. Without
this, each MUL_MAT op has 1-6% relative error that compounds through the
27-layer SigLIP network.

**Location**: `ggml_vk_get_mul_mat_mat_pipeline()` — adds
`!ctx->device->force_f32_matmul` condition before returning f16acc variants.

### 3. `disable_fusion = true`
Disables operation fusion (e.g. MUL_MAT+ADD, RMS_NORM+MUL) which bypasses
barriers between ops.

## Additional Requirements

### i915 Kernel Timeouts

The GPU serialization significantly increases per-graph execution time.
The i915 kernel timeout defaults (preempt_timeout_ms=640, heartbeat=2500)
will kill the process during long operations. **Both must be set to 0**:

```bash
# NixOS: via systemd service in /etc/nixos/configuration.nix
# Other: sudo scripts/setup-intel-gpu-timeout.sh

# Verify:
cat /sys/class/drm/card*/engine/*/preempt_timeout_ms  # should be 0
cat /sys/class/drm/card*/engine/*/heartbeat_interval_ms  # should be 0
```

## Performance

| Configuration | Vision Encode | Total Time | Output |
|---|---|---|---|
| CPU mmproj (`--no-mmproj-offload`) | ~20 min | ~21 min | ✅ Correct |
| GPU mmproj (no fix) | ~0.7s | ~134s | ❌ Garbage |
| GPU mmproj + serialize_graph + f32acc | ~14s | ~336s | ✅ Correct |

Note: The total time with serialization is ~5.5 min because per-node fence
waits add overhead to both the vision encoder (917 nodes) and LLM (1369 nodes).
This is still much faster than CPU-only mmproj (~21 min).

## Architecture Detection

Intel Gen9/Gen9.5 is detected in `get_device_architecture()` by:
- `subgroupProperties.minSubgroupSize == 8`
- `subgroupProperties.maxSubgroupSize == 32`

This covers Intel UHD 620, 630, CML GT2, and similar Gen9/Gen9.5 iGPUs.

## Files Modified

- `ggml/src/ggml-vulkan/ggml-vulkan.cpp`:
  - `INTEL_GEN9` enum value in `vk_device_architecture`
  - `get_device_architecture()` detection
  - `serialize_graph`, `force_f32_matmul`, `coarse_sync` fields in `vk_device_struct`
  - `ggml_vk_get_mul_mat_mat_pipeline()` — f32 accumulation forcing
  - `ggml_backend_vk_graph_compute()` — serialized execution
- `src/backends/mtmd_cli.py`: `mmproj_offload` default changed to `True`
