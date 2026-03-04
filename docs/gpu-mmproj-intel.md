# GPU mmproj Offloading on Intel iGPU

## Overview

The SigLIP vision encoder (mmproj) can run on either CPU or GPU:

| Mode | Encode Time | Total OCR | Quality |
|------|------------|-----------|---------|
| CPU mmproj (default) | ~20 min | ~21 min | ✅ Correct |
| GPU mmproj (optimized) | ~22 s | ~90 s | ✅ Correct (with fix) |

GPU mmproj is **14x faster** but requires a kernel-level fix on Intel iGPUs.

## The Problem

Intel i915's default GPU preemption timeout is **640 ms**. The SigLIP
encoder's Vulkan compute dispatches exceed this, causing:

1. **GPU hang** → `vk::DeviceLostError: ErrorDeviceLost`
2. **Corrupted embeddings** → garbage text output (Devanagari, Greek, etc.)

The kernel log shows:
```
Fence expiration time out i915-0000:00:02.0:llama-mtmd-cli
Resetting rcs0 for preemption time out
GPU HANG: ecode 9:1:8ed9fff2
```

## The Fix

Disable the i915 preemption timeout:

```bash
# One-time (requires root)
sudo ./scripts/setup-intel-gpu-timeout.sh

# Or manually:
sudo sh -c 'echo 0 > /sys/class/drm/card1/engine/rcs0/preempt_timeout_ms'
```

The backend auto-detects this setting and enables GPU mmproj when safe.

### Make Persistent (NixOS)

Add to your `configuration.nix`:

```nix
# Option 1: tmpfiles.d rule (recommended)
systemd.tmpfiles.rules = [
  "w /sys/class/drm/card1/engine/rcs0/preempt_timeout_ms - - - - 0"
  "w /sys/class/drm/card1/engine/rcs0/heartbeat_interval_ms - - - - 60000"
];

# Option 2: Kernel parameter (disables ALL hang detection)
# boot.kernelParams = [ "i915.enable_hangcheck=0" ];
```

### Restore Defaults

```bash
sudo ./scripts/setup-intel-gpu-timeout.sh restore
```

## Prerequisites

Both of these must be satisfied for GPU mmproj to work:

1. **IM2COL workgroup overflow fix** — llama.cpp PR [#18180](https://github.com/ggml-org/llama.cpp/pull/18180)
   (merged Dec 21 2025, included in build b8182+). Prevents corrupted CLIP
   embeddings from the Vulkan IM2COL shader overflowing `maxComputeWorkGroupCount`.

2. **i915 preemption timeout disabled** — set `preempt_timeout_ms=0`
   via sysfs. Prevents the kernel from interrupting/resetting the GPU
   during long compute dispatches.

## How Auto-Detection Works

In `src/backends/mtmd_cli.py`, the `_i915_preempt_timeout_disabled()`
function reads `/sys/class/drm/card*/engine/rcs0/preempt_timeout_ms` and:

- If `0` → enables GPU mmproj offload (fast path, ~90 s)
- If `>0` → falls back to CPU mmproj (slow path, ~21 min)
- If not an i915 system → defaults to GPU offload

## ⚠ Warning

With preemption disabled, a truly hung GPU shader will block the render
engine until reboot. This is safe for OCR workloads since the SigLIP
shaders are known to terminate, but **do not leave it disabled if you're
doing GPU development** where shaders might have infinite loops.

## Related Issues

- [llama.cpp #18164](https://github.com/ggml-org/llama.cpp/issues/18164) — IM2COL corruption (fixed by PR #18180)
- [llama.cpp #17389](https://github.com/ggml-org/llama.cpp/issues/17389) — DeviceLostError on Intel (same root cause)
- [llama.cpp #19327](https://github.com/ggml-org/llama.cpp/issues/19327) — Intel xe driver timeout (similar issue, different driver)
- [llama.cpp #15846](https://github.com/ggml-org/llama.cpp/issues/15846) — subgroup_arithmetic bugs on some Intel drivers

## Hardware Tested

- Intel Core i3-10110U (CML GT2, Gen 9.5 iGPU)
- Mesa ANV Vulkan driver
- NixOS 25.11
- llama.cpp build b8182 (commit 05728db)
