#!/usr/bin/env bash
# ──────────────────────────────────────────────────────────────────────
# setup-intel-gpu-timeout.sh — Adjust i915 GPU preemption timeout
#
# The SigLIP vision encoder (27 transformer layers, 1536-dim, 896×896
# image) runs heavy Vulkan compute dispatches that exceed the default
# i915 preemption timeout (640 ms).  When the timeout fires the kernel
# either resets the GPU (vk::DeviceLostError) or corrupts the in-flight
# computation, producing garbage embeddings.
#
# Disabling the preemption timeout (setting it to 0) allows the GPU to
# finish each dispatch without interruption.  Combined with the IM2COL
# workgroup overflow fix (llama.cpp PR #18180, already in our build),
# this enables full GPU-accelerated vision encoding:
#
#   ~22 s encode + ~40 s prompt eval + ~27 s generation ≈ 90 s total
#
# vs the current CPU fallback path:
#
#   ~20 min encode + ~40 s prompt eval + ~27 s generation ≈ 21 min total
#
# Usage (requires root):
#
#   sudo ./scripts/setup-intel-gpu-timeout.sh          # disable timeout
#   sudo ./scripts/setup-intel-gpu-timeout.sh restore  # restore default
#   sudo ./scripts/setup-intel-gpu-timeout.sh status   # show current
#
# ⚠  WARNING: With preemption disabled a truly hung GPU shader will
#    block the render engine until reboot.  This is safe for the OCR
#    workload since the shaders are known to terminate.
# ──────────────────────────────────────────────────────────────────────

set -euo pipefail

# Detect the correct DRM card for the Intel i915 GPU
find_engine_sysfs() {
    local card
    for card in /sys/class/drm/card*; do
        [[ -d "$card/engine/rcs0" ]] || continue
        local driver
        driver=$(basename "$(readlink -f "$card/device/driver")" 2>/dev/null) || continue
        if [[ "$driver" == "i915" ]]; then
            echo "$card/engine/rcs0"
            return 0
        fi
    done
    echo "ERROR: No i915 GPU engine found" >&2
    return 1
}

ENGINE=$(find_engine_sysfs)
PREEMPT="$ENGINE/preempt_timeout_ms"
HEARTBEAT="$ENGINE/heartbeat_interval_ms"

show_status() {
    echo "Engine:     $ENGINE"
    echo "preempt_timeout_ms:    $(cat "$PREEMPT")"
    echo "heartbeat_interval_ms: $(cat "$HEARTBEAT")"
}

case "${1:-enable}" in
    enable|disable-timeout)
        echo "Disabling i915 preemption timeout for Vulkan vision encoder..."
        show_status
        echo ""
        echo 0 > "$PREEMPT"
        # Also increase heartbeat to avoid spurious warnings
        echo 60000 > "$HEARTBEAT"
        echo "→ Updated:"
        show_status
        echo ""
        echo "GPU mmproj offloading should now work correctly."
        echo "To restore defaults: sudo $0 restore"
        ;;
    restore)
        echo "Restoring i915 default timeouts..."
        echo 640 > "$PREEMPT"
        echo 2500 > "$HEARTBEAT"
        show_status
        ;;
    status)
        show_status
        ;;
    *)
        echo "Usage: sudo $0 [enable|restore|status]"
        exit 1
        ;;
esac
