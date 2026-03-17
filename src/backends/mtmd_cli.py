"""
llama-mtmd-cli wrapper for multimodal (vision) inference.

This backend calls llama-mtmd-cli as a subprocess for vision OCR tasks.
It auto-detects the best GPU backend (Vulkan, CUDA, etc.) and uses the
appropriate binary.

For text-only tasks, use LlamaCppServer (llama_cpp_server.py) instead.
"""

import os
import platform
import subprocess
import logging
import time
from pathlib import Path
from typing import Optional

from backends.auto_detect import detect, Backend, DetectionResult, _opencl_env

log = logging.getLogger(__name__)


def _i915_preempt_timeout_disabled() -> bool:
    """Check if the i915 GPU preemption timeout is disabled (set to 0).

    **Diagnostic only** — disabling the preemption timeout prevents GPU
    hangs (``vk::DeviceLostError``) but does NOT fix the corrupted CLIP
    embeddings.  The corruption is a Vulkan shader computation bug on
    Intel Gen9.5, likely related to subgroup arithmetic (#15846).

    Returns ``True`` if the timeout is disabled.
    """
    import glob

    for card in glob.glob("/sys/class/drm/card*/engine/rcs0/preempt_timeout_ms"):
        # Only check i915 devices
        card_dir = str(Path(card).parents[2])
        driver_link = Path(card_dir) / "device" / "driver"
        try:
            driver = driver_link.resolve().name
        except OSError:
            continue
        if driver != "i915":
            continue
        try:
            val = int(Path(card).read_text().strip())
            if val == 0:
                log.info(
                    "i915 preempt_timeout_ms=0 → enabling GPU mmproj offload "
                    "(~22 s encode vs ~20 min on CPU)"
                )
                return True
            else:
                log.info(
                    "i915 preempt_timeout_ms=%d (>0) → using CPU mmproj "
                    "(run: sudo scripts/setup-intel-gpu-timeout.sh)",
                    val,
                )
                return False
        except (OSError, ValueError):
            continue
    # Not an Intel i915 system — default to GPU offload
    return True


class LlamaMtmdCli:
    """
    Wraps llama-mtmd-cli for vision OCR inference.

    Usage::

        cli = LlamaMtmdCli()
        text = cli.run_vision("path/to/image.jpg", "What text is in this image?")
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        mmproj_path: Optional[str] = None,
        binary_path: Optional[str] = None,
        backend: Optional[Backend] = None,
        n_gpu_layers: int = -1,
        ctx_size: int = 4096,
        threads: Optional[int] = None,
        temp: float = 0.1,
        top_k: int = 40,
        top_p: float = 0.9,
        max_tokens: int = 512,
        mmproj_offload: Optional[bool] = None,
    ):
        models_dir = Path(
            os.getenv("LLAMA_CPP_MODELS", Path.home() / ".cache" / "llama.cpp" / "models")
        )

        # Model files
        self.model_path = Path(model_path) if model_path else models_dir / "gemma-3-4b-it-q4_0_s.gguf"
        self.mmproj_path = Path(mmproj_path) if mmproj_path else models_dir / "mmproj-model-f16-4B.gguf"

        if not self.model_path.exists():
            raise FileNotFoundError(f"Model not found: {self.model_path}")
        if not self.mmproj_path.exists():
            raise FileNotFoundError(f"Vision projector not found: {self.mmproj_path}")

        # Auto-detect backend and binary
        if binary_path:
            self.binary_path = Path(binary_path)
            self.detection = None
        else:
            self.detection = detect(prefer=backend)
            if not self.detection.binary_path:
                raise FileNotFoundError(
                    "llama-mtmd-cli not found. Build with:\n"
                    "  ./scripts/build-llama-mtmd-vulkan.sh\n"
                    "  ./scripts/build-llama-mtmd-opencl.sh  (recommended for Intel iGPUs)"
                )
            self.binary_path = Path(self.detection.binary_path)

        # Generation params
        self.n_gpu_layers = n_gpu_layers
        self.ctx_size = ctx_size
        self.threads = threads or max(1, (os.cpu_count() or 4) // 2)
        self.temp = temp
        self.top_k = top_k
        self.top_p = top_p
        self.max_tokens = max_tokens
        # Intel iGPU vision encoder (mmproj) offloading
        # ───────────────────────────────────────────────
        # GPU mmproj on Intel Gen9.5 Vulkan works correctly when using
        # a patched ggml-vulkan.cpp with these workarounds:
        #   - serialize_graph: per-node submit + fence wait (fixes
        #     broken intra-command-buffer barriers in Mesa i915)
        #   - force_f32_matmul: f32 accumulation in matmul shaders
        #     (fixes 1-6% per-op error from fp16 accumulation)
        #   - disable_fusion: prevents fused ops that bypass barriers
        #
        # Also requires i915 preempt_timeout_ms=0 and heartbeat_interval_ms=0
        # (set via NixOS systemd service or scripts/setup-intel-gpu-timeout.sh).
        #
        # With these fixes, GPU vision encoder runs in ~14s vs ~20 min on CPU.
        # Total inference time: ~5.5 min (vision 14s + prompt eval 69s + gen 248s).
        #
        # OpenCL: needs CPU mmproj — lacks GGML_OP_POOL_2D needed
        # by Gemma 3's SigLIP vision encoder (crashes during warmup).

        # Detect if we're using the OpenCL backend
        self._is_opencl = (
            self.binary_path.name == "llama-mtmd-cli-opencl"
            or (self.detection and self.detection.recommended_backend == Backend.OPENCL)
        )

        self.mmproj_offload = mmproj_offload if mmproj_offload is not None else True

        # Handle for the currently running subprocess so it can be killed
        # on cancellation from another thread.
        self._current_process: Optional[subprocess.Popen] = None

    def _build_cmd(self, image_path: str, prompt: str) -> list[str]:
        """Build the llama-mtmd-cli command."""
        cmd = [
            str(self.binary_path),
            "-m", str(self.model_path),
            "--mmproj", str(self.mmproj_path),
            "--image", str(image_path),
            "-p", prompt,
            "--jinja",
            "--ctx-size", str(self.ctx_size),
            "--threads", str(self.threads),
            "--temp", str(self.temp),
            "--top-k", str(self.top_k),
            "--top-p", str(self.top_p),
            "-ngl", str(self.n_gpu_layers),
            "-n", str(self.max_tokens),
        ]
        if not self.mmproj_offload:
            cmd.append("--no-mmproj-offload")
        # Disable flash attention on backends / platforms where it crashes
        # during the vision encoder (SigLIP) warmup or image encoding:
        #   - OpenCL: lacks FA kernel support entirely.
        #   - Windows Vulkan: FA crashes with STATUS_STACK_BUFFER_OVERRUN
        #     (0xC0000409) during "encoding image slice" on many drivers.
        if self._is_opencl or platform.system() == "Windows":
            cmd.extend(["-fa", "off"])
        return cmd

    def run_vision(
        self,
        image_path: str,
        prompt: str = "Extract all visible text from this image. List each word or phrase you can read.",
        timeout: int = 2700,
    ) -> dict:
        """
        Run vision OCR on an image.

        The default timeout is 2700 s (45 min) because the vision encoder
        runs on GPU with the patched ggml-vulkan.cpp (serialize_graph +
        force_f32_matmul workarounds for Intel Gen9.5). Vision encoding
        takes ~14s on GPU vs ~20 min on CPU. Total inference ~5.5 min
        on i3-10110U with all 34 LLM layers on Vulkan GPU.

        Returns
        -------
        dict
            Keys: text, elapsed_s, backend, image
        """
        image_path = str(Path(image_path).resolve())
        if not Path(image_path).exists():
            raise FileNotFoundError(f"Image not found: {image_path}")

        cmd = self._build_cmd(image_path, prompt)
        log.info(
            "Running: %s ... (mmproj_offload=%s)",
            " ".join(cmd[:6]),
            self.mmproj_offload,
        )

        # Set up environment (OpenCL needs OCL_ICD_VENDORS)
        env = _opencl_env() if self._is_opencl else None

        t0 = time.monotonic()
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            env=env,
        )
        self._current_process = proc
        try:
            stdout, stderr = proc.communicate(timeout=timeout)
            returncode = proc.returncode
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=10)
            raise TimeoutError(f"Vision OCR timed out after {timeout}s")
        finally:
            # Only clear if we're still the active process (avoids
            # clobbering a newer request's handle in concurrent use).
            if self._current_process is proc:
                self._current_process = None

        elapsed = time.monotonic() - t0

        # Negative or None return code means killed by signal (e.g. cancel).
        if returncode is None or returncode < 0:
            raise RuntimeError("Vision OCR was cancelled.")

        if returncode != 0:
            log.error("llama-mtmd-cli stderr:\n%s", stderr[-500:] if stderr else "(empty)")
            raise RuntimeError(
                f"llama-mtmd-cli failed (exit {returncode}): {stderr[-300:]}"
            )

        # The model output is on stdout; stderr has loading/timing info.
        # With --jinja the prompt is NOT echoed, so stdout is clean.
        text = stdout.strip()
        backend = self.detection.recommended_backend.value if self.detection else "unknown"

        # Parse timings from stderr for diagnostics
        timings = {}
        if stderr:
            for line in stderr.splitlines():
                if "image slice encoded in" in line:
                    try:
                        ms = int(line.split("encoded in")[1].split("ms")[0].strip())
                        timings["image_encode_ms"] = ms
                    except (ValueError, IndexError):
                        pass
                elif "image decoded" in line:
                    try:
                        ms = int(line.split("in")[1].split("ms")[0].strip())
                        timings["image_decode_ms"] = ms
                    except (ValueError, IndexError):
                        pass

        return {
            "text": text,
            "elapsed_s": round(elapsed, 2),
            "backend": backend,
            "image": image_path,
            "timings": timings,
        }

    def cancel(self):
        """Kill any running vision OCR subprocess."""
        proc = self._current_process
        if proc is not None:
            log.info("Cancelling vision OCR subprocess (pid=%s)...", proc.pid)
            try:
                proc.kill()
                proc.wait(timeout=10)
            except Exception as exc:
                log.warning("Error killing subprocess: %s", exc)

    def info(self) -> dict:
        """Return diagnostic info about this backend."""
        return {
            "binary": str(self.binary_path),
            "model": str(self.model_path),
            "mmproj": str(self.mmproj_path),
            "backend": self.detection.recommended_backend.value if self.detection else "manual",
            "is_opencl": self._is_opencl,
            "devices": [
                {"name": d.name, "backend": d.backend.value}
                for d in (self.detection.devices if self.detection else [])
            ],
            "threads": self.threads,
            "ctx_size": self.ctx_size,
            "n_gpu_layers": self.n_gpu_layers,
            "max_tokens": self.max_tokens,
            "mmproj_offload": self.mmproj_offload,
            "i915_preempt_timeout_disabled": _i915_preempt_timeout_disabled(),
        }


def main():
    """Quick CLI test."""
    import sys
    import json as _json

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    if len(sys.argv) < 2:
        print("Usage: python -m backends.mtmd_cli <image> [prompt]")
        sys.exit(1)

    image = sys.argv[1]
    prompt = sys.argv[2] if len(sys.argv) > 2 else "Extract all visible text from this image."

    cli = LlamaMtmdCli()
    print(f"Backend: {cli.detection.recommended_backend.value if cli.detection else 'manual'}")
    print(f"Binary:  {cli.binary_path}")
    print(f"Running vision OCR on {image} ...")
    print()

    result = cli.run_vision(image, prompt)
    print(_json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
