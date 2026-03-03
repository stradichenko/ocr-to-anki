"""
llama-mtmd-cli wrapper for multimodal (vision) inference.

This backend calls llama-mtmd-cli as a subprocess for vision OCR tasks.
It auto-detects the best GPU backend (Vulkan, CUDA, etc.) and uses the
appropriate binary.

For text-only tasks, use LlamaCppServer (llama_cpp_server.py) instead.
"""

import os
import subprocess
import logging
import time
from pathlib import Path
from typing import Optional

from backends.auto_detect import detect, Backend, DetectionResult, _opencl_env

log = logging.getLogger(__name__)


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
        mmproj_offload: bool = False,
    ):
        models_dir = Path(
            os.getenv("LLAMA_CPP_MODELS", Path.home() / ".cache" / "llama.cpp" / "models")
        )

        # Model files
        self.model_path = Path(model_path) if model_path else models_dir / "gemma-3-4b-it-qat-q4_0_s.gguf"
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
        # Intel iGPU Vulkan produces corrupted vision encoder output;
        # keep the CLIP/mmproj on CPU and only offload text layers to GPU.
        # The OpenCL backend does NOT have this issue -- GPU vision works.
        self.mmproj_offload = mmproj_offload

        # Detect if we're using the OpenCL backend
        self._is_opencl = (
            self.binary_path.name == "llama-mtmd-cli-opencl"
            or (self.detection and self.detection.recommended_backend == Backend.OPENCL)
        )

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
        runs on CPU when ``--no-mmproj-offload`` is set (required for
        Intel iGPU Vulkan which corrupts CLIP outputs).

        Returns
        -------
        dict
            Keys: text, elapsed_s, backend, image
        """
        image_path = str(Path(image_path).resolve())
        if not Path(image_path).exists():
            raise FileNotFoundError(f"Image not found: {image_path}")

        cmd = self._build_cmd(image_path, prompt)
        log.info("Running: %s", " ".join(cmd[:6]) + " ...")

        # Set up environment (OpenCL needs OCL_ICD_VENDORS)
        env = _opencl_env() if self._is_opencl else None

        t0 = time.monotonic()
        try:
            proc = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=timeout,
                env=env,
            )
        except subprocess.TimeoutExpired:
            raise TimeoutError(f"Vision OCR timed out after {timeout}s")

        elapsed = time.monotonic() - t0

        if proc.returncode != 0:
            log.error("llama-mtmd-cli stderr:\n%s", proc.stderr[-500:] if proc.stderr else "(empty)")
            raise RuntimeError(
                f"llama-mtmd-cli failed (exit {proc.returncode}): {proc.stderr[-300:]}"
            )

        # The model output is on stdout; stderr has loading/timing info.
        # With --jinja the prompt is NOT echoed, so stdout is clean.
        text = proc.stdout.strip()
        backend = self.detection.recommended_backend.value if self.detection else "unknown"

        # Parse timings from stderr for diagnostics
        timings = {}
        if proc.stderr:
            for line in proc.stderr.splitlines():
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
