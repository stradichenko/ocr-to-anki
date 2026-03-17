"""
llama.cpp server wrapper for fully offline local inference.

Uses llama-server (or llama-server-opencl) with Gemma 3 4B for text and
vision tasks.  Manages the server lifecycle (start / health / stop) and
provides a ``generate()`` helper that works around the LCP prompt-cache
bug that silently drops image embeddings on consecutive identical prompts.

Dependencies: Python stdlib only (urllib, json, subprocess, …).
"""

import json
import logging
import os
import platform
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.request
import uuid
from pathlib import Path
from typing import Optional

from backends.auto_detect import Backend, _opencl_env, _llama_bin_cache, detect  # noqa: F401

log = logging.getLogger(__name__)


# ── Binary resolution ────────────────────────────────────────────────

_SERVER_BINARY_NAMES: dict[Backend, list[str]] = {
    Backend.CUDA:   ["llama-server-cuda",   "llama-server"],
    Backend.METAL:  ["llama-server-metal",  "llama-server"],
    Backend.VULKAN: ["llama-server-vulkan", "llama-server"],
    Backend.OPENCL: ["llama-server-opencl"],
    Backend.SYCL:   ["llama-server-sycl",   "llama-server"],
    Backend.CPU:    ["llama-server-cpu",     "llama-server"],
}

_SEARCH_PATHS = [
    _llama_bin_cache(),                # Auto-downloaded binaries
    Path.home() / ".local" / "bin",    # User-installed (Linux/macOS)
    Path("/usr/local/bin"),
    Path("/usr/bin"),
]


def _find_server_binary(backend: Backend) -> Optional[Path]:
    """Locate a llama-server binary for *backend*.

    Search order mirrors :func:`auto_detect._find_binary`:
    explicit dirs → recursive cache search → PATH.
    On Windows ``.exe`` is tried automatically.
    """
    import shutil

    is_win = platform.system() == "Windows"
    cache = _llama_bin_cache()

    for name in _SERVER_BINARY_NAMES.get(backend, ["llama-server"]):
        ext_names = [f"{name}.exe", name] if is_win else [name]

        for n in ext_names:
            # 1. Flat search in explicit dirs
            for d in _SEARCH_PATHS:
                candidate = d / n
                if candidate.is_file() and (is_win or os.access(candidate, os.X_OK)):
                    return candidate

            # 2. Recursive search in cache dir
            if cache.exists():
                for f in cache.rglob(n):
                    if f.is_file() and (is_win or os.access(f, os.X_OK)):
                        return f

        # 3. PATH lookup
        found = shutil.which(name)
        if found:
            return Path(found)

    return None


class LlamaCppServer:
    """Manages a llama-server process for local inference with Gemma 3."""

    def __init__(
        self,
        model_path: Optional[str] = None,
        mmproj_path: Optional[str] = None,
        binary_path: Optional[str] = None,
        backend: Optional[Backend] = None,
        host: str = "127.0.0.1",
        port: int = 8090,
        context_size: int = 4096,
        n_gpu_layers: Optional[int] = None,
        threads: Optional[int] = None,
        verbose: bool = False,
    ):
        self.host = host
        self.port = port
        self.context_size = context_size
        # n_gpu_layers: None = auto (CPU on Windows, GPU elsewhere),
        #               -1   = force all layers on GPU,
        #                0   = force CPU-only.
        if n_gpu_layers is None:
            if platform.system() == "Windows":
                self.n_gpu_layers = 0
                log.info("Windows detected — defaulting to CPU-only (-ngl 0)")
            else:
                self.n_gpu_layers = -1
        else:
            self.n_gpu_layers = n_gpu_layers
        self.threads = threads or max(1, (os.cpu_count() or 4) // 2)
        self.verbose = verbose
        self.process: Optional[subprocess.Popen] = None

        # ── Model files ──────────────────────────────────────────────
        models_dir = Path(
            os.getenv(
                "LLAMA_CPP_MODELS",
                Path.home() / ".cache" / "llama.cpp" / "models",
            )
        )

        self.model_path = (
            Path(model_path) if model_path
            else models_dir / "gemma-3-4b-it-q4_0_s.gguf"
        )
        if not self.model_path.exists():
            raise FileNotFoundError(
                f"Model not found: {self.model_path}\n"
                f"Run: ./scripts/setup-llama-cpp.sh"
            )

        self.mmproj_path = (
            Path(mmproj_path) if mmproj_path
            else models_dir / "mmproj-model-f16-4B.gguf"
        )
        self.has_vision = self.mmproj_path.exists()

        # ── Backend / binary detection ───────────────────────────────
        if binary_path:
            self.binary_path = Path(binary_path)
            self.detection = None
            self._is_opencl = "opencl" in self.binary_path.name
        else:
            self.detection = detect(prefer=backend)
            be = self.detection.recommended_backend
            self.binary_path = _find_server_binary(be)
            if self.binary_path is None:
                raise FileNotFoundError(
                    f"llama-server binary not found for {be.value}.\n"
                    "Build with e.g. scripts/build-llama-mtmd-opencl.sh"
                )
            self._is_opencl = be == Backend.OPENCL

        log.info("Server binary: %s", self.binary_path)

    # ── URL helpers ──────────────────────────────────────────────────

    @property
    def base_url(self) -> str:
        return f"http://{self.host}:{self.port}"

    def _http_post(self, path: str, payload: dict, timeout: int = 600) -> dict:
        """POST JSON to the server, return parsed response."""
        data = json.dumps(payload).encode()
        req = urllib.request.Request(
            f"{self.base_url}{path}",
            data=data,
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read())

    def _http_get(self, path: str, timeout: int = 5) -> dict:
        """GET JSON from the server."""
        with urllib.request.urlopen(
            f"{self.base_url}{path}", timeout=timeout
        ) as resp:
            return json.loads(resp.read())

    # ── Lifecycle ────────────────────────────────────────────────────

    def is_running(self) -> bool:
        """Return True if the server responds to /health."""
        try:
            return self._http_get("/health").get("status") == "ok"
        except Exception:
            return False

    def start(self, wait_ready: bool = True, timeout: int = 120):
        """Start the llama-server subprocess."""
        if self.is_running():
            log.info("Server already running at %s", self.base_url)
            return

        cmd = [
            str(self.binary_path),
            "-m", str(self.model_path),
            "--host", self.host,
            "--port", str(self.port),
            "--ctx-size", str(self.context_size),
            "-t", str(self.threads),
            "-ngl", str(self.n_gpu_layers),
            "--parallel", "1",
            "--jinja",
            # Disable the new prompt cache -- it causes LCP slot reuse that
            # corrupts multimodal image embeddings on consecutive requests.
            "--cache-ram", "0",
        ]

        if self.has_vision:
            cmd.extend(["--mmproj", str(self.mmproj_path)])

        log.info("Starting: %s", " ".join(cmd[:8]) + " ...")

        # Environment (OpenCL needs OCL_ICD_VENDORS)
        env = _opencl_env() if self._is_opencl else None

        self.process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            env=env,
        )

        if wait_ready:
            self._wait_for_ready(timeout)

    def _wait_for_ready(self, timeout: int):
        """Block until the server is healthy or raise."""
        t0 = time.monotonic()
        while time.monotonic() - t0 < timeout:
            if self.process and self.process.poll() is not None:
                out = self.process.stdout.read() if self.process.stdout else ""
                raise RuntimeError(f"Server died on startup:\n{out[:1000]}")
            if self.is_running():
                elapsed = time.monotonic() - t0
                log.info("Server ready in %.1fs", elapsed)
                return
            time.sleep(1)
        self.stop()
        raise RuntimeError(f"Server failed to start within {timeout}s")

    def stop(self):
        """Gracefully stop the server."""
        if self.process is not None:
            log.info("Stopping llama-server (pid %d)...", self.process.pid)
            self.process.terminate()
            try:
                self.process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self.process.kill()
            self.process = None

    # ── Generation ───────────────────────────────────────────────────

    def generate(
        self,
        prompt: str,
        image_data: Optional[str] = None,
        max_tokens: int = 512,
        temperature: float = 0.1,
        top_k: int = 40,
        top_p: float = 0.9,
        repeat_penalty: float = 1.1,
        stop: Optional[list[str]] = None,
        system: Optional[str] = None,
        timeout: int = 600,
    ) -> dict:
        """
        Generate text, with optional image input.

        Returns dict with keys: content, tokens_evaluated, tokens_predicted,
        timings, wall_s.
        """
        if not self.is_running():
            raise RuntimeError("Server is not running. Call start() first.")

        gemma_stops = ["<end_of_turn>", "<eos>"]
        if stop:
            gemma_stops.extend(stop)

        # -- Build prompt ------------------------------------------------
        # A unique request-id tag is prepended to every prompt.  This
        # prevents the llama.cpp slot LCP (Longest Common Prefix) cache
        # from matching two consecutive requests that share the same text
        # prefix -- a scenario that silently corrupts multimodal image
        # embeddings and produces empty output.
        rid = uuid.uuid4().hex[:8]

        if image_data and self.has_vision:
            formatted = (
                f"<start_of_turn>user\n"
                f"<start_of_image>\n"
                f"[rid:{rid}] {prompt}"
                f"<end_of_turn>\n"
                f"<start_of_turn>model\n"
            )
            payload: dict = {
                "prompt": formatted,
                "image_data": [{"data": image_data, "id": 10}],
                "n_predict": min(max_tokens, 512),
                "temperature": temperature,
                "top_k": top_k,
                "top_p": top_p,
                "stop": gemma_stops,
                "cache_prompt": False,
                "stream": False,
            }
        else:
            # Text-only
            parts = ""
            if system:
                parts += f"<start_of_turn>system\n{system}<end_of_turn>\n"
            parts += (
                f"<start_of_turn>user\n"
                f"[rid:{rid}] {prompt}"
                f"<end_of_turn>\n"
                f"<start_of_turn>model\n"
            )
            payload = {
                "prompt": parts,
                "n_predict": max_tokens,
                "temperature": temperature,
                "top_k": top_k,
                "top_p": top_p,
                "repeat_penalty": repeat_penalty,
                "stop": gemma_stops,
                "cache_prompt": False,
                "stream": False,
            }

        if self.verbose:
            mode = "vision" if image_data else "text"
            log.debug("[%s] rid=%s n_predict=%d", mode, rid, payload["n_predict"])

        # -- Send request ------------------------------------------------
        t0 = time.monotonic()
        try:
            result = self._http_post("/completion", payload, timeout=timeout)
        except (urllib.error.URLError, OSError) as exc:
            # OSError catches ConnectionError / RemoteDisconnected (server crash)
            raise RuntimeError(f"Request to llama-server failed: {exc}") from exc
        wall = time.monotonic() - t0

        content = result.get("content", "").strip()
        for tok in gemma_stops:
            content = content.replace(tok, "")
        content = content.strip()

        timings = result.get("timings", {})
        return {
            "content": content,
            "tokens_evaluated": result.get("tokens_evaluated", 0),
            "tokens_predicted": result.get("tokens_predicted", 0),
            "timings": {
                "prompt_ms": timings.get("prompt_ms", 0),
                "predicted_ms": timings.get("predicted_ms", 0),
                "predicted_per_second": timings.get("predicted_per_second", 0),
            },
            "wall_s": round(wall, 2),
        }

    # ── Vision convenience method ────────────────────────────────────

    def run_vision(
        self,
        image_path: str,
        prompt: str = (
            "Extract all visible text from this image exactly as written. "
            "Output ONLY the raw text, nothing else."
        ),
        max_tokens: int = 256,
        timeout: int = 600,
    ) -> dict:
        """
        Run vision OCR on a single image.

        Mirrors ``LlamaMtmdCli.run_vision()`` so callers can swap backends
        transparently.

        Returns dict with keys: text, elapsed_s, backend, image, timings.
        """
        import base64

        image_path = str(Path(image_path).resolve())
        if not Path(image_path).exists():
            raise FileNotFoundError(f"Image not found: {image_path}")

        with open(image_path, "rb") as f:
            b64 = base64.b64encode(f.read()).decode()

        result = self.generate(
            prompt=prompt,
            image_data=b64,
            max_tokens=max_tokens,
            timeout=timeout,
        )

        backend = (
            self.detection.recommended_backend.value
            if self.detection else "unknown"
        )
        return {
            "text": result["content"],
            "elapsed_s": result["wall_s"],
            "backend": backend,
            "image": image_path,
            "timings": result["timings"],
        }

    # ── Info / diagnostics ───────────────────────────────────────────

    def info(self) -> dict:
        return {
            "binary": str(self.binary_path),
            "model": str(self.model_path),
            "mmproj": str(self.mmproj_path),
            "backend": (
                self.detection.recommended_backend.value
                if self.detection else "manual"
            ),
            "is_opencl": self._is_opencl,
            "host": self.host,
            "port": self.port,
            "threads": self.threads,
            "ctx_size": self.context_size,
            "n_gpu_layers": self.n_gpu_layers,
            "has_vision": self.has_vision,
        }

    # ── Context manager ──────────────────────────────────────────────

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()


# ── CLI entrypoint ───────────────────────────────────────────────────

def main():
    """Run llama-server in standalone mode or do a quick vision test."""
    import argparse

    logging.basicConfig(
        level=logging.INFO, format="%(levelname)s: %(message)s"
    )

    parser = argparse.ArgumentParser(
        description="llama.cpp server with Gemma 3 4B"
    )
    parser.add_argument("--model", type=str, help="Path to GGUF model")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8090)
    parser.add_argument("--ctx-size", type=int, default=4096)
    parser.add_argument("-v", "--verbose", action="store_true")
    sub = parser.add_subparsers(dest="cmd")

    sub.add_parser("serve", help="Start server and keep running")

    test_p = sub.add_parser(
        "test", help="Quick vision test (starts server if needed)"
    )
    test_p.add_argument("image", help="Image file to OCR")

    args = parser.parse_args()

    server = LlamaCppServer(
        model_path=args.model,
        host=args.host,
        port=args.port,
        context_size=args.ctx_size,
        verbose=args.verbose,
    )

    if args.cmd == "test":
        managed = False
        if not server.is_running():
            print("Starting server...")
            server.start()
            managed = True
        try:
            result = server.run_vision(args.image)
            print(json.dumps(result, indent=2, ensure_ascii=False))
        finally:
            if managed:
                server.stop()
        return

    # Default: start and keep running
    print(f"Server binary : {server.binary_path}")
    print(f"Model         : {server.model_path.name}")
    print(f"Vision        : {'yes' if server.has_vision else 'no'}")
    print()

    try:
        server.start()
        print(f"\nServer running at {server.base_url}")
        print("Press Ctrl+C to stop\n")

        def _handle_signal(sig, frame):
            server.stop()
            sys.exit(0)

        signal.signal(signal.SIGINT, _handle_signal)
        signal.signal(signal.SIGTERM, _handle_signal)

        while True:
            if not server.is_running():
                print("Server stopped unexpectedly")
                break
            time.sleep(2)
    except Exception as e:
        print(f"Error: {e}")
        server.stop()
        sys.exit(1)


if __name__ == "__main__":
    main()
