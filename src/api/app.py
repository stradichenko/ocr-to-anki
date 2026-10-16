"""
OCR-to-Anki FastAPI Application.

Hybrid architecture:
  • Vision OCR  → llama-mtmd-cli (subprocess, auto-detected GPU backend)
  • Text tasks  → llama-server   (persistent HTTP server)

Run:
  uvicorn api.app:app --host 0.0.0.0 --port 8000 --reload
"""

import asyncio
import base64
import hashlib
import logging
import os
import platform
import re
import tempfile
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from fastapi import FastAPI, HTTPException, UploadFile, File, Form, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

from api.models import (
    BackendInfoResponse,
    BackendName,
    EnrichRequest,
    EnrichResponse,
    EnrichWordResult,
    GenerateRequest,
    GenerateResponse,
    HealthResponse,
    ModelFileInfo,
    ModelStatusResponse,
    PipelineImageToCardsResponse,
    UpdateCheckResponse,
    VisionOCRRequest,
    VisionOCRResponse,
)
from api.enrichment import (
    DEFAULT_ENRICH_TOKENS_PER_WORD,
    build_enrich_prompt,
    build_enrich_system,
    compute_batch_size,
    compute_batch_timeout,
    parse_enrich_response,
    sanitize_words,
)
from backends.auto_detect import detect, Backend, find_binary, llama_bin_cache
from backends.mtmd_cli import LlamaMtmdCli
from backends.llama_cpp_server import LlamaCppServer

log = logging.getLogger(__name__)

# -------------------------------------------------------------------
# Model constants (must match scripts/setup-llama-cpp.sh)
# -------------------------------------------------------------------
_MODEL_FILE = "gemma-3-4b-it-q4_0_s.gguf"
_MMPROJ_FILE = "mmproj-model-f16-4B.gguf"
_MODEL_URL = (
    "https://huggingface.co/stduhpf/google-gemma-3-4b-it-qat-q4_0-gguf-small"
    f"/resolve/main/{_MODEL_FILE}"
)
_MMPROJ_URL = (
    "https://huggingface.co/stduhpf/google-gemma-3-4b-it-qat-q4_0-gguf-small"
    "/resolve/main/mmproj-google_gemma-3-4b-it-f16.gguf"
)


def _models_dir() -> Path:
    """Return the directory where model files are stored."""
    return Path(
        os.getenv("LLAMA_CPP_MODELS",
                   Path.home() / ".cache" / "llama.cpp" / "models")
    )


# -------------------------------------------------------------------
# llama.cpp binary auto-download
# -------------------------------------------------------------------
_LLAMA_TAG = "b8292"
_LLAMA_RELEASE_BASE = (
    f"https://github.com/ggml-org/llama.cpp/releases/download/{_LLAMA_TAG}"
)


def _llama_asset_url() -> str:
    """Return the download URL for the platform-appropriate llama.cpp release."""
    system = platform.system()
    machine = platform.machine().lower()
    if system == "Windows":
        return f"{_LLAMA_RELEASE_BASE}/llama-{_LLAMA_TAG}-bin-win-vulkan-x64.zip"
    elif system == "Darwin":
        arch = "arm64" if machine in ("arm64", "aarch64") else "x64"
        return f"{_LLAMA_RELEASE_BASE}/llama-{_LLAMA_TAG}-bin-macos-{arch}.tar.gz"
    else:  # Linux
        return f"{_LLAMA_RELEASE_BASE}/llama-{_LLAMA_TAG}-bin-ubuntu-vulkan-x64.tar.gz"


def _has_any_llama_binary() -> bool:
    """Return True if any llama-mtmd-cli binary is already available."""
    for backend in Backend:
        if find_binary(backend):
            return True
    return False


async def _download_llama_binary(
    on_progress=None,
) -> None:
    """Download and extract llama.cpp pre-built binaries from GitHub.

    Parameters
    ----------
    on_progress : callable, optional
        Called with (downloaded_bytes, total_bytes) for progress reporting.
    """
    import httpx

    bin_dir = llama_bin_cache()
    bin_dir.mkdir(parents=True, exist_ok=True)

    url = _llama_asset_url()
    is_zip = url.endswith(".zip")
    archive_path = bin_dir / ("llama.zip" if is_zip else "llama.tar.gz")

    log.info("Downloading llama.cpp binaries from %s", url)

    async with httpx.AsyncClient(follow_redirects=True, timeout=None) as client:
        async with client.stream("GET", url) as resp:
            resp.raise_for_status()
            total = int(resp.headers.get("content-length", 0))
            downloaded = 0
            with open(archive_path, "wb") as f:
                async for chunk in resp.aiter_bytes(chunk_size=256 * 1024):
                    f.write(chunk)
                    downloaded += len(chunk)
                    if on_progress:
                        on_progress(downloaded, total)

    log.info("Downloaded %d MB, extracting...", downloaded // (1024 * 1024))

    # Extract
    if is_zip:
        import zipfile
        with zipfile.ZipFile(archive_path) as zf:
            zf.extractall(bin_dir)
    else:
        import tarfile
        with tarfile.open(archive_path) as tf:
            tf.extractall(bin_dir)

    archive_path.unlink(missing_ok=True)

    # Make binaries executable on Unix
    if platform.system() != "Windows":
        for f in bin_dir.rglob("llama-*"):
            if f.is_file():
                f.chmod(f.stat().st_mode | 0o755)

    # Verify
    binary_name = "llama-mtmd-cli.exe" if platform.system() == "Windows" else "llama-mtmd-cli"
    found = list(bin_dir.rglob(binary_name))
    if found:
        log.info("llama-mtmd-cli ready: %s", found[0])
    else:
        log.warning("llama-mtmd-cli not found after extraction")


# -------------------------------------------------------------------
# Globals (set during lifespan)
# -------------------------------------------------------------------
_vision: Optional[LlamaMtmdCli] = None
_text: Optional[LlamaCppServer] = None
_detection = None
# GPU acceleration mode: 'auto' | 'gpu' | 'cpu'
# 'auto' = platform default (CPU on Windows, GPU elsewhere)
# 'gpu'  = force all layers on GPU (-ngl -1)
# 'cpu'  = force CPU-only (-ngl 0)
_gpu_mode: str = "auto"
# Serialize vision OCR: only one llama-mtmd-cli process at a time.
# Running multiple simultaneously exhausts GPU VRAM and crashes on
# Windows/iGPUs (STATUS_STACK_BUFFER_OVERRUN 0xC0000409).
_vision_sem = asyncio.Semaphore(1)

# Text server context. This must match the model actually in use: the Flutter
# registry advertises 8192 for Gemma 4 E2B, and the enrichment batch size is
# derived from it rather than hardcoded. It was previously 1024 here, which
# contradicted both the class default and config/settings.yaml and silently
# capped enrichment at two words per call.
DEFAULT_CONTEXT_SIZE = 8192


def _ngl_for_mode() -> Optional[int]:
    """Return n_gpu_layers based on the current _gpu_mode."""
    if _gpu_mode == "gpu":
        return -1   # all layers on GPU
    if _gpu_mode == "cpu":
        return 0    # CPU-only
    # 'auto' — let the backend constructor decide (CPU on Windows)
    return None


def _init_vision() -> Optional[LlamaMtmdCli]:
    """Try to initialise the vision backend; return None on failure."""
    try:
        ngl = _ngl_for_mode()
        # In 'auto' mode the constructor defaults to 0 on Windows;
        # in 'gpu' mode we force -1 to override that safety net.
        force_gpu = _gpu_mode == "gpu"
        cli = LlamaMtmdCli(
            n_gpu_layers=ngl,
            mmproj_offload=True if force_gpu else None,
        )
        log.info(
            "Vision backend ready: %s (%s) ngl=%s mode=%s",
            cli.detection.recommended_backend.value if cli.detection else "manual",
            cli.binary_path,
            cli.n_gpu_layers,
            _gpu_mode,
        )
        return cli
    except Exception as e:
        log.warning("Vision backend unavailable: %s", e)
        return None


def _init_text() -> Optional[LlamaCppServer]:
    """Try to initialise the text backend; return None on failure."""
    try:
        ngl = _ngl_for_mode()
        server = LlamaCppServer(context_size=DEFAULT_CONTEXT_SIZE, n_gpu_layers=ngl)
        log.info("Text backend configured: %s (ngl=%s mode=%s)",
                 server.model_path.name, ngl, _gpu_mode)
        return server
    except Exception as e:
        log.warning("Text backend unavailable: %s", e)
        return None


async def _reinit_backends() -> dict:
    """Stop text server, re-detect hardware, and re-init vision + text backends.

    Returns a dict with status keys for use by endpoints.
    """
    global _vision, _text, _detection

    if _text is not None:
        _text.stop()
    _detection = detect()
    _vision = _init_vision()
    _text = _init_text()

    text_ok = _text is not None
    vision_ok = _vision is not None
    return {
        "status": "ok" if (text_ok or vision_ok) else "degraded",
        "vision_available": vision_ok,
        "text_available": text_ok,
    }


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup / shutdown hooks."""
    global _vision, _text, _detection

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(name)s: %(message)s")
    log.info("Starting OCR-to-Anki API...")

    await _reinit_backends()

    yield

    # Shutdown
    if _text is not None:
        _text.stop()
    log.info("API shut down.")


# -------------------------------------------------------------------
# App
# -------------------------------------------------------------------

app = FastAPI(
    title="OCR-to-Anki API",
    description=(
        "Fully offline OCR and vocabulary enrichment powered by "
        "Gemma 3 4B via llama.cpp.  Vision tasks use llama-mtmd-cli "
        "(GPU-accelerated), text tasks use llama-server."
    ),
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# -------------------------------------------------------------------
# Health & info
# -------------------------------------------------------------------

@app.get("/health", response_model=HealthResponse, tags=["system"])
async def health():
    """Health check with backend status."""
    text_ok = _text is not None
    vision_ok = _vision is not None

    mdir = _models_dir()
    model_exists = (mdir / _MODEL_FILE).exists()
    mmproj_exists = (mdir / _MMPROJ_FILE).exists()
    all_present = model_exists and mmproj_exists

    return HealthResponse(
        status="ok" if (text_ok or vision_ok) else "degraded",
        vision_available=vision_ok,
        text_available=text_ok,
        backend=_detection.recommended_backend.value if _detection else "unknown",
        model=_text.model_path.name if _text else "-",
        devices=[
            {"name": d.name, "backend": d.backend.value}
            for d in (_detection.devices if _detection else [])
        ],
        models_downloaded=all_present,
        models_dir=str(mdir),
        llama_binary_available=_has_any_llama_binary(),
        gpu_mode=_gpu_mode,
    )


# -------------------------------------------------------------------
# Model management
# -------------------------------------------------------------------

@app.get("/models/status", response_model=ModelStatusResponse, tags=["models"])
async def models_status():
    """Check which model files are present on disk."""
    mdir = _models_dir()
    files = []
    for name, url in [(_MODEL_FILE, _MODEL_URL), (_MMPROJ_FILE, _MMPROJ_URL)]:
        p = mdir / name
        files.append(ModelFileInfo(
            name=name,
            size_bytes=p.stat().st_size if p.exists() else 0,
            exists=p.exists(),
            url=url,
        ))
    return ModelStatusResponse(
        all_present=all(f.exists for f in files),
        models_dir=str(mdir),
        files=files,
    )


@app.post("/models/download", tags=["models"])
async def models_download():
    """Download missing model files.  Returns an SSE stream with progress.

    Each SSE event is a JSON object:
      {"file": "...", "downloaded": <bytes>, "total": <bytes>, "done": false}
    The final event has ``done: true``.
    """
    import httpx

    mdir = _models_dir()
    mdir.mkdir(parents=True, exist_ok=True)

    to_download: list[tuple[str, str, Path]] = []
    for name, url in [(_MODEL_FILE, _MODEL_URL), (_MMPROJ_FILE, _MMPROJ_URL)]:
        dest = mdir / name
        if not dest.exists():
            to_download.append((name, url, dest))

    if not to_download:
        import json as _json

        async def _already_done():
            yield f"data: {_json.dumps({'file': '', 'downloaded': 0, 'total': 0, 'done': True, 'error': None})}\n\n"

        return StreamingResponse(_already_done(), media_type="text/event-stream")

    async def _stream():
        import json as _json

        for name, url, dest in to_download:
            tmp = dest.with_suffix(".part")
            downloaded = 0
            total = 0
            try:
                async with httpx.AsyncClient(follow_redirects=True, timeout=None) as client:
                    async with client.stream("GET", url) as resp:
                        resp.raise_for_status()
                        total = int(resp.headers.get("content-length", 0))
                        with open(tmp, "wb") as f:
                            async for chunk in resp.aiter_bytes(chunk_size=1024 * 256):
                                f.write(chunk)
                                downloaded += len(chunk)
                                yield f"data: {_json.dumps({'file': name, 'downloaded': downloaded, 'total': total, 'done': False, 'error': None})}\n\n"
                tmp.rename(dest)
                log.info("Downloaded model: %s (%d bytes)", name, downloaded)
            except Exception as exc:
                tmp.unlink(missing_ok=True)
                yield f"data: {_json.dumps({'file': name, 'downloaded': downloaded, 'total': total, 'done': True, 'error': str(exc)})}\n\n"
                return

        yield f"data: {_json.dumps({'file': '', 'downloaded': 0, 'total': 0, 'done': True, 'error': None})}\n\n"

    return StreamingResponse(_stream(), media_type="text/event-stream")


@app.post("/models/reinit", tags=["models"])
async def models_reinit():
    """Re-initialise vision and text backends after model download."""
    return await _reinit_backends()


@app.post("/config/gpu", tags=["system"])
async def config_gpu(body: dict):
    """Set the GPU acceleration mode and reinitialise backends.

    Body: ``{"mode": "auto" | "gpu" | "cpu"}``

    * **auto** — platform default (CPU on Windows, GPU on Linux/macOS).
    * **gpu**  — force all layers on GPU (overrides Windows safety net).
    * **cpu**  — force CPU-only inference on all platforms.
    """
    global _gpu_mode, _vision, _text, _detection

    mode = body.get("mode", "auto")
    if mode not in ("auto", "gpu", "cpu"):
        raise HTTPException(400, f"Invalid GPU mode: {mode!r}")

    old = _gpu_mode
    _gpu_mode = mode
    log.info("GPU mode changed: %s -> %s", old, mode)

    # Reinitialise backends with new GPU settings.
    result = await _reinit_backends()
    result["gpu_mode"] = _gpu_mode
    return result


# -------------------------------------------------------------------
# llama.cpp binary management
# -------------------------------------------------------------------

@app.get("/llama/status", tags=["llama"])
async def llama_status():
    """Check if llama.cpp binaries are available."""
    available = _has_any_llama_binary()
    bin_dir = str(llama_bin_cache())
    return {"available": available, "bin_dir": bin_dir}


@app.post("/llama/download", tags=["llama"])
async def llama_download():
    """Download llama.cpp binaries.  Returns an SSE stream with progress.

    Each SSE event is a JSON object:
      {"file": "llama.cpp", "downloaded": <bytes>, "total": <bytes>,
       "done": false, "error": null}
    The final event has ``done: true``.
    """
    if _has_any_llama_binary():
        import json as _json

        async def _already():
            yield f"data: {_json.dumps({'file': '', 'downloaded': 0, 'total': 0, 'done': True, 'error': None})}\n\n"

        return StreamingResponse(_already(), media_type="text/event-stream")

    async def _download_llama_binary_queue(progress_queue: asyncio.Queue) -> None:
        """Wrap _download_llama_binary and feed progress into an asyncio.Queue."""

        def _on_progress(d: int, t: int):
            progress_queue.put_nowait((d, t))

        try:
            await _download_llama_binary(on_progress=_on_progress)
            progress_queue.put_nowait(None)
        except Exception:
            progress_queue.put_nowait(None)
            raise

    async def _stream():
        import json as _json

        queue: asyncio.Queue = asyncio.Queue()
        task = asyncio.create_task(_download_llama_binary_queue(queue))

        try:
            while True:
                item = await queue.get()
                if item is None:
                    break
                downloaded_bytes, total_bytes = item
                yield f"data: {_json.dumps({'file': 'llama.cpp', 'downloaded': downloaded_bytes, 'total': total_bytes, 'done': False, 'error': None})}\n\n"

            await task
            yield f"data: {_json.dumps({'file': 'llama.cpp', 'downloaded': 0, 'total': 0, 'done': True, 'error': None})}\n\n"
        except Exception as exc:
            yield f"data: {_json.dumps({'file': 'llama.cpp', 'downloaded': 0, 'total': 0, 'done': True, 'error': str(exc)})}\n\n"

    return StreamingResponse(_stream(), media_type="text/event-stream")


@app.get("/backends", response_model=BackendInfoResponse, tags=["system"])
async def backends():
    """Show detected hardware and available backends."""
    det = _detection or detect()
    return BackendInfoResponse(
        os_name=det.os_name,
        arch=det.arch,
        recommended_backend=det.recommended_backend.value,
        binary_path=det.binary_path,
        devices=[
            {"name": d.name, "backend": d.backend.value, "vram_mb": d.vram_mb}
            for d in det.devices
        ],
    )


# -------------------------------------------------------------------
# Update checking
# -------------------------------------------------------------------
_GITHUB_REPO = os.getenv("GITHUB_REPO", "stradichenko/ocr-to-anki")


def _parse_version(v: str) -> tuple[int, ...]:
    """Parse a version string like '0.1.0' into a numeric tuple."""
    parts = []
    for p in v.strip().lstrip("v").split("."):
        # Take only leading digits from each part (e.g. "0+1" -> 0)
        digits = ""
        for ch in p:
            if ch.isdigit():
                digits += ch
            else:
                break
        parts.append(int(digits) if digits else 0)
    return tuple(parts)


def _update_asset_url(version: str) -> str:
    """Build the download URL for the current platform's release archive."""
    system = platform.system()
    machine = platform.machine().lower()
    base = f"https://github.com/{_GITHUB_REPO}/releases/download/v{version}"
    if system == "Windows":
        return f"{base}/ocr-to-anki-v{version}-windows-x86_64.zip"
    elif system == "Darwin":
        arch = "arm64" if machine in ("arm64", "aarch64") else "x64"
        return f"{base}/ocr-to-anki-v{version}-macos-{arch}.zip"
    else:
        return f"{base}/ocr-to-anki-v{version}-linux-x86_64.tar.gz"


@app.get("/update/check", response_model=UpdateCheckResponse, tags=["system"])
async def update_check(current_version: str = Query(..., description="Current app version, e.g. 0.1.0")):
    """Check whether a newer release is available on GitHub."""
    import httpx

    api_url = f"https://api.github.com/repos/{_GITHUB_REPO}/releases/latest"
    try:
        async with httpx.AsyncClient(timeout=30) as client:
            resp = await client.get(api_url)
            resp.raise_for_status()
            data = resp.json()
    except Exception as e:
        log.warning("Update check failed: %s", e)
        return UpdateCheckResponse(
            has_update=False,
            current_version=current_version,
            latest_version="unknown",
        )

    tag = data.get("tag_name", "")
    latest = tag.lstrip("v")
    has_update = _parse_version(latest) > _parse_version(current_version)

    download_url = ""
    if has_update:
        download_url = _update_asset_url(latest)

    return UpdateCheckResponse(
        has_update=has_update,
        current_version=current_version,
        latest_version=latest,
        download_url=download_url,
        release_notes=data.get("body", ""),
        published_at=data.get("published_at", ""),
    )


# -------------------------------------------------------------------
# GPU coordination helpers
# -------------------------------------------------------------------

def _pause_text_server():
    """Stop the text server to free iGPU memory for vision OCR.

    On devices with a single GPU (e.g. Intel iGPU) the text server
    (llama-server) holds VRAM that prevents llama-mtmd-cli from running
    the vision encoder.  We stop it here and let _ensure_text_server()
    lazily restart it when text generation is needed again.
    """
    if _text is not None and _text.is_running():
        log.info("Pausing text server to free GPU for vision OCR...")
        _text.stop()


def _downscale_image(raw: bytes, max_dim: int = 768) -> bytes:
    """Downscale an image so its longest side is at most *max_dim* pixels.

    Gemma 3's SigLIP vision encoder tiles images into 896x896 chunks.
    A 1133x1348 image creates multiple tiles, multiplying the number of
    vision tokens that must go through prompt eval (the slowest phase on
    Intel iGPU without flash attention).  Fitting the image into a
    single tile can cut prompt eval time by 50-75%.

    Returns the original bytes unchanged if already within bounds.
    """
    arr = np.frombuffer(raw, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        return raw  # not a decodable image -- pass through

    h, w = img.shape[:2]
    if max(h, w) <= max_dim:
        return raw  # already small enough

    scale = max_dim / max(h, w)
    new_w, new_h = int(w * scale), int(h * scale)
    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".jpg", resized, [cv2.IMWRITE_JPEG_QUALITY, 90])
    if not ok:
        return raw

    log.info(
        "Downscaled image %dx%d -> %dx%d (%.0f KB -> %.0f KB)",
        w, h, new_w, new_h, len(raw) / 1024, len(buf) / 1024,
    )
    return bytes(buf)


# -------------------------------------------------------------------
# Vision OCR
# -------------------------------------------------------------------

async def _run_vision_ocr(
    raw_image_bytes: bytes,
    prompt: str,
    timeout: int,
    suffix: str = ".jpg",
    max_image_dim: int = 0,
) -> VisionOCRResponse:
    """Shared helper: downscale, hash, serialize, pause text server, run vision OCR.

    Raises HTTPException on invalid input, timeout, or backend failure.
    """
    if _vision is None:
        raise HTTPException(503, "Vision backend not available")

    img_hash = hashlib.md5(raw_image_bytes).hexdigest()[:12]

    # Downscale large images to reduce vision tokens (fewer SigLIP tiles).
    if max_image_dim > 0:
        raw_image_bytes = await asyncio.to_thread(
            _downscale_image, raw_image_bytes, max_image_dim
        )

    # Serialize: only one vision subprocess at a time (GPU VRAM).
    async with _vision_sem:
        # Free the GPU -- stop text server if it's occupying the iGPU.
        await asyncio.to_thread(_pause_text_server)

        # delete=False + manual cleanup: on Windows the subprocess cannot
        # open a NamedTemporaryFile that is still held open by Python.
        tmp = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)
        tmp_path = tmp.name
        try:
            tmp.write(raw_image_bytes)
            tmp.flush()
            tmp.close()

            try:
                result = await asyncio.to_thread(
                    _vision.run_vision,
                    image_path=tmp_path,
                    prompt=prompt,
                    timeout=timeout,
                )
            except TimeoutError:
                raise HTTPException(504, f"Vision OCR timed out after {timeout}s")
            except Exception as e:
                log.exception("Vision OCR failed")
                raise HTTPException(500, str(e))
        finally:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass

    return VisionOCRResponse(
        text=result["text"],
        elapsed_s=result["elapsed_s"],
        backend=result["backend"],
        image_hash=img_hash,
    )


@app.post("/ocr/vision", response_model=VisionOCRResponse, tags=["ocr"])
async def ocr_vision(req: VisionOCRRequest):
    """
    Run vision OCR on a base64-encoded image.

    The image is decoded, written to a temp file, and passed to
    llama-mtmd-cli with the chosen GPU backend.
    """
    try:
        raw = base64.b64decode(req.image_base64)
    except Exception:
        raise HTTPException(400, "Invalid base64 image data")

    return await _run_vision_ocr(
        raw_image_bytes=raw,
        prompt=req.prompt,
        timeout=req.timeout,
        suffix=".jpg",
        max_image_dim=req.max_image_dim,
    )


@app.post("/ocr/cancel", tags=["ocr"])
async def ocr_cancel():
    """Cancel any running vision OCR subprocess."""
    if _vision is not None:
        await asyncio.to_thread(_vision.cancel)
        return {"status": "cancelled"}
    return {"status": "no_vision_backend"}


@app.post("/ocr/vision/upload", response_model=VisionOCRResponse, tags=["ocr"])
async def ocr_vision_upload(
    file: UploadFile = File(...),
    prompt: str = Form(
        default="Extract all visible text from this image. List each word or phrase you can read."
    ),
    timeout: int = Form(default=2700),
    max_image_dim: int = Form(default=768),
):
    """
    Run vision OCR on an uploaded image file.
    """
    raw = await file.read()
    if not raw:
        raise HTTPException(400, "Empty file")

    suffix = Path(file.filename).suffix if file.filename else ".jpg"

    return await _run_vision_ocr(
        raw_image_bytes=raw,
        prompt=prompt,
        timeout=timeout,
        suffix=suffix,
        max_image_dim=max_image_dim,
    )


# -------------------------------------------------------------------
# Text generation & enrichment
# -------------------------------------------------------------------

def _ensure_text_server():
    """Lazily start the text server on first use."""
    if _text is None:
        raise HTTPException(503, "Text backend not available")
    if not _text.is_running():
        log.info("Starting llama-server on first text request...")
        _text.start(wait_ready=True, timeout=90)


@app.post("/generate", response_model=GenerateResponse, tags=["text"])
async def generate(req: GenerateRequest):
    """Raw text generation with Gemma 3 4B."""
    _ensure_text_server()

    try:
        result = await asyncio.to_thread(
            _text.generate,
            prompt=req.prompt,
            system=req.system,
            max_tokens=req.max_tokens,
            temperature=req.temperature,
            top_k=req.top_k,
            top_p=req.top_p,
        )
    except TimeoutError:
        raise HTTPException(504, "Generation timed out")
    except Exception as e:
        log.exception("Generation failed")
        raise HTTPException(500, str(e))

    return GenerateResponse(
        content=result["content"],
        tokens_evaluated=result.get("tokens_evaluated", 0),
        tokens_predicted=result.get("tokens_predicted", 0),
    )


def _strip_md(text: str) -> str:
    """Strip stray markdown asterisks the model emits despite instructions."""
    return text.replace("*", "")


async def _enrich_words(
    words: list[str],
    *,
    definition_language: str,
    examples_language: str,
    term_language: str,
    temperature: float,
    context_size: int = DEFAULT_CONTEXT_SIZE,
) -> tuple[list[dict], dict]:
    """Enrich words in batches, then retry whatever came back unusable.

    Shared by /enrich and the image pipeline so both get identical batching,
    the same retry pass, and the same timing. It also means a failure reports
    the same `not_found` warning in both places: previously the pipeline filled
    failures with no warning at all, which the client cache then treated as a
    good result and never retried.

    The batch size comes from the context rather than a fixed number, so the
    two cannot drift apart the way they did when the context was pinned to 1024
    while the batch stayed at two words.

    Returns the per-word dicts in input order, plus a summary of the stages.
    """
    empty = lambda w: {  # noqa: E731 - a one-line literal, not logic
        "word": w, "definition": "", "examples": "",
        "warning": "not_found", "corrected_word": "",
    }

    if not words or _text is None:
        return [empty(w) for w in words], {}

    batch_size = compute_batch_size(context_size)
    batches = [words[i:i + batch_size] for i in range(0, len(words), batch_size)]
    system = build_enrich_system(definition_language, examples_language, term_language)
    stages = {
        "batches": 0,
        "retry_batches": 0,
        "prompt_ms": 0,
        "predicted_ms": 0,
        "tokens_evaluated": 0,
        "tokens_predicted": 0,
    }

    log.info(
        "Enrichment: %d word(s) in %d batch(es) of up to %d (ctx=%d)",
        len(words), len(batches), batch_size, context_size,
    )

    async def _run_batch(chunk: list[str], label: str):
        """One generate call, with a single restart-and-retry on failure."""
        prompt = build_enrich_prompt(
            chunk, definition_language, examples_language, term_language,
        )
        timeout = compute_batch_timeout(len(chunk))
        started = time.monotonic()
        for attempt in range(2):
            try:
                result = await asyncio.to_thread(
                    _text.generate,
                    prompt=prompt,
                    system=system,
                    max_tokens=DEFAULT_ENRICH_TOKENS_PER_WORD * len(chunk),
                    temperature=temperature,
                    timeout=timeout,
                )
                timings = result.get("timings") or {}
                stages["prompt_ms"] += int(timings.get("prompt_ms") or 0)
                stages["predicted_ms"] += int(timings.get("predicted_ms") or 0)
                stages["tokens_evaluated"] += int(result.get("tokens_evaluated") or 0)
                stages["tokens_predicted"] += int(result.get("tokens_predicted") or 0)
                log.info(
                    "%s: %d word(s) wall=%.1fs prompt=%dms decode=%dms "
                    "prompt_tokens=%d gen_tokens=%d",
                    label, len(chunk), time.monotonic() - started,
                    int(timings.get("prompt_ms") or 0),
                    int(timings.get("predicted_ms") or 0),
                    int(result.get("tokens_evaluated") or 0),
                    int(result.get("tokens_predicted") or 0),
                )
                return parse_enrich_response(result["content"], chunk)
            except Exception as exc:
                if attempt == 0:
                    log.warning("%s failed (%s); restarting text server", label, exc)
                    try:
                        _text.stop()
                    except Exception:
                        pass
                    try:
                        _ensure_text_server()
                    except Exception as restart_err:
                        log.error("Server restart failed: %s", restart_err)
                else:
                    log.error("%s failed after retry: %s", label, exc)
        return None

    all_parsed: list[dict] = []
    for idx, batch in enumerate(batches, 1):
        stages["batches"] += 1
        parsed = await _run_batch(batch, f"enrich batch {idx}/{len(batches)}")
        all_parsed.extend(parsed if parsed else [empty(w) for w in batch])

    # Retry anything missing or cut off. Truncated definitions are retried too:
    # a definition cut mid-sentence is not usable on a flashcard, and it is
    # otherwise cached as a valid result and never revisited.
    retryable = [p["word"] for p in all_parsed if p["warning"] in ("not_found", "truncated")]
    if retryable:
        log.info("Retrying %d word(s) after the first pass", len(retryable))
        recovered_map: dict[str, dict] = {}
        for idx in range(0, len(retryable), batch_size):
            chunk = retryable[idx:idx + batch_size]
            stages["retry_batches"] += 1
            parsed = await _run_batch(chunk, f"enrich retry {idx // batch_size + 1}")
            if not parsed:
                continue
            for entry in parsed:
                if entry["definition"] and not entry["warning"]:
                    recovered_map[entry["word"].lower()] = entry
        recovered = 0
        for i, entry in enumerate(all_parsed):
            if entry["warning"] in ("not_found", "truncated"):
                replacement = recovered_map.get(entry["word"].lower())
                if replacement:
                    all_parsed[i] = replacement
                    recovered += 1
        if recovered:
            log.info("Retry recovered %d of %d word(s)", recovered, len(retryable))

    return all_parsed, stages


@app.post("/enrich", response_model=EnrichResponse, tags=["vocabulary"])
async def enrich(req: EnrichRequest):
    """Enrich a list of vocabulary words with definitions and example sentences.

    Batching is derived from the configured context (see ``compute_batch_size``)
    and words that come back unusable are retried once.
    """
    _ensure_text_server()

    # Sanitize words — remove OCR/LLM artefacts before sending to the model
    sanitized = sanitize_words(req.words)
    if not sanitized:
        return EnrichResponse(results=[], elapsed_s=0.0)
    log.info("Enrichment: %d raw words → %d after cleanup", len(req.words), len(sanitized))

    t0 = time.monotonic()
    parsed, stages = await _enrich_words(
        sanitized,
        definition_language=req.definition_language,
        examples_language=req.examples_language,
        term_language=req.term_language,
        temperature=req.temperature,
    )
    elapsed = time.monotonic() - t0

    results = [
        EnrichWordResult(
            word=p["word"],
            definition=_strip_md(p["definition"]),
            examples=_strip_md(p["examples"]),
            warning=p.get("warning", ""),
            corrected_word=p.get("corrected_word", ""),
        )
        for p in parsed
    ]

    log.info(
        "Enrichment complete: %d word(s) in %d batch(es) (+%d retry), %.1fs total",
        len(sanitized), stages.get("batches", 0), stages.get("retry_batches", 0), elapsed,
    )

    return EnrichResponse(results=results, elapsed_s=round(elapsed, 2))


# -------------------------------------------------------------------
# Pipeline: image → OCR → enrich → Anki-ready cards
# -------------------------------------------------------------------

@app.post("/pipeline/image-to-cards", response_model=PipelineImageToCardsResponse, tags=["pipeline"])
async def pipeline_image_to_cards(
    file: UploadFile = File(...),
    definition_language: str = Form(default="english"),
    examples_language: str = Form(default="english"),
    term_language: str = Form(default="auto"),
    ocr_prompt: str = Form(
        default="Extract all visible text from this image. List each word or phrase, one per line."
    ),
):
    """
    Full pipeline: Upload an image → extract text via vision OCR →
    enrich each word with definitions and examples → return Anki-ready cards.
    """
    t0 = time.monotonic()

    # Step 1: Vision OCR
    raw = await file.read()
    if not raw:
        raise HTTPException(400, "Empty file")

    suffix = Path(file.filename).suffix if file.filename else ".jpg"

    ocr_resp = await _run_vision_ocr(
        raw_image_bytes=raw,
        prompt=ocr_prompt,
        timeout=2700,
        suffix=suffix,
        max_image_dim=768,
    )

    ocr_text = ocr_resp.text

    # Step 2: Parse words from OCR output
    words = [w.strip().strip("-•·") for w in ocr_text.splitlines() if w.strip()]
    words = [w for w in words if len(w) > 1]  # filter noise

    if not words:
        return PipelineImageToCardsResponse(
            ocr_text=ocr_text,
            ocr_backend=ocr_resp.backend,
            ocr_elapsed_s=ocr_resp.elapsed_s,
            cards=[],
            total_elapsed_s=round(time.monotonic() - t0, 2),
            message="No words extracted from image",
        )

    # Step 3: Enrich (skip if text server unavailable)
    capped = words[:20]
    if _text is not None and capped:
        _ensure_text_server()
        cards, stages = await _enrich_words(
            capped,
            definition_language=definition_language,
            examples_language=examples_language,
            term_language=term_language,
            temperature=0.1,
        )
        log.info(
            "Pipeline enrichment: %d batch(es) (+%d retry)",
            stages.get("batches", 0), stages.get("retry_batches", 0),
        )
    else:
        cards = [
            {"word": w, "definition": "", "examples": "",
             "warning": "not_found", "corrected_word": ""}
            for w in words
        ]

    elapsed = round(time.monotonic() - t0, 2)

    results = [
        EnrichWordResult(
            word=c["word"],
            definition=c.get("definition", ""),
            examples=c.get("examples", ""),
            warning=c.get("warning", ""),
            corrected_word=c.get("corrected_word", ""),
        )
        for c in cards
    ]

    return PipelineImageToCardsResponse(
        ocr_text=ocr_text,
        ocr_backend=ocr_resp.backend,
        ocr_elapsed_s=ocr_resp.elapsed_s,
        cards=results,
        total_elapsed_s=elapsed,
    )
