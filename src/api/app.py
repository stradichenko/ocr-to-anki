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
    VisionOCRRequest,
    VisionOCRResponse,
)
from backends.auto_detect import detect, Backend, find_binary, llama_bin_cache
from backends.mtmd_cli import LlamaMtmdCli
from backends.llama_cpp_server import LlamaCppServer

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(name)s: %(message)s")

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
        server = LlamaCppServer(context_size=1024, n_gpu_layers=ngl)
        log.info("Text backend configured: %s (ngl=%s mode=%s)",
                 server.model_path.name, ngl, _gpu_mode)
        return server
    except Exception as e:
        log.warning("Text backend unavailable: %s", e)
        return None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup / shutdown hooks."""
    global _vision, _text, _detection

    log.info("Starting OCR-to-Anki API...")

    _detection = detect()
    _vision = _init_vision()
    _text = _init_text()

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
    if _text is not None:
        _text.stop()
    _detection = detect()
    _vision = _init_vision()
    _text = _init_text()

    text_ok = _text is not None
    vision_ok = _vision is not None
    return {
        "gpu_mode": _gpu_mode,
        "status": "ok" if (text_ok or vision_ok) else "degraded",
        "vision_available": vision_ok,
        "text_available": text_ok,
    }


# -------------------------------------------------------------------
# llama.cpp binary management
# -------------------------------------------------------------------

@app.get("/llama/status", tags=["llama"])
async def llama_status():
    """Check if llama.cpp binaries are available."""
    available = _has_any_llama_binary()
    bin_dir = str(_llama_bin_cache())
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

    async def _stream():
        import json as _json
        downloaded_bytes = 0
        total_bytes = 0

        def _on_progress(d: int, t: int):
            nonlocal downloaded_bytes, total_bytes
            downloaded_bytes = d
            total_bytes = t

        try:
            # Run download in background; emit progress at intervals
            import asyncio as _aio
            task = _aio.create_task(_download_llama_binary(on_progress=_on_progress))

            while not task.done():
                yield f"data: {_json.dumps({'file': 'llama.cpp', 'downloaded': downloaded_bytes, 'total': total_bytes, 'done': False, 'error': None})}\n\n"
                await _aio.sleep(0.3)

            # Ensure task errors propagate
            await task

            yield f"data: {_json.dumps({'file': 'llama.cpp', 'downloaded': downloaded_bytes, 'total': total_bytes, 'done': True, 'error': None})}\n\n"
        except Exception as exc:
            yield f"data: {_json.dumps({'file': 'llama.cpp', 'downloaded': downloaded_bytes, 'total': total_bytes, 'done': True, 'error': str(exc)})}\n\n"

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

@app.post("/ocr/vision", response_model=VisionOCRResponse, tags=["ocr"])
async def ocr_vision(req: VisionOCRRequest):
    """
    Run vision OCR on a base64-encoded image.

    The image is decoded, written to a temp file, and passed to
    llama-mtmd-cli with the chosen GPU backend.
    """
    if _vision is None:
        raise HTTPException(503, "Vision backend not available")

    # Decode image to temp file
    try:
        raw = base64.b64decode(req.image_base64)
    except Exception:
        raise HTTPException(400, "Invalid base64 image data")

    img_hash = hashlib.md5(raw).hexdigest()[:12]

    # Downscale large images to reduce vision tokens (fewer SigLIP tiles).
    if req.max_image_dim > 0:
        raw = await asyncio.to_thread(_downscale_image, raw, req.max_image_dim)

    # Serialize: only one vision subprocess at a time (GPU VRAM).
    async with _vision_sem:
        # Free the GPU -- stop text server if it's occupying the iGPU.
        await asyncio.to_thread(_pause_text_server)

        # delete=False + manual cleanup: on Windows the subprocess cannot
        # open a NamedTemporaryFile that is still held open by Python.
        tmp = tempfile.NamedTemporaryFile(suffix=".jpg", delete=False)
        tmp_path = tmp.name
        try:
            tmp.write(raw)
            tmp.flush()
            tmp.close()

            try:
                result = await asyncio.to_thread(
                    _vision.run_vision,
                    image_path=tmp_path,
                    prompt=req.prompt,
                    timeout=req.timeout,
                )
            except TimeoutError:
                raise HTTPException(504, f"Vision OCR timed out after {req.timeout}s")
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
    if _vision is None:
        raise HTTPException(503, "Vision backend not available")

    raw = await file.read()
    if not raw:
        raise HTTPException(400, "Empty file")

    img_hash = hashlib.md5(raw).hexdigest()[:12]
    suffix = Path(file.filename).suffix if file.filename else ".jpg"

    # Downscale large images to reduce vision tokens.
    if max_image_dim > 0:
        raw = await asyncio.to_thread(_downscale_image, raw, max_image_dim)

    # Serialize: only one vision subprocess at a time (GPU VRAM).
    async with _vision_sem:
        # Free the GPU -- stop text server if it's occupying the iGPU.
        await asyncio.to_thread(_pause_text_server)

        tmp = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)
        tmp_path = tmp.name
        try:
            tmp.write(raw)
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


def _build_enrich_prompt(
    words: list[str],
    def_lang: str,
    ex_lang: str,
    term_lang: str = "auto",
) -> str:
    """Build a single batched prompt for all words."""
    word_list = "\n".join(f"- {w}" for w in words)
    lang_hint = (
        f"The word is in {term_lang}. "
        if term_lang and term_lang.lower() != "auto"
        else ""
    )
    return (
        f"{lang_hint}"
        f"For each word, give a definition in {def_lang} and 2 example sentences in {ex_lang}.\n"
        f"WORD: must contain the EXACT original word, unchanged. Do NOT translate, correct, or modify it.\n"
        f"No asterisks, no bold, no markdown. Plain text only.\n\n"
        f"If unrecognizable, write DEF: UNKNOWN and skip examples.\n\n"
        f"Format (labels must be in English):\n"
        f"WORD: <original word, unchanged>\n"
        f"DEF: <{def_lang} definition>\n"
        f"EX1: <{ex_lang} sentence>\n"
        f"EX2: <{ex_lang} sentence>\n\n"
        f"Example — word: 'amigo' (spanish), def_lang={def_lang}, ex_lang={ex_lang}:\n"
        f"WORD: amigo\n"
        f"DEF: A friend; a person with whom one has a bond of mutual affection.\n"
        f"EX1: I met a new friend at school today.\n"
        f"EX2: She is my best friend and we always help each other.\n\n"
        f"Words:\n{word_list}"
    )


def _build_enrich_system(
    def_lang: str,
    ex_lang: str,
    term_lang: str = "auto",
) -> str:
    """Build a system prompt for enrichment to reinforce language and formatting rules."""
    lang_hint = (
        f"The words are in {term_lang}. "
        if term_lang and term_lang.lower() != "auto"
        else ""
    )
    return (
        f"You are a dictionary. {lang_hint}"
        f"For each word, output its definition in "
        f"{def_lang} and two example sentences in {ex_lang}. "
        f"The WORD: line must contain the EXACT original word, unchanged. "
        f"Do NOT translate, correct, or replace the word itself. "
        f"No markdown, no asterisks. "
        f"Use labels: WORD:, DEF:, EX1:, EX2:."
    )


# Regex patterns for multilingual format markers the LLM might produce
# when asked for definitions in non-English languages.
_WORD_SPLIT_RE = re.compile(
    r"(?m)^(?:WORD|WORT|MOT|PALABRA|PALAVRA|PAROLA|СЛОВО|単語|단어):\s*",
    re.IGNORECASE,
)
_DEF_RE = re.compile(
    r"^(?:DEF|DEFINITION|BEDEUTUNG|DÉFINITION|DEFINICIÓN|DEFINIÇÃO|DEFINIZIONE):\s*",
    re.IGNORECASE,
)
_EX1_RE = re.compile(
    r"^(?:EX1|BEISPIEL\s*1|EXEMPLE\s*1|EJEMPLO\s*1|EXEMPLO\s*1|ESEMPIO\s*1):\s*",
    re.IGNORECASE,
)
_EX2_RE = re.compile(
    r"^(?:EX2|BEISPIEL\s*2|EXEMPLE\s*2|EJEMPLO\s*2|EXEMPLO\s*2|ESEMPIO\s*2):\s*",
    re.IGNORECASE,
)
_CORRECTED_RE = re.compile(
    r"^(?:CORRECTED|KORRIGIERT|CORRIGÉ|CORREGIDO|CORRIGIDO|CORRETTO):\s*",
    re.IGNORECASE,
)


def _parse_enrich_response(text: str, words: list[str]) -> list[dict]:
    """Parse batched enrichment response into per-word dicts."""
    log.debug("Raw LLM response (%d chars):\n%s", len(text), text[:800])
    results = []
    # Split by WORD: markers (supports translated labels)
    blocks = _WORD_SPLIT_RE.split(text)
    parsed: dict[str, dict] = {}
    parsed_ordered: list[dict] = []  # positional fallback
    for block in blocks:
        block = block.strip()
        if not block:
            continue
        lines = block.split("\n", 1)
        word_key = lines[0].strip()
        body = lines[1] if len(lines) > 1 else ""
        defn = ""
        examples = ""
        for line in body.split("\n"):
            line = line.strip()
            if _CORRECTED_RE.match(line):
                pass  # Ignore any CORRECTED: the model emits
            elif _DEF_RE.match(line):
                defn = _DEF_RE.sub("", line).strip()
            elif _EX1_RE.match(line):
                examples = _EX1_RE.sub("", line).strip()
            elif _EX2_RE.match(line):
                ex2 = _EX2_RE.sub("", line).strip()
                if ex2:
                    examples += "\n" + ex2
        data = {"definition": defn, "examples": examples}
        # Strip markdown-style asterisks the LLM may insert despite instructions.
        for key in ("definition", "examples"):
            data[key] = data[key].replace("*", "")
        parsed[word_key.lower()] = data
        parsed_ordered.append(data)

    # Match back to original word list.
    for idx, w in enumerate(words):
        # 1. Exact case-insensitive match.
        entry = parsed.get(w.lower())
        # 2. Positional fallback — if the model reformulated the word
        #    (e.g. singular vs plural, added accent), use the Nth block.
        if entry is None and idx < len(parsed_ordered):
            entry = parsed_ordered[idx]
            log.debug("Positional match for %r → block %d", w, idx)
        if entry is None:
            entry = {}
        defn = entry.get("definition", "")
        examples = entry.get("examples", "")

        # Detect UNKNOWN marker from LLM
        if (
            defn.strip().upper() == "UNKNOWN"
            or defn.strip().upper().startswith("UNKNOWN")
            or "UNKNOWN" in defn.upper()
        ):
            defn = ""
            examples = ""
            warning = "not_found"
        # Detect truncated definitions (no terminal punctuation)
        elif defn and not defn.rstrip().endswith(('.', '!', '?', '"', ')')):
            warning = "truncated"
        elif not defn:
            warning = "not_found"
        else:
            warning = ""

        # Detect untranslated: original word appears verbatim in examples.
        # Skip very short words (<=2 chars) to avoid false positives.
        if not warning and len(w) > 2 and examples:
            w_lower = w.lower()
            ex_lower = examples.lower()
            # Check if the original word (or a close variant) leaked into examples.
            if re.search(r'\b' + re.escape(w_lower) + r'\w{0,2}\b', ex_lower):
                warning = "untranslated"

        results.append({
            "word": w,
            "definition": defn,
            "examples": examples,
            "warning": warning,
            "corrected_word": "",
        })
    return results


def _sanitize_words(raw: list[str]) -> list[str]:
    """Clean OCR artefacts from word list before enrichment."""
    cleaned: list[str] = []
    for w in raw:
        # Strip markdown bullets, numbering, leading punctuation
        w = re.sub(r'^[\s*\-\u2022\u00b7#>]+', '', w)
        w = re.sub(r'^\d+\.\s*', '', w)
        w = w.strip()
        if len(w) < 2:
            continue
        # Skip LLM commentary: sentences with >4 words
        if len(w.split()) > 4:
            log.debug("Skipping non-word: %r", w)
            continue
        cleaned.append(w)
    return cleaned


@app.post("/enrich", response_model=EnrichResponse, tags=["vocabulary"])
async def enrich(req: EnrichRequest):
    """
    Enrich a list of vocabulary words with definitions and example sentences.

    Processes words in small batches (3 at a time) to keep each LLM call
    short enough for the iGPU.  Each batch generates ~180 tokens instead
    of ~1600 for 20 words, avoiding timeouts and keeping the model on-format.
    """
    _ensure_text_server()

    # Sanitize words — remove OCR/LLM artefacts before sending to the model
    sanitized = _sanitize_words(req.words)
    if not sanitized:
        return EnrichResponse(results=[], elapsed_s=0.0)
    log.info("Enrichment: %d raw words → %d after cleanup", len(req.words), len(sanitized))

    t0 = time.monotonic()
    all_parsed: list[dict] = []

    BATCH_SIZE = 2  # small batches — 2 words × 150 tok ≈ 300 output, fits 1024
    # Token budget per word: definitions (1-2 sentences) + 2 example sentences
    # typically need 90-120 tokens each.  The old budget of 60 caused
    # truncation and lost words when the response was cut mid-batch.
    TOKENS_PER_WORD = 150
    batches = [
        sanitized[i : i + BATCH_SIZE]
        for i in range(0, len(sanitized), BATCH_SIZE)
    ]

    for batch_idx, batch in enumerate(batches, 1):
        log.info(
            "Enrichment batch %d/%d  (%d words: %s)",
            batch_idx, len(batches), len(batch), ", ".join(batch),
        )
        prompt = _build_enrich_prompt(
            batch, req.definition_language, req.examples_language, req.term_language,
        )
        system = _build_enrich_system(
            req.definition_language, req.examples_language, req.term_language,
        )
        max_tok = TOKENS_PER_WORD * len(batch)
        # Timeout: iGPU needs ~200s prompt eval + ~200s/word generation.
        # First batch may also include server restart overhead (~90s).
        batch_timeout = 900

        batch_parsed = None
        for attempt in range(2):  # one retry on failure
            try:
                result = await asyncio.to_thread(
                    _text.generate,
                    prompt=prompt,
                    system=system,
                    max_tokens=max_tok,
                    temperature=req.temperature,
                    timeout=batch_timeout,
                )
                batch_parsed = _parse_enrich_response(result["content"], batch)
                break
            except Exception as e:
                if attempt == 0:
                    log.warning(
                        "Batch %d/%d failed (attempt 1: %s), restarting server...",
                        batch_idx, len(batches), e,
                    )
                    # Server may have crashed / timed out — restart it
                    try:
                        _text.stop()
                    except Exception:
                        pass
                    try:
                        _ensure_text_server()
                    except Exception as restart_err:
                        log.error("Server restart failed: %s", restart_err)
                else:
                    log.error(
                        "Batch %d/%d failed after retry: %s",
                        batch_idx, len(batches), e,
                    )

        if batch_parsed:
            all_parsed.extend(batch_parsed)
        else:
            # Fill with empty entries so we don't lose words
            for w in batch:
                all_parsed.append({"word": w, "definition": "", "examples": "", "warning": "not_found"})

        log.info(
            "Batch %d/%d done in %.1fs",
            batch_idx, len(batches), time.monotonic() - t0,
        )

    # ── Retry not_found words in small batches ─────────────────────
    not_found_words = [p["word"] for p in all_parsed if p["warning"] == "not_found"]
    if not_found_words:
        retry_list = not_found_words[:4]  # cap to 2 retry batches of 2
        log.info(
            "Retrying %d/%d not_found word(s): %s",
            len(retry_list), len(not_found_words), ", ".join(retry_list),
        )
        retry_batches = [
            retry_list[i : i + 2]
            for i in range(0, len(retry_list), 2)
        ]
        retry_lookup: dict[str, dict] = {}
        for rb_idx, rb in enumerate(retry_batches, 1):
            prompt = _build_enrich_prompt(
                rb, req.definition_language, req.examples_language, req.term_language,
            )
            try:
                result = await asyncio.to_thread(
                    _text.generate,
                    prompt=prompt,
                    system=system,
                    max_tokens=TOKENS_PER_WORD * len(rb),
                    temperature=req.temperature,
                    timeout=batch_timeout,
                )
                rp = _parse_enrich_response(result["content"], rb)
                for entry in rp:
                    if entry["definition"]:
                        retry_lookup[entry["word"].lower()] = entry
            except Exception as e:
                log.warning("Retry batch %d failed: %s", rb_idx, e)

        # Merge recovered words back.
        recovered = 0
        for i, p in enumerate(all_parsed):
            replacement = retry_lookup.get(p["word"].lower())
            if p["warning"] == "not_found" and replacement:
                all_parsed[i] = replacement
                recovered += 1
        if recovered:
            log.info("Retry recovered %d/%d word(s)", recovered, len(not_found_words))

    # Final sanitisation: strip any stray asterisks the LLM may have emitted.
    def _strip_md(s: str) -> str:
        return s.replace("*", "")

    results = [
        EnrichWordResult(
            word=p["word"],
            definition=_strip_md(p["definition"]),
            examples=_strip_md(p["examples"]),
            warning=p.get("warning", ""),
            corrected_word=p.get("corrected_word", ""),
        )
        for p in all_parsed
    ]

    elapsed = time.monotonic() - t0
    log.info(
        "Enrichment complete: %d words in %d batches, %.1fs total",
        len(sanitized), len(batches), elapsed,
    )

    return EnrichResponse(results=results, elapsed_s=round(elapsed, 2))


# -------------------------------------------------------------------
# Pipeline: image → OCR → enrich → Anki-ready cards
# -------------------------------------------------------------------

@app.post("/pipeline/image-to-cards", tags=["pipeline"])
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
    if _vision is None:
        raise HTTPException(503, "Vision backend not available")

    t0 = time.monotonic()

    # Step 1: Vision OCR
    raw = await file.read()
    if not raw:
        raise HTTPException(400, "Empty file")

    suffix = Path(file.filename).suffix if file.filename else ".jpg"

    # Downscale large images to reduce vision tokens.
    raw = await asyncio.to_thread(_downscale_image, raw, 768)

    # Serialize: only one vision subprocess at a time (GPU VRAM).
    async with _vision_sem:
        # Free the GPU -- stop text server if it's occupying the iGPU.
        await asyncio.to_thread(_pause_text_server)

        tmp = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)
        tmp_path = tmp.name
        try:
            tmp.write(raw)
            tmp.flush()
            tmp.close()
            ocr_result = await asyncio.to_thread(
                _vision.run_vision, tmp_path, prompt=ocr_prompt, timeout=2700,
            )
        finally:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass

    ocr_text = ocr_result["text"]

    # Step 2: Parse words from OCR output
    words = [w.strip().strip("-•·") for w in ocr_text.splitlines() if w.strip()]
    words = [w for w in words if len(w) > 1]  # filter noise

    if not words:
        return {
            "ocr_text": ocr_text,
            "cards": [],
            "elapsed_s": round(time.monotonic() - t0, 2),
            "message": "No words extracted from image",
        }

    # Step 3: Enrich (skip if text server unavailable)
    cards = []
    capped = words[:20]
    if _text is not None and capped:
        _ensure_text_server()
        BATCH_SIZE = 3  # small batches to fit in 1024 context window
        batches = [capped[i : i + BATCH_SIZE] for i in range(0, len(capped), BATCH_SIZE)]
        try:
            for batch in batches:
                prompt = _build_enrich_prompt(batch, definition_language, examples_language, term_language)
                sys_prompt = _build_enrich_system(definition_language, examples_language, term_language)
                max_tok = 60 * len(batch)
                result = await asyncio.to_thread(
                    _text.generate,
                    prompt=prompt,
                    system=sys_prompt,
                    max_tokens=max_tok,
                    temperature=0.1,
                    timeout=len(batch) * 60 + 60,
                )
                cards.extend(_parse_enrich_response(result["content"], batch))
        except Exception as e:
            log.warning("Batched enrichment failed: %s", e)
            # Fill remaining words with empty entries
            enriched = {c["word"].lower() for c in cards}
            for w in capped:
                if w.lower() not in enriched:
                    cards.append({"word": w, "definition": "", "examples": ""})
    else:
        cards = [{"word": w, "definition": "", "examples": ""} for w in words]

    elapsed = round(time.monotonic() - t0, 2)

    return {
        "ocr_text": ocr_text,
        "ocr_backend": ocr_result["backend"],
        "ocr_elapsed_s": ocr_result["elapsed_s"],
        "cards": cards,
        "total_elapsed_s": elapsed,
    }
