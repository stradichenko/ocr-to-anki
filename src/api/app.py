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
import tempfile
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException, UploadFile, File, Form, Query
from fastapi.middleware.cors import CORSMiddleware

from api.models import (
    BackendInfoResponse,
    BackendName,
    EnrichRequest,
    EnrichResponse,
    EnrichWordResult,
    GenerateRequest,
    GenerateResponse,
    HealthResponse,
    VisionOCRRequest,
    VisionOCRResponse,
)
from backends.auto_detect import detect, Backend
from backends.mtmd_cli import LlamaMtmdCli
from backends.llama_cpp_server import LlamaCppServer

log = logging.getLogger(__name__)

# -------------------------------------------------------------------
# Globals (set during lifespan)
# -------------------------------------------------------------------
_vision: Optional[LlamaMtmdCli] = None
_text: Optional[LlamaCppServer] = None
_detection = None


def _init_vision() -> Optional[LlamaMtmdCli]:
    """Try to initialise the vision backend; return None on failure."""
    try:
        cli = LlamaMtmdCli()
        log.info(
            "Vision backend ready: %s (%s)",
            cli.detection.recommended_backend.value if cli.detection else "manual",
            cli.binary_path,
        )
        return cli
    except Exception as e:
        log.warning("Vision backend unavailable: %s", e)
        return None


def _init_text() -> Optional[LlamaCppServer]:
    """Try to initialise the text backend; return None on failure."""
    try:
        server = LlamaCppServer()
        # Don't auto-start here; start lazily on first request
        log.info("Text backend configured: %s", server.model_path.name)
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

    return HealthResponse(
        status="ok" if (text_ok or vision_ok) else "degraded",
        vision_available=vision_ok,
        text_available=text_ok,
        backend=_detection.recommended_backend.value if _detection else "unknown",
        model=_text.model_path.name if _text else "—",
        devices=[
            {"name": d.name, "backend": d.backend.value}
            for d in (_detection.devices if _detection else [])
        ],
    )


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

    # Free the GPU — stop text server if it's occupying the iGPU.
    await asyncio.to_thread(_pause_text_server)

    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=True) as tmp:
        tmp.write(raw)
        tmp.flush()

        try:
            result = await asyncio.to_thread(
                _vision.run_vision,
                image_path=tmp.name,
                prompt=req.prompt,
                timeout=req.timeout,
            )
        except TimeoutError:
            raise HTTPException(504, f"Vision OCR timed out after {req.timeout}s")
        except Exception as e:
            log.exception("Vision OCR failed")
            raise HTTPException(500, str(e))

    return VisionOCRResponse(
        text=result["text"],
        elapsed_s=result["elapsed_s"],
        backend=result["backend"],
        image_hash=img_hash,
    )


@app.post("/ocr/vision/upload", response_model=VisionOCRResponse, tags=["ocr"])
async def ocr_vision_upload(
    file: UploadFile = File(...),
    prompt: str = Form(
        default="Extract all visible text from this image. List each word or phrase you can read."
    ),
    timeout: int = Form(default=600),
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

    # Free the GPU — stop text server if it's occupying the iGPU.
    await asyncio.to_thread(_pause_text_server)

    with tempfile.NamedTemporaryFile(suffix=suffix, delete=True) as tmp:
        tmp.write(raw)
        tmp.flush()

        try:
            result = await asyncio.to_thread(
                _vision.run_vision,
                image_path=tmp.name,
                prompt=prompt,
                timeout=timeout,
            )
        except TimeoutError:
            raise HTTPException(504, f"Vision OCR timed out after {timeout}s")
        except Exception as e:
            log.exception("Vision OCR failed")
            raise HTTPException(500, str(e))

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


@app.post("/enrich", response_model=EnrichResponse, tags=["vocabulary"])
async def enrich(req: EnrichRequest):
    """
    Enrich a list of vocabulary words with definitions and example sentences.
    """
    _ensure_text_server()

    t0 = time.monotonic()
    results = []

    for word in req.words:
        # Definition
        def_result = await asyncio.to_thread(
            _text.generate,
            prompt=f'Define the word "{word}" in {req.definition_language}. Be concise (1\u20132 sentences).',
            max_tokens=req.max_tokens,
            temperature=req.temperature,
        )

        # Examples
        ex_result = await asyncio.to_thread(
            _text.generate,
            prompt=f'Write 2 short example sentences using "{word}" in {req.examples_language}.',
            max_tokens=req.max_tokens,
            temperature=max(0.5, req.temperature),
        )

        results.append(EnrichWordResult(
            word=word,
            definition=def_result["content"],
            examples=ex_result["content"],
        ))

    elapsed = time.monotonic() - t0

    return EnrichResponse(results=results, elapsed_s=round(elapsed, 2))


# -------------------------------------------------------------------
# Pipeline: image → OCR → enrich → Anki-ready cards
# -------------------------------------------------------------------

@app.post("/pipeline/image-to-cards", tags=["pipeline"])
async def pipeline_image_to_cards(
    file: UploadFile = File(...),
    definition_language: str = Form(default="english"),
    examples_language: str = Form(default="english"),
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

    # Free the GPU -- stop text server if it's occupying the iGPU.
    await asyncio.to_thread(_pause_text_server)

    with tempfile.NamedTemporaryFile(suffix=suffix, delete=True) as tmp:
        tmp.write(raw)
        tmp.flush()
        ocr_result = await asyncio.to_thread(
            _vision.run_vision, tmp.name, prompt=ocr_prompt, timeout=600,
        )

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
    if _text is not None:
        _ensure_text_server()
        for word in words[:20]:  # cap at 20 words
            try:
                defn = await asyncio.to_thread(
                    _text.generate,
                    prompt=f'Define "{word}" in {definition_language}. 1-2 sentences.',
                    max_tokens=128,
                    temperature=0.1,
                )
                exs = await asyncio.to_thread(
                    _text.generate,
                    prompt=f'Write 2 example sentences using "{word}" in {examples_language}.',
                    max_tokens=150,
                    temperature=0.5,
                )
                cards.append({
                    "word": word,
                    "definition": defn["content"],
                    "examples": exs["content"],
                })
            except Exception as e:
                log.warning("Failed to enrich '%s': %s", word, e)
                cards.append({"word": word, "definition": "", "examples": ""})
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
