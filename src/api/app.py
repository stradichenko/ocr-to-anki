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
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(name)s: %(message)s")

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
        # Context window: prompt (~100 tok) + output (2 × 150 = 300 tok) ≈ 400.
        # 1024 leaves plenty of room and keeps KV cache small for the iGPU.
        server = LlamaCppServer(context_size=1024)
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
        model=_text.model_path.name if _text else "-",
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

    # Free the GPU -- stop text server if it's occupying the iGPU.
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

    # Free the GPU -- stop text server if it's occupying the iGPU.
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


def _build_enrich_prompt(words: list[str], def_lang: str, ex_lang: str) -> str:
    """Build a single batched prompt for all words."""
    word_list = "\n".join(f"- {w}" for w in words)
    return (
        f"For each word below, provide:\n"
        f"1) A concise definition (1-2 sentences) written in {def_lang}\n"
        f"2) Two short example sentences written in {ex_lang}\n\n"
        f"IMPORTANT: These words come from OCR and may contain spelling errors.\n"
        f"If the word is misspelled or has OCR artifacts, add a CORRECTED: line\n"
        f"with the correct spelling. Use the corrected word in the definition and examples.\n"
        f"If the word is already correct, omit the CORRECTED: line.\n\n"
        f"CRITICAL: You MUST use these EXACT English labels for formatting:\n"
        f"WORD:, CORRECTED:, DEF:, EX1:, EX2:\n"
        f"Do NOT translate the labels into {def_lang} or any other language.\n"
        f"Only the definition text and example sentences should be in the "
        f"requested language.\n\n"
        f"If you do not recognize a word or it is not a real word, "
        f"write DEF: UNKNOWN and leave the examples empty.\n\n"
        f"Use EXACTLY this format for each word (no extra text):\n\n"
        f"WORD: <word as given>\n"
        f"CORRECTED: <correct spelling>  (only if different from WORD)\n"
        f"DEF: <definition in {def_lang}>\n"
        f"EX1: <example sentence in {ex_lang}>\n"
        f"EX2: <example sentence in {ex_lang}>\n\n"
        f"Words:\n{word_list}"
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
        corrected = ""
        for line in body.split("\n"):
            line = line.strip()
            if _CORRECTED_RE.match(line):
                corrected = _CORRECTED_RE.sub("", line).strip()
            elif _DEF_RE.match(line):
                defn = _DEF_RE.sub("", line).strip()
            elif _EX1_RE.match(line):
                examples = _EX1_RE.sub("", line).strip()
            elif _EX2_RE.match(line):
                ex2 = _EX2_RE.sub("", line).strip()
                if ex2:
                    examples += "\n" + ex2
        data = {"definition": defn, "examples": examples, "corrected_word": corrected}
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

        corrected = entry.get("corrected_word", "")
        # Only keep correction if it actually differs from the original word.
        if corrected and corrected.lower() == w.lower():
            corrected = ""

        results.append({
            "word": w,
            "definition": defn,
            "examples": examples,
            "warning": warning,
            "corrected_word": corrected,
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
            batch, req.definition_language, req.examples_language,
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
                rb, req.definition_language, req.examples_language,
            )
            try:
                result = await asyncio.to_thread(
                    _text.generate,
                    prompt=prompt,
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

    results = [
        EnrichWordResult(
            word=p["word"],
            definition=p["definition"],
            examples=p["examples"],
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

    # Free the GPU -- stop text server if it's occupying the iGPU.
    await asyncio.to_thread(_pause_text_server)

    with tempfile.NamedTemporaryFile(suffix=suffix, delete=True) as tmp:
        tmp.write(raw)
        tmp.flush()
        ocr_result = await asyncio.to_thread(
            _vision.run_vision, tmp.name, prompt=ocr_prompt, timeout=2700,
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
    capped = words[:20]
    if _text is not None and capped:
        _ensure_text_server()
        BATCH_SIZE = 3  # small batches to fit in 1024 context window
        batches = [capped[i : i + BATCH_SIZE] for i in range(0, len(capped), BATCH_SIZE)]
        try:
            for batch in batches:
                prompt = _build_enrich_prompt(batch, definition_language, examples_language)
                max_tok = 60 * len(batch)
                result = await asyncio.to_thread(
                    _text.generate,
                    prompt=prompt,
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
