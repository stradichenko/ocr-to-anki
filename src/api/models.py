"""
Pydantic models for the OCR-to-Anki API.
"""

from enum import Enum
from typing import Optional
from pydantic import BaseModel, Field


# -------------------------------------------------------------------
# Enums
# -------------------------------------------------------------------

class BackendName(str, Enum):
    cuda = "cuda"
    metal = "metal"
    vulkan = "vulkan"
    sycl = "sycl"
    cpu = "cpu"


class JobStatus(str, Enum):
    pending = "pending"
    running = "running"
    completed = "completed"
    failed = "failed"


# -------------------------------------------------------------------
# Requests
# -------------------------------------------------------------------

class VisionOCRRequest(BaseModel):
    """Request body for vision OCR (base64 image upload)."""
    image_base64: str = Field(..., description="Base64-encoded image (JPEG or PNG)")
    prompt: str = Field(
        default="Extract all visible text from this image. List each word or phrase you can read.",
        description="Prompt for the vision model",
    )
    timeout: int = Field(default=2700, ge=10, le=7200, description="Timeout in seconds")
    max_image_dim: int = Field(
        default=768, ge=0, le=4096,
        description=(
            "Downscale image so longest side is at most this many pixels. "
            "Reduces vision tokens and speeds up prompt eval.  0 = no limit."
        ),
    )


class EnrichRequest(BaseModel):
    """Request body for vocabulary enrichment."""
    words: list[str] = Field(..., min_length=1, max_length=50)
    definition_language: str = Field(default="english")
    examples_language: str = Field(default="english")
    term_language: str = Field(
        default="auto",
        description=(
            "Language the words are in (e.g. 'french', 'spanish'). "
            "'auto' = let the model detect automatically."
        ),
    )
    max_tokens: int = Field(default=256, ge=32, le=2048)
    temperature: float = Field(default=0.1, ge=0.0, le=2.0)


class GenerateRequest(BaseModel):
    """Request body for raw text generation."""
    prompt: str
    system: Optional[str] = None
    max_tokens: int = Field(default=512, ge=1, le=4096)
    temperature: float = Field(default=0.1, ge=0.0, le=2.0)
    top_k: int = Field(default=40, ge=1, le=100)
    top_p: float = Field(default=0.9, ge=0.0, le=1.0)


# -------------------------------------------------------------------
# Responses
# -------------------------------------------------------------------

class VisionOCRResponse(BaseModel):
    text: str
    elapsed_s: float
    backend: str
    image_hash: str = ""


class EnrichWordResult(BaseModel):
    word: str
    definition: str
    examples: str
    warning: str = ""  # "not_found", "truncated", or "" (ok)
    corrected_word: str = ""  # LLM-suggested correct spelling (if OCR error detected)


class EnrichResponse(BaseModel):
    results: list[EnrichWordResult]
    elapsed_s: float


class GenerateResponse(BaseModel):
    content: str
    tokens_evaluated: int = 0
    tokens_predicted: int = 0


class HealthResponse(BaseModel):
    status: str
    vision_available: bool
    text_available: bool
    backend: str
    model: str
    devices: list[dict] = []
    models_downloaded: bool = True
    models_dir: str = ""
    llama_binary_available: bool = True
    gpu_mode: str = "auto"


class PipelineImageToCardsResponse(BaseModel):
    ocr_text: str
    ocr_backend: str = ""
    ocr_elapsed_s: float = 0.0
    cards: list[EnrichWordResult]
    total_elapsed_s: float
    message: str = ""


class ModelFileInfo(BaseModel):
    name: str
    size_bytes: int = 0
    exists: bool
    url: str


class ModelStatusResponse(BaseModel):
    all_present: bool
    models_dir: str
    files: list[ModelFileInfo]


class BackendInfoResponse(BaseModel):
    os_name: str
    arch: str
    recommended_backend: str
    binary_path: Optional[str]
    devices: list[dict]


class UpdateCheckResponse(BaseModel):
    has_update: bool
    current_version: str
    latest_version: str
    download_url: str = ""
    release_notes: str = ""
    published_at: str = ""
