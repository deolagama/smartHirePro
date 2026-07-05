"""
SmartHirePro - FastAPI Backend
================================
RESTful API exposing the RAG pipeline for resume evaluation.

Endpoints:
  POST /api/v1/ingest          – Ingest PDF resumes
  POST /api/v1/evaluate        – Evaluate candidates for a job description
  GET  /api/v1/health          – Health check
  GET  /api/v1/store/stats     – Vector store statistics
  DELETE /api/v1/store/clear   – Wipe the vector store

Run:
    uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
"""

from __future__ import annotations

import io
import sys
import time
from pathlib import Path
from typing import Annotated

# Ensure project root is on path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fastapi import FastAPI, File, Form, HTTPException, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from utils.logger import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------

app = FastAPI(
    title="SmartHirePro API",
    description=(
        "AI-powered resume screening and ranking via Retrieval-Augmented Generation. "
        "Upload resumes, set a job description, and receive structured candidate evaluations."
    ),
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Pydantic schemas
# ---------------------------------------------------------------------------

class EvaluateRequest(BaseModel):
    job_description: str = Field(
        ...,
        min_length=50,
        description="Full job description text (at least 50 characters).",
        example=(
            "We are hiring a Senior Python Engineer with 5+ years of experience in "
            "FastAPI, PostgreSQL, and cloud infrastructure (AWS/GCP). Experience with "
            "LLM APIs and vector databases is a strong plus."
        ),
    )
    top_k: int = Field(
        default=5,
        ge=1,
        le=20,
        description="Number of top candidates to return.",
    )
    include_summary: bool = Field(
        default=True,
        description="Whether to include an executive batch summary.",
    )


class IngestResponse(BaseModel):
    success: bool
    resumes_ingested: int
    chunks_stored: int
    elapsed_seconds: float
    message: str


class EvaluateResponse(BaseModel):
    success: bool
    job_description_preview: str
    candidates_evaluated: int
    results: list[dict]
    summary: dict | None
    elapsed_seconds: float


class HealthResponse(BaseModel):
    status: str
    vector_store_count: int
    provider: str


class StoreStatsResponse(BaseModel):
    backend: str
    vector_count: int


# ---------------------------------------------------------------------------
# Dependency helpers (singleton-like lazy init)
# ---------------------------------------------------------------------------

_store = None
_engine = None
_retriever = None
_evaluator = None


def get_store():
    global _store
    if _store is None:
        from vector_store.store import VectorStoreFactory
        _store = VectorStoreFactory.create()
    return _store


def get_engine():
    global _engine
    if _engine is None:
        from embedding.embedder import EmbeddingEngine
        _engine = EmbeddingEngine()
    return _engine


def get_evaluator():
    global _evaluator
    if _evaluator is None:
        from evaluation.evaluator import ResumeEvaluator
        _evaluator = ResumeEvaluator()
    return _evaluator


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/api/v1/health", response_model=HealthResponse, tags=["System"])
async def health():
    """Liveness probe. Returns vector store size and configured LLM provider."""
    from config import settings

    return HealthResponse(
        status="ok",
        vector_store_count=get_store().count,
        provider=settings.LLM_PROVIDER,
    )


@app.get("/api/v1/store/stats", response_model=StoreStatsResponse, tags=["Vector Store"])
async def store_stats():
    """Return vector store metadata."""
    store = get_store()
    return StoreStatsResponse(
        backend=type(store).__name__,
        vector_count=store.count,
    )


@app.delete("/api/v1/store/clear", tags=["Vector Store"])
async def clear_store():
    """
    **Warning**: Wipes all stored resume vectors.
    You will need to re-run ingestion after this.
    """
    get_store().clear()
    logger.warning("Vector store cleared via API.")
    return {"success": True, "message": "Vector store cleared."}


@app.post(
    "/api/v1/ingest",
    response_model=IngestResponse,
    status_code=status.HTTP_201_CREATED,
    tags=["Ingestion"],
)
async def ingest_resumes(
    files: list[UploadFile] = File(..., description="One or more PDF resume files."),
):
    """
    Upload and ingest PDF resumes into the vector store.

    Accepts multipart/form-data with one or more `files` fields.
    Each file must be a PDF. Non-PDFs and empty files are skipped.
    """
    from ingestion.ingestor import ResumeIngestor, ResumeDocument
    from chunking.chunker import ResumeChunker

    start = time.perf_counter()

    ingestor = ResumeIngestor(english_only=False)  # API users may upload any PDF
    chunker = ResumeChunker()
    engine = get_engine()
    store = get_store()

    all_chunks = []
    ingested_count = 0

    for upload in files:
        if not upload.filename or not upload.filename.lower().endswith(".pdf"):
            logger.warning("Skipping non-PDF file: %s", upload.filename)
            continue

        pdf_bytes = await upload.read()

        if not pdf_bytes:
            logger.warning("Empty file: %s", upload.filename)
            continue

        # Extract text from uploaded bytes
        from ingestion.ingestor import PDFExtractor

        raw_text = PDFExtractor.from_bytes(pdf_bytes, upload.filename)
        if not raw_text or len(raw_text) < 50:
            logger.warning("Could not extract text from %s.", upload.filename)
            continue

        doc = ResumeDocument(
            resume_id=upload.filename,
            raw_text=raw_text,
            source="api_upload",
        )

        chunks = chunker.chunk(doc)
        all_chunks.extend(chunks)
        ingested_count += 1
        logger.info("Ingested %s → %d chunk(s).", upload.filename, len(chunks))

    if not all_chunks:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="No valid PDF content could be extracted from the uploaded files.",
        )

    chunk_vector_pairs = engine.embed_chunks(all_chunks)
    store.add(chunk_vector_pairs)

    elapsed = time.perf_counter() - start

    return IngestResponse(
        success=True,
        resumes_ingested=ingested_count,
        chunks_stored=len(all_chunks),
        elapsed_seconds=round(elapsed, 2),
        message=f"Successfully ingested {ingested_count} resume(s) into the vector store.",
    )


@app.post(
    "/api/v1/evaluate",
    response_model=EvaluateResponse,
    tags=["Evaluation"],
)
async def evaluate_resumes(request: EvaluateRequest):
    """
    Evaluate top candidates against a job description using the RAG pipeline.

    Requires resumes to be ingested first via `POST /api/v1/ingest`.
    """
    if get_store().count == 0:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="The vector store is empty. Please ingest resumes first.",
        )

    start = time.perf_counter()
    evaluator = get_evaluator()

    results, summary = evaluator.evaluate(
        job_description=request.job_description,
        top_k=request.top_k,
        include_summary=request.include_summary,
    )

    elapsed = time.perf_counter() - start

    return EvaluateResponse(
        success=True,
        job_description_preview=request.job_description[:200] + "…",
        candidates_evaluated=len(results),
        results=[r.to_dict() for r in results],
        summary=summary.to_dict() if summary else None,
        elapsed_seconds=round(elapsed, 2),
    )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn
    from config import settings

    uvicorn.run(
        "api.main:app",
        host=settings.API_HOST,
        port=settings.API_PORT,
        reload=settings.API_RELOAD,
        log_level=settings.LOG_LEVEL.lower(),
    )
