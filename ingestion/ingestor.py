"""
SmartHirePro - Data Ingestion Module
======================================
Handles loading resume PDFs from:
  1. A local directory of PDF files.
  2. The Hugging Face dataset  ``d4rk3r/resumes-raw-pdf``.

Each ingested resume is returned as a standardised :class:`ResumeDocument`
dataclass so that downstream modules remain decoupled from the source.

Usage:
    from ingestion.ingestor import ResumeIngestor
    ingestor = ResumeIngestor()
    docs = ingestor.ingest_from_directory("data/raw_resumes")
    docs = ingestor.ingest_from_huggingface()
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Generator

import fitz  # PyMuPDF
from langdetect import detect, LangDetectException

from utils.logger import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Data model ✅
# ---------------------------------------------------------------------------

@dataclass
class ResumeDocument:
    """Standardised representation of a single ingested resume."""

    resume_id: str                          # Unique identifier (usually the filename)
    raw_text: str                           # Full extracted text
    source: str                             # "local" | "huggingface"
    metadata: dict = field(default_factory=dict)  # Extra fields (page_count, etc.)

    def __repr__(self) -> str:
        preview = self.raw_text[:80].replace("\n", " ")
        return f"<ResumeDocument id={self.resume_id!r} source={self.source!r} text_len={len(self.raw_text)} preview={preview!r}>"


# ---------------------------------------------------------------------------
# PDF text extraction helper ✅
# ---------------------------------------------------------------------------

class PDFExtractor:
    """
    Low-level utility that extracts plain text from PDF bytes or file paths
    using PyMuPDF (fitz).  Falls back gracefully on corrupt or encrypted PDFs.
    """

    @staticmethod
    def from_bytes(pdf_bytes: bytes, resume_id: str = "unknown") -> str | None:
        """Extract text from raw PDF bytes."""
        doc = None
        try:
            doc = fitz.open(stream=pdf_bytes, filetype="pdf")
            return PDFExtractor._read_doc(doc, resume_id)
        except Exception as exc:
            logger.warning("PDFExtractor.from_bytes failed for %s: %s", resume_id, exc)
            return None
        finally:
            if doc:
                doc.close()

    @staticmethod
    def from_path(pdf_path: str | Path, resume_id: str = "unknown") -> str | None:
        """Extract text from a PDF file on disk."""
        doc = None
        try:
            doc = fitz.open(str(pdf_path))
            return PDFExtractor._read_doc(doc, resume_id)
        except Exception as exc:
            logger.warning("PDFExtractor.from_path failed for %s: %s", resume_id, exc)
            return None
        finally:
            if doc:
                doc.close()

    @staticmethod
    def _read_doc(doc: fitz.Document, resume_id: str) -> str:
        """Concatenate text from all pages."""
        pages: list[str] = []
        for page_num, page in enumerate(doc):
            try:
                pages.append(page.get_text())
            except Exception as exc:
                logger.debug("Skipping page %d of %s: %s", page_num, resume_id, exc)
        return "\n\n".join(pages).strip()


# ---------------------------------------------------------------------------
# Language filter ✅
# ---------------------------------------------------------------------------

def _is_english(text: str) -> bool:
    """Return True only if ``text`` is detected as English."""
    try:
        return detect(text) == "en"
    except LangDetectException:
        return False


# ---------------------------------------------------------------------------
# Main ingestor ✅
# ---------------------------------------------------------------------------

class ResumeIngestor:
    """
    Orchestrates PDF loading, text extraction, language filtering, and
    produces :class:`ResumeDocument` objects ready for chunking.

    Args:
        english_only: If True (default), non-English resumes are discarded.
        min_text_len: Resumes with fewer characters are skipped as empty.
    """

    def __init__(self, english_only: bool = True, min_text_len: int = 100) -> None:
        self.english_only = english_only
        self.min_text_len = min_text_len
        self._extractor = PDFExtractor()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def ingest_from_directory(self, directory: str | Path) -> list[ResumeDocument]:
        """
        Recursively scan *directory* for PDF files and ingest all of them.

        Args:
            directory: Path to a folder containing ``*.pdf`` files.

        Returns:
            List of successfully ingested :class:`ResumeDocument` objects.
        """
        directory = Path(directory)
        if not directory.exists():
            logger.error("Directory does not exist: %s", directory)
            return []

        pdf_files = sorted(directory.rglob("*.pdf"))
        logger.info("Found %d PDF file(s) in %s", len(pdf_files), directory)

        docs: list[ResumeDocument] = []
        for pdf_path in pdf_files:
            doc = self._process_file(pdf_path)
            if doc:
                docs.append(doc)

        logger.info(
            "Ingested %d / %d resumes from directory (english_only=%s)",
            len(docs),
            len(pdf_files),
            self.english_only,
        )
        return docs

    def ingest_from_huggingface(self, dataset_name: str | None = None) -> list[ResumeDocument]:
        """
        Stream the HuggingFace PDF resume dataset and ingest all English resumes.

        Args:
            dataset_name: HF dataset identifier. Defaults to the value in settings.

        Returns:
            List of successfully ingested :class:`ResumeDocument` objects.
        """
        from config import settings

        _dataset_name = dataset_name or settings.HF_DATASET_NAME

        try:
            from datasets import load_dataset  # optional dependency
        except ImportError:
            logger.error("Install 'datasets' to use ingest_from_huggingface().")
            return []

        logger.info("Loading HuggingFace dataset: %s", _dataset_name)
        try:
            ds = load_dataset(_dataset_name, split="train")
        except Exception as exc:
            logger.error("Failed to load HF dataset %s: %s", _dataset_name, exc)
            return []

        docs: list[ResumeDocument] = []
        for idx, item in enumerate(ds):
            resume_id = item.get("file_name", f"hf_sample_{idx+1}.pdf")
            doc = self._process_hf_item(item, resume_id)
            if doc:
                docs.append(doc)

        logger.info("Ingested %d resumes from HuggingFace dataset.", len(docs))
        return docs

    def ingest_single_pdf(self, pdf_path: str | Path) -> ResumeDocument | None:
        """
        Ingest a single PDF file, bypassing language filtering.

        Args:
            pdf_path: Path to the PDF file.

        Returns:
            A :class:`ResumeDocument` or None if extraction fails.
        """
        path = Path(pdf_path)
        if not path.exists():
            logger.error("File not found: %s", path)
            return None
        return self._process_file(path, skip_language_filter=True)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _process_file(
        self, pdf_path: Path, skip_language_filter: bool = False
    ) -> ResumeDocument | None:
        """Extract text from a PDF file and apply filters."""
        resume_id = pdf_path.name
        raw_text = self._extractor.from_path(pdf_path, resume_id)

        return self._validate_and_build(
            raw_text, resume_id, source="local",
            skip_language_filter=skip_language_filter,
            metadata={"file_path": str(pdf_path)},
        )

    def _process_hf_item(self, item: dict, resume_id: str) -> ResumeDocument | None:
        """Extract text from a HuggingFace dataset item."""
        pdf_data = item.get("pdf")
        if not pdf_data:
            logger.debug("No PDF data in item %s, skipping.", resume_id)
            return None

        raw_text: str | None = None
        if hasattr(pdf_data, "bytes") and pdf_data.bytes:
            raw_text = self._extractor.from_bytes(pdf_data.bytes, resume_id)
        elif hasattr(pdf_data, "path") and pdf_data.path:
            raw_text = self._extractor.from_path(pdf_data.path, resume_id)

        return self._validate_and_build(
            raw_text, resume_id, source="huggingface",
            metadata={"hf_index": item.get("__index_level_0__", -1)},
        )

    def _validate_and_build(
        self,
        raw_text: str | None,
        resume_id: str,
        source: str,
        metadata: dict | None = None,
        skip_language_filter: bool = False,
    ) -> ResumeDocument | None:
        """Common validation and document construction logic."""
        if not raw_text:
            logger.debug("No text extracted for %s, skipping.", resume_id)
            return None

        if len(raw_text) < self.min_text_len:
            logger.debug(
                "Resume %s too short (%d chars), skipping.", resume_id, len(raw_text)
            )
            return None

        if self.english_only and not skip_language_filter:
            if not _is_english(raw_text):
                logger.debug("Non-English resume skipped: %s", resume_id)
                return None

        return ResumeDocument(
            resume_id=resume_id,
            raw_text=raw_text,
            source=source,
            metadata=metadata or {},
        )
