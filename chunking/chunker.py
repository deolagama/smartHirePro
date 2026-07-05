"""
SmartHirePro - Resume Chunking Module
=======================================
Splits a raw resume text into semantically meaningful chunks suitable for
embedding and vector retrieval.  Two strategies are provided:

1. **SectionChunker** – Regex-based section detection (EXPERIENCE, EDUCATION,
   SKILLS, etc.).  Each section becomes its own chunk, then oversized sections
   are further split with a sliding window.  *Preferred* for well-structured
   resumes.

2. **SlidingWindowChunker** – Pure token-count sliding window.  Used as a
   fallback when section detection yields too few segments or for
   less-structured resumes.

Usage:
    from chunking.chunker import ResumeChunker
    from ingestion.ingestor import ResumeDocument

    chunker = ResumeChunker()
    chunks  = chunker.chunk(resume_doc)
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import List

from utils.logger import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Common section header patterns found in English resumes
# ---------------------------------------------------------------------------
_SECTION_PATTERNS = re.compile(
    r"(?im)"                          # case-insensitive, multiline
    r"^("
    r"(?:professional\s+)?summary|objective|profile|"
    r"(?:work\s+)?experience|employment(?:\s+history)?|career\s+history|"
    r"education(?:al)?\s*(?:background)?|"
    r"(?:technical\s+)?skills?|competenc(?:y|ies)|expertise|"
    r"certifications?|licens(?:es?|ure)|"
    r"projects?|portfolio|"
    r"publications?|research|"
    r"awards?|honors?|achievements?|"
    r"volunteering?|extra-?curricular|activities|"
    r"languages?|"
    r"references?"
    r")"
    r"\s*:?\s*$"                      # optional colon at end of line
)


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class ResumeChunk:
    """A single chunk derived from a resume, ready to be embedded."""

    chunk_id: str            # e.g.  "john_doe.pdf::skills::0"
    resume_id: str           # parent resume identifier
    section: str             # section label ("skills", "experience", "full", …)
    text: str                # chunk text content
    chunk_index: int         # ordinal within the resume
    metadata: dict = field(default_factory=dict)

    def __repr__(self) -> str:
        preview = self.text[:60].replace("\n", " ")
        return (
            f"<ResumeChunk id={self.chunk_id!r} section={self.section!r} "
            f"len={len(self.text)} preview={preview!r}>"
        )


# ---------------------------------------------------------------------------
# Section-aware chunker
# ---------------------------------------------------------------------------

class SectionChunker:
    """
    Splits resume text into logical sections based on header detection.

    Each detected section becomes its own chunk.  Sections exceeding
    *max_section_chars* are further sub-split using a sliding window so that
    no chunk fed to the embedder is excessively large.

    Args:
        max_section_chars: Maximum character length per chunk before
                           further splitting occurs.
        overlap_chars:     Overlap between sub-splits (in characters).
    """

    def __init__(
        self,
        max_section_chars: int = 1500,
        overlap_chars: int = 150,
    ) -> None:
        self.max_section_chars = max_section_chars
        self.overlap_chars = overlap_chars

    def split(self, resume_id: str, text: str) -> list[ResumeChunk]:
        """
        Split *text* into section-aware chunks.

        Args:
            resume_id: Identifier of the parent resume.
            text:      Full resume text.

        Returns:
            List of :class:`ResumeChunk` objects.
        """
        sections = self._detect_sections(text)

        if len(sections) < 2:
            logger.debug(
                "Section detection yielded too few sections for %s, "
                "falling back to sliding-window.",
                resume_id,
            )
            return []   # signal caller to use fallback

        chunks: list[ResumeChunk] = []
        chunk_index = 0

        for section_name, section_text in sections:
            # Further split oversized sections
            sub_texts = self._slide(section_text)
            for sub in sub_texts:
                chunk = ResumeChunk(
                    chunk_id=f"{resume_id}::{section_name}::{chunk_index}",
                    resume_id=resume_id,
                    section=section_name,
                    text=sub.strip(),
                    chunk_index=chunk_index,
                )
                chunks.append(chunk)
                chunk_index += 1

        logger.debug(
            "SectionChunker produced %d chunks for %s.", len(chunks), resume_id
        )
        return chunks

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _detect_sections(self, text: str) -> list[tuple[str, str]]:
        """
        Return a list of (section_name, section_text) tuples by splitting
        the resume on detected section headers.
        """
        lines = text.splitlines()
        sections: list[tuple[str, str]] = []
        current_section = "header"
        current_lines: list[str] = []

        for line in lines:
            if _SECTION_PATTERNS.match(line.strip()):
                # Flush previous section
                body = "\n".join(current_lines).strip()
                if body:
                    sections.append((current_section, body))
                # Start new section
                current_section = line.strip().lower().rstrip(":")
                current_lines = []
            else:
                current_lines.append(line)

        # Flush the final section
        body = "\n".join(current_lines).strip()
        if body:
            sections.append((current_section, body))

        return sections

    def _slide(self, text: str) -> list[str]:
        """
        Break *text* into overlapping windows of at most *max_section_chars*.
        Returns a list with a single element if the text is already small enough.
        """
        if len(text) <= self.max_section_chars:
            return [text]

        parts: list[str] = []
        start = 0
        while start < len(text):
            end = start + self.max_section_chars
            parts.append(text[start:end])
            start = end - self.overlap_chars
        return parts


# ---------------------------------------------------------------------------
# Sliding-window fallback chunker
# ---------------------------------------------------------------------------

class SlidingWindowChunker:
    """
    Token-count based sliding-window chunker.

    Splits text into chunks of *chunk_size* words with *overlap* words of
    context overlap between consecutive chunks.

    Args:
        chunk_size: Approximate number of words per chunk.
        overlap:    Number of overlapping words between consecutive chunks.
    """

    def __init__(self, chunk_size: int = 150, overlap: int = 20) -> None:
        self.chunk_size = chunk_size
        self.overlap = overlap

    def split(self, resume_id: str, text: str) -> list[ResumeChunk]:
        """
        Split *text* into overlapping word-count chunks.

        Args:
            resume_id: Identifier of the parent resume.
            text:      Full resume text.

        Returns:
            List of :class:`ResumeChunk` objects.
        """
        words = text.split()
        if not words:
            return []

        chunks: list[ResumeChunk] = []
        start = 0
        chunk_index = 0

        while start < len(words):
            end = start + self.chunk_size
            chunk_words = words[start:end]
            chunk_text = " ".join(chunk_words).strip()

            if chunk_text:
                chunks.append(
                    ResumeChunk(
                        chunk_id=f"{resume_id}::window::{chunk_index}",
                        resume_id=resume_id,
                        section="full",
                        text=chunk_text,
                        chunk_index=chunk_index,
                    )
                )
                chunk_index += 1

            start = end - self.overlap

        logger.debug(
            "SlidingWindowChunker produced %d chunks for %s.", len(chunks), resume_id
        )
        return chunks


# ---------------------------------------------------------------------------
# Unified chunker facade
# ---------------------------------------------------------------------------

class ResumeChunker:
    """
    High-level chunker that:
    1. Tries section-based chunking first.
    2. Falls back to sliding-window if section detection is insufficient.

    Args:
        chunk_size:    Word count per sliding-window chunk (fallback).
        chunk_overlap: Word overlap for sliding-window chunks (fallback).
        max_section_chars: Max chars per section chunk before sub-splitting.
    """

    def __init__(
        self,
        chunk_size: int | None = None,
        chunk_overlap: int | None = None,
        max_section_chars: int = 1500,
    ) -> None:
        from config import settings

        _size = chunk_size or settings.CHUNK_SIZE
        _overlap = chunk_overlap or settings.CHUNK_OVERLAP

        # Convert token counts to approximate word counts (1 token ≈ 0.75 words)
        word_size = max(50, int(_size * 0.75))
        word_overlap = max(10, int(_overlap * 0.75))

        self._section_chunker = SectionChunker(
            max_section_chars=max_section_chars,
            overlap_chars=150,
        )
        self._window_chunker = SlidingWindowChunker(
            chunk_size=word_size,
            overlap=word_overlap,
        )

    def chunk(self, resume) -> list[ResumeChunk]:
        """
        Chunk a :class:`~ingestion.ingestor.ResumeDocument`.

        Args:
            resume: A ResumeDocument instance.

        Returns:
            List of :class:`ResumeChunk` objects.
        """
        resume_id = resume.resume_id
        text = resume.raw_text.strip()

        if not text:
            logger.warning("Empty text for resume %s, skipping.", resume_id)
            return []

        # Try section chunking first
        chunks = self._section_chunker.split(resume_id, text)

        if not chunks:
            logger.info(
                "Falling back to sliding-window chunking for %s.", resume_id
            )
            chunks = self._window_chunker.split(resume_id, text)

        # Attach resume-level metadata to each chunk
        for chunk in chunks:
            chunk.metadata.update(resume.metadata)

        logger.info(
            "Chunked resume %s → %d chunk(s).", resume_id, len(chunks)
        )
        return chunks

    def chunk_text(self, resume_id: str, text: str) -> list[ResumeChunk]:
        """
        Chunk arbitrary text without a ResumeDocument wrapper.
        Useful for chunking job descriptions.
        """
        from ingestion.ingestor import ResumeDocument

        dummy = ResumeDocument(
            resume_id=resume_id, raw_text=text, source="inline"
        )
        return self.chunk(dummy)
