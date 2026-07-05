"""
SmartHirePro - Retrieval Module
=================================
Turns a natural-language job description into a ranked list of
:class:`~vector_store.store.SearchResult` objects, then groups them by resume
to produce a de-duplicated, per-candidate context ready for LLM reasoning.

The retriever works in two stages:
  1. **Chunk retrieval** – embed the job description and fetch the *top_k_chunks*
     most similar chunks from the vector store.
  2. **Resume grouping** – aggregate chunks by ``resume_id``, deduplicate,
     and rank candidates by their best chunk similarity score.

Usage:
    from retrieval.retriever import ResumeRetriever
    retriever = ResumeRetriever()
    results = retriever.retrieve(job_description="Python engineer …", top_k=5)
    for candidate in results:
        print(candidate.resume_id, candidate.overall_score)
        print(candidate.context_text)
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from utils.logger import get_logger
from vector_store.store import SearchResult

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Per-candidate result container
# ---------------------------------------------------------------------------

@dataclass
class CandidateContext:
    """
    All retrieved context for a single candidate / resume.

    Attributes:
        resume_id:     Unique identifier (usually the PDF filename).
        chunks:        Individual retrieved :class:`SearchResult` objects.
        overall_score: Composite relevance score (0.0 – 1.0).
        context_text:  Concatenated, deduplicated chunk text for the LLM prompt.
    """

    resume_id: str
    chunks: list[SearchResult] = field(default_factory=list)
    overall_score: float = 0.0
    context_text: str = ""

    def __repr__(self) -> str:
        return (
            f"<CandidateContext resume={self.resume_id!r} "
            f"score={self.overall_score:.4f} chunks={len(self.chunks)}>"
        )


# ---------------------------------------------------------------------------
# Retriever
# ---------------------------------------------------------------------------

class ResumeRetriever:
    """
    Retrieves and ranks candidate resumes for a given job description.

    Args:
        embedding_engine: An :class:`~embedding.embedder.EmbeddingEngine` instance.
                          Created from settings if not provided.
        vector_store:     A :class:`~vector_store.store.BaseVectorStore` instance.
                          Created from settings if not provided.
        top_k_chunks:     Number of raw chunks to pull from the vector store.
        top_k_resumes:    Number of unique candidates to return after grouping.
        score_strategy:   "max" (best chunk score), "mean", or "weighted_mean".
    """

    def __init__(
        self,
        embedding_engine=None,
        vector_store=None,
        top_k_chunks: int | None = None,
        top_k_resumes: int | None = None,
        score_strategy: str = "weighted_mean",
    ) -> None:
        from config import settings

        self.top_k_chunks = top_k_chunks or settings.TOP_K_CHUNKS
        self.top_k_resumes = top_k_resumes or settings.TOP_K_RESUMES
        self.score_strategy = score_strategy

        # Lazy dependencies
        self._engine = embedding_engine
        self._store = vector_store

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def retrieve(
        self,
        job_description: str,
        top_k: int | None = None,
    ) -> list[CandidateContext]:
        """
        Return the top-ranked candidates for the given job description.

        Args:
            job_description: Full job description text (or just key skills).
            top_k:           Override for the number of candidates returned.

        Returns:
            List of :class:`CandidateContext` objects, ranked best-first.
        """
        _top_k = top_k or self.top_k_resumes

        logger.info("Retrieving candidates for job description (len=%d) …", len(job_description))

        # Step 1 – embed the job description
        query_vector = self._get_engine().embed_query(job_description)

        # Step 2 – search vector store
        raw_results = self._get_store().search(query_vector, top_k=self.top_k_chunks)
        logger.info("Vector store returned %d chunk(s).", len(raw_results))

        if not raw_results:
            logger.warning("No results found. Is the vector store populated?")
            return []

        # Step 3 – group by resume and rank
        candidates = self._group_by_resume(raw_results)
        ranked = sorted(candidates, key=lambda c: c.overall_score, reverse=True)

        top = ranked[:_top_k]
        logger.info("Returning %d ranked candidate(s).", len(top))
        return top

    def retrieve_raw_chunks(
        self, job_description: str, top_k: int | None = None
    ) -> list[SearchResult]:
        """
        Return raw chunk-level search results without grouping.

        Useful for debugging or for custom re-ranking strategies.
        """
        query_vector = self._get_engine().embed_query(job_description)
        return self._get_store().search(query_vector, top_k=top_k or self.top_k_chunks)

    # ------------------------------------------------------------------
    # Grouping & scoring
    # ------------------------------------------------------------------

    def _group_by_resume(
        self, results: list[SearchResult]
    ) -> list[CandidateContext]:
        """
        Group chunk-level results by ``resume_id`` and compute a composite score.
        """
        groups: dict[str, list[SearchResult]] = {}
        for r in results:
            groups.setdefault(r.resume_id, []).append(r)

        candidates: list[CandidateContext] = []
        for resume_id, chunks in groups.items():
            # Deduplicate chunks by chunk_id (can happen with overlapping windows)
            seen = set()
            unique_chunks: list[SearchResult] = []
            for c in sorted(chunks, key=lambda x: x.score, reverse=True):
                if c.chunk_id not in seen:
                    seen.add(c.chunk_id)
                    unique_chunks.append(c)

            score = self._compute_score(unique_chunks)
            context_text = self._build_context(unique_chunks)

            candidates.append(
                CandidateContext(
                    resume_id=resume_id,
                    chunks=unique_chunks,
                    overall_score=score,
                    context_text=context_text,
                )
            )

        return candidates

    def _compute_score(self, chunks: list[SearchResult]) -> float:
        """Aggregate chunk scores into a single candidate score."""
        if not chunks:
            return 0.0

        scores = [c.score for c in chunks]

        if self.score_strategy == "max":
            return float(max(scores))
        elif self.score_strategy == "mean":
            return float(np.mean(scores))
        elif self.score_strategy == "weighted_mean":
            # Weight by rank position (top chunk counts more)
            weights = [1.0 / (i + 1) for i in range(len(scores))]
            return float(np.average(scores, weights=weights))
        else:
            return float(max(scores))

    def _build_context(self, chunks: list[SearchResult]) -> str:
        """
        Concatenate chunks into a single context block for the LLM.

        Chunks are ordered by section type so the LLM sees a coherent resume.
        """
        # Preferred section order for readability
        section_order = [
            "summary", "objective", "profile",
            "experience", "employment", "career",
            "education",
            "skills", "competencies",
            "projects",
            "certifications",
            "header", "full", "window",
        ]

        def section_key(chunk: SearchResult) -> int:
            section = chunk.section.lower()
            for i, s in enumerate(section_order):
                if s in section:
                    return i
            return len(section_order)

        ordered = sorted(chunks, key=section_key)
        parts = [f"[{c.section.upper()}]\n{c.text}" for c in ordered]
        return "\n\n---\n\n".join(parts)

    # ------------------------------------------------------------------
    # Lazy dependency resolution
    # ------------------------------------------------------------------

    def _get_engine(self):
        if self._engine is None:
            from embedding.embedder import EmbeddingEngine

            self._engine = EmbeddingEngine()
        return self._engine

    def _get_store(self):
        if self._store is None:
            from vector_store.store import VectorStoreFactory

            self._store = VectorStoreFactory.create()
        return self._store
