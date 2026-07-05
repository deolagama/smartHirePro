"""
SmartHirePro - Resume Evaluation Engine
=========================================
Orchestrates the full RAG pipeline to evaluate one or more candidates
against a job description.

Pipeline:
  retrieve → build prompt → LLM inference → parse + validate → return

The evaluator wires together:
  :class:`~retrieval.retriever.ResumeRetriever`
  :class:`~llm.client.LLMClient`
  :mod:`~prompting.prompts`

Usage:
    from evaluation.evaluator import ResumeEvaluator, EvaluationResult

    evaluator = ResumeEvaluator()
    results = evaluator.evaluate(
        job_description="We are looking for a Senior Python engineer …",
        top_k=5,
    )
    for r in results:
        print(r.resume_id, r.candidate_score, r.hiring_recommendation)
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, List

from utils.logger import get_logger
from prompting.prompts import (
    SYSTEM_PROMPT,
    EVALUATION_PROMPT_TEMPLATE,
    BATCH_SUMMARY_PROMPT_TEMPLATE,
)
from retrieval.retriever import CandidateContext

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Output schema
# ---------------------------------------------------------------------------

@dataclass
class EvaluationResult:
    """
    Structured output for a single candidate evaluation.

    All fields mirror the JSON schema defined in :mod:`~prompting.prompts`.
    """

    resume_id: str
    candidate_score: int                        # 0 – 100
    matching_skills: list[str] = field(default_factory=list)
    missing_skills: list[str] = field(default_factory=list)
    strengths: list[str] = field(default_factory=list)
    weaknesses: list[str] = field(default_factory=list)
    explanation: str = ""
    hiring_recommendation: str = "UNKNOWN"     # STRONG_YES | YES | MAYBE | NO | STRONG_NO
    interview_questions: list[str] = field(default_factory=list)
    retrieval_score: float = 0.0               # Vector similarity score
    raw_llm_response: str = ""                 # For debugging
    error: str | None = None

    @classmethod
    def from_dict(cls, data: dict, retrieval_score: float = 0.0) -> "EvaluationResult":
        """Construct an EvaluationResult from a parsed LLM JSON response."""
        return cls(
            resume_id=data.get("resume_id", ""),
            candidate_score=int(data.get("candidate_score", 0)),
            matching_skills=data.get("matching_skills", []),
            missing_skills=data.get("missing_skills", []),
            strengths=data.get("strengths", []),
            weaknesses=data.get("weaknesses", []),
            explanation=data.get("explanation", ""),
            hiring_recommendation=data.get("hiring_recommendation", "UNKNOWN"),
            interview_questions=data.get("interview_questions", []),
            retrieval_score=retrieval_score,
        )

    def to_dict(self) -> dict:
        """Serialise to a plain dict (safe for JSON / API response)."""
        return {
            "resume_id": self.resume_id,
            "candidate_score": self.candidate_score,
            "matching_skills": self.matching_skills,
            "missing_skills": self.missing_skills,
            "strengths": self.strengths,
            "weaknesses": self.weaknesses,
            "explanation": self.explanation,
            "hiring_recommendation": self.hiring_recommendation,
            "interview_questions": self.interview_questions,
            "retrieval_score": round(self.retrieval_score, 4),
            "error": self.error,
        }

    def __repr__(self) -> str:
        return (
            f"<EvaluationResult resume={self.resume_id!r} "
            f"score={self.candidate_score} "
            f"rec={self.hiring_recommendation!r}>"
        )


@dataclass
class BatchSummary:
    """Executive summary across all evaluated candidates."""

    total_evaluated: int
    recommended_candidates: list[str] = field(default_factory=list)
    top_candidate: str = ""
    summary: str = ""
    key_talent_gaps: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "total_evaluated": self.total_evaluated,
            "recommended_candidates": self.recommended_candidates,
            "top_candidate": self.top_candidate,
            "summary": self.summary,
            "key_talent_gaps": self.key_talent_gaps,
        }


# ---------------------------------------------------------------------------
# Evaluator
# ---------------------------------------------------------------------------

class ResumeEvaluator:
    """
    Full RAG pipeline orchestrator for resume evaluation.

    Args:
        retriever:  :class:`~retrieval.retriever.ResumeRetriever` instance.
                    Created from settings if not provided.
        llm_client: :class:`~llm.client.LLMClient` instance.
                    Created from settings if not provided.
    """

    def __init__(
        self,
        retriever=None,
        llm_client=None,
    ) -> None:
        self._retriever = retriever
        self._llm = llm_client

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def evaluate(
        self,
        job_description: str,
        top_k: int | None = None,
        include_summary: bool = True,
    ) -> tuple[list[EvaluationResult], BatchSummary | None]:
        """
        Full pipeline: retrieve → evaluate each candidate → batch summary.

        Args:
            job_description: Free-text job description.
            top_k:           Number of candidates to evaluate.
            include_summary: Whether to generate a batch executive summary.

        Returns:
            Tuple of ``(list[EvaluationResult], BatchSummary | None)``.
        """
        logger.info("Starting evaluation pipeline for job description (len=%d).", len(job_description))

        # 1 – Retrieve top candidates
        candidates = self._get_retriever().retrieve(job_description, top_k=top_k)

        if not candidates:
            logger.warning("No candidates retrieved. Ensure the vector store is populated.")
            return [], None

        logger.info("Retrieved %d candidate(s). Beginning LLM evaluation …", len(candidates))

        # 2 – Evaluate each candidate
        results: list[EvaluationResult] = []
        for i, candidate in enumerate(candidates, start=1):
            logger.info(
                "[%d/%d] Evaluating %s (retrieval_score=%.4f) …",
                i,
                len(candidates),
                candidate.resume_id,
                candidate.overall_score,
            )
            result = self._evaluate_single(candidate, job_description)
            results.append(result)

        # 3 – Rank by LLM score (secondary sort: retrieval score)
        results.sort(
            key=lambda r: (r.candidate_score, r.retrieval_score),
            reverse=True,
        )

        # 4 – Optional batch summary
        summary: BatchSummary | None = None
        if include_summary and results:
            logger.info("Generating batch summary for %d candidates …", len(results))
            summary = self._batch_summary(job_description, results)

        logger.info("Evaluation complete. Top candidate: %s", results[0].resume_id if results else "N/A")
        return results, summary

    def evaluate_candidate(
        self,
        candidate: CandidateContext,
        job_description: str,
    ) -> EvaluationResult:
        """
        Evaluate a single :class:`~retrieval.retriever.CandidateContext`.

        Args:
            candidate:       Pre-retrieved candidate context.
            job_description: Job description text.

        Returns:
            :class:`EvaluationResult`
        """
        return self._evaluate_single(candidate, job_description)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _evaluate_single(
        self,
        candidate: CandidateContext,
        job_description: str,
    ) -> EvaluationResult:
        """Build prompt, call LLM, parse response for one candidate."""
        user_prompt = EVALUATION_PROMPT_TEMPLATE.format(
            job_description=job_description.strip(),
            resume_id=candidate.resume_id,
            resume_context=candidate.context_text,
        )

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": user_prompt},
        ]

        try:
            data = self._get_llm().chat_json(messages)

            if "error" in data and data.get("error") == "json_parse_failed":
                logger.warning("JSON parse failed for %s.", candidate.resume_id)
                result = EvaluationResult(
                    resume_id=candidate.resume_id,
                    candidate_score=0,
                    retrieval_score=candidate.overall_score,
                    raw_llm_response=data.get("raw", ""),
                    error="json_parse_failed",
                )
            else:
                # Ensure resume_id is correct (LLM might hallucinate)
                data["resume_id"] = candidate.resume_id
                result = EvaluationResult.from_dict(data, retrieval_score=candidate.overall_score)

        except Exception as exc:
            logger.error("LLM call failed for %s: %s", candidate.resume_id, exc)
            result = EvaluationResult(
                resume_id=candidate.resume_id,
                candidate_score=0,
                retrieval_score=candidate.overall_score,
                error=str(exc),
            )

        return result

    def _batch_summary(
        self,
        job_description: str,
        results: list[EvaluationResult],
    ) -> BatchSummary:
        """Generate an executive summary across all evaluated candidates."""
        evals_json = json.dumps([r.to_dict() for r in results], indent=2)

        user_prompt = BATCH_SUMMARY_PROMPT_TEMPLATE.format(
            n_candidates=len(results),
            job_description=job_description.strip(),
            evaluations_json=evals_json,
        )

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": user_prompt},
        ]

        try:
            data = self._get_llm().chat_json(messages)
            return BatchSummary(
                total_evaluated=data.get("total_evaluated", len(results)),
                recommended_candidates=data.get("recommended_candidates", []),
                top_candidate=data.get("top_candidate", ""),
                summary=data.get("summary", ""),
                key_talent_gaps=data.get("key_talent_gaps", []),
            )
        except Exception as exc:
            logger.error("Batch summary generation failed: %s", exc)
            return BatchSummary(total_evaluated=len(results))

    # ------------------------------------------------------------------
    # Lazy dependency resolution
    # ------------------------------------------------------------------

    def _get_retriever(self):
        if self._retriever is None:
            from retrieval.retriever import ResumeRetriever

            self._retriever = ResumeRetriever()
        return self._retriever

    def _get_llm(self):
        if self._llm is None:
            from llm.client import LLMClient

            self._llm = LLMClient()
        return self._llm
