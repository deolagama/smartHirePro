"""
SmartHirePro - Embedding Generation Module
============================================
Generates dense vector embeddings from text using SentenceTransformers.
Designed for both batch (ingest time) and single-query (retrieval time) use.

Supports:
  - SentenceTransformers local models (default: all-MiniLM-L6-v2)
  - OpenAI text-embedding-3-small (optional, requires OPENAI_API_KEY)

Usage:
    from embedding.embedder import EmbeddingEngine
    engine = EmbeddingEngine()
    vectors = engine.embed_chunks(chunks)
    query_vec = engine.embed_query("Python engineer with 5 years experience")
"""

from __future__ import annotations

import time
from typing import List

# pyrefly: ignore [missing-import]
import numpy as np

from utils.logger import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Base class
# ---------------------------------------------------------------------------

class BaseEmbedder:
    """Abstract interface for all embedding backends."""

    def embed_texts(self, texts: list[str]) -> np.ndarray:
        raise NotImplementedError

    def embed_query(self, query: str) -> np.ndarray:
        return self.embed_texts([query])[0]


# ---------------------------------------------------------------------------
# SentenceTransformers backend (default)
# ---------------------------------------------------------------------------

class SentenceTransformerEmbedder(BaseEmbedder):
    """
    Local embedding engine backed by SentenceTransformers.

    The model is lazy-loaded on first call, so importing this module is cheap.

    Args:
        model_name:   HuggingFace model ID.
        device:       "cpu" | "cuda" | "mps".  Auto-detected if None.
        batch_size:   Number of texts processed per forward pass.
        show_progress: Whether to print tqdm progress bars during batch encoding.
    """

    def __init__(
        self,
        model_name: str | None = None,
        device: str | None = None,
        batch_size: int = 64,
        show_progress: bool = False,
    ) -> None:
        from config import settings

        self.model_name = model_name or settings.EMBEDDING_MODEL
        self.device = device or self._auto_device()
        self.batch_size = batch_size
        self.show_progress = show_progress
        self._model = None  # lazy init

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    def embed_texts(self, texts: list[str]) -> np.ndarray:
        """
        Encode a list of strings into a 2-D float32 numpy array.

        Args:
            texts: Strings to embed.

        Returns:
            ``np.ndarray`` of shape ``(len(texts), embedding_dim)``.
        """
        model = self._get_model()
        if not texts:
            return np.empty((0, model.get_sentence_embedding_dimension()), dtype=np.float32)

        start = time.perf_counter()
        logger.info("Embedding %d text(s) with %s …", len(texts), self.model_name)

        vectors = model.encode(
            texts,
            batch_size=self.batch_size,
            show_progress_bar=self.show_progress,
            convert_to_numpy=True,
            normalize_embeddings=True,  # L2-normalise → cosine ≡ dot product
        )

        elapsed = time.perf_counter() - start
        logger.info(
            "Embedding done: shape=%s, elapsed=%.2fs", vectors.shape, elapsed
        )
        return vectors.astype(np.float32)

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _get_model(self):
        """Lazy-load the SentenceTransformer model."""
        if self._model is None:

            # pyrefly: ignore [missing-import]
            from sentence_transformers import SentenceTransformer

            logger.info(
                "Loading SentenceTransformer model '%s' on %s …",
                self.model_name,
                self.device,
            )
            self._model = SentenceTransformer(self.model_name, device=self.device)
            logger.info("Model loaded (dim=%d).", self._model.get_sentence_embedding_dimension())
        return self._model

    @staticmethod
    def _auto_device() -> str:
        """Choose the best available device."""
        try:
            # pyrefly: ignore [missing-import]
            import torch

            if torch.cuda.is_available():
                return "cuda"
            if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                return "mps"
        except ImportError:
            pass
        return "cpu"


# ---------------------------------------------------------------------------
# OpenAI backend (optional)
# ---------------------------------------------------------------------------

class OpenAIEmbedder(BaseEmbedder):
    """
    Remote embedding engine using OpenAI's text-embedding-3-small model.

    Requires:
        - ``OPENAI_API_KEY`` environment variable.
        - ``openai`` package installed.

    Args:
        model: OpenAI model name.
        batch_size: Texts per API call (max 2048 for OpenAI).
    """

    OPENAI_EMBED_MODEL = "text-embedding-3-small"

    def __init__(self, model: str | None = None, batch_size: int = 256) -> None:
        self.model = model or self.OPENAI_EMBED_MODEL
        self.batch_size = batch_size
        self._client = None

    def embed_texts(self, texts: list[str]) -> np.ndarray:
        client = self._get_client()
        vectors: list[list[float]] = []

        for i in range(0, len(texts), self.batch_size):
            batch = texts[i : i + self.batch_size]
            logger.info("OpenAI embedding batch %d – %d", i, i + len(batch))
            response = client.embeddings.create(model=self.model, input=batch)
            vectors.extend([d.embedding for d in response.data])

        return np.array(vectors, dtype=np.float32)

    def _get_client(self):
        if self._client is None:
            # pyrefly: ignore [missing-import]
            from openai import OpenAI
            from config import settings

            if not settings.OPENAI_API_KEY:
                raise EnvironmentError(
                    "OPENAI_API_KEY is not set. Add it to your .env file."
                )
            self._client = OpenAI(api_key=settings.OPENAI_API_KEY)
        return self._client


# ---------------------------------------------------------------------------
# High-level facade
# ---------------------------------------------------------------------------

class EmbeddingEngine:
    """
    Unified embedding engine that delegates to the configured backend.

    Backend is selected from ``settings.LLM_PROVIDER`` but can be overridden
    with the *backend* argument.

    Args:
        backend: "sentence_transformers" (default) | "openai"
        **kwargs: Forwarded to the underlying embedder constructor.
    """

    _BACKENDS = {
        "sentence_transformers": SentenceTransformerEmbedder,
        "openai": OpenAIEmbedder,
    }

    def __init__(self, backend: str = "sentence_transformers", **kwargs) -> None:
        klass = self._BACKENDS.get(backend)
        if klass is None:
            raise ValueError(
                f"Unknown backend '{backend}'. Choose from: {list(self._BACKENDS)}"
            )
        self._embedder: BaseEmbedder = klass(**kwargs)
        logger.info("EmbeddingEngine initialised with backend '%s'.", backend)

    @property
    def dimension(self) -> int:
        """Return the embedding dimension (resolved lazily)."""
        from config import settings

        return settings.EMBEDDING_DIMENSION

    def embed_chunks(self, chunks: list) -> list[tuple]:
        """
        Embed a list of :class:`~chunking.chunker.ResumeChunk` objects.

        Args:
            chunks: List of ResumeChunk objects.

        Returns:
            List of ``(chunk, vector)`` tuples.
        """
        if not chunks:
            return []

        texts = [c.text for c in chunks]
        vectors = self._embedder.embed_texts(texts)

        return list(zip(chunks, vectors))

    def embed_query(self, query: str) -> np.ndarray:
        """
        Embed a single query string (e.g. a job description or skill).

        Args:
            query: The query text.

        Returns:
            1-D float32 numpy array.
        """
        return self._embedder.embed_query(query)

    def embed_texts(self, texts: list[str]) -> np.ndarray:
        """Directly embed arbitrary texts (batch)."""
        return self._embedder.embed_texts(texts)
