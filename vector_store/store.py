"""
SmartHirePro - Vector Store Module
=====================================
Provides a unified interface to persist and search vector embeddings.

Two backends are supported:
  1. **FAISSVectorStore** – Local, high-performance ANN search using Facebook FAISS.
  2. **ChromaVectorStore** – Persistent, metadata-rich vector store using ChromaDB.

Both expose the same :class:`BaseVectorStore` interface so that the retrieval
module is completely decoupled from the storage backend.

Usage:
    from vector_store.store import VectorStoreFactory
    store = VectorStoreFactory.create()           # uses settings.VECTOR_STORE_TYPE
    store.add(chunk_vector_pairs)
    results = store.search(query_vector, top_k=10)
"""

from __future__ import annotations

import json
import os
import pickle
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import numpy as np

from utils.logger import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Shared result dataclass
# ---------------------------------------------------------------------------

class SearchResult:
    """A single vector search result."""

    def __init__(
        self,
        chunk_id: str,
        resume_id: str,
        section: str,
        text: str,
        score: float,
        metadata: dict | None = None,
    ) -> None:
        self.chunk_id = chunk_id
        self.resume_id = resume_id
        self.section = section
        self.text = text
        self.score = score
        self.metadata = metadata or {}

    def __repr__(self) -> str:
        return (
            f"<SearchResult resume={self.resume_id!r} section={self.section!r} "
            f"score={self.score:.4f}>"
        )


# ---------------------------------------------------------------------------
# Abstract base
# ---------------------------------------------------------------------------

class BaseVectorStore(ABC):
    """Abstract interface that every vector store backend must implement."""

    @abstractmethod
    def add(self, chunk_vector_pairs: list[tuple]) -> None:
        """Persist (chunk, vector) pairs to the store."""

    @abstractmethod
    def search(self, query_vector: np.ndarray, top_k: int = 10) -> list[SearchResult]:
        """Return the *top_k* most similar chunks to *query_vector*."""

    @abstractmethod
    def clear(self) -> None:
        """Wipe all stored vectors and metadata."""

    @property
    @abstractmethod
    def count(self) -> int:
        """Number of vectors currently stored."""


# ---------------------------------------------------------------------------
# FAISS Backend
# ---------------------------------------------------------------------------

class FAISSVectorStore(BaseVectorStore):
    """
    FAISS-backed vector store with persistent metadata sidecar.

    Persists:
      - ``<index_path>.faiss`` – the FAISS binary index.
      - ``<index_path>.meta``  – JSON-serialised chunk metadata list.

    Args:
        index_path: Path prefix for persisted files (without extension).
        dimension:  Embedding dimension.  Must match embedder output.
        index_type: "flat_l2" | "flat_ip" | "ivf" (ivf = faster for large corpora).
    """

    def __init__(
        self,
        index_path: str | None = None,
        dimension: int | None = None,
        index_type: str = "flat_ip",
    ) -> None:
        import faiss  # noqa: F401 (imported for side-effect check)

        from config import settings

        self._faiss = faiss
        self.index_path = Path(index_path or settings.FAISS_INDEX_PATH)
        self.dimension = dimension or settings.EMBEDDING_DIMENSION
        self.index_type = index_type
        self._index = None
        self._metadata: list[dict] = []

        # Try to load an existing index from disk
        self._load_if_exists()

    # ------------------------------------------------------------------
    # Interface implementation
    # ------------------------------------------------------------------

    def add(self, chunk_vector_pairs: list[tuple]) -> None:
        """
        Add (ResumeChunk, np.ndarray) pairs to the FAISS index.

        If no index exists yet, one is created using the vector dimension.
        """
        if not chunk_vector_pairs:
            return

        if self._index is None:
            self._index = self._build_index(self.dimension)

        chunks, vectors = zip(*chunk_vector_pairs)
        matrix = np.vstack(vectors).astype(np.float32)

        self._index.add(matrix)

        for chunk in chunks:
            self._metadata.append({
                "chunk_id": chunk.chunk_id,
                "resume_id": chunk.resume_id,
                "section": chunk.section,
                "text": chunk.text,
                "metadata": chunk.metadata,
            })

        logger.info(
            "FAISS: added %d vector(s). Total: %d.", len(vectors), self._index.ntotal
        )
        self._save()

    def search(self, query_vector: np.ndarray, top_k: int = 10) -> list[SearchResult]:
        """Return the *top_k* nearest neighbours of *query_vector*."""
        if self._index is None or self._index.ntotal == 0:
            logger.warning("FAISS index is empty. Run ingestion first.")
            return []

        q = query_vector.reshape(1, -1).astype(np.float32)
        k = min(top_k, self._index.ntotal)
        scores, indices = self._index.search(q, k)

        results: list[SearchResult] = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0:
                continue
            meta = self._metadata[idx]
            results.append(
                SearchResult(
                    chunk_id=meta["chunk_id"],
                    resume_id=meta["resume_id"],
                    section=meta["section"],
                    text=meta["text"],
                    score=float(score),
                    metadata=meta.get("metadata", {}),
                )
            )

        return results

    def clear(self) -> None:
        """Wipe the index and metadata, and delete persisted files."""
        self._index = None
        self._metadata = []
        for suffix in (".faiss", ".meta"):
            p = self.index_path.with_suffix(suffix)
            if p.exists():
                p.unlink()
        logger.info("FAISS store cleared.")

    @property
    def count(self) -> int:
        return self._index.ntotal if self._index is not None else 0

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _save(self) -> None:
        self.index_path.parent.mkdir(parents=True, exist_ok=True)
        self._faiss.write_index(self._index, str(self.index_path.with_suffix(".faiss")))
        with open(self.index_path.with_suffix(".meta"), "w", encoding="utf-8") as fh:
            json.dump(self._metadata, fh, ensure_ascii=False)
        logger.debug("FAISS index persisted to %s.", self.index_path)

    def _load_if_exists(self) -> None:
        faiss_file = self.index_path.with_suffix(".faiss")
        meta_file = self.index_path.with_suffix(".meta")
        if faiss_file.exists() and meta_file.exists():
            logger.info("Loading existing FAISS index from %s …", faiss_file)
            self._index = self._faiss.read_index(str(faiss_file))
            with open(meta_file, encoding="utf-8") as fh:
                self._metadata = json.load(fh)
            logger.info("FAISS index loaded: %d vector(s).", self._index.ntotal)

    def _build_index(self, dim: int):
        """Construct the FAISS index structure."""
        if self.index_type == "flat_ip":
            # Inner product (cosine-equivalent when vectors are L2-normalised)
            return self._faiss.IndexFlatIP(dim)
        elif self.index_type == "flat_l2":
            return self._faiss.IndexFlatL2(dim)
        elif self.index_type == "ivf":
            quantiser = self._faiss.IndexFlatL2(dim)
            return self._faiss.IndexIVFFlat(quantiser, dim, 100)
        else:
            raise ValueError(f"Unknown index_type: {self.index_type}")


# ---------------------------------------------------------------------------
# ChromaDB Backend
# ---------------------------------------------------------------------------

class ChromaVectorStore(BaseVectorStore):
    """
    ChromaDB-backed vector store with rich metadata filtering support.

    Args:
        persist_dir:      Local directory for ChromaDB persistence.
        collection_name:  Name of the Chroma collection.
    """

    def __init__(
        self,
        persist_dir: str | None = None,
        collection_name: str | None = None,
    ) -> None:
        from config import settings
        import chromadb
        from chromadb.config import Settings

        _persist_dir = persist_dir or settings.CHROMA_PERSIST_DIR
        _collection = collection_name or settings.CHROMA_COLLECTION_NAME

        Path(_persist_dir).mkdir(parents=True, exist_ok=True)

        self._client = chromadb.PersistentClient(
            path=_persist_dir,
            settings=Settings(anonymized_telemetry=False),
        )
        self._collection = self._client.get_or_create_collection(
            name=_collection,
            metadata={"hnsw:space": "cosine"},
        )
        logger.info(
            "ChromaDB store ready: collection=%s, count=%d",
            _collection,
            self._collection.count(),
        )

    def add(self, chunk_vector_pairs: list[tuple]) -> None:
        if not chunk_vector_pairs:
            return

        ids, embeddings, documents, metadatas = [], [], [], []

        for chunk, vector in chunk_vector_pairs:
            ids.append(chunk.chunk_id)
            embeddings.append(vector.tolist())
            documents.append(chunk.text)
            metadatas.append({
                "resume_id": chunk.resume_id,
                "section": chunk.section,
                **{k: str(v) for k, v in (chunk.metadata or {}).items()},
            })

        self._collection.upsert(
            ids=ids,
            embeddings=embeddings,
            documents=documents,
            metadatas=metadatas,
        )
        logger.info(
            "Chroma: upserted %d chunk(s). Total: %d.",
            len(ids),
            self._collection.count(),
        )

    def search(
        self,
        query_vector: np.ndarray,
        top_k: int = 10,
    ) -> list[SearchResult]:
        if self._collection.count() == 0:
            logger.warning("Chroma collection is empty. Run ingestion first.")
            return []

        response = self._collection.query(
            query_embeddings=[query_vector.tolist()],
            n_results=min(top_k, self._collection.count()),
        )

        results: list[SearchResult] = []
        for chunk_id, doc, meta, dist in zip(
            response["ids"][0],
            response["documents"][0],
            response["metadatas"][0],
            response["distances"][0],
        ):
            # Chroma cosine distance → similarity = 1 - distance
            results.append(
                SearchResult(
                    chunk_id=chunk_id,
                    resume_id=meta.get("resume_id", ""),
                    section=meta.get("section", ""),
                    text=doc,
                    score=1.0 - float(dist),
                    metadata=meta,
                )
            )

        return results

    def clear(self) -> None:
        name = self._collection.name
        self._client.delete_collection(name)
        self._collection = self._client.create_collection(name)
        logger.info("Chroma collection '%s' cleared.", name)

    @property
    def count(self) -> int:
        return self._collection.count()


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

class VectorStoreFactory:
    """
    Creates the configured vector store backend.

    Reads ``settings.VECTOR_STORE_TYPE`` by default.

    Usage:
        store = VectorStoreFactory.create()
        store = VectorStoreFactory.create("chroma")
    """

    @staticmethod
    def create(backend: str | None = None, **kwargs) -> BaseVectorStore:
        from config import settings

        _backend = (backend or settings.VECTOR_STORE_TYPE).lower()

        if _backend == "faiss":
            return FAISSVectorStore(**kwargs)
        elif _backend in ("chroma", "chromadb"):
            return ChromaVectorStore(**kwargs)
        else:
            raise ValueError(
                f"Unknown vector store backend '{_backend}'. Use 'faiss' or 'chroma'."
            )
