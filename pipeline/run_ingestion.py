"""
SmartHirePro - Ingestion Pipeline Runner
==========================================
CLI script that runs the full ingestion pipeline:
  1. Load PDFs (local directory or HuggingFace)
  2. Chunk each resume
  3. Generate embeddings
  4. Persist to vector store

Run:
    python -m pipeline.run_ingestion --source huggingface
    python -m pipeline.run_ingestion --source local --dir data/raw_resumes
    python -m pipeline.run_ingestion --source local --dir data/raw_resumes --clear
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

# Ensure the project root is on sys.path when run as a script
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ingestion.ingestor import ResumeIngestor
from chunking.chunker import ResumeChunker
from embedding.embedder import EmbeddingEngine
from vector_store.store import VectorStoreFactory
from utils.logger import get_logger

logger = get_logger(__name__)


def run_ingestion(
    source: str = "local",
    directory: str | None = None,
    clear_store: bool = False,
) -> None:
    """
    Execute the full ingestion pipeline.

    Args:
        source:      "local" | "huggingface"
        directory:   Path to local PDF directory (used when source="local").
        clear_store: If True, wipe the vector store before ingesting.
    """
    total_start = time.perf_counter()

    # ------------------------------------------------------------------
    # 1. Initialise components
    # ------------------------------------------------------------------
    logger.info("=" * 60)
    logger.info("SmartHirePro – Ingestion Pipeline")
    logger.info("=" * 60)

    ingestor = ResumeIngestor(english_only=True)
    chunker = ResumeChunker()
    engine = EmbeddingEngine()
    store = VectorStoreFactory.create()

    if clear_store:
        logger.warning("Clearing existing vector store as requested.")
        store.clear()

    logger.info(
        "Vector store: %s | Existing vectors: %d",
        type(store).__name__,
        store.count,
    )

    # ------------------------------------------------------------------
    # 2. Load resumes
    # ------------------------------------------------------------------
    logger.info("Step 1/4 – Loading resumes (source='%s') …", source)

    if source == "huggingface":
        docs = ingestor.ingest_from_huggingface()
    elif source == "local":
        if not directory:
            raise ValueError("--dir is required when --source=local")
        docs = ingestor.ingest_from_directory(directory)
    else:
        raise ValueError(f"Unknown source '{source}'. Use 'local' or 'huggingface'.")

    if not docs:
        logger.error("No resumes ingested. Aborting.")
        return

    logger.info("Loaded %d resume(s).", len(docs))

    # ------------------------------------------------------------------
    # 3. Chunk resumes
    # ------------------------------------------------------------------
    logger.info("Step 2/4 – Chunking resumes …")
    all_chunks = []
    for doc in docs:
        chunks = chunker.chunk(doc)
        all_chunks.extend(chunks)

    logger.info("Total chunks generated: %d.", len(all_chunks))

    if not all_chunks:
        logger.error("No chunks produced. Aborting.")
        return

    # ------------------------------------------------------------------
    # 4. Generate embeddings (batched)
    # ------------------------------------------------------------------
    logger.info("Step 3/4 – Generating embeddings …")
    chunk_vector_pairs = engine.embed_chunks(all_chunks)
    logger.info("Embeddings generated: %d pairs.", len(chunk_vector_pairs))

    # ------------------------------------------------------------------
    # 5. Store in vector database
    # ------------------------------------------------------------------
    logger.info("Step 4/4 – Storing vectors …")
    store.add(chunk_vector_pairs)

    elapsed = time.perf_counter() - total_start
    logger.info("=" * 60)
    logger.info(
        "Ingestion complete in %.1fs | Resumes: %d | Chunks: %d | Vectors stored: %d",
        elapsed,
        len(docs),
        len(all_chunks),
        store.count,
    )
    logger.info("=" * 60)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="SmartHirePro ingestion pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--source",
        choices=["local", "huggingface"],
        default="local",
        help="Resume source (default: local)",
    )
    parser.add_argument(
        "--dir",
        default=None,
        help="Path to local PDF directory (required when --source=local)",
    )
    parser.add_argument(
        "--clear",
        action="store_true",
        help="Clear existing vector store before ingestion",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_ingestion(
        source=args.source,
        directory=args.dir,
        clear_store=args.clear,
    )
