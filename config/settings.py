"""
SmartHirePro - Central Configuration
=====================================
All environment variables, model names, paths, and system-level
constants are loaded and exposed from this single module.

Author: Deola Gama
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# ---------------------------------------------------------------------------
# Base Paths
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent.parent

# Load .env from project root (overrides any existing env vars only if not set)
load_dotenv(BASE_DIR / ".env", override=False)

# ---------------------------------------------------------------------------
# API Keys
# ---------------------------------------------------------------------------
OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
HUGGINGFACE_TOKEN: str = os.getenv("HUGGINGFACE_TOKEN", "")

# ---------------------------------------------------------------------------
# LLM Settings
# ---------------------------------------------------------------------------
LLM_PROVIDER: str = os.getenv("LLM_PROVIDER", "openai")   # "openai" | "ollama"
OPENAI_MODEL: str = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
OLLAMA_MODEL: str = os.getenv("OLLAMA_MODEL", "llama3")
OLLAMA_BASE_URL: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
LLM_TEMPERATURE: float = float(os.getenv("LLM_TEMPERATURE", "0.2"))
LLM_MAX_TOKENS: int = int(os.getenv("LLM_MAX_TOKENS", "2048"))

# ---------------------------------------------------------------------------
# Embedding Settings
# ---------------------------------------------------------------------------
EMBEDDING_MODEL: str = os.getenv(
    "EMBEDDING_MODEL", "all-MiniLM-L6-v2"
)  # Fast & good quality
EMBEDDING_DIMENSION: int = int(os.getenv("EMBEDDING_DIMENSION", "384"))

# ---------------------------------------------------------------------------
# Vector Store Settings
# ---------------------------------------------------------------------------
VECTOR_STORE_TYPE: str = os.getenv("VECTOR_STORE_TYPE", "faiss")  # "faiss" | "chroma"
FAISS_INDEX_PATH: str = os.getenv(
    "FAISS_INDEX_PATH", str(BASE_DIR / "data" / "vector_store" / "faiss_index")
)
CHROMA_PERSIST_DIR: str = os.getenv(
    "CHROMA_PERSIST_DIR", str(BASE_DIR / "data" / "vector_store" / "chroma")
)
CHROMA_COLLECTION_NAME: str = os.getenv("CHROMA_COLLECTION_NAME", "resumes")

# ---------------------------------------------------------------------------
# Chunking Settings
# ---------------------------------------------------------------------------
CHUNK_SIZE: int = int(os.getenv("CHUNK_SIZE", "512"))
CHUNK_OVERLAP: int = int(os.getenv("CHUNK_OVERLAP", "64"))

# ---------------------------------------------------------------------------
# Retrieval Settings
# ---------------------------------------------------------------------------
TOP_K_CHUNKS: int = int(os.getenv("TOP_K_CHUNKS", "10"))
TOP_K_RESUMES: int = int(os.getenv("TOP_K_RESUMES", "5"))

# ---------------------------------------------------------------------------
# Data Paths
# ---------------------------------------------------------------------------
RAW_RESUME_DIR: str = os.getenv(
    "RAW_RESUME_DIR", str(BASE_DIR / "data" / "raw_resumes")
)
PROCESSED_RESUME_DIR: str = os.getenv(
    "PROCESSED_RESUME_DIR", str(BASE_DIR / "data" / "processed_resumes_en")
)
HF_DATASET_NAME: str = os.getenv("HF_DATASET_NAME", "d4rk3r/resumes-raw-pdf")

# ---------------------------------------------------------------------------
# API Settings
# ---------------------------------------------------------------------------
API_HOST: str = os.getenv("API_HOST", "0.0.0.0")
API_PORT: int = int(os.getenv("API_PORT", "8000"))
API_RELOAD: bool = os.getenv("API_RELOAD", "true").lower() == "true"

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")
LOG_FILE: str = os.getenv("LOG_FILE", str(BASE_DIR / "logs" / "smarthirepro.log"))
