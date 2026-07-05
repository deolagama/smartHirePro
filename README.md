# Smart Hire Pro 🧠

> **AI-powered Resume Intelligence Platform · RAG Architecture · Production-Ready**

SmartHirePro v2 is a complete **Retrieval-Augmented Generation (RAG)** system that intelligently screens, ranks, and evaluates resumes against a job description. It replaces the previous TF-IDF/SBERT/GloVe pipeline with a modern, LLM-driven architecture suitable for an **AI Engineer portfolio**.

---

## ✨ What's New in v2

| Feature | v1 (Old) | v2 (New) |
|---|---|---|
| Ranking | TF-IDF + SBERT + GloVe + RF | RAG (Retrieve → Reason) |
| Output | Similarity score only | Score + Skills + Strengths + Weaknesses + Explanation + Interview Qs |
| LLM | None | GPT-4o-mini / Llama 3 |
| Vector DB | None | FAISS or ChromaDB |
| Architecture | Single scripts | Clean modular architecture |
| API | None | FastAPI REST API |
| Frontend | None | Streamlit dashboard |

---

## 🏗️ Architecture

```
PDF Resumes
     │
     ▼
┌──────────────────────────────────────────────────────────────┐
│                        INGESTION                             │
│  PDFExtractor (PyMuPDF) → Language Filter → ResumeDocument  │
└─────────────────────────────┬────────────────────────────────┘
                              │
                              ▼
┌──────────────────────────────────────────────────────────────┐
│                        CHUNKING                              │
│  SectionChunker (header/experience/skills/…) → ResumeChunk  │
│  ↳ Falls back to SlidingWindowChunker if no sections found  │
└─────────────────────────────┬────────────────────────────────┘
                              │
                              ▼
┌──────────────────────────────────────────────────────────────┐
│                      EMBEDDING                               │
│  SentenceTransformers (all-MiniLM-L6-v2) → float32 vectors  │
└─────────────────────────────┬────────────────────────────────┘
                              │
                              ▼
┌──────────────────────────────────────────────────────────────┐
│                     VECTOR STORE                             │
│  FAISSVectorStore or ChromaVectorStore (persistent)          │
└─────────────────────────────┬────────────────────────────────┘
                              │ (at query time)
                              ▼
┌──────────────────────────────────────────────────────────────┐
│                      RETRIEVAL                               │
│  Embed job description → ANN search → Group by resume →      │
│  Rank by weighted similarity → CandidateContext              │
└─────────────────────────────┬────────────────────────────────┘
                              │
                              ▼
┌──────────────────────────────────────────────────────────────┐
│                  LLM EVALUATION (RAG)                        │
│  System prompt + resume context + job description            │
│  → GPT-4o-mini / Llama 3                                    │
│  → Structured JSON: score, skills, strengths, weaknesses,   │
│     explanation, recommendation, interview questions         │
└─────────────────────────────┬────────────────────────────────┘
                              │
                              ▼
          ┌───────────────────┴───────────────────┐
          │                                       │
    FastAPI REST API                   Streamlit Dashboard
    POST /api/v1/ingest               Drag-and-drop PDF upload
    POST /api/v1/evaluate             Candidate cards + scoring
    GET  /api/v1/health               Skill pills + interview Qs
```

---

## 📂 Project Structure

```
smartHirePro/
│
├── config/
│   └── settings.py          # Centralised config (env vars + .env)
│
├── utils/
│   └── logger.py            # Rotating file + console logger
│
├── ingestion/
│   └── ingestor.py          # PDF extraction, HF dataset, language filter
│
├── chunking/
│   └── chunker.py           # Section-aware + sliding-window chunker
│
├── embedding/
│   └── embedder.py          # SentenceTransformers + OpenAI embedding engine
│
├── vector_store/
│   └── store.py             # FAISS + ChromaDB backends, factory
│
├── retrieval/
│   └── retriever.py         # ANN search, resume grouping, ranked context
│
├── prompting/
│   └── prompts.py           # All LLM prompt templates
│
├── llm/
│   └── client.py            # OpenAI + Ollama clients, JSON parsing
│
├── evaluation/
│   └── evaluator.py         # RAG pipeline orchestrator, EvaluationResult
│
├── pipeline/
│   └── run_ingestion.py     # CLI ingestion pipeline runner
│
├── api/
│   └── main.py              # FastAPI application
│
├── frontend/
│   └── app.py               # Streamlit dashboard
│
├── src/
│   ├── preprocess_data.py   # (Legacy v1 – kept for reference)
│   └── train_model.py       # (Legacy v1 – kept for reference)
│
├── data/
│   ├── raw_resumes/         # Place PDF files here for local ingestion
│   └── vector_store/        # Auto-generated FAISS/Chroma index
│
├── logs/                    # Auto-generated log files
├── .env.example             # Environment variable template
├── requirements.txt
└── README.md
```

---

## ⚙️ Setup

### 1. Clone & create virtual environment

```bash
git clone https://github.com/deolagama/smartHirePro.git
cd smartHirePro
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Configure environment

```bash
cp .env.example .env
# Edit .env and set OPENAI_API_KEY (or configure Ollama)
```

### 4. Create required directories

```bash
mkdir -p data/raw_resumes data/vector_store logs
```

---

## ▶️ Running the System

### Step 1: Ingest Resumes

**Option A — Local PDF files:**
```bash
# Place your PDFs in data/raw_resumes/
python -m pipeline.run_ingestion --source local --dir data/raw_resumes
```

**Option B — HuggingFace dataset:**
```bash
huggingface-cli login
python -m pipeline.run_ingestion --source huggingface
```

### Step 2: Start the API

```bash
uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
```

API docs: http://localhost:8000/docs

### Step 3: Start the Frontend

```bash
streamlit run frontend/app.py
```

Dashboard: http://localhost:8501

---

## 🔌 API Reference

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/api/v1/health` | API health + store count |
| `POST` | `/api/v1/ingest` | Upload PDF resumes (multipart) |
| `POST` | `/api/v1/evaluate` | Evaluate candidates for a job |
| `GET` | `/api/v1/store/stats` | Vector store statistics |
| `DELETE` | `/api/v1/store/clear` | Clear all vectors |

### Evaluate Request Body

```json
{
  "job_description": "We are hiring a Senior Python Engineer...",
  "top_k": 5,
  "include_summary": true
}
```

### Evaluation Response (per candidate)

```json
{
  "resume_id": "john_doe.pdf",
  "candidate_score": 87,
  "matching_skills": ["Python", "FastAPI", "PostgreSQL", "Docker"],
  "missing_skills": ["Kubernetes", "Terraform"],
  "strengths": ["5 years of backend Python experience", "Led ML platform team"],
  "weaknesses": ["No cloud infrastructure experience mentioned"],
  "explanation": "John demonstrates strong backend skills aligned with the role...",
  "hiring_recommendation": "STRONG_YES",
  "interview_questions": [
    "Describe your experience designing microservices with FastAPI.",
    "How have you handled database migrations in production?",
    ...
  ],
  "retrieval_score": 0.8921
}
```

---

## 🔧 Configuration

All settings are controlled via `.env`:

| Variable | Default | Description |
|----------|---------|-------------|
| `LLM_PROVIDER` | `openai` | `openai` or `ollama` |
| `OPENAI_API_KEY` | — | Your OpenAI API key |
| `OPENAI_MODEL` | `gpt-4o-mini` | OpenAI model name |
| `OLLAMA_MODEL` | `llama3` | Ollama model tag |
| `EMBEDDING_MODEL` | `all-MiniLM-L6-v2` | SentenceTransformers model |
| `VECTOR_STORE_TYPE` | `faiss` | `faiss` or `chroma` |
| `CHUNK_SIZE` | `512` | Tokens per chunk |
| `CHUNK_OVERLAP` | `64` | Overlap between chunks |
| `TOP_K_CHUNKS` | `10` | Chunks retrieved per query |
| `TOP_K_RESUMES` | `5` | Candidates returned per query |

---

## 🚀 Use with Local Llama (Ollama)

```bash
# Install Ollama: https://ollama.com
ollama pull llama3
ollama serve

# Set in .env:
# LLM_PROVIDER=ollama
# OLLAMA_MODEL=llama3
```

---

## 👩‍💻 Author

**Deola Gama** — AI Engineer  
Building production-grade AI systems for modern recruitment.

---

⭐ Star this repo if it helped you build your AI portfolio!
