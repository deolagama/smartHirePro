"""
SmartHirePro - Streamlit Frontend
====================================
A rich, interactive recruiter dashboard that communicates with the
SmartHirePro FastAPI backend.

Features:
  - Upload PDF resumes via drag-and-drop
  - Enter a job description and configure evaluation parameters
  - View ranked candidate cards with scores and recommendations
  - Drill down into matching/missing skills, strengths, weaknesses
  - Read the AI-generated hiring explanation
  - Copy suggested interview questions
  - Executive batch summary panel

Run:
    streamlit run frontend/app.py
"""

import time
import json
import sys
from pathlib import Path
import requests
import streamlit as st

# ---------------------------------------------------------------------------
# Page config – MUST be the very first Streamlit call
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="SmartHirePro | AI Resume Screener",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
API_BASE = "http://localhost:8000/api/v1"

RECOMMENDATION_COLORS = {
    "STRONG_YES": "#00c853",
    "YES":        "#64dd17",
    "MAYBE":      "#ffab00",
    "NO":         "#ff6d00",
    "STRONG_NO":  "#d50000",
    "UNKNOWN":    "#9e9e9e",
}

RECOMMENDATION_LABELS = {
    "STRONG_YES": "✅ Strong Hire",
    "YES":        "👍 Hire",
    "MAYBE":      "🤔 Maybe",
    "NO":         "👎 Not Recommended",
    "STRONG_NO":  "❌ Reject",
    "UNKNOWN":    "❓ Unknown",
}

# ---------------------------------------------------------------------------
# Custom CSS
# ---------------------------------------------------------------------------
st.markdown("""
<style>
/* ── Google Fonts ── */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

/* ── Dark gradient background ── */
.stApp {
    background: linear-gradient(135deg, #0f0c29, #302b63, #24243e);
    color: #e8eaf6;
}

/* ── Hero header ── */
.hero-title {
    font-size: 3rem;
    font-weight: 700;
    background: linear-gradient(90deg, #7c4dff, #40c4ff, #00e5ff);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin-bottom: 0.2rem;
}
.hero-subtitle {
    color: #b0bec5;
    font-size: 1.1rem;
    margin-bottom: 2rem;
}

/* ── Glassmorphism cards ── */
.glass-card {
    background: rgba(255, 255, 255, 0.06);
    border: 1px solid rgba(255, 255, 255, 0.12);
    border-radius: 16px;
    padding: 1.5rem;
    backdrop-filter: blur(10px);
    margin-bottom: 1.2rem;
    transition: transform 0.2s ease, box-shadow 0.2s ease;
}
.glass-card:hover {
    transform: translateY(-2px);
    box-shadow: 0 8px 32px rgba(124, 77, 255, 0.25);
}

/* ── Score badge ── */
.score-badge {
    font-size: 2.5rem;
    font-weight: 700;
    display: inline-block;
    width: 80px;
    height: 80px;
    line-height: 80px;
    text-align: center;
    border-radius: 50%;
    background: linear-gradient(135deg, #7c4dff, #40c4ff);
    color: white;
    box-shadow: 0 4px 20px rgba(124, 77, 255, 0.5);
}

/* ── Skill pill ── */
.skill-pill-match {
    display: inline-block;
    background: rgba(0, 200, 83, 0.2);
    border: 1px solid rgba(0, 200, 83, 0.5);
    color: #69f0ae;
    border-radius: 20px;
    padding: 3px 12px;
    font-size: 0.8rem;
    margin: 3px;
}
.skill-pill-missing {
    display: inline-block;
    background: rgba(255, 109, 0, 0.2);
    border: 1px solid rgba(255, 109, 0, 0.5);
    color: #ffab40;
    border-radius: 20px;
    padding: 3px 12px;
    font-size: 0.8rem;
    margin: 3px;
}

/* ── Section header ── */
.section-header {
    font-size: 1rem;
    font-weight: 600;
    color: #90caf9;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    margin-bottom: 0.5rem;
    margin-top: 1rem;
}

/* ── Recommendation tag ── */
.rec-tag {
    display: inline-block;
    border-radius: 8px;
    padding: 4px 14px;
    font-size: 0.85rem;
    font-weight: 600;
    margin-left: 8px;
}

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: rgba(15, 12, 41, 0.8) !important;
    border-right: 1px solid rgba(255,255,255,0.08);
}

/* ── Buttons ── */
.stButton > button {
    background: linear-gradient(90deg, #7c4dff, #40c4ff);
    color: white;
    border: none;
    border-radius: 10px;
    padding: 0.5rem 1.5rem;
    font-weight: 600;
    transition: opacity 0.2s;
}
.stButton > button:hover {
    opacity: 0.85;
}

/* ── Progress bar ── */
.stProgress > div > div > div {
    background: linear-gradient(90deg, #7c4dff, #40c4ff) !important;
}

/* ── Expander ── */
.streamlit-expanderHeader {
    background: rgba(255,255,255,0.04) !important;
    border-radius: 8px !important;
}

/* ── Divider ── */
hr {
    border-color: rgba(255,255,255,0.08) !important;
}
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# API helpers
# ---------------------------------------------------------------------------

def api_health() -> dict | None:
    try:
        r = requests.get(f"{API_BASE}/health", timeout=5)
        r.raise_for_status()
        return r.json()
    except Exception:
        return None


def api_ingest(files: list) -> dict | None:
    try:
        file_tuples = [("files", (f.name, f.getvalue(), "application/pdf")) for f in files]
        r = requests.post(f"{API_BASE}/ingest", files=file_tuples, timeout=300)
        r.raise_for_status()
        return r.json()
    except requests.HTTPError as e:
        st.error(f"Ingestion API error: {e.response.text}")
        return None
    except Exception as e:
        st.error(f"Ingestion failed: {e}")
        return None


def api_evaluate(job_description: str, top_k: int, include_summary: bool) -> dict | None:
    try:
        r = requests.post(
            f"{API_BASE}/evaluate",
            json={
                "job_description": job_description,
                "top_k": top_k,
                "include_summary": include_summary,
            },
            timeout=600,
        )
        r.raise_for_status()
        return r.json()
    except requests.HTTPError as e:
        st.error(f"Evaluation API error: {e.response.text}")
        return None
    except Exception as e:
        st.error(f"Evaluation failed: {e}")
        return None


def api_clear_store() -> bool:
    try:
        r = requests.delete(f"{API_BASE}/store/clear", timeout=30)
        r.raise_for_status()
        return True
    except Exception as e:
        st.error(f"Clear failed: {e}")
        return False


# ---------------------------------------------------------------------------
# UI helpers
# ---------------------------------------------------------------------------

def score_color(score: int) -> str:
    if score >= 80:
        return "#00c853"
    elif score >= 60:
        return "#64dd17"
    elif score >= 40:
        return "#ffab00"
    else:
        return "#ff6d00"


def render_skill_pills(skills: list[str], pill_class: str) -> str:
    return "".join(f'<span class="{pill_class}">{s}</span>' for s in skills)


def render_candidate_card(result: dict, rank: int):
    rec = result.get("hiring_recommendation", "UNKNOWN")
    score = result.get("candidate_score", 0)
    resume_id = result.get("resume_id", "Unknown")

    rec_color = RECOMMENDATION_COLORS.get(rec, "#9e9e9e")
    rec_label = RECOMMENDATION_LABELS.get(rec, rec)

    with st.container():
        st.markdown(f"""
        <div class="glass-card">
            <div style="display:flex; align-items:center; gap:1rem; flex-wrap:wrap;">
                <div class="score-badge" style="background: linear-gradient(135deg, {score_color(score)}, {rec_color});">
                    {score}
                </div>
                <div style="flex:1; min-width:200px;">
                    <div style="font-size:1.2rem; font-weight:600; color:#e8eaf6;">
                        #{rank} &nbsp; {resume_id}
                    </div>
                    <div style="margin-top:4px;">
                        <span class="rec-tag" style="background:{rec_color}22; border:1px solid {rec_color}; color:{rec_color};">
                            {rec_label}
                        </span>
                        <span style="color:#78909c; font-size:0.85rem; margin-left:12px;">
                            Vector score: {result.get('retrieval_score', 0):.3f}
                        </span>
                    </div>
                </div>
                <div style="text-align:right; min-width:120px;">
                    <div style="font-size:0.8rem; color:#78909c;">Candidate Score</div>
                    <div style="font-size:1.8rem; font-weight:700; color:{score_color(score)};">{score}/100</div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        with st.expander(f"📋 Full Analysis — {resume_id}", expanded=(rank == 1)):
            # Score bar
            st.progress(score / 100)

            col_a, col_b = st.columns(2)

            with col_a:
                st.markdown('<div class="section-header">✅ Matching Skills</div>', unsafe_allow_html=True)
                matching = result.get("matching_skills", [])
                if matching:
                    st.markdown(render_skill_pills(matching, "skill-pill-match"), unsafe_allow_html=True)
                else:
                    st.caption("No matching skills detected.")

                st.markdown('<div class="section-header">💪 Strengths</div>', unsafe_allow_html=True)
                for s in result.get("strengths", []):
                    st.markdown(f"- {s}")

            with col_b:
                st.markdown('<div class="section-header">❌ Missing Skills</div>', unsafe_allow_html=True)
                missing = result.get("missing_skills", [])
                if missing:
                    st.markdown(render_skill_pills(missing, "skill-pill-missing"), unsafe_allow_html=True)
                else:
                    st.caption("No critical skills missing.")

                st.markdown('<div class="section-header">⚠️ Weaknesses</div>', unsafe_allow_html=True)
                for w in result.get("weaknesses", []):
                    st.markdown(f"- {w}")

            st.markdown('<div class="section-header">🧠 AI Explanation</div>', unsafe_allow_html=True)
            st.info(result.get("explanation", "No explanation provided."))

            st.markdown('<div class="section-header">🎤 Suggested Interview Questions</div>', unsafe_allow_html=True)
            for i, q in enumerate(result.get("interview_questions", []), start=1):
                st.markdown(f"**Q{i}.** {q}")


def render_batch_summary(summary: dict):
    st.markdown("---")
    st.markdown("""
    <div style="font-size:1.4rem; font-weight:700; color:#40c4ff; margin-bottom:1rem;">
        📊 Executive Hiring Summary
    </div>
    """, unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)
    col1.metric("Total Evaluated", summary.get("total_evaluated", 0))
    col2.metric("Top Candidate", summary.get("top_candidate", "N/A"))
    col3.metric("Recommended", len(summary.get("recommended_candidates", [])))

    st.markdown('<div class="section-header">📝 Summary</div>', unsafe_allow_html=True)
    st.success(summary.get("summary", "No summary available."))

    rec_candidates = summary.get("recommended_candidates", [])
    if rec_candidates:
        st.markdown('<div class="section-header">👥 Recommended Candidates</div>', unsafe_allow_html=True)
        st.markdown(" · ".join([f"`{c}`" for c in rec_candidates]))

    gaps = summary.get("key_talent_gaps", [])
    if gaps:
        st.markdown('<div class="section-header">🔍 Key Talent Gaps in Candidate Pool</div>', unsafe_allow_html=True)
        for g in gaps:
            st.markdown(f"- {g}")


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

with st.sidebar:
    st.markdown("""
    <div style="text-align:center; padding:1rem 0;">
        <div style="font-size:2.5rem;">🧠</div>
        <div style="font-size:1.3rem; font-weight:700; color:#40c4ff;">SmartHirePro</div>
        <div style="color:#78909c; font-size:0.85rem;">RAG Resume Intelligence</div>
    </div>
    """, unsafe_allow_html=True)
    st.divider()

    # API status
    health = api_health()
    if health:
        st.success("🟢 API Connected")
        st.caption(f"Vector Store: **{health.get('vector_store_count', 0)}** chunks indexed")
        st.caption(f"LLM Provider: **{health.get('provider', 'unknown').upper()}**")
    else:
        st.error("🔴 API Offline")
        st.caption("Start the API: `uvicorn api.main:app --reload`")

    st.divider()
    st.markdown("### ⚙️ Settings")
    top_k = st.slider("Max Candidates to Evaluate", min_value=1, max_value=15, value=5)
    include_summary = st.toggle("Include Executive Summary", value=True)

    st.divider()
    st.markdown("### 🗄️ Vector Store")
    if st.button("🗑️ Clear Store", use_container_width=True, type="secondary"):
        if api_clear_store():
            st.success("Store cleared!")
            st.rerun()

    st.divider()
    st.markdown("""
    <div style="color:#546e7a; font-size:0.75rem; text-align:center;">
    SmartHirePro v2.0 · RAG Architecture<br/>
    Built with LangChain · FAISS · SentenceTransformers
    </div>
    """, unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Main content
# ---------------------------------------------------------------------------

st.markdown('<div class="hero-title">SmartHirePro</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="hero-subtitle">AI-Powered Resume Intelligence · RAG Architecture</div>',
    unsafe_allow_html=True,
)

tab_ingest, tab_evaluate = st.tabs(["📥 Ingest Resumes", "🔍 Evaluate Candidates"])

# ── Tab 1: Ingest ──────────────────────────────────────────────────────────

with tab_ingest:
    st.markdown("### Upload Resume PDFs")
    st.markdown(
        "Drop one or more PDF resumes below. They will be chunked, embedded, "
        "and stored in the vector database for retrieval."
    )

    uploaded_files = st.file_uploader(
        "Upload PDFs",
        type=["pdf"],
        accept_multiple_files=True,
        label_visibility="collapsed",
    )

    if uploaded_files:
        st.markdown(f"**{len(uploaded_files)} file(s) selected:**")
        for f in uploaded_files:
            st.markdown(f"- 📄 `{f.name}` ({f.size / 1024:.1f} KB)")

        if st.button("🚀 Ingest Resumes", use_container_width=True):
            if health is None:
                st.error("API is offline. Cannot ingest.")
            else:
                with st.spinner("Ingesting resumes… this may take a minute."):
                    result = api_ingest(uploaded_files)

                if result and result.get("success"):
                    st.balloons()
                    col1, col2, col3 = st.columns(3)
                    col1.metric("✅ Resumes Ingested", result["resumes_ingested"])
                    col2.metric("📦 Chunks Stored", result["chunks_stored"])
                    col3.metric("⏱️ Time", f"{result['elapsed_seconds']}s")
                    st.success(result["message"])
    else:
        st.markdown("""
        <div class="glass-card" style="text-align:center; padding:3rem;">
            <div style="font-size:3rem;">📂</div>
            <div style="color:#78909c;">Drag and drop PDF files here</div>
        </div>
        """, unsafe_allow_html=True)

# ── Tab 2: Evaluate ────────────────────────────────────────────────────────

with tab_evaluate:
    st.markdown("### Job Description")
    st.markdown("Paste the job description below. The AI will retrieve and evaluate the best matching candidates.")

    job_description = st.text_area(
        "Job Description",
        height=220,
        placeholder=(
            "We are hiring a Senior AI/ML Engineer with 5+ years of experience in Python, "
            "LLMs, and MLOps. The ideal candidate has hands-on experience with LangChain, "
            "vector databases (FAISS, Pinecone, or Weaviate), RAG architectures, and "
            "deploying models to production using FastAPI and Docker.\n\n"
            "Preferred: OpenAI API, HuggingFace, PyTorch, AWS SageMaker."
        ),
        label_visibility="collapsed",
    )

    col_btn, col_info = st.columns([1, 3])
    with col_btn:
        evaluate_clicked = st.button(
            "🔍 Evaluate Candidates",
            use_container_width=True,
            disabled=(not job_description or len(job_description) < 50),
        )

    with col_info:
        if job_description and len(job_description) < 50:
            st.warning("Please enter at least 50 characters for the job description.")

    if evaluate_clicked:
        if health is None:
            st.error("API is offline. Cannot evaluate.")
        elif health.get("vector_store_count", 0) == 0:
            st.warning("The vector store is empty. Please ingest resumes first.")
        else:
            with st.spinner("Retrieving and evaluating candidates… this may take a few minutes."):
                evaluation = api_evaluate(job_description, top_k, include_summary)

            if evaluation and evaluation.get("success"):
                results = evaluation.get("results", [])
                summary = evaluation.get("summary")

                st.markdown("---")
                col_m1, col_m2, col_m3 = st.columns(3)
                col_m1.metric("Candidates Evaluated", evaluation.get("candidates_evaluated", 0))
                col_m2.metric(
                    "Top Score",
                    f"{results[0]['candidate_score']}/100" if results else "N/A"
                )
                col_m3.metric("Time Taken", f"{evaluation.get('elapsed_seconds', 0):.1f}s")

                st.markdown("### 🏆 Candidate Rankings")
                for rank, result in enumerate(results, start=1):
                    render_candidate_card(result, rank)

                if summary and include_summary:
                    render_batch_summary(summary)

                # Export
                st.markdown("---")
                st.markdown("### 📤 Export Results")
                export_data = {
                    "job_description": job_description,
                    "results": results,
                    "summary": summary,
                }
                st.download_button(
                    label="⬇️ Download JSON Report",
                    data=json.dumps(export_data, indent=2),
                    file_name="smarthirepro_evaluation.json",
                    mime="application/json",
                )
