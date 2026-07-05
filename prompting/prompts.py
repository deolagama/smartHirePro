"""
SmartHirePro - Prompt Templates
=================================
All LLM prompts are centralised here for easy tuning and versioning.
No business logic lives in this file — only string templates.

Templates use Python str.format() compatible placeholders.
"""

# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """\
You are an expert AI talent acquisition specialist with deep knowledge of technical \
hiring, skills assessment, and workforce analytics. You analyse candidate resumes \
against job descriptions with precision, fairness, and actionable insight.

Your evaluations are:
- Objective and evidence-based (cite specific resume content)
- Structured and consistent across all candidates
- Free from bias related to gender, ethnicity, age, or nationality
- Focused on hard and soft skills relevant to the role

Always respond with valid JSON matching the schema provided.
"""

# ---------------------------------------------------------------------------
# Main evaluation prompt
# ---------------------------------------------------------------------------

EVALUATION_PROMPT_TEMPLATE = """\
## Job Description
{job_description}

---

## Candidate Resume (ID: {resume_id})
{resume_context}

---

## Your Task
Evaluate this candidate against the job description and return a structured JSON \
analysis. Be specific — quote or paraphrase concrete evidence from the resume.

Return **only** a JSON object (no markdown fences, no commentary) with exactly \
this schema:

{{
  "resume_id": "{resume_id}",
  "candidate_score": <integer 0-100>,
  "matching_skills": [<string>, ...],
  "missing_skills": [<string>, ...],
  "strengths": [<string>, ...],
  "weaknesses": [<string>, ...],
  "explanation": "<2-4 sentence narrative explaining the score>",
  "hiring_recommendation": "<STRONG_YES | YES | MAYBE | NO | STRONG_NO>",
  "interview_questions": [<string>, ...]
}}

Scoring rubric:
  90-100 → Exceptional match, exceed requirements
  75-89  → Strong match, minor gaps
  60-74  → Moderate match, some important gaps
  40-59  → Weak match, significant gaps
  0-39   → Poor match, does not meet requirements

Interview questions: generate 5 targeted questions based on the candidate's \
specific background and the role's requirements (mix technical + behavioural).
"""

# ---------------------------------------------------------------------------
# Batch summary prompt
# ---------------------------------------------------------------------------

BATCH_SUMMARY_PROMPT_TEMPLATE = """\
You have evaluated {n_candidates} candidates for the following role:

{job_description}

Here are the individual evaluations (JSON array):
{evaluations_json}

---

Return a concise hiring summary as valid JSON with this schema:

{{
  "total_evaluated": {n_candidates},
  "recommended_candidates": [<resume_id>, ...],
  "top_candidate": "<resume_id>",
  "summary": "<3-5 sentence executive summary for the hiring manager>",
  "key_talent_gaps": [<skill or attribute missing across most candidates>, ...]
}}
"""
