import os
import joblib
import numpy as np
from datasets import load_from_disk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans

DATA_PATH = "../data/processed_resumes_en"
MODEL_DIR = "../models"
GLOVE_PATH = "glove.6B.100d.txt"

def load_resumes():
    print("Loading resumes from disk...")
    ds = load_from_disk(DATA_PATH)
    texts = ds["train"]["clean_text"]
    filenames = ds["train"]["original_file"]
    print(f"Loaded {len(texts)} resumes.")
    return texts, filenames

def train_tfidf(resumes):
    print("Training TF-IDF...")
    vectorizer = TfidfVectorizer(max_features=5000, stop_words="english")
    resume_tfidf_matrix = vectorizer.fit_transform(resumes)
    
    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(vectorizer, os.path.join(MODEL_DIR, "tfidf_vectorizer.pkl"))
    joblib.dump(resume_tfidf_matrix, os.path.join(MODEL_DIR, "resume_tfidf_matrix.pkl"))
    print("TF-IDF training complete.")
    return vectorizer, resume_tfidf_matrix

def tfidf_score(job_description, vectorizer, resume_tfidf_matrix):
    job_vec = vectorizer.transform([job_description])
    sim = cosine_similarity(job_vec, resume_tfidf_matrix).flatten()
    return sim

def sbert_score(job_description, resumes):
    print("Computing SBERT embeddings...")
    model = SentenceTransformer("all-mpnet-base-v2")
    resume_vec = model.encode(resumes, show_progress_bar=True)
    job_vec = model.encode([job_description])
    sim = cosine_similarity(job_vec, resume_vec).flatten()
    print("SBERT done.")
    return sim
def load_glove(path=GLOVE_PATH):
    print("Loading GloVe vectors...")
    glove = {}
    with open(path, encoding="utf8") as f:
        for line in f:
            w, *v = line.split()
            glove[w] = np.array(v, dtype=float)
    print("GloVe loaded.")
    return glove

def embed_glove(text, glove):
    words = text.lower().split()
    vecs = [glove[w] for w in words if w in glove]
    return np.mean(vecs, axis=0) if vecs else np.zeros(100)

def glove_score(job_description, resumes, glove):
    print("Computing GloVe similarity...")
    job_vec = embed_glove(job_description, glove)
    resume_vec = np.array([embed_glove(r, glove) for r in resumes])
    sim = cosine_similarity([job_vec], resume_vec).flatten()
    print("GloVe done.")
    return sim

def rf_score(resumes):
    print("Training Random Forest on SBERT embeddings...")
    model = SentenceTransformer("all-mpnet-base-v2")
    resume_vec = model.encode(resumes, show_progress_bar=True)
    
    km = KMeans(n_clusters=2, random_state=42).fit(resume_vec)
    labels = km.labels_
    
    rf = RandomForestClassifier()
    rf.fit(resume_vec, labels)
    pred = rf.predict_proba(resume_vec)[:,1]
    print("Random Forest done.")
    return pred

def rank_resumes(job_description, top_n=5):
    resumes, filenames = load_resumes()
    vectorizer, resume_tfidf_matrix = train_tfidf(resumes)
    glove = load_glove()

    tfidf_s = tfidf_score(job_description, vectorizer, resume_tfidf_matrix)
    sbert_s = sbert_score(job_description, resumes)
    glove_s = glove_score(job_description, resumes, glove)
    rf_s = rf_score(resumes)

    scores = np.mean([tfidf_s, sbert_s, glove_s, rf_s], axis=0)

    results = []
    for idx, score in enumerate(scores):
        results.append({"resume_id": filenames[idx], "final_score": score})

    ranked = sorted(results, key=lambda x: x["final_score"], reverse=True)
    
    print("\n=== Top Matches ===")
    for i, r in enumerate(ranked[:top_n]):
        print(f"{i+1}. Resume: {r['resume_id']}  |  Score: {r['final_score']:.4f}")

if __name__ == "__main__":
    SAMPLE_JOB_DESCRIPTION = """
    We are seeking a proactive Software Engineer to design, develop, and maintain 
    our Python-based applications. The ideal candidate will have strong experience 
    with web frameworks like Django or Flask, cloud platforms such as AWS, 
    and a good understanding of machine learning principles.
    """
    
    rank_resumes(SAMPLE_JOB_DESCRIPTION, top_n=5)