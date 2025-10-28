import os
import joblib
from datasets import load_from_disk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

DATA_PATH = "../data/processed_resumes_en"
MODEL_DIR = "../models"

def train_vectorizer():
    print("Loading preprocessed resume text...")
    try:
        ds = load_from_disk(DATA_PATH)
        texts = ds["train"]["clean_text"]
        filenames = ds["train"]["original_file"]
    except FileNotFoundError:
        print(f"Error: Could not find dataset at {DATA_PATH}")
        print("Please run the 'simple_extractor.py' script first.")
        return
    except KeyError:
        print("Error: Dataset does not contain 'clean_text' or 'original_file' columns.")
        print("Please ensure 'simple_extractor.py' ran correctly.")
        return

    print(f"Found {len(texts)} resumes. Training TF-IDF model...")
    
    vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')
    
    resume_tfidf_matrix = vectorizer.fit_transform(texts)

    print(f"Saving models to {MODEL_DIR}...")
    os.makedirs(MODEL_DIR, exist_ok=True)
    
    joblib.dump(vectorizer, os.path.join(MODEL_DIR, "tfidf_vectorizer.pkl"))
    
    joblib.dump(resume_tfidf_matrix, os.path.join(MODEL_DIR, "resume_tfidf_matrix.pkl"))
    
    joblib.dump(filenames, os.path.join(MODEL_DIR, "resume_filenames.pkl"))
    
    print("Training complete. Vectorizer and resume matrix saved.")

def find_top_matches(job_description, top_n=5):
    print(f"\nFinding top {top_n} matches for job description...")
    
    try:
        vectorizer = joblib.load(os.path.join(MODEL_DIR, "tfidf_vectorizer.pkl"))
        resume_tfidf_matrix = joblib.load(os.path.join(MODEL_DIR, "resume_tfidf_matrix.pkl"))
        filenames = joblib.load(os.path.join(MODEL_DIR, "resume_filenames.pkl"))
    except FileNotFoundError:
        print(f"Error: Could not load models from {MODEL_DIR}")
        print("Please run the 'train_vectorizer()' function first (e.g., run this script).")
        return

    job_vec = vectorizer.transform([job_description])
    
    sim_scores = cosine_similarity(job_vec, resume_tfidf_matrix)
    
    top_indices = sim_scores[0].argsort()[-top_n:][::-1]

    print("Top Matches Found")
    for i in top_indices:
        score = sim_scores[0][i]
        filename = filenames[i]
        print(f"  - Score: {score:.4f}  |  File: {filename}")

if __name__ == "__main__":
    train_vectorizer()
    
    SAMPLE_JOB_DESCRIPTION = """
    We are seeking a proactive Software Engineer to design, develop, and maintain 
    our Python-based applications. The ideal candidate will have strong experience 
    with web frameworks like Django or Flask, cloud platforms such as AWS, 
    and a good understanding of machine learning principles.
    """
    
    find_top_matches(SAMPLE_JOB_DESCRIPTION, top_n=5)