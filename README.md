# Smart Hire Pro 🚀

Smart Hire Pro is an **ML-powered resume screening and ranking system** that automatically matches resumes to a given job description using multiple NLP and machine learning techniques. It is designed to help recruiters and hiring teams efficiently shortlist the most relevant candidates from a large pool of resumes.

This project demonstrates an end-to-end **resume processing + semantic matching pipeline**, starting from raw PDF resumes to ranked candidate outputs.

---

## ✨ Key Highlights

* 📄 Processes **raw PDF resumes** at scale
* 🌍 Filters resumes by **English language detection**
* 🧹 Extracts and cleans resume text using **PyMuPDF**
* 🧠 Uses **multiple scoring models** for better matching accuracy
* 📊 Produces a final **aggregated relevance score** for ranking
* 🔬 Designed for real-world recruitment and AI experimentation

---

## 🧠 Matching & Scoring Techniques Used

Smart Hire Pro combines **four independent similarity models** and averages their scores for robust ranking:

1. **TF-IDF + Cosine Similarity**

   * Captures keyword-level relevance between resumes and job descriptions

2. **SBERT (Sentence-BERT)**

   * Uses `all-mpnet-base-v2` for deep semantic understanding

3. **GloVe Embeddings**

   * Computes word-embedding-based similarity using pre-trained GloVe vectors

4. **Random Forest over SBERT Embeddings**

   * Uses KMeans clustering to generate pseudo-labels and trains a Random Forest for probabilistic relevance scoring

The final score is the **mean of all four models**, improving reliability over single-model approaches.

---

## 🗂️ Project Workflow

```
Raw Resume PDFs
      ↓
Text Extraction (PyMuPDF)
      ↓
Language Filtering (English only)
      ↓
Processed Dataset (Hugging Face Dataset)
      ↓
Feature Extraction & Embeddings
      ↓
Similarity Scoring (TF-IDF, SBERT, GloVe, RF)
      ↓
Final Resume Ranking
```

---

## 🛠️ Tech Stack

### Core Technologies

* **Python 3**
* **PyMuPDF (fitz)** – PDF text extraction
* **Hugging Face Datasets** – Dataset handling & storage
* **langdetect** – Language detection

### Machine Learning & NLP

* **scikit-learn** – TF-IDF, Random Forest, KMeans
* **Sentence-Transformers (SBERT)**
* **GloVe (100d embeddings)**
* **NumPy**

### Utilities

* **joblib** – Model persistence
* **PyArrow** – Efficient dataset storage

---

## 📂 Project Structure

```
Smart-Hire-Pro/
│── data/
│   └── processed_resumes_en/
│── models/
│   ├── tfidf_vectorizer.pkl
│   ├── resume_tfidf_matrix.pkl
│── src/
│   ├── preprocess_resumes.py
│   ├── train_and_rank.py
│── README.md
```

---

## 📥 Dataset Used

* **Source:** Hugging Face dataset – `d4rk3r/resumes-raw-pdf`
* **Format:** Raw PDF resumes
* **Processing:**

  * Text extraction from PDF
  * Language filtering (English only)
  * Saved as a Hugging Face Dataset on disk

---

## ⚙️ Installation & Setup

### 1️⃣ Clone the Repository

```bash
git clone https://github.com/deolagama/smartHirePro.git
cd smart-hire-pro
```

### 2️⃣ Create Virtual Environment

```bash
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
```

### 3️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

### 4️⃣ Login to Hugging Face (Required)

```bash
huggingface-cli login
```

### 5️⃣ Download GloVe Embeddings

Download **glove.6B.100d.txt** and place it in the project root or update the path in code.

---

## ▶️ Running the Project

### Step 1: Preprocess Resumes

```bash
python src/preprocess_resumes.py
```

* Extracts text from PDFs
* Filters English resumes
* Saves processed dataset to disk

### Step 2: Rank Resumes for a Job Description

```bash
python src/train_and_rank.py
```

* Computes similarity scores
* Outputs top-ranked resumes with relevance scores

---

## 📊 Sample Output

```
=== Top Matches ===
1. Resume: john_doe.pdf   | Score: 0.8421
2. Resume: jane_smith.pdf | Score: 0.8164
3. Resume: alex_k.pdf     | Score: 0.8012
```

---

## 🚀 Future Enhancements

* 🧠 Supervised learning with labeled resume-job matches
* 📌 Skill extraction & weighting
* 🌐 Web-based recruiter dashboard (Flask/React)
* 📧 Automated shortlisting notifications
* 🗄️ Resume database indexing for faster retrieval

---

## 🎯 Use Cases

* ML-based resume shortlisting
* Recruitment automation systems
* NLP research & experimentation
* Academic mini / major projects

---

## 👩‍💻 Author

**Deola Gama**
Full-stack developer exploring ML-driven recruitment solutions.

---

⭐ If you find this project useful, consider starring the repository!
