import torch
import numpy as np
import joblib
from tqdm import tqdm
from datasets import load_from_disk
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

print("📂 Loading preprocessed resumes...")
ds = load_from_disk("../data/processed_resumes_en")
texts = ds["train"]["clean_text"]
labels = ds["train"]["label"]


print("Training TF-IDF + Logistic Regression model")
vectorizer = TfidfVectorizer(max_features=5000) #ransforms text into numbers based on how important a word is to a document in a corpus
X = vectorizer.fit_transform(texts)
clf = LogisticRegression(max_iter=1000)
clf.fit(X, labels)
tfidf_preds = clf.predict_proba(X) #instead of just giving one label, it outputs a probability for each class

print("Fine-tuning BERT model")
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased") #BERT can’t work with raw text; it only understands token IDs plus attention masks
model = AutoModelForSequenceClassification.from_pretrained(
    "bert-base-uncased",
    num_labels=len(set(labels))
)

inputs = tokenizer(texts, padding=True, truncation=True, return_tensors="pt", max_length=256)
labels_tensor = torch.tensor(labels)
optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)

model.train()
for epoch in range(1): 
    optimizer.zero_grad()
    outputs = model(**inputs, labels=labels_tensor)
    loss = outputs.loss
    loss.backward()
    optimizer.step()
    print(f"Epoch {epoch+1}, Loss: {loss.item()}")


model.eval()
with torch.no_grad():
    outputs = model(**inputs)
    bert_probs = torch.nn.functional.softmax(outputs.logits, dim=-1).numpy()

final_preds = 0.5 * tfidf_preds + 0.5 * bert_probs #ensembelling - each model 50 percent weight
final_labels = np.argmax(final_preds, axis=1)

print("Ensemble model accuracy:", accuracy_score(labels, final_labels))

print("Saving models")
joblib.dump(vectorizer, "../models/tfidf_vectorizer.pkl")
joblib.dump(clf, "../models/logreg_model.pkl")
model.save_pretrained("../models/bert_model")
tokenizer.save_pretrained("../models/bert_tokenizer")

print("Training complete. Models saved in '../models/'") 