import torch
import numpy as np
import joblib
from tqdm import tqdm
from datasets import load_from_disk
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import xgboost as xgb
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

print("Loading preprocessed resumes")
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

print("Training XGBoost model on TF-IDF features")#XGBoost is a boosting algorithm, meaning it combines many weak models (small decision trees) into a strong model
booster = xgb.XGBClassifier(n_estimators=100, max_depth=6, learning_rate=0.1) #Builds 100 decision trees sequentially
booster.fit(X, labels)
boost_preds = booster.predict_proba(X)

final_preds = (0.4 * tfidf_preds) + (0.4 * bert_probs) + (0.2 * boost_preds) #ensembelling - each model w percent weight
# Resume scoring on a scale of 0-100
predicted_prob = np.max(final_preds, axis=1)  # probability of the best-matching class
resume_scores = predicted_prob * 100
print("Resume scores (0-100):", resume_scores)

print("Saving models")
joblib.dump(vectorizer, "../models/tfidf_vectorizer.pkl")
joblib.dump(clf, "../models/logreg_model.pkl")
model.save_pretrained("../models/bert_model")
tokenizer.save_pretrained("../models/bert_tokenizer")

print("Training complete. Models saved in '../models/'") 