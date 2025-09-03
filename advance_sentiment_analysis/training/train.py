# training/train.py

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import joblib
import os

# ============================
# 1. Load dataset
# ============================
DATA_PATH = "D:\\VSCode\\Python\\ML_tp\\advance_sentiment_analysis\\data.csv"  # adjust if needed
df = pd.read_csv(DATA_PATH)

print("Dataset shape:", df.shape)
print("Sample rows:\n", df.head())

# Drop rows with missing values
df = df.dropna()

# ============================
# 2. Split features & labels
# ============================
X = df["text"]
y_sentiment = df["sentiment"]
y_ctype = df["ctype"]

# Train-test split
X_train, X_test, y_sent_train, y_sent_test = train_test_split(
    X, y_sentiment, test_size=0.2, random_state=42
)
_, _, y_ctype_train, y_ctype_test = train_test_split(
    X, y_ctype, test_size=0.2, random_state=42
)

# ============================
# 3. Vectorization
# ============================
vectorizer = TfidfVectorizer(stop_words="english", max_features=5000)
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# ============================
# 4. Train models
# ============================
# Sentiment model
sentiment_model = LogisticRegression(max_iter=1000)
sentiment_model.fit(X_train_vec, y_sent_train)

# CType model
ctype_model = LogisticRegression(max_iter=1000)
ctype_model.fit(X_train_vec, y_ctype_train)

# ============================
# 5. Evaluate
# ============================
print("\n=== Sentiment Model ===")
y_sent_pred = sentiment_model.predict(X_test_vec)
print("Accuracy:", accuracy_score(y_sent_test, y_sent_pred))
print(classification_report(y_sent_test, y_sent_pred))

print("\n=== CType Model ===")
y_ctype_pred = ctype_model.predict(X_test_vec)
print("Accuracy:", accuracy_score(y_ctype_test, y_ctype_pred))
print(classification_report(y_ctype_test, y_ctype_pred))

# ============================
# 6. Save models
# ============================
os.makedirs("../models", exist_ok=True)

joblib.dump(sentiment_model, "../models/sentiment_model.pkl")
joblib.dump(ctype_model, "../models/ctype_model.pkl")
joblib.dump(vectorizer, "../models/tfidf_vectorizer.pkl")

print("\n✅ Models and vectorizer saved in ../models/")
