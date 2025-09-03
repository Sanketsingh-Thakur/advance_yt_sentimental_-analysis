# app/utils.py

import re
import joblib
import os
from youtube_comment_downloader import YoutubeCommentDownloader

# Path to models
MODEL_PATH = "../models/"

def load_models():
    """Load sentiment model, ctype model, and vectorizer."""
    sentiment_model = joblib.load(os.path.join(MODEL_PATH, "sentiment_model.pkl"))
    ctype_model = joblib.load(os.path.join(MODEL_PATH, "ctype_model.pkl"))
    vectorizer = joblib.load(os.path.join(MODEL_PATH, "tfidf_vectorizer.pkl"))
    return sentiment_model, ctype_model, vectorizer


def clean_text(text: str) -> str:
    """Basic text preprocessing."""
    text = text.lower()
    text = re.sub(r"http\S+", "", text)  # remove URLs
    text = re.sub(r"@\w+", "", text)     # remove mentions
    text = re.sub(r"[^a-z\s]", "", text) # remove non-alphabets
    text = re.sub(r"\s+", " ", text).strip()
    return text


def fetch_comments(video_url: str, limit: int = 50):
    """
    Fetch YouTube comments using youtube-comment-downloader.
    Returns a list of comment texts.
    """
    downloader = YoutubeCommentDownloader()
    comments_gen = downloader.get_comments_from_url(video_url, sort_by=0)  # 0 = popular, 1 = recent

    comments = []
    for c in comments_gen:
        if "text" in c:
            comments.append(c["text"])
        if len(comments) >= limit:
            break
    return comments
