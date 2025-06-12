# utils.py
from summarize_issue import generate_user_story

import numpy as np
import joblib
from pathlib import Path
from sentence_transformers import SentenceTransformer

models_dir = Path(__file__).resolve().parent.parent / 'models'

VEC_OUT = "story_vectors.data.gz"
META_OUT = "story_metadata.data.gz"
MODEL_OUT = "classifier.data.gz"
LABELS_OUT = "label_encoder.data.gz"

known_models = {
    "bge-base": ("BAAI/bge-base-en-v1.5", 768),
    "bge-large": ("BAAI/bge-large-en", 1024),
    "mini-lm": ("sentence-transformers/all-MiniLM-L6-v2", 384),
}
preferred_model = "bge-large"

_model, _size = None, None

def get_model():
    global _model
    global _size
    if _model and _size:
        return _model, _size

    name, _size = known_models[preferred_model]
    _model = SentenceTransformer(name)
    return _model, _size


max_text_len = 1024  # arbitrary, but try 512-2048

def get_embedding(text: str):
    model, size = get_model()
    cleaned = text.strip().replace("\r\n", " ").replace("\n", " ").replace("{", "").replace("}", "").replace("**", "")
    if len(cleaned) < 5:  # Arbitrary length threshold
        print("⚠️ Warning: cleaned text is very short; returning zeros!")
        return np.zeros(size)
    reinterpreted = generate_user_story(text)
    truncated = reinterpreted[:max_text_len]  # crude safety cutoff
    try:
        emb = model.encode(truncated, normalize_embeddings=True)
        if np.all(emb == 0):  # Check if the embedding is a zero vector
            raise ValueError("Embedding is zero vector")
        return np.array(emb)
    except Exception as e:
        print("🛑 Embedding error:", e)
        return np.zeros(size)

def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def save_model(data, filename):
    joblib.dump(data, models_dir/filename)

def load_model(filename):
    return joblib.load(models_dir/filename)
