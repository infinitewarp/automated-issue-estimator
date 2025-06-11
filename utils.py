# utils.py
import numpy as np
import pickle
from sentence_transformers import SentenceTransformer

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
    truncated = cleaned[:max_text_len]  # crude safety cutoff
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

def save_pickle(data, filename):
    with open(filename, 'wb') as f:
        pickle.dump(data, f)

def load_pickle(filename):
    with open(filename, 'rb') as f:
        return pickle.load(f)
