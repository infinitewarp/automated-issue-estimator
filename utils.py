# utils.py
import numpy as np
import pickle
from sentence_transformers import SentenceTransformer

# Load globally once
model = SentenceTransformer("BAAI/bge-base-en-v1.5")  # Swap with bge-large-en if desired

def get_embedding(text: str):
    cleaned = text.strip().replace("\n", " ")
    if len(cleaned) < 5:  # Arbitrary length threshold
        print("⚠️ Warning: cleaned text is very short; returning zeros!")
        return np.zeros(768)
    truncated = cleaned[:2000]  # crude safety cutoff
    try:
        emb = model.encode(truncated, normalize_embeddings=True)
        if np.all(emb == 0):  # Check if the embedding is a zero vector
            raise ValueError("Embedding is zero vector")
        return np.array(emb)
    except Exception as e:
        print("🛑 Embedding error:", e)
        return np.zeros(768)  # 768 for bge-base, 1024 for bge-large

def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def save_pickle(data, filename):
    with open(filename, 'wb') as f:
        pickle.dump(data, f)

def load_pickle(filename):
    with open(filename, 'rb') as f:
        return pickle.load(f)
