# utils.py
import subprocess
import pickle
import numpy as np

def get_embedding(text: str):
    try:
        result = subprocess.run(
            ["ollama", "run", "nomic-embed-text", text],
            capture_output=True, text=True, check=True
        )
        output = result.stdout.strip()
        return np.array(eval(output))  # expecting output like "[0.12, 0.34, ...]"
    except Exception as e:
        print("Failed to get embedding:", e)
        return np.zeros(384)  # fallback size

def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def save_pickle(data, filename):
    with open(filename, 'wb') as f:
        pickle.dump(data, f)

def load_pickle(filename):
    with open(filename, 'rb') as f:
        return pickle.load(f)
