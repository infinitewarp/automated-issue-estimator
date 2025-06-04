# train_embeddings.py
import json
import os
import numpy as np
from utils import get_embedding, save_pickle

DATA_PATH = "stories.json"        # expects list of dicts with id, title, description, size
EMBEDDING_OUTPUT = "story_vectors.pkl"
METADATA_OUTPUT = "story_metadata.pkl"

def main():
    with open(DATA_PATH, 'r') as f:
        stories = json.load(f)

    vectors, metadata = [], []
    for story in stories:
        text = story['title'] + " " + story['description']
        emb = get_embedding(text)
        vectors.append(emb)
        metadata.append({
            "id": story["id"],
            "title": story["title"],
            "size": story["size"],
            "raw": text
        })

    save_pickle(vectors, EMBEDDING_OUTPUT)
    save_pickle(metadata, METADATA_OUTPUT)
    print(f"Saved {len(vectors)} embeddings.")

if __name__ == "__main__":
    main()
