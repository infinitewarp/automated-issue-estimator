# train_embeddings.py
import json
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from utils import get_embedding, save_pickle
import numpy as np

DATA_PATH = "stories.json"
VEC_OUT = "story_vectors.pkl"
META_OUT = "story_metadata.pkl"
MODEL_OUT = "classifier.pkl"
LABELS_OUT = "label_encoder.pkl"

def main():
    with open(DATA_PATH, 'r') as f:
        stories = json.load(f)

    vectors, labels, metadata = [], [], []

    for story in stories:
        text = story['title'] + " " + story['description']
        emb = get_embedding(text)
        vectors.append(emb)
        labels.append(story['size'])
        metadata.append({
            "id": story["id"],
            "title": story["title"],
            "size": story["size"],
            "raw": text
        })

    vectors = np.array(vectors)

    # Encode size labels to integers
    encoder = LabelEncoder()
    y = encoder.fit_transform(labels)

    # Train a classifier
    clf = LogisticRegression(max_iter=1000)
    clf.fit(vectors, y)

    # Save everything
    save_pickle(vectors, VEC_OUT)
    save_pickle(metadata, META_OUT)
    save_pickle(clf, MODEL_OUT)
    save_pickle(encoder, LABELS_OUT)

    print(f"Saved {len(vectors)} embeddings and classifier.")

if __name__ == "__main__":
    main()
