# train_embeddings.py
import json
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report
from sklearn.linear_model import LogisticRegression
from utils import get_embedding, save_pickle

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
    encoder = LabelEncoder()
    y = encoder.fit_transform(labels)

    clf = LogisticRegression(max_iter=1000)
    clf.fit(vectors, y)

    # Save all relevant artifacts
    save_pickle(vectors, VEC_OUT)
    save_pickle(metadata, META_OUT)
    save_pickle(clf, MODEL_OUT)
    save_pickle(encoder, LABELS_OUT)

    print(f"\n✅ Trained and saved model using {len(vectors)} stories.")

if __name__ == "__main__":
    main()
