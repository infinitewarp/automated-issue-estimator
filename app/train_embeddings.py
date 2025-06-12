# train_embeddings.py
import json
import numpy as np
from alive_progress import alive_bar
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from utils import get_embedding, save_model, VEC_OUT, META_OUT, MODEL_OUT, LABELS_OUT

DATA_PATH = "stories.json"


def train_embeddings():
    with open(DATA_PATH, "r") as f:
        stories = json.load(f)

    vectors, labels, metadata = [], [], []

    with alive_bar(len(stories)) as bar:
        for story in stories:
            text = story["title"].strip().rstrip(".?!") + ". " + story["description"]
            emb = get_embedding(text)
            vectors.append(emb)
            labels.append(story["size"])
            metadata.append(
                {
                    "id": story["id"],
                    "title": story["title"],
                    "size": story["size"],
                    "raw": text,
                }
            )
            bar.text = story["id"]
            bar()

    vectors = np.array(vectors)

    if np.any(np.all(vectors == 0, axis=1)):
        print(
            "⚠️ Warning: Some embeddings are zero vectors and will be excluded from training!"
        )
        # Filter out problematic zero vectors for training.
        # It's unclear when or how this could happen with current training data.
        non_zero_mask = ~np.all(vectors == 0, axis=1)
        vectors = vectors[non_zero_mask]
        labels = [label for i, label in enumerate(labels) if non_zero_mask[i]]
        metadata = [meta for i, meta in enumerate(metadata) if non_zero_mask[i]]
        print(
            f"Removed {len(non_zero_mask) - np.sum(non_zero_mask)} zero-vector stories."
        )

    encoder = LabelEncoder()
    y = encoder.fit_transform(labels)

    clf = RandomForestClassifier(
        n_estimators=100,
        max_depth=None,
        min_samples_split=2,
        random_state=42,
        class_weight="balanced",
    )
    clf.fit(vectors, y)

    # Save all relevant artifacts
    save_model(vectors, VEC_OUT)
    save_model(metadata, META_OUT)
    save_model(clf, MODEL_OUT)
    save_model(encoder, LABELS_OUT)

    print(
        f"\n✅ Trained and saved model using {len(vectors)} embeddings for {len(stories)} stories."
    )
