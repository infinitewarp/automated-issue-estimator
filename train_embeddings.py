# train_embeddings.py
import json
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
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

    # Encode labels to integers
    encoder = LabelEncoder()
    y = encoder.fit_transform(labels)

    # Split for evaluation (80/20)
    X_train, X_test, y_train, y_test = train_test_split(
        vectors, y, test_size=0.2, random_state=42, stratify=y
    )

    # XGBoost classifier
    clf = XGBClassifier(
        n_estimators=100,
        learning_rate=0.1,
        use_label_encoder=False,
        eval_metric='mlogloss',
        max_depth=6,
        random_state=42
    )

    clf.fit(X_train, y_train)

    # Evaluate performance
    y_pred = clf.predict(X_test)
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=encoder.classes_))

    # Save model & artifacts
    save_pickle(vectors, VEC_OUT)
    save_pickle(metadata, META_OUT)
    save_pickle(clf, MODEL_OUT)
    save_pickle(encoder, LABELS_OUT)

    print(f"\nTraining complete. Model and data saved.")

if __name__ == "__main__":
    main()
