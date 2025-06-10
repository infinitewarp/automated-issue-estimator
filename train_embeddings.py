# train_embeddings.py
import json
import numpy as np
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from utils import get_embedding, save_pickle
from sklearn.svm import SVC
import sys

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

    # Attempt to elimintate:
    # > sklearn/linear_model/_linear_loss.py:203: RuntimeWarning: divide by zero encountered in matmul
    # >   raw_prediction = X @ weights.T + intercept  # ndarray, likely C-contiguous
    # and
    # > sklearn/utils/extmath.py:44: RuntimeWarning: divide by zero encountered in dot
    # >   return np.dot(x, x)

    # scaler = StandardScaler()
    # vectors = scaler.fit_transform(vectors)  # Normalize embeddings
    # ...but that did not help.

    # Diagnose if there are extreme vectors...
    # print(f"Embedding range: min={np.min(vectors)}, max={np.max(vectors)}")
    # np.clip(vectors, -10, 10, out=vectors)  # Clipping to a range between -10 and 10
    # ...but they're all between |5|.

    if np.any(np.all(vectors == 0, axis=1)):
        print("⚠️ Warning: Some embeddings are zero vectors!")

    encoder = LabelEncoder()
    y = encoder.fit_transform(labels)

    # clf = LogisticRegression(max_iter=1000)
    # clf = LogisticRegression(max_iter=1000, C=1.0)  # Default regularization strength
    # clf = LogisticRegression(max_iter=1000, C=0.1)  # Try a lower C (stronger regularization) ...but no help.
    # clf = LogisticRegression(max_iter=1000, solver='liblinear')  # ...but liblinear is deprecated

    # solver='saga' produces no warnings, but it any is better?
    # clf = LogisticRegression(max_iter=1000, solver='saga')

    # https://machinelearningmastery.com/one-vs-rest-and-one-vs-one-for-multi-class-classification/
    # https://scikit-learn.org/stable/modules/multiclass.html#ovo-classification
    # https://www.geeksforgeeks.org/understanding-the-predictproba-function-in-scikit-learns-svc/#the-role-of-predict_proba
    # clf = SVC(decision_function_shape='ovo', probability=True)  # works pretty well!


    # clf = RandomForestClassifier(
    #     n_estimators=200,       # more trees for stability
    #     max_depth=None,         # or limit to avoid overfitting
    #     random_state=42
    # )  # works pretty well!

    # params = {
    #     "n_estimators": [100, 200],
    #     "max_depth": [None, 10, 20],
    #     "min_samples_split": [2, 5]
    # }
    # clf = GridSearchCV(RandomForestClassifier(random_state=42), params, cv=3)
    # clf.fit(vectors, y)
    # print("Best params:", clf.best_params_)

    # recommended from best params above
    clf = RandomForestClassifier(
        n_estimators=100,
        max_depth=None,
        min_samples_split=2,
        random_state=42
    )

    clf.fit(vectors, y)

    # Save all relevant artifacts
    save_pickle(vectors, VEC_OUT)
    save_pickle(metadata, META_OUT)
    save_pickle(clf, MODEL_OUT)
    save_pickle(encoder, LABELS_OUT)

    print(f"\n✅ Trained and saved model using {len(vectors)} stories.")

if __name__ == "__main__":
    main()
