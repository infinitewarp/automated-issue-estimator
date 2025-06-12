# estimate_size.py
import numpy as np
from utils import (
    get_embedding,
    load_model,
    cosine_similarity,
    VEC_OUT,
    META_OUT,
    MODEL_OUT,
    LABELS_OUT,
)

from alive_progress import alive_bar

TOP_K = 5


def estimate_size(user_story: str):
    with alive_bar(
        spinner="dots_waves", bar=None, monitor=False, elapsed=False, stats=False
    ):
        vectors = load_model(VEC_OUT)
        metadata = load_model(META_OUT)
        clf = load_model(MODEL_OUT)
        encoder = load_model(LABELS_OUT)

        emb = get_embedding(user_story)

        # Predict with classifier and get probability
        probs = clf.predict_proba([emb])[0]
        best_idx = probs.argmax()
        predicted_label = encoder.inverse_transform([best_idx])[0]
        confidence = probs[best_idx]

        # Similarity search for reference stories
        similarities = [cosine_similarity(emb, v) for v in vectors]
        top_indices = np.argsort(similarities)[-TOP_K:][::-1]
        top_stories = [metadata[i] for i in top_indices]

    print(f"\n🔮 Predicted size: {predicted_label} (confidence: {confidence:.2%})\n")
    print("📚 Most similar existing stories:")
    for story in top_stories:
        print(f"- [{story['id']}] {story['title']} (Size: {story['size']})")
