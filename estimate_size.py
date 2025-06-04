# estimate_size.py
import sys
import numpy as np
from utils import get_embedding, load_pickle, cosine_similarity

TOP_K = 5

def main():
    if len(sys.argv) < 2:
        print("Usage: python estimate_size.py '<user story text>'")
        return

    user_story = sys.argv[1]

    vectors = load_pickle("story_vectors.pkl")
    metadata = load_pickle("story_metadata.pkl")
    clf = load_pickle("classifier.pkl")
    encoder = load_pickle("label_encoder.pkl")

    emb = get_embedding(user_story)

    # Predict with classifier and get probability
    probs = clf.predict_proba([emb])[0]
    best_idx = probs.argmax()
    predicted_label = encoder.inverse_transform([best_idx])[0]
    confidence = probs[best_idx]

    print(f"\n🔮 Predicted size: {predicted_label} (confidence: {confidence:.2%})\n")

    # Similarity search for reference stories
    similarities = [cosine_similarity(emb, v) for v in vectors]
    top_indices = np.argsort(similarities)[-TOP_K:][::-1]
    top_stories = [metadata[i] for i in top_indices]

    print("📚 Most similar existing stories:")
    for story in top_stories:
        print(f"- [{story['id']}] {story['title']} (Size: {story['size']})")

if __name__ == "__main__":
    main()
