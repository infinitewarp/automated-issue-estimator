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

    # Predict with classifier
    predicted_index = clf.predict([emb])[0]
    predicted_label = encoder.inverse_transform([predicted_index])[0]

    # Similarity for context
    similarities = [cosine_similarity(emb, v) for v in vectors]
    top_indices = np.argsort(similarities)[-TOP_K:][::-1]
    top_stories = [metadata[i] for i in top_indices]

    print(f"Predicted size: {predicted_label}\n")
    print("Similar stories:")
    for story in top_stories:
        print(f"- [{story['id']}] {story['title']} ({story['size']})")

if __name__ == "__main__":
    main()
