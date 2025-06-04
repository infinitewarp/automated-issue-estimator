# estimate_size.py
import sys
import numpy as np
from utils import get_embedding, load_pickle, cosine_similarity
from collections import Counter

TOP_K = 5

def main():
    if len(sys.argv) < 2:
        print("Usage: python estimate_size.py '<user story text>'")
        return

    user_story = sys.argv[1]

    vectors = load_pickle("story_vectors.pkl")
    metadata = load_pickle("story_metadata.pkl")

    new_vec = get_embedding(user_story)
    similarities = [cosine_similarity(new_vec, v) for v in vectors]

    top_indices = np.argsort(similarities)[-TOP_K:][::-1]
    top_sizes = [metadata[i]["size"] for i in top_indices]
    top_stories = [metadata[i] for i in top_indices]

    predicted_size = Counter(top_sizes).most_common(1)[0][0]

    print(f"Predicted size: {predicted_size}")
    print("\nSimilar stories:")
    for story in top_stories:
        print(f"- [{story['id']}] {story['title']} ({story['size']})")

if __name__ == "__main__":
    main()
