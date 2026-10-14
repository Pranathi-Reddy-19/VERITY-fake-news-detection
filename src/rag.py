import pickle
import faiss
from sentence_transformers import SentenceTransformer

INDEX_PATH = "./data/processed/news_faiss.index"
TEXT_PATH = "./data/processed/news_texts.pkl"
LABEL_PATH = "./data/processed/news_labels.pkl"

model = SentenceTransformer("all-MiniLM-L6-v2")
index = faiss.read_index(INDEX_PATH)

with open(TEXT_PATH, "rb") as f:
    texts = pickle.load(f)

with open(LABEL_PATH, "rb") as f:
    labels = pickle.load(f)


def retrieve_articles(query, top_k=5, exclude_index=None):
    search_k = top_k + 5 if exclude_index is not None else top_k

    query_embedding = model.encode(
        [query],
        normalize_embeddings=True
    )

    scores, indices = index.search(query_embedding, search_k)

    results = []

    for score, idx in zip(scores[0], indices[0]):

        if idx == -1:
            continue

        if exclude_index is not None and int(idx) == int(exclude_index):
            continue

        similarity = float(score)

        if similarity >= 0.60:
            evidence_strength = "STRONG"
        elif similarity >= 0.45:
            evidence_strength = "MODERATE"
        else:
            evidence_strength = "WEAK"

        label = int(labels[idx])

        results.append({
            "text": texts[idx],
            "similarity": similarity,
            "index": int(idx),
            "label": label,
            "label_name": "FAKE" if label == 1 else "REAL",
            "evidence_strength": evidence_strength
        })

        if len(results) >= top_k:
            break

    return results