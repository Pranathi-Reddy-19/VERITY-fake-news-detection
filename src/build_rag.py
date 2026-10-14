import os
import pickle
import pandas as pd
import faiss
from sentence_transformers import SentenceTransformer

DATA_PATH = "./data/processed/clean_news.csv"

INDEX_PATH = "./data/processed/news_faiss.index"
TEXT_PATH = "./data/processed/news_texts.pkl"
LABEL_PATH = "./data/processed/news_labels.pkl"

print("Loading dataset...")

df = pd.read_csv(DATA_PATH)

texts = df["content"].fillna("").tolist()
labels = df["label"].tolist()

print(f"Loaded {len(texts)} articles")

print("Loading embedding model...")

model = SentenceTransformer("all-MiniLM-L6-v2")

print("Creating embeddings...")

embeddings = model.encode(
    texts,
    batch_size=32,
    show_progress_bar=True,
    normalize_embeddings=True
)

print("Creating FAISS index...")

dimension = embeddings.shape[1]

index = faiss.IndexFlatIP(dimension)

index.add(embeddings)

os.makedirs("./data/processed", exist_ok=True)

faiss.write_index(index, INDEX_PATH)

with open(TEXT_PATH, "wb") as f:
    pickle.dump(texts, f)

with open(LABEL_PATH, "wb") as f:
    pickle.dump(labels, f)

print("\nRAG index created successfully!")

print(f"Articles indexed: {index.ntotal}")
print(f"Embedding dimension: {dimension}")
print(f"Labels stored: {len(labels)}")
print(f"Index saved to: {INDEX_PATH}")