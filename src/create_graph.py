import pandas as pd
import torch
from sklearn.neighbors import NearestNeighbors

DATA_PATH = "./data/processed/clean_news.csv"
EMBEDDING_PATH = "./data/processed/bert_embeddings.pt"
OUTPUT_PATH = "./data/processed/news_graph.pt"

print("Loading data...")

df = pd.read_csv(DATA_PATH)
embeddings = torch.load(EMBEDDING_PATH, weights_only=True)

print("Embeddings shape:", embeddings.shape)

# Number of nearest neighbors
K = 5

print("Finding similar news articles...")

# Convert embeddings to numpy
X = embeddings.numpy()

# Find K nearest neighbors using cosine distance
knn = NearestNeighbors(
    n_neighbors=K + 1,
    metric="cosine"
)

knn.fit(X)

distances, indices = knn.kneighbors(X)

print("Building graph edges...")

edges = []

for i in range(len(indices)):
    for j in range(1, K + 1):
        neighbor = indices[i][j]

        # Add edge in both directions
        edges.append([i, neighbor])
        edges.append([neighbor, i])

edge_index = torch.tensor(
    edges,
    dtype=torch.long
).t().contiguous()

labels = torch.tensor(
    df["label"].values,
    dtype=torch.long
)

graph = {
    "x": embeddings,
    "edge_index": edge_index,
    "y": labels
}

torch.save(graph, OUTPUT_PATH)

print("\nGraph created successfully!")
print("Nodes:", graph["x"].shape[0])
print("Features per node:", graph["x"].shape[1])
print("Edges:", graph["edge_index"].shape[1])
print("Labels:", graph["y"].shape)
print("Saved to:", OUTPUT_PATH)