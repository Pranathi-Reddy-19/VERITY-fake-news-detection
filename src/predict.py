import torch
import torch.nn.functional as F
import pandas as pd
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics.pairwise import cosine_similarity
from torch_geometric.nn import GCNConv

MODEL_PATH = "./models/gnn_model.pt"
BERT_PATH = "./models/bert-base-uncased"
DATA_PATH = "./data/processed/clean_news.csv"
EMBEDDING_PATH = "./data/processed/bert_embeddings.pt"
GRAPH_PATH = "./data/processed/news_graph.pt"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class GCN(torch.nn.Module):

    def __init__(self):
        super().__init__()

        self.conv1 = GCNConv(768, 128)
        self.conv2 = GCNConv(128, 64)
        self.conv3 = GCNConv(64, 2)

    def forward(self, x, edge_index):

        x = self.conv1(x, edge_index)
        x = F.relu(x)

        x = self.conv2(x, edge_index)
        x = F.relu(x)

        x = self.conv3(x, edge_index)

        return x


print("Loading BERT...")

tokenizer = AutoTokenizer.from_pretrained(BERT_PATH)
bert = AutoModel.from_pretrained(BERT_PATH)

bert.to(device)
bert.eval()

print("Loading GNN...")

model = GCN()
model.load_state_dict(
    torch.load(MODEL_PATH, map_location=device)
)

model.to(device)
model.eval()

print("Loading graph...")

graph = torch.load(GRAPH_PATH, weights_only=False)

existing_embeddings = graph["x"]
existing_edges = graph["edge_index"]

print("\nModel loaded successfully.")

news = input("\nEnter news article: ")

inputs = tokenizer(
    news,
    return_tensors="pt",
    truncation=True,
    max_length=256
)

inputs = {
    key: value.to(device)
    for key, value in inputs.items()
}

with torch.no_grad():

    output = bert(**inputs)

    new_embedding = output.last_hidden_state[:, 0, :]

new_embedding_cpu = new_embedding.cpu()

# Find 5 most similar existing articles
similarities = cosine_similarity(
    new_embedding_cpu.numpy(),
    existing_embeddings.numpy()
)[0]

top_indices = similarities.argsort()[-5:][::-1]

print("\nMost similar articles:")

for i in top_indices:
    print(
        f"Article {i} - similarity: {similarities[i]:.4f}"
    )

# Create new node
new_node_index = existing_embeddings.shape[0]

all_features = torch.cat(
    [
        existing_embeddings,
        new_embedding_cpu
    ],
    dim=0
)

# Connect new article to its 5 nearest articles
new_edges = []

for i in top_indices:

    new_edges.append(
        [new_node_index, int(i)]
    )

    new_edges.append(
        [int(i), new_node_index]
    )

new_edges = torch.tensor(
    new_edges,
    dtype=torch.long
).t()

all_edges = torch.cat(
    [
        existing_edges,
        new_edges
    ],
    dim=1
)

all_features = all_features.to(device)
all_edges = all_edges.to(device)

with torch.no_grad():

    predictions = model(
        all_features,
        all_edges
    )

    probabilities = F.softmax(
        predictions[new_node_index],
        dim=0
    )

    prediction = probabilities.argmax().item()

if prediction == 1:
    label = "FAKE"
else:
    label = "REAL"

confidence = probabilities[prediction].item() * 100

print("\n" + "=" * 40)
print("PREDICTION:", label)
print(f"CONFIDENCE: {confidence:.2f}%")
print("=" * 40)