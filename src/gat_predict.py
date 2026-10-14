import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel
from torch_geometric.nn import GATConv

MODEL_PATH = "./models/gat_model.pt"
BERT_PATH = "./models/bert-base-uncased"
GRAPH_PATH = "./data/processed/news_graph.pt"

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


class GAT(torch.nn.Module):

    def __init__(self):

        super().__init__()

        self.gat1 = GATConv(
            in_channels=768,
            out_channels=128,
            heads=4,
            dropout=0.2
        )

        self.gat2 = GATConv(
            in_channels=128 * 4,
            out_channels=64,
            heads=2,
            dropout=0.2
        )

        self.gat3 = GATConv(
            in_channels=64 * 2,
            out_channels=2,
            heads=1,
            concat=False,
            dropout=0.2
        )

    def forward(self, x, edge_index):

        x = self.gat1(x, edge_index)
        x = F.elu(x)

        x = self.gat2(x, edge_index)
        x = F.elu(x)

        x = self.gat3(x, edge_index)

        return x


print("Loading GAT...")

tokenizer = AutoTokenizer.from_pretrained(BERT_PATH)
bert = AutoModel.from_pretrained(BERT_PATH)

bert.to(device)
bert.eval()

gat = GAT().to(device)

gat.load_state_dict(
    torch.load(
        MODEL_PATH,
        map_location=device
    )
)

gat.eval()

graph = torch.load(
    GRAPH_PATH,
    weights_only=False
)

existing_embeddings = graph["x"].cpu()
existing_edges = graph["edge_index"].cpu()

print("GAT loaded successfully.")


def predict_gat(text):

    text = str(text).strip()

    if not text:
        return {
            "prediction": "UNCERTAIN",
            "confidence": 0.0
        }

    inputs = tokenizer(
        text,
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

    # Calculate similarity to existing graph nodes
    similarities = torch.nn.functional.cosine_similarity(
        new_embedding_cpu,
        existing_embeddings,
        dim=1
    )

    top_k = min(5, len(similarities))

    top_indices = torch.topk(
        similarities,
        top_k
    ).indices

    new_node_index = existing_embeddings.shape[0]

    all_features = torch.cat(
        [
            existing_embeddings,
            new_embedding_cpu
        ],
        dim=0
    )

    new_edges = []

    for idx in top_indices.tolist():

        new_edges.append(
            [new_node_index, idx]
        )

        new_edges.append(
            [idx, new_node_index]
        )

    if new_edges:

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

    else:

        all_edges = existing_edges

    all_features = all_features.to(device)
    all_edges = all_edges.to(device)

    with torch.no_grad():

        logits = gat(
            all_features,
            all_edges
        )

        probabilities = F.softmax(
            logits[new_node_index],
            dim=0
        )

    prediction = probabilities.argmax().item()

    confidence = (
        probabilities[prediction].item() * 100
    )

    label = (
        "FAKE"
        if prediction == 1
        else "REAL"
    )

    return {
        "prediction": label,
        "confidence": confidence
    }