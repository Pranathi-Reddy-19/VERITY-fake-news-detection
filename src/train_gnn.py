import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv
from sklearn.model_selection import train_test_split

GRAPH_PATH = "./data/processed/news_graph.pt"
MODEL_PATH = "./models/gnn_model.pt"

print("Loading graph...")

graph = torch.load(GRAPH_PATH, weights_only=False)

x = graph["x"]
edge_index = graph["edge_index"]
y = graph["y"]

print("Nodes:", x.shape[0])
print("Features:", x.shape[1])
print("Edges:", edge_index.shape[1])

# Train / validation / test split
indices = list(range(len(y)))

train_idx, temp_idx = train_test_split(
    indices,
    test_size=0.30,
    random_state=42,
    stratify=y.numpy()
)

val_idx, test_idx = train_test_split(
    temp_idx,
    test_size=0.50,
    random_state=42,
    stratify=y.numpy()[temp_idx]
)

train_mask = torch.zeros(len(y), dtype=torch.bool)
val_mask = torch.zeros(len(y), dtype=torch.bool)
test_mask = torch.zeros(len(y), dtype=torch.bool)

train_mask[train_idx] = True
val_mask[val_idx] = True
test_mask[test_idx] = True

data = Data(
    x=x,
    edge_index=edge_index,
    y=y
)

data.train_mask = train_mask
data.val_mask = val_mask
data.test_mask = test_mask


class GCN(torch.nn.Module):

    def __init__(self):
        super().__init__()

        self.conv1 = GCNConv(768, 128)
        self.conv2 = GCNConv(128, 64)
        self.conv3 = GCNConv(64, 2)

    def forward(self, x, edge_index):

        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=0.3, training=self.training)

        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=0.3, training=self.training)

        x = self.conv3(x, edge_index)

        return x


device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Using device:", device)

data = data.to(device)

model = GCN().to(device)

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.01,
    weight_decay=5e-4
)

print("\nTraining GNN...")

for epoch in range(1, 101):

    model.train()

    optimizer.zero_grad()

    output = model(
        data.x,
        data.edge_index
    )

    loss = F.cross_entropy(
        output[data.train_mask],
        data.y[data.train_mask]
    )

    loss.backward()

    optimizer.step()

    if epoch % 10 == 0:

        model.eval()

        with torch.no_grad():

            pred = output.argmax(dim=1)

            train_acc = (
                pred[data.train_mask] ==
                data.y[data.train_mask]
            ).float().mean().item()

            val_acc = (
                pred[data.val_mask] ==
                data.y[data.val_mask]
            ).float().mean().item()

        print(
            f"Epoch {epoch:03d} | "
            f"Loss: {loss.item():.4f} | "
            f"Train Acc: {train_acc:.4f} | "
            f"Val Acc: {val_acc:.4f}"
        )

# Final test evaluation
model.eval()

with torch.no_grad():

    output = model(
        data.x,
        data.edge_index
    )

    predictions = output.argmax(dim=1)

    test_accuracy = (
        predictions[data.test_mask] ==
        data.y[data.test_mask]
    ).float().mean().item()

print("\nTraining completed.")

print(
    f"Test Accuracy: {test_accuracy * 100:.2f}%"
)

torch.save(
    model.state_dict(),
    MODEL_PATH
)

print("Model saved to:", MODEL_PATH)