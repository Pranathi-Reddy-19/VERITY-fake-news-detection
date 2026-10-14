import torch
import torch.nn.functional as F
import pandas as pd

from torch_geometric.data import Data
from torch_geometric.nn import GATConv
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


GRAPH_PATH = "./data/processed/news_graph.pt"
DATA_PATH = "./data/processed/clean_news.csv"
MODEL_PATH = "./models/gat_model.pt"


print("Loading graph...")
graph = torch.load(GRAPH_PATH, weights_only=False)

x = graph["x"]
edge_index = graph["edge_index"]

print("Loading labels...")
df = pd.read_csv(DATA_PATH)

y = torch.tensor(
    df["label"].values,
    dtype=torch.long
)

print(f"Nodes: {x.shape[0]}")
print(f"Features: {x.shape[1]}")
print(f"Edges: {edge_index.shape[1]}")


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


print("Creating train/validation/test split...")

indices = torch.arange(len(y))

train_idx, temp_idx = train_test_split(
    indices.numpy(),
    test_size=0.30,
    random_state=42,
    stratify=y.numpy()
)

val_idx, test_idx = train_test_split(
    temp_idx,
    test_size=0.50,
    random_state=42,
    stratify=y[temp_idx].numpy()
)

train_idx = torch.tensor(train_idx)
val_idx = torch.tensor(val_idx)
test_idx = torch.tensor(test_idx)


train_mask = torch.zeros(len(y), dtype=torch.bool)
val_mask = torch.zeros(len(y), dtype=torch.bool)
test_mask = torch.zeros(len(y), dtype=torch.bool)

train_mask[train_idx] = True
val_mask[val_idx] = True
test_mask[test_idx] = True


device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print(f"Using device: {device}")

x = x.to(device)
edge_index = edge_index.to(device)
y = y.to(device)

train_mask = train_mask.to(device)
val_mask = val_mask.to(device)
test_mask = test_mask.to(device)


model = GAT().to(device)

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.001,
    weight_decay=5e-4
)


best_val_accuracy = 0

print("\nStarting GAT training...\n")


for epoch in range(1, 101):

    model.train()

    optimizer.zero_grad()

    output = model(x, edge_index)

    loss = F.cross_entropy(
        output[train_mask],
        y[train_mask]
    )

    loss.backward()

    optimizer.step()


    model.eval()

    with torch.no_grad():

        predictions = output.argmax(dim=1)

        train_accuracy = (
            predictions[train_mask] == y[train_mask]
        ).float().mean().item()

        val_accuracy = (
            predictions[val_mask] == y[val_mask]
        ).float().mean().item()


    if val_accuracy > best_val_accuracy:

        best_val_accuracy = val_accuracy

        torch.save(
            model.state_dict(),
            MODEL_PATH
        )


    if epoch % 10 == 0:

        print(
            f"Epoch {epoch:03d} | "
            f"Loss: {loss.item():.4f} | "
            f"Train Acc: {train_accuracy * 100:.2f}% | "
            f"Val Acc: {val_accuracy * 100:.2f}%"
        )


print("\nLoading best GAT model...")

model.load_state_dict(
    torch.load(
        MODEL_PATH,
        map_location=device
    )
)

model.eval()


with torch.no_grad():

    output = model(x, edge_index)

    predictions = output.argmax(dim=1)


y_true = y[test_mask].cpu().numpy()
y_pred = predictions[test_mask].cpu().numpy()


accuracy = accuracy_score(
    y_true,
    y_pred
)

precision = precision_score(
    y_true,
    y_pred,
    zero_division=0
)

recall = recall_score(
    y_true,
    y_pred,
    zero_division=0
)

f1 = f1_score(
    y_true,
    y_pred,
    zero_division=0
)


print("\n" + "=" * 60)
print("GAT TEST RESULTS")
print("=" * 60)

print(f"Accuracy : {accuracy * 100:.2f}%")
print(f"Precision: {precision * 100:.2f}%")
print(f"Recall   : {recall * 100:.2f}%")
print(f"F1 Score : {f1 * 100:.2f}%")

print("=" * 60)

print(f"\nBest validation accuracy: {best_val_accuracy * 100:.2f}%")
print(f"Model saved to: {MODEL_PATH}")