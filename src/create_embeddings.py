import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm
import os

DATA_PATH = "./data/processed/clean_news.csv"
MODEL_PATH = "./models/bert-base-uncased"
OUTPUT_PATH = "./data/processed/bert_embeddings.pt"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Using device:", DEVICE)

df = pd.read_csv(DATA_PATH)

print("Loading BERT...")

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModel.from_pretrained(MODEL_PATH)

model.to(DEVICE)
model.eval()

embeddings = []

print("Creating BERT embeddings...")

for i in tqdm(range(len(df))):
    text = str(df.iloc[i]["content"])

    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=256,
        padding=True
    )

    inputs = {key: value.to(DEVICE) for key, value in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)

    # [CLS] token embedding
    embedding = outputs.last_hidden_state[:, 0, :]

    embeddings.append(embedding.cpu())

embeddings = torch.cat(embeddings, dim=0)

torch.save(embeddings, OUTPUT_PATH)

print("\nBERT embeddings created successfully.")
print("Embedding shape:", embeddings.shape)
print("Saved to:", OUTPUT_PATH)