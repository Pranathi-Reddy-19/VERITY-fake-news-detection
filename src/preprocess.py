import pandas as pd
import os

INPUT_PATH = "./data/raw/fake_news.csv"
OUTPUT_PATH = "./data/processed/clean_news.csv"

os.makedirs("./data/processed", exist_ok=True)

print("Loading dataset...")

df = pd.read_csv(INPUT_PATH)

print("Original shape:", df.shape)

# Keep only required columns
df = df[["title", "text", "label"]]

# Remove missing values
df = df.dropna(subset=["title", "text", "label"])

# Combine title and article text
df["content"] = df["title"].astype(str) + " " + df["text"].astype(str)

# Convert labels to numbers
df["label"] = df["label"].map({
    "REAL": 0,
    "FAKE": 1
})

# Remove rows with invalid labels
df = df.dropna(subset=["label"])

df["label"] = df["label"].astype(int)

# Keep only what we need
df = df[["content", "label"]]

# Reset index
df = df.reset_index(drop=True)

df.to_csv(OUTPUT_PATH, index=False)

print("Preprocessing completed.")
print("Final shape:", df.shape)
print("\nLabel distribution:")
print(df["label"].value_counts())

print("\nSaved to:", OUTPUT_PATH)