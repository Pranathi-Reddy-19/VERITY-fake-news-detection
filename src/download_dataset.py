import pandas as pd
import os

os.makedirs("./data/raw", exist_ok=True)

url = "https://raw.githubusercontent.com/lutzhamel/fake-news/master/data/fake_or_real_news.csv"

print("Downloading dataset...")

df = pd.read_csv(url)

print("Dataset downloaded successfully.")
print("Shape:", df.shape)
print("Columns:", df.columns.tolist())

print("\nFirst 5 rows:")
print(df.head())

output_path = "./data/raw/fake_news.csv"
df.to_csv(output_path, index=False)

print("\nSaved to:", output_path)