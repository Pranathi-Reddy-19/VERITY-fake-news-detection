from transformers import AutoTokenizer, AutoModel

MODEL_NAME = "bert-base-uncased"
MODEL_PATH = "./models/bert-base-uncased"

print("Downloading BERT tokenizer...")

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
tokenizer.save_pretrained(MODEL_PATH)

print("Downloading BERT model...")

model = AutoModel.from_pretrained(MODEL_NAME)
model.save_pretrained(MODEL_PATH)

print("BERT downloaded successfully.")
print("Saved at:", MODEL_PATH)