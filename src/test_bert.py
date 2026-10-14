from transformers import AutoTokenizer, AutoModel
import torch

MODEL_PATH = "./models/bert-base-uncased"

print("Loading BERT...")

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModel.from_pretrained(MODEL_PATH)

text = "This is a fake news detection test."

inputs = tokenizer(
    text,
    return_tensors="pt",
    truncation=True,
    padding=True
)

with torch.no_grad():
    outputs = model(**inputs)

print("BERT loaded successfully.")
print("Input shape:", inputs["input_ids"].shape)
print("Output shape:", outputs.last_hidden_state.shape)