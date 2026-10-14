import ollama

response = ollama.chat(
    model="llama3.2:3b",
    messages=[
        {
            "role": "user",
            "content": "Explain fake news detection in two simple sentences."
        }
    ]
)

print("\nLLM RESPONSE:\n")
print(response["message"]["content"])