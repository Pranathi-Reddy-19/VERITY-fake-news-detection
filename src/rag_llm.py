import ollama
from src.rag import retrieve_articles


def analyze_with_rag(article):

    retrieved = retrieve_articles(article, top_k=5)

    context = "\n\n".join(
        [
            f"RELATED ARTICLE {i + 1}:\n{item['text'][:2000]}"
            for i, item in enumerate(retrieved)
        ]
    )

    prompt = f"""
You are a news analysis assistant.

Analyze the following news article using the retrieved articles
as supporting context.

Do not assume that retrieved articles are automatically true.
Do not invent facts.

NEWS ARTICLE:
{article}

RETRIEVED CONTEXT:
{context}

Provide:

1. A short assessment of whether the article appears suspicious,
   credible, or uncertain.
2. The main reasons for your assessment.
3. Which retrieved articles are relevant.
4. A short explanation suitable for a user.

Keep the answer concise.
"""

    response = ollama.chat(
        model="llama3.2:3b",
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ]
    )

    return response["message"]["content"], retrieved