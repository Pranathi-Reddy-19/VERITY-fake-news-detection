from src.verity_pipeline import (
    classify_article,
    generate_explanation
)


article = """
Scientists have announced a revolutionary new technology
that can completely eliminate pollution from every car
within one year. The researchers claim the technology
requires no infrastructure changes and will be free for
everyone.
"""


print("\nAnalyzing article...\n")

result = classify_article(article)

print("\n" + "=" * 60)
print("VERITY RESULT")
print("=" * 60)

print(
    f"Prediction: {result['prediction']}"
)

print(
    f"Confidence: {result['confidence']:.2f}%"
)

print(
    "\nProbabilities:"
)

print(
    f"REAL: {result['probabilities'][0] * 100:.2f}%"
)

print(
    f"FAKE: {result['probabilities'][1] * 100:.2f}%"
)


print("\nGenerating LLM explanation...")

explanation = generate_explanation(
    article,
    result
)

print("\n" + "=" * 60)
print("LLM EXPLANATION")
print("=" * 60)

print(explanation)