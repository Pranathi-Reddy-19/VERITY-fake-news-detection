from src.verity_pipeline import classify_article

tests = [
    ("POLITICS", "Donald Trump passes away"),
    ("TECHNOLOGY", "Apple released an iPhone 17"),
    ("SCIENCE", "Water boils at 100 degrees Celsius at sea level"),
    ("SPACE", "NASA discovered life on Mars"),
    ("HEALTH", "Drinking bleach cures COVID-19"),
    ("SPORTS", "Lionel Messi won the 2026 World Cup"),
    ("CELEBRITY", "Taylor Swift married Elon Musk"),
    ("WORLD NEWS", "A major earthquake happened in Japan"),
    ("HISTORY", "The Titanic sank in 1912"),
    ("FINANCE", "Bitcoin reached 1 million dollars"),
]

print("\n" + "=" * 80)
print("VERITY MULTI-TOPIC TEST")
print("=" * 80)

for category, article in tests:

    print("\n" + "-" * 80)
    print("CATEGORY:", category)
    print("CLAIM:", article)

    result = classify_article(article)

    print("FINAL:", result["prediction"])
    print("CONFIDENCE:", f'{result["confidence"]:.2f}%')

    print(
        "GAT:",
        result["gat_prediction"],
        f'{result["gat_confidence"]:.2f}%'
    )

    print(
        "RAG:",
        result["evidence_status"],
        f'{result["rag_fake_ratio"] * 100:.2f}% fake'
    )

    print(
        "RETRIEVED:",
        [
            (
                round(x["similarity"], 3),
                x["label_name"]
            )
            for x in result["retrieved"]
        ]
    )

print("\n" + "=" * 80)
print("TEST COMPLETE")
print("=" * 80)