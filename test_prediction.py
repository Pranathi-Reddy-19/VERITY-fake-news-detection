from src.verity_pipeline import classify_article


tests = [

    "Donald Trump passes away",

    "Scientists discover a new species of bird in the Amazon rainforest.",

    "The United States government has announced a new policy affecting international trade.",

    "A completely fabricated story claims that the moon will disappear tomorrow."

]


for i, article in enumerate(tests, 1):

    print("\n\n")
    print("=" * 80)
    print(f"TEST {i}")
    print("=" * 80)

    print("\nARTICLE:")
    print(article)

    result = classify_article(article)

    print("\nFINAL RESULT")
    print("-" * 40)
    print("Prediction :", result["prediction"])
    print(
        "Confidence :",
        f"{result['confidence']:.2f}%"
    )
    print(
        "GAT        :",
        result["gat_prediction"]
    )
    print(
        "RAG Fake   :",
        f"{result['rag_fake_ratio'] * 100:.2f}%"
    )
    print(
        "Evidence   :",
        result["evidence_status"]
    )