# from src.rag import retrieve_articles

# query = """
# Scientists have developed a new technology that can dramatically
# reduce carbon emissions from vehicles.
# """

# results = retrieve_articles(query, top_k=5)

# print("\nTOP RETRIEVED ARTICLES\n")

# for i, result in enumerate(results, 1):

#     print("=" * 80)
#     print(f"RESULT {i}")
#     print(f"Similarity: {result['similarity']:.4f}")
#     print()
#     print(result["text"][:1000])
#     print()

from src.rag import retrieve_articles

query = "Donald Trump passes away"

results = retrieve_articles(query, top_k=10)

print("\nTOP RETRIEVED ARTICLES\n")

for i, item in enumerate(results, 1):

    print("=" * 80)

    print(f"ARTICLE {i}")
    print(f"Similarity : {item['similarity']:.4f}")
    print(f"Label      : {item['label_name']}")
    print()

    print(item["text"][:500])