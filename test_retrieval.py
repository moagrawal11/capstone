import logging

from app.retrieval.retriever import retrieve

# Configure logging
logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

# Representative customer queries covering different document categories
TEST_QUERIES = [
    {
        "query": "What is the payload capacity of the AX-200 robot?",
        "expected_source_hint": "products",
        "category": "Product Specification",
    },
    {
        "query": "How do I reset my robot to factory settings?",
        "expected_source_hint": "support",
        "category": "Troubleshooting",
    },
    {
        "query": "What is the warranty period for Axiom robots?",
        "expected_source_hint": "policies",
        "category": "Warranty & Policy",
    },
    {
        "query": "How do I connect the robot to WiFi?",
        "expected_source_hint": "faq",
        "category": "FAQ / Setup",
    },
    {
        "query": "What safety certifications do Axiom robots have?",
        "expected_source_hint": "products",
        "category": "Product Safety",
    },
]


def run_retrieval_tests() -> None:
    """Run retrieval tests for all representative queries."""
    print("=" * 70)
    print("  RoboDesk — Retrieval Validation")
    print("=" * 70)
    print()

    for i, test in enumerate(TEST_QUERIES, 1):
        query = test["query"]
        hint = test["expected_source_hint"]
        category = test["category"]

        print(f"Query {i} [{category}]: {query}")
        print(f"  Expected source contains: '{hint}'")
        print("-" * 70)

        results = retrieve(query, top_k=3)

        if not results:
            print("  !!  No results found!")
        else:
            for j, result in enumerate(results, 1):
                relevance = "[Y]" if hint in result.source_file else "[?]"
                print(f"  [{j}] {relevance} Source: {result.source_file} "
                      f"(chunk {result.chunk_index}, score: {result.similarity_score:.4f})")
                # Print first 150 chars of chunk text
                preview = result.chunk_text[:150].replace("\n", " ")
                # Safely encode for Windows console (replace unsupported chars)
                preview = preview.encode("ascii", errors="replace").decode("ascii")
                print(f"      Preview: {preview}...")

        print()

    print("=" * 70)
    print("  Retrieval validation complete!")
    print("=" * 70)


if __name__ == "__main__":
    run_retrieval_tests()
