import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

from app.services.retrieval import (
    calculate_retrieval_score,
    format_context,
    retrieve_chunks,
)


def main() -> None:
    query = "What is the core definition of Agentic AI?"

    print("=" * 70)
    print(f"QUERY: {query}")
    print("=" * 70)

    chunks = retrieve_chunks(query)
    retrieval_score = calculate_retrieval_score(chunks)

    print(f"\nRetrieved chunks: {len(chunks)}")
    print(f"Average retrieval score: {retrieval_score}")

    if not chunks:
        print("\nNo chunks passed the similarity threshold.")
        return

    print("\nRETRIEVED CONTEXT")
    print("-" * 70)
    print(format_context(chunks))


if __name__ == "__main__":
    main()