import json
import sys
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.graph import rag_graph


def run_benchmark_question(category: str, query: str) -> dict:
    print("\n" + "=" * 80)
    print(f"[{category.upper()}]")
    print(f"QUERY: {query}")
    print("=" * 80)

    result = rag_graph.invoke({"query": query})

    raw_chunks = result.get("retrieved_chunks", [])
    text_chunks = [chunk.get("text", "") for chunk in raw_chunks]

    payload = {
        "query": query,
        "final_answer": result.get(
            "answer",
            "I cannot find this information in the provided eBook.",
        ),
        "retrieved_context_chunks": text_chunks,
        "confidence_score": result.get("confidence_score", 0.0),
    }

    print(f"\nStatus:           {result.get('status')}")
    print(f"Confidence Score: {payload['confidence_score']}")
    print(f"Grounded:         {result.get('is_grounded')}")
    print(f"Retrieved Chunks: {len(raw_chunks)}")

    print("\nFINAL ANSWER:")
    print(payload["final_answer"])

    if raw_chunks:
        print("\nTOP RETRIEVED SOURCES:")
        for idx, chunk in enumerate(raw_chunks, start=1):
            print(
                f"  [{idx}] Page {chunk.get('page_number')} | "
                f"Similarity: {chunk.get('similarity_score')} | "
                f"Preview: {chunk.get('text')[:120].replace(chr(10), ' ')}..."
            )

    print("\nSTRUCTURED PAYLOAD (Section 3.C):")
    print(json.dumps(payload, indent=2))

    return payload


def main() -> None:
    test_cases = [
        ("Definition & Scope", "What is the core definition of Agentic AI as outlined in the eBook?"),
        ("Architecture & Paradigms", "What are the main architectural components required to build agentic systems?"),
        ("Use Cases", "What real-world industry use cases for Agentic AI are discussed in the eBook?"),
        ("Comparison", "How does Agentic AI differ from traditional generative AI chatbots according to the text?"),
        ("Challenges & Considerations", "What key challenges or limitations of Agentic AI are mentioned in the document?"),
        ("Out-of-Scope Test", "What is the capital of France?"),
    ]

    print("=" * 80)
    print("STARTING BENCHMARK EVALUATION SUITE")
    print("=" * 80)

    results = []
    for category, query in test_cases:
        res = run_benchmark_question(category, query)
        results.append(res)

    print("\n" + "=" * 80)
    print("BENCHMARK EVALUATION SUMMARY")
    print("=" * 80)
    for idx, (cat, q) in enumerate(test_cases):
        r = results[idx]
        ans_preview = r['final_answer'][:60].replace('\n', ' ')
        print(f"{idx+1}. [{cat}] Score: {r['confidence_score']:.2f} | Chunks: {len(r['retrieved_context_chunks'])} | Answer: {ans_preview}...")


if __name__ == "__main__":
    main()