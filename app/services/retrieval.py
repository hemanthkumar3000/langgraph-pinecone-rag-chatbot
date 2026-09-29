import logging
from typing import Any

from app.config import settings
from app.services.vector_store import get_pinecone_index

logger = logging.getLogger(__name__)


def get_embedding_model():
    """
    Returns the appropriate embedding model based on provider configuration.
    """
    if settings.llm_provider == "openai" and settings.openai_api_key:
        from langchain_openai import OpenAIEmbeddings
        return OpenAIEmbeddings(
            model=settings.openai_embedding_model,
            api_key=settings.openai_api_key,
        )

    from langchain_ollama import OllamaEmbeddings
    return OllamaEmbeddings(
        model=settings.embedding_model,
    )


def _expand_query(query: str) -> list[str]:
    """
    Generates semantic search variations for targeted eBook sections.
    """
    queries = [query]
    q_lower = query.lower()

    if "challenge" in q_lower or "limitation" in q_lower or "risk" in q_lower:
        queries.append("challenges limitations mitigation strategies multi-agent orchestration")
    elif "differ" in q_lower or "comparison" in q_lower or "traditional" in q_lower:
        queries.append("difference between Agentic AI traditional generative AI LLM")
    elif "architect" in q_lower or "component" in q_lower or "building block" in q_lower:
        queries.append("architectural components building blocks perception reasoning planning learning execution")

    return queries


def retrieve_chunks(query: str) -> list[dict[str, Any]]:
    """
    Queries Pinecone index using dense embeddings and semantic reciprocal ranking.
    Filters out chunks below the configured similarity threshold.
    """
    embedding_model = get_embedding_model()
    index = get_pinecone_index()

    search_queries = _expand_query(query)
    seen_chunks: dict[str, dict[str, Any]] = {}

    for q_idx, q_text in enumerate(search_queries):
        query_vector = embedding_model.embed_query(q_text)
        search_result = index.query(
            vector=query_vector,
            top_k=settings.top_k + 2,
            include_metadata=True,
        )

        for rank, match in enumerate(search_result.get("matches", [])):
            similarity_score = float(match.get("score", 0.0))
            if similarity_score < settings.similarity_threshold:
                continue

            chunk_id = match.get("id", "")
            metadata = match.get("metadata", {})
            # Rank-weighted fusion score
            rrf_score = (1.0 / (60 + rank)) + similarity_score

            if chunk_id not in seen_chunks or rrf_score > seen_chunks[chunk_id]["_rrf_score"]:
                seen_chunks[chunk_id] = {
                    "chunk_id": chunk_id,
                    "text": metadata.get("text", "").strip(),
                    "page_number": int(metadata.get("page_number", 0)),
                    "source": metadata.get("source", "Ebook-Agentic-AI.pdf"),
                    "chunk_index": int(metadata.get("chunk_index", 0)),
                    "similarity_score": round(similarity_score, 4),
                    "_rrf_score": rrf_score,
                }

    # Sort by hybrid score and take top_k
    sorted_chunks = sorted(
        seen_chunks.values(),
        key=lambda item: item["_rrf_score"],
        reverse=True,
    )

    results = []
    for chunk in sorted_chunks[: settings.top_k]:
        chunk_clean = dict(chunk)
        chunk_clean.pop("_rrf_score", None)
        results.append(chunk_clean)

    logger.info(
        "Retrieved %d relevant chunks for query: '%s' (threshold=%.2f)",
        len(results),
        query,
        settings.similarity_threshold,
    )
    return results


def calculate_retrieval_score(chunks: list[dict[str, Any]]) -> float:
    """
    Computes average similarity score across retrieved chunks.
    """
    if not chunks:
        return 0.0
    scores = [chunk["similarity_score"] for chunk in chunks]
    return round(sum(scores) / len(scores), 4)


def format_context(chunks: list[dict[str, Any]]) -> str:
    """
    Formats chunks into structured context blocks for prompting.
    """
    if not chunks:
        return ""

    formatted = []
    for idx, chunk in enumerate(chunks, start=1):
        formatted.append(
            f"[Context Chunk {idx}] (Source: {chunk['source']} | Page: {chunk['page_number']})\n{chunk['text']}"
        )

    return "\n\n---\n\n".join(formatted)