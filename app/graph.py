import logging
from typing import TypedDict

from langgraph.graph import END, START, StateGraph

from app.config import settings
from app.prompts import ANSWER_GENERATION_PROMPT
from app.services.graders import (
    grade_answer_grounding,
    grade_retrieval_relevance,
)
from app.services.retrieval import (
    calculate_retrieval_score,
    format_context,
    retrieve_chunks,
)

logger = logging.getLogger(__name__)

REFUSAL_MESSAGE = "I cannot find this information in the provided eBook."


class RAGState(TypedDict, total=False):
    """
    Shared graph state passed between LangGraph nodes.
    """
    query: str
    retrieved_chunks: list[dict]
    context: str

    retrieval_score: float
    is_relevant: bool
    relevance_reason: str

    answer: str

    is_grounded: bool
    grounding_score: float
    grounding_reason: str

    confidence_score: float
    status: str
    refusal_reason: str


def get_answer_llm():
    """
    Returns the configured LLM for grounded answer generation.
    """
    if settings.llm_provider == "openai" and settings.openai_api_key:
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(
            model=settings.openai_chat_model,
            temperature=0,
            api_key=settings.openai_api_key,
        )

    from langchain_ollama import ChatOllama
    return ChatOllama(
        model=settings.chat_model,
        temperature=0,
    )


def retrieve_node(state: RAGState) -> dict:
    """
    Node 1: Retrieve top matching chunks from Pinecone.
    """
    query = state["query"]
    logger.info("Executing retrieve_node for query: %s", query)

    chunks = retrieve_chunks(query)
    context = format_context(chunks)
    retrieval_score = calculate_retrieval_score(chunks)

    return {
        "retrieved_chunks": chunks,
        "context": context,
        "retrieval_score": retrieval_score,
    }


def decide_after_retrieval(state: RAGState) -> str:
    """
    Conditional edge: if no chunks passed the similarity threshold, refuse early.
    """
    if not state.get("retrieved_chunks"):
        logger.info("No chunks passed similarity threshold. Routing to refuse node.")
        return "refuse"

    return "grade_relevance"


def relevance_node(state: RAGState) -> dict:
    """
    Node 2: Grade whether retrieved chunks can address the query.
    """
    query = state["query"]
    context = state["context"]

    grade = grade_retrieval_relevance(query=query, context=context)
    logger.info("Relevance grade: relevant=%s | reason=%s", grade.relevant, grade.reason)

    return {
        "is_relevant": grade.relevant,
        "relevance_reason": grade.reason,
    }


def decide_after_relevance(state: RAGState) -> str:
    """
    Conditional edge: if retrieved chunks are irrelevant, route to refuse.
    """
    if not state.get("is_relevant", False):
        logger.info("Retrieved context graded irrelevant. Routing to refuse node.")
        return "refuse"

    return "generate"


def generate_node(state: RAGState) -> dict:
    """
    Node 3: Synthesize a grounded answer strictly from eBook context.
    """
    query = state["query"]
    context = state["context"]
    logger.info("Executing generate_node for query: %s", query)

    llm = get_answer_llm()
    prompt = ANSWER_GENERATION_PROMPT.format(
        query=query,
        context=context,
    )

    response = llm.invoke(prompt)
    answer_text = response.content.strip() if hasattr(response, "content") else str(response).strip()

    return {
        "answer": answer_text,
    }


def grounding_node(state: RAGState) -> dict:
    """
    Node 4: Evaluate groundedness of the generated answer against context.
    """
    query = state["query"]
    context = state["context"]
    answer = state.get("answer", "")

    grade = grade_answer_grounding(
        query=query,
        context=context,
        answer=answer,
    )
    logger.info(
        "Grounding grade: grounded=%s | score=%.2f | reason=%s",
        grade.grounded,
        grade.score,
        grade.reason,
    )

    return {
        "is_grounded": grade.grounded,
        "grounding_score": grade.score,
        "grounding_reason": grade.reason,
    }


def decide_after_grounding(state: RAGState) -> str:
    """
    Conditional edge: route to refuse if not grounded or if answer is refusal.
    """
    answer = state.get("answer", "").strip().lower()
    is_refusal = REFUSAL_MESSAGE.lower() in answer

    if is_refusal or not state.get("is_grounded", False):
        logger.info("Answer ungrounded or explicit refusal. Routing to refuse node.")
        return "refuse"

    return "finalize"


def finalize_node(state: RAGState) -> dict:
    """
    Node 5: Calculate combined confidence score and prepare success payload.
    Confidence formula: 40% retrieval score + 60% grounding score.
    """
    retrieval_score = state.get("retrieval_score", 0.0)
    grounding_score = state.get("grounding_score", 0.0)

    confidence_score = round(
        (0.4 * retrieval_score) + (0.6 * grounding_score),
        2,
    )

    return {
        "confidence_score": confidence_score,
        "status": "answered",
        "refusal_reason": None,
    }


def refuse_node(state: RAGState) -> dict:
    """
    Safe fallback node for out-of-scope, irrelevant, or ungrounded queries.
    """
    if not state.get("retrieved_chunks"):
        reason = "No eBook chunks met the configured similarity threshold."
    elif not state.get("is_relevant", True):
        reason = state.get(
            "relevance_reason",
            "Retrieved context was not relevant to the question.",
        )
    elif not state.get("is_grounded", True):
        reason = state.get(
            "grounding_reason",
            "Generated answer was not supported by retrieved context.",
        )
    else:
        reason = "The eBook does not contain sufficient information to answer this query."

    return {
        "answer": REFUSAL_MESSAGE,
        "is_grounded": False,
        "grounding_score": 0.0,
        "confidence_score": 0.0,
        "status": "refused",
        "refusal_reason": reason,
    }


def build_rag_graph():
    """
    Constructs and compiles the cyclic/branching LangGraph RAG workflow.
    """
    workflow = StateGraph(RAGState)

    # Register nodes
    workflow.add_node("retrieve", retrieve_node)
    workflow.add_node("grade_relevance", relevance_node)
    workflow.add_node("generate", generate_node)
    workflow.add_node("grade_grounding", grounding_node)
    workflow.add_node("finalize", finalize_node)
    workflow.add_node("refuse", refuse_node)

    # Build flow edges
    workflow.add_edge(START, "retrieve")

    workflow.add_conditional_edges(
        "retrieve",
        decide_after_retrieval,
        {
            "grade_relevance": "grade_relevance",
            "refuse": "refuse",
        },
    )

    workflow.add_conditional_edges(
        "grade_relevance",
        decide_after_relevance,
        {
            "generate": "generate",
            "refuse": "refuse",
        },
    )

    workflow.add_edge("generate", "grade_grounding")

    workflow.add_conditional_edges(
        "grade_grounding",
        decide_after_grounding,
        {
            "finalize": "finalize",
            "refuse": "refuse",
        },
    )

    workflow.add_edge("finalize", END)
    workflow.add_edge("refuse", END)

    return workflow.compile()


rag_graph = build_rag_graph()