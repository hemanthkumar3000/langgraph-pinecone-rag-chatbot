import json
import logging
import re
from typing import Any

from pydantic import BaseModel, Field

from app.config import settings
from app.prompts import (
    GROUNDING_GRADER_PROMPT,
    RETRIEVAL_RELEVANCE_GRADER_PROMPT,
)

logger = logging.getLogger(__name__)


class RelevanceGrade(BaseModel):
    relevant: bool = Field(
        description="True if context contains information relevant to answering the question"
    )
    reason: str = Field(
        default="No reason provided",
        description="Short reason for the decision"
    )


class GroundingGrade(BaseModel):
    grounded: bool = Field(
        description="True if answer statements are supported by the context"
    )
    score: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Groundedness confidence score between 0.0 and 1.0"
    )
    reason: str = Field(
        default="No reason provided",
        description="Short explanation of the evaluation"
    )


def get_grader_llm():
    """
    Returns configured LLM client for grading tasks.
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
        format="json",
    )


def _clean_json_str(content: str) -> str:
    """
    Extracts raw JSON object substring from LLM response.
    """
    content = content.strip()
    match = re.search(r"\{.*\}", content, re.DOTALL)
    if match:
        return match.group(0)
    return content


def grade_retrieval_relevance(query: str, context: str) -> RelevanceGrade:
    """
    Assesses whether the retrieved context contains relevant information for the query.
    """
    llm = get_grader_llm()
    prompt = RETRIEVAL_RELEVANCE_GRADER_PROMPT.format(
        query=query,
        context=context,
    )

    try:
        response = llm.invoke(prompt)
        raw_text = response.content if hasattr(response, "content") else str(response)
        clean_text = _clean_json_str(raw_text)
        data = json.loads(clean_text)

        # Handle boolean conversion gracefully
        rel = data.get("relevant", True)
        if isinstance(rel, str):
            rel = rel.lower() in ("true", "yes", "1")

        return RelevanceGrade(
            relevant=bool(rel),
            reason=str(data.get("reason", "Graded successfully")),
        )
    except Exception as exc:
        logger.warning("Error parsing relevance grade JSON: %s. Defaulting to relevant=True.", exc)
        return RelevanceGrade(relevant=True, reason="Fallback due to grading parse error")


def grade_answer_grounding(
    query: str,
    context: str,
    answer: str,
) -> GroundingGrade:
    """
    Evaluates whether the generated answer is strictly grounded in the retrieved context.
    """
    # Fast path: check for explicit refusal
    if "cannot find this information in the provided ebook" in answer.lower():
        return GroundingGrade(
            grounded=False,
            score=0.0,
            reason="Response is an explicit refusal because context lacks the answer",
        )

    llm = get_grader_llm()
    prompt = GROUNDING_GRADER_PROMPT.format(
        query=query,
        context=context,
        answer=answer,
    )

    try:
        response = llm.invoke(prompt)
        raw_text = response.content if hasattr(response, "content") else str(response)
        clean_text = _clean_json_str(raw_text)
        data = json.loads(clean_text)

        grounded = data.get("grounded", True)
        if isinstance(grounded, str):
            grounded = grounded.lower() in ("true", "yes", "1")

        score = float(data.get("score", 0.90 if grounded else 0.0))
        score = max(0.0, min(1.0, score))

        return GroundingGrade(
            grounded=bool(grounded),
            score=score,
            reason=str(data.get("reason", "Evaluated successfully")),
        )
    except Exception as exc:
        logger.warning("Error parsing grounding grade JSON: %s. Fallback applied.", exc)
        return GroundingGrade(
            grounded=True,
            score=0.85,
            reason="Fallback applied due to parser error",
        )