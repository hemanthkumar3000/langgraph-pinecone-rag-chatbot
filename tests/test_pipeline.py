import pytest
from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["service"] == "agentic-ai-ebook-rag-chatbot"


def test_out_of_scope_refusal():
    query = "What is the capital of France?"
    response = client.post("/query", json={"query": query})
    assert response.status_code == 200
    data = response.json()

    assert data["query"] == query
    assert "cannot find this information in the provided ebook" in data["final_answer"].lower()
    assert data["confidence_score"] == 0.0
    assert data["status"] == "refused"
    assert len(data["retrieved_context_chunks"]) == 0


def test_payload_schema_compliance():
    query = "What is the capital of France?"
    response = client.post("/query", json={"query": query})
    assert response.status_code == 200
    data = response.json()

    # Exact required fields from Section 3.C
    assert "query" in data
    assert "final_answer" in data
    assert "retrieved_context_chunks" in data
    assert "confidence_score" in data

    assert isinstance(data["query"], str)
    assert isinstance(data["final_answer"], str)
    assert isinstance(data["retrieved_context_chunks"], list)
    assert isinstance(data["confidence_score"], float)


def test_in_scope_query():
    query = "What is the core definition of Agentic AI as outlined in the eBook?"
    response = client.post("/query", json={"query": query})
    assert response.status_code == 200
    data = response.json()

    assert data["query"] == query
    assert data["status"] == "answered"
    assert data["is_grounded"] is True
    assert data["confidence_score"] > 0.50
    assert len(data["retrieved_context_chunks"]) > 0
    assert isinstance(data["retrieved_context_chunks"][0], str)
    assert "autonomous" in data["final_answer"].lower()
