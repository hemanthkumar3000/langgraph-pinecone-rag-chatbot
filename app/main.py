import logging
from fastapi import FastAPI, HTTPException, status
from fastapi.responses import HTMLResponse, JSONResponse

from app.graph import rag_graph
from app.schemas import ChunkMetadata, QueryRequest, QueryResponse

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)

logger = logging.getLogger(__name__)

app = FastAPI(
    title="Agentic AI eBook RAG Chatbot",
    description=(
        "A LangGraph and Pinecone powered RAG chatbot built to answer queries "
        "strictly based on the 'Agentic AI eBook' (Konverge AI)."
    ),
    version="1.0.0",
)


@app.get(
    "/health",
    tags=["System"],
    summary="Health check endpoint",
)
def health_check() -> dict:
    """
    Lightweight health check endpoint.
    """
    return {
        "status": "healthy",
        "service": "agentic-ai-ebook-rag-chatbot",
        "version": "1.0.0",
    }


@app.post(
    "/query",
    response_model=QueryResponse,
    tags=["RAG Chatbot"],
    summary="Ask a document-grounded question",
)
def query_ebook(request: QueryRequest) -> QueryResponse:
    """
    Executes the LangGraph RAG workflow:
    1. Pinecone Vector Retrieval (top-k semantic matching + reciprocal rank fusion).
    2. Retrieval Relevance Grading (Self-RAG check).
    3. Grounded Answer Synthesis (constrained strictly to eBook context).
    4. Groundedness / Hallucination Verification.
    5. Final Confidence Scoring & Structured JSON Payload Delivery.
    """
    query = request.query.strip()
    if not query:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Query cannot be empty.",
        )

    try:
        logger.info("Executing RAG graph for query: '%s'", query)
        graph_result = rag_graph.invoke({"query": query})

        raw_chunks = graph_result.get("retrieved_chunks", [])

        # Exact required payload field: array of strings
        context_chunks_text = [chunk.get("text", "") for chunk in raw_chunks]

        # Rich metadata field for detailed inspection
        sources_meta = [
            ChunkMetadata(
                chunk_id=chunk.get("chunk_id", ""),
                text=chunk.get("text", ""),
                page_number=chunk.get("page_number", 0),
                similarity_score=chunk.get("similarity_score", 0.0),
            )
            for chunk in raw_chunks
        ]

        response = QueryResponse(
            query=query,
            final_answer=graph_result.get(
                "answer",
                "I cannot find this information in the provided eBook.",
            ),
            retrieved_context_chunks=context_chunks_text,
            confidence_score=graph_result.get("confidence_score", 0.0),
            is_grounded=graph_result.get("is_grounded", False),
            status=graph_result.get("status", "refused"),
            sources=sources_meta,
        )

        logger.info(
            "Query finished | status=%s | confidence=%.2f | chunks=%d",
            response.status,
            response.confidence_score,
            len(response.retrieved_context_chunks),
        )

        return response

    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Unexpected server error during query execution")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal workflow error occurred while processing query.",
        ) from exc


@app.get("/", response_class=HTMLResponse, tags=["System"])
def root_console():
    """
    Lightweight web UI console served directly by FastAPI.
    """
    return """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Agentic AI eBook RAG Chatbot</title>
  <style>
    body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif; background: #0f172a; color: #f8fafc; margin: 0; padding: 2rem; }
    .container { max-width: 900px; margin: 0 auto; }
    h1 { color: #38bdf8; margin-bottom: 0.2rem; }
    p.subtitle { color: #94a3b8; margin-top: 0; margin-bottom: 2rem; }
    .badge { display: inline-block; background: #1e293b; color: #38bdf8; padding: 0.25rem 0.75rem; border-radius: 9999px; font-size: 0.85rem; font-weight: 600; margin-right: 0.5rem; }
    .card { background: #1e293b; border: 1px solid #334155; border-radius: 0.75rem; padding: 1.5rem; margin-bottom: 1.5rem; }
    textarea { width: 100%; box-sizing: border-box; background: #0f172a; color: #fff; border: 1px solid #475569; border-radius: 0.5rem; padding: 0.75rem; font-size: 1rem; resize: vertical; min-height: 80px; }
    button { background: #0284c7; color: white; border: none; padding: 0.75rem 1.5rem; border-radius: 0.5rem; font-weight: 600; cursor: pointer; margin-top: 0.75rem; }
    button:hover { background: #0369a1; }
    .presets { margin: 1rem 0; }
    .preset-btn { background: #334155; color: #cbd5e1; font-size: 0.85rem; padding: 0.35rem 0.75rem; border-radius: 0.375rem; border: none; margin: 0.25rem; cursor: pointer; }
    .preset-btn:hover { background: #475569; color: #fff; }
    pre { background: #020617; border: 1px solid #1e293b; border-radius: 0.5rem; padding: 1rem; overflow-x: auto; color: #e2e8f0; font-size: 0.9rem; }
    .links a { color: #38bdf8; text-decoration: none; margin-right: 1.5rem; }
    .links a:hover { text-decoration: underline; }
  </style>
</head>
<body>
  <div class="container">
    <h1>Agentic AI eBook RAG Chatbot</h1>
    <p class="subtitle">LangGraph &bull; Pinecone &bull; Vector Embeddings &bull; Grounded Generation</p>
    <div style="margin-bottom: 1.5rem;" class="links">
      <span class="badge">LangGraph Cyclic RAG</span>
      <span class="badge">Pinecone Serverless</span>
      <a href="/docs" target="_blank">&rarr; Interactive Swagger API Docs</a>
    </div>

    <div class="card">
      <label for="queryInput"><strong>Enter your question:</strong></label>
      <div class="presets">
        <span style="font-size:0.85rem; color:#94a3b8;">Presets:</span>
        <button class="preset-btn" onclick="setQuery(1)">Definition & Scope</button>
        <button class="preset-btn" onclick="setQuery(2)">Architectural Components</button>
        <button class="preset-btn" onclick="setQuery(3)">Industry Use Cases</button>
        <button class="preset-btn" onclick="setQuery(4)">Comparison with GenAI</button>
        <button class="preset-btn" onclick="setQuery(5)">Challenges & Limitations</button>
        <button class="preset-btn" onclick="setQuery(6)">Out-of-Scope (France)</button>
      </div>
      <textarea id="queryInput" rows="3">What is the core definition of Agentic AI as outlined in the eBook?</textarea>
      <button onclick="sendQuery()" id="submitBtn">Submit Query</button>
    </div>

    <div id="resultCard" class="card" style="display:none;">
      <h3>Response Payload:</h3>
      <pre id="outputJson">Processing...</pre>
    </div>
  </div>

  <script>
    const presets = {
      1: "What is the core definition of Agentic AI as outlined in the eBook?",
      2: "What are the main architectural components required to build agentic systems?",
      3: "What real-world industry use cases for Agentic AI are discussed in the eBook?",
      4: "How does Agentic AI differ from traditional generative AI chatbots according to the text?",
      5: "What key challenges or limitations of Agentic AI are mentioned in the document?",
      6: "What is the capital of France?"
    };

    function setQuery(id) {
      document.getElementById('queryInput').value = presets[id];
    }

    async function sendQuery() {
      const q = document.getElementById('queryInput').value;
      const btn = document.getElementById('submitBtn');
      const resCard = document.getElementById('resultCard');
      const out = document.getElementById('outputJson');

      btn.disabled = true;
      btn.innerText = "Querying RAG Graph...";
      resCard.style.display = "block";
      out.innerText = "Running LangGraph workflow (Pinecone retrieval -> grading -> generation -> hallucination check)...";

      try {
        const response = await fetch('/query', {
          method: 'POST',
          headers: {'Content-Type': 'application/json'},
          body: JSON.stringify({query: q})
        });
        const data = await response.json();
        out.innerText = JSON.stringify(data, null, 2);
      } catch (err) {
        out.innerText = "Error: " + err;
      } finally {
        btn.disabled = false;
        btn.innerText = "Submit Query";
      }
    }
  </script>
</body>
</html>
"""


@app.exception_handler(Exception)
async def unhandled_exception_handler(request, error):
    logger.exception("Unhandled server error: %s", error)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"detail": "An unexpected server error occurred."},
    )