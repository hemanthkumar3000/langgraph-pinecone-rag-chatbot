import json
import streamlit as st

from app.config import settings
from app.graph import rag_graph

st.set_page_config(
    page_title="Agentic AI eBook RAG Chatbot",
    page_icon="🤖",
    layout="wide",
)

st.title("🤖 Agentic AI eBook RAG Chatbot")
st.caption(
    "Custom LangGraph Cyclic RAG • Pinecone Vector DB • Strict Document Grounding"
)

with st.sidebar:
    st.header("⚙️ System Architecture")
    st.markdown(
        """
        - **Orchestration:** LangGraph StateGraph
        - **Vector DB:** Pinecone (`agentic-ai-ebook-index-local`)
        - **Embeddings:** Nomic / OpenAI
        - **LLM Engine:** Llama 3.2 / GPT-4o-mini
        - **Knowledge Base:** *Agentic AI eBook* (Konverge AI)
        """
    )
    st.divider()
    st.subheader("Configuration")
    st.write(f"**Provider:** `{settings.llm_provider}`")
    st.write(f"**Top-K Chunks:** `{settings.top_k}`")
    st.write(f"**Threshold:** `{settings.similarity_threshold}`")

preset_queries = [
    "What is the core definition of Agentic AI as outlined in the eBook?",
    "What are the main architectural components required to build agentic systems?",
    "What real-world industry use cases for Agentic AI are discussed in the eBook?",
    "How does Agentic AI differ from traditional generative AI chatbots according to the text?",
    "What key challenges or limitations of Agentic AI are mentioned in the document?",
    "What is the capital of France?",
]

st.subheader("Select a Benchmark Sample Query:")
cols = st.columns(3)
selected_query = None

for i, query_text in enumerate(preset_queries):
    col = cols[i % 3]
    label = f"Query {i+1}: {query_text[:35]}..."
    if col.button(label, use_container_width=True, key=f"btn_{i}"):
        st.session_state["query_input"] = query_text

user_query = st.text_area(
    "Or enter your custom question:",
    value=st.session_state.get(
        "query_input",
        "What is the core definition of Agentic AI as outlined in the eBook?",
    ),
    height=80,
)

if st.button("Run LangGraph RAG Pipeline", type="primary"):
    if not user_query.strip():
        st.warning("Please enter a valid query.")
    else:
        with st.spinner("Executing LangGraph RAG workflow..."):
            try:
                result = rag_graph.invoke({"query": user_query.strip()})

                raw_chunks = result.get("retrieved_chunks", [])
                text_chunks = [c.get("text", "") for c in raw_chunks]

                payload = {
                    "query": user_query.strip(),
                    "final_answer": result.get(
                        "answer",
                        "I cannot find this information in the provided eBook.",
                    ),
                    "retrieved_context_chunks": text_chunks,
                    "confidence_score": result.get("confidence_score", 0.0),
                }

                status = result.get("status", "answered")
                is_grounded = result.get("is_grounded", False)
                conf = payload["confidence_score"]

                st.divider()
                st.subheader("Results")

                m1, m2, m3 = st.columns(3)
                m1.metric("Status", status.upper())
                m2.metric("Confidence Score", f"{conf:.2f}")
                m3.metric("Grounded in eBook", "YES" if is_grounded else "NO")

                st.markdown("### Final Answer")
                if status == "refused":
                    st.error(payload["final_answer"])
                    if result.get("refusal_reason"):
                        st.info(f"Reason: {result.get('refusal_reason')}")
                else:
                    st.success(payload["final_answer"])

                st.markdown("### Retrieved Context Chunks")
                if not raw_chunks:
                    st.write("No chunks passed the similarity threshold.")
                else:
                    for idx, chunk in enumerate(raw_chunks, start=1):
                        with st.expander(
                            f"Chunk {idx} (Page {chunk.get('page_number')} | Similarity: {chunk.get('similarity_score')})"
                        ):
                            st.write(chunk.get("text"))

                st.markdown("### Output Payload (JSON)")
                st.json(payload)

            except Exception as e:
                st.error(f"Execution Error: {e}")
