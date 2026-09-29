from pydantic import BaseModel, Field


class QueryRequest(BaseModel):
    query: str = Field(
        ...,
        min_length=2,
        max_length=1000,
        description="User question regarding the Agentic AI eBook",
        examples=["What is the core definition of Agentic AI as outlined in the eBook?"]
    )


class ChunkMetadata(BaseModel):
    chunk_id: str = Field(description="Unique vector ID")
    text: str = Field(description="Chunk raw text")
    page_number: int = Field(description="PDF page number")
    similarity_score: float = Field(description="Cosine similarity score from Pinecone")


class QueryResponse(BaseModel):
    query: str = Field(
        description="The submitted user question",
        examples=["What is Agentic AI?"]
    )
    final_answer: str = Field(
        description="Grounded response synthesized solely from the eBook context",
        examples=["Agentic AI refers to systems capable of autonomous decision-making and action..."]
    )
    retrieved_context_chunks: list[str] = Field(
        description="Text content of the retrieved context chunks from the PDF",
        examples=[["Chunk 1 text from PDF...", "Chunk 2 text from PDF..."]]
    )
    confidence_score: float = Field(
        description="Confidence score computed from retrieval similarity and answer grounding",
        examples=[0.92]
    )
    is_grounded: bool = Field(
        default=True,
        description="Indicates whether the response is fully supported by the retrieved context"
    )
    status: str = Field(
        default="answered",
        description="Workflow outcome status: 'answered' or 'refused'"
    )
    sources: list[ChunkMetadata] = Field(
        default_factory=list,
        description="Detailed metadata for each retrieved context chunk"
    )