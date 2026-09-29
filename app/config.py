from typing import Literal, Optional
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # Pinecone
    pinecone_api_key: str
    pinecone_index_name: str = "agentic-ai-ebook-index-local"
    pinecone_cloud: str = "aws"
    pinecone_region: str = "us-east-1"

    # Provider: 'ollama' (default local/free) or 'openai' (cloud)
    llm_provider: Literal["ollama", "openai"] = "ollama"

    # OpenAI settings
    openai_api_key: Optional[str] = None
    openai_chat_model: str = "gpt-4o-mini"
    openai_embedding_model: str = "text-embedding-3-small"
    openai_embedding_dimension: int = 1536

    # Ollama settings
    embedding_model: str = "nomic-embed-text"
    chat_model: str = "llama3.2"
    embedding_dimension: int = 768

    # Ingestion & Chunking
    chunk_size: int = 800
    chunk_overlap: int = 120

    # Retrieval
    top_k: int = 5
    similarity_threshold: float = 0.50
    max_retries: int = 1

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


settings = Settings()