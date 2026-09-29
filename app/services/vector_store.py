import time
import logging
from pinecone import Pinecone, ServerlessSpec

from app.config import settings

logger = logging.getLogger(__name__)


def get_dimension() -> int:
    if settings.llm_provider == "openai":
        return settings.openai_embedding_dimension
    return settings.embedding_dimension


def get_pinecone_index():
    """
    Connect to Pinecone, creating the serverless index if it doesn't already exist.
    Returns the ready Pinecone Index client.
    """
    pc = Pinecone(api_key=settings.pinecone_api_key)
    existing_indexes = pc.list_indexes().names()
    target_dimension = get_dimension()

    if settings.pinecone_index_name not in existing_indexes:
        logger.info(
            "Creating Pinecone index: %s (dimension=%s, metric=cosine)",
            settings.pinecone_index_name,
            target_dimension,
        )

        pc.create_index(
            name=settings.pinecone_index_name,
            dimension=target_dimension,
            metric="cosine",
            spec=ServerlessSpec(
                cloud=settings.pinecone_cloud,
                region=settings.pinecone_region,
            ),
        )

        while not pc.describe_index(settings.pinecone_index_name).status["ready"]:
            logger.info("Waiting for Pinecone index to become ready...")
            time.sleep(2)

    return pc.Index(settings.pinecone_index_name)