import logging
import os
import sys
from pathlib import Path
import requests

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import settings
from app.services.ingestion import load_and_chunk_pdf
from app.services.retrieval import get_embedding_model
from app.services.vector_store import get_dimension, get_pinecone_index

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

PDF_URL = "https://konverge.ai/pdf/Ebook-Agentic-AI.pdf"
DATA_DIR = Path("data")
PDF_PATH = DATA_DIR / "Ebook-Agentic-AI.pdf"
BATCH_SIZE = 25


def ensure_pdf_downloaded() -> Path:
    """
    Downloads the eBook PDF from the source URL if not already present.
    """
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    if not PDF_PATH.exists():
        logger.info("PDF not found locally. Downloading from %s...", PDF_URL)
        response = requests.get(PDF_URL, stream=True, timeout=60)
        response.raise_for_status()
        with open(PDF_PATH, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        logger.info("Downloaded %s successfully.", PDF_PATH)
    else:
        logger.info("Using existing PDF at %s", PDF_PATH)
    return PDF_PATH


def build_vectors(chunks: list) -> list[dict]:
    """
    Generate dense embeddings for each text chunk using the configured embedding model.
    """
    embedding_model = get_embedding_model()
    texts = [chunk.page_content for chunk in chunks]

    logger.info("Generating embeddings for %d text chunks using %s...", len(texts), settings.embedding_model)
    vectors = embedding_model.embed_documents(texts)

    records = []
    for chunk, vector in zip(chunks, vectors):
        records.append(
            {
                "id": chunk.metadata["chunk_id"],
                "values": vector,
                "metadata": {
                    "text": chunk.page_content,
                    "source": chunk.metadata.get("source", "Ebook-Agentic-AI.pdf"),
                    "page_number": int(chunk.metadata.get("page_number", 0)),
                    "chunk_index": int(chunk.metadata.get("chunk_index", 0)),
                },
            }
        )
    return records


def upsert_in_batches(index, records: list[dict]) -> None:
    """
    Uploads vector records to Pinecone in controlled batches.
    """
    total = len(records)
    logger.info("Beginning batch upsert of %d records to Pinecone...", total)
    for start in range(0, total, BATCH_SIZE):
        batch = records[start : start + BATCH_SIZE]
        index.upsert(vectors=batch)
        completed = min(start + BATCH_SIZE, total)
        logger.info("Upserted %d/%d vectors", completed, total)


def main() -> None:
    print("=" * 60)
    print("AGENTIC AI EBOOK INGESTION PIPELINE")
    print("=" * 60)

    pdf_file = ensure_pdf_downloaded()
    chunks = load_and_chunk_pdf(str(pdf_file))

    print(f"\nExtracted chunks: {len(chunks)}")
    print("Sample chunk metadata:", chunks[0].metadata)

    records = build_vectors(chunks)
    expected_dim = get_dimension()

    actual_dim = len(records[0]["values"])
    if actual_dim != expected_dim:
        raise ValueError(
            f"Embedding dimension mismatch: expected {expected_dim}, got {actual_dim}"
        )

    index = get_pinecone_index()
    upsert_in_batches(index, records)

    stats = index.describe_index_stats()
    print("\n" + "=" * 60)
    print("INGESTION COMPLETED SUCCESSFULLY")
    print("=" * 60)
    print(f"Pinecone total vector count: {stats.get('total_vector_count')}")


if __name__ == "__main__":
    main()