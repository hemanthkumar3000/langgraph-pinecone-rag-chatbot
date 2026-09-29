import hashlib
from pathlib import Path

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader

from app.config import settings


def create_chunk_id(page_number: int, chunk_index: int, text: str) -> str:
    """
    Create a stable vector ID.

    A stable ID prevents duplicates when the same PDF is indexed again.
    """
    content_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()[:12]
    return f"agentic-ai-page-{page_number}-chunk-{chunk_index}-{content_hash}"


def load_pdf_pages(pdf_path: str) -> list[Document]:
    """
    Extract the PDF one page at a time.

    Keeping page boundaries allows us to return page numbers
    in API retrieval results.
    """
    file_path = Path(pdf_path)

    if not file_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    reader = PdfReader(str(file_path))
    documents: list[Document] = []

    for page_number, page in enumerate(reader.pages, start=1):
        text = (page.extract_text() or "").strip()

        if not text:
            print(f"Skipping page {page_number}: no extractable text.")
            continue

        documents.append(
            Document(
                page_content=text,
                metadata={
                    "source": file_path.name,
                    "page_number": page_number,
                },
            )
        )

    if not documents:
        raise ValueError(
            "No text could be extracted from the PDF. "
            "The PDF may be image-based or protected."
        )

    return documents


def chunk_pdf_documents(documents: list[Document]) -> list[Document]:
    """
    Split page documents into smaller overlapping chunks.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    chunks = splitter.split_documents(documents)

    for chunk_index, chunk in enumerate(chunks, start=1):
        page_number = chunk.metadata["page_number"]
        chunk.metadata["chunk_id"] = create_chunk_id(
            page_number=page_number,
            chunk_index=chunk_index,
            text=chunk.page_content,
        )
        chunk.metadata["chunk_index"] = chunk_index

    return chunks


def load_and_chunk_pdf(pdf_path: str) -> list[Document]:
    """
    Complete local preprocessing pipeline:
    PDF → page documents → overlapping text chunks.
    """
    page_documents = load_pdf_pages(pdf_path)
    chunks = chunk_pdf_documents(page_documents)

    print(f"Extracted pages with text: {len(page_documents)}")
    print(f"Created chunks: {len(chunks)}")

    return chunks