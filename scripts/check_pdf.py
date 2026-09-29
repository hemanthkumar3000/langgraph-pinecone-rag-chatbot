from pathlib import Path

from pypdf import PdfReader


PDF_PATH = Path("data/Ebook-Agentic-AI.pdf")


def main() -> None:
    if not PDF_PATH.exists():
        raise FileNotFoundError(
            f"PDF not found at: {PDF_PATH}. "
            "Download the assignment eBook and save it there."
        )

    reader = PdfReader(str(PDF_PATH))

    print(f"Total pages: {len(reader.pages)}")

    for page_number, page in enumerate(reader.pages[:3], start=1):
        text = page.extract_text() or ""

        print(f"\n{'=' * 60}")
        print(f"PAGE {page_number} | Extracted characters: {len(text)}")
        print(f"{'=' * 60}")
        print(text[:1000])


if __name__ == "__main__":
    main()