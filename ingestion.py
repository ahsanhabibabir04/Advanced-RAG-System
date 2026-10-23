"""PDF ingestion with page-level source tracking."""

from pathlib import Path

import pymupdf

from retrieval import split_text


def extract_pdf_pages(pdf_path):
    """Extract text from a PDF while preserving page numbers."""
    path = Path(pdf_path)

    if not path.exists():
        raise FileNotFoundError(f"PDF not found: {path}")

    if path.suffix.lower() != ".pdf":
        raise ValueError("Input file must be a PDF")

    pages = []

    with pymupdf.open(path) as document:
        for page_number, page in enumerate(document, start=1):
            text = page.get_text("text").strip()

            if text:
                pages.append({
                    "source": path.name,
                    "page": page_number,
                    "text": text,
                })

    return pages


def chunk_pdf_pages(pdf_path, chunk_size=1000, overlap=150):
    """Split extracted PDF pages into chunks with source metadata."""
    pages = extract_pdf_pages(pdf_path)
    chunks = []

    for page in pages:
        page_chunks = split_text(
            page["text"],
            chunk_size=chunk_size,
            overlap=overlap,
        )

        for chunk in page_chunks:
            chunks.append({
                "source": page["source"],
                "page": page["page"],
                "text": chunk,
            })

    return chunks