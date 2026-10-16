
"""
OmniBrain - Step 3: PDF Text Extraction

Responsibilities:
- Extract text page by page using PyMuPDF.
- Preserve 1-based page numbers.
- Keep document ID and source filename.
- Save extracted content as JSON in Backend/Data/processed/.
"""

import json
from pathlib import Path
from typing import Any

import pymupdf

# Backend directory: OmniBrain/Backend/
BACKEND_DIR = Path(__file__).resolve().parents[2]

# Default Location for extracted text
PROCESSED_DIR = BACKEND_DIR / "Data" / "processed"

def extract_pdf_text(
        pdf_path: str | Path,
        document_id: str | None = None,
        output_dir: str | Path | None = None,
) -> dict[str,any]:
    """
    Extract text from a PDF and save the result as JSON.

    Args:
        pdf_path: Path to the original PDF.
        document_id: Existing document ID; defaults to the PDF filename.
        output_dir: Optional output directory for the JSON file.

    Returns:
        Dictionary containing document metadata, extracted pages,
        and the path of the generated JSON file.

    Raises:
        FileNotFoundError: If the PDF does not exist.
        ValueError: If the file is not a PDF or contains no extractable text.
        RuntimeError: If PyMuPDF cannot read the PDF.
    """
    source_path = Path(pdf_path).expanduser().resolve()

    # Validate the input file.
    if not source_path.is_file():
        raise FileNotFoundError(
            f"PDF file not found: {source_path}"
        )

    if source_path.suffix.lower() != ".pdf":
        raise ValueError("The input file must have a .pdf extension.")

    # Preserve the document ID generated during step 2
    doc_id = document_id or source_path.stem

    # Prevent document Ids from creating paths outside the output folder.
    if(
        not doc_id
        or Path(doc_id).name != doc_id
        or doc_id in {".",".."}
        or "/" in doc_id
        or "\\" in doc_id
    ):
        raise ValueError("Invalid document ID.")

    destination_dir = (
        Path(output_dir).expanduser().resolve()
        if output_dir is not None
        else PROCESSED_DIR  
    )

    pages = []

    try:
        with pymupdf.open(source_path) as pdf:
            for page_number, page in enumerate(pdf, start=1):
                # Extract text while preserving the page association.
                text = page.get_text("text").strip()

                pages.append(
                    {
                        "page_number": page_number,
                        "text": text,
                    }
                )

    except Exception as exc:
        raise RuntimeError(
            f"Failed to extarct text from {source_path.name}: {exc}"
        ) from exc

    #  Detect scanned PDFs or documents with no embedded text.
    if not any(page["text"] for page in pages):
        raise ValueError(
            "No extractable text was found. "
            "This PDF may be scanned and require OCR."
        )

    # Build the structured extraction result
    result = {
        "document_id": doc_id,
        "source_file": source_path.name,
        "page_count": len(pages),
        "pages": pages,
    }

    # Create the processed directory when it does not exit.
    destination_dir.mkdir(parents=True, exist_ok=True)

    output_path = destination_dir / f"{doc_id}.json"
    temporary_path = destination_dir / f".{doc_id}.json.tmp"

    try:
        # Write to a temporary file before replacing the final output
        with temporary_path.open("w", encoding="utf-8") as file:
            json.dump(
                result,
                file,
                ensure_ascii=False,
                indent = 2,
            )

        temporary_path.replace(output_path)

    finally:
        # Remove on incomplete temporary file if writing fails.
        if temporary_path.exists():
            temporary_path.unlink()

    # Return the result and the saved JSON Location.
    return{
        **result,
        "output_json": str(output_path),
    }