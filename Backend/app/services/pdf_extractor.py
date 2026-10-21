
"""
OmniBrain - Step 3: PDF Text Extraction

Responsibilities:
- Extract text page by page using PyMuPDF.
- Preserve 1-based page numbers.
- Keep document ID and source filename.
- Save extracted content as JSON in Backend/Data/processed/.
"""

import json
import logging
from pathlib import Path
from typing import Any

import pymupdf

logger = logging.getLogger(__name__)

# Backend directory: OmniBrain/Backend/
BACKEND_DIR = Path(__file__).resolve().parents[2]

# Default location for extracted text
PROCESSED_DIR = BACKEND_DIR / "Data" / "processed"


def extract_pdf_text(
    pdf_path: str | Path,
    document_id: str | None = None,
    output_dir: str | Path | None = None,
) -> dict[str, Any]:
    """
    Extract embedded PDF text page by page and save it as JSON.

    Note:
        This function does not perform OCR or visual chart/table analysis.
    """
    source_path = Path(pdf_path).expanduser().resolve()

    # 1. Validate the source file.
    if not source_path.is_file():
        raise FileNotFoundError(f"PDF file not found: {source_path}")

    if source_path.suffix.lower() != ".pdf":
        raise ValueError("The input file must have a .pdf extension.")

    # 2. Validate the document ID.
    doc_id = document_id or source_path.stem

    if (
        not isinstance(doc_id, str)
        or not doc_id.strip()
        or Path(doc_id).name != doc_id
        or doc_id in {".", ".."}
        or "/" in doc_id
        or "\\" in doc_id
    ):
        raise ValueError("Invalid document ID.")

    # 3. Resolve the output directory.
    destination_dir = (
        Path(output_dir).expanduser().resolve()
        if output_dir is not None
        else PROCESSED_DIR
    )

    pages: list[dict[str, Any]] = []
    pdf_metadata: dict[str, Any] = {}

    # 4. Extract text while preserving page numbers.
    try:
        with pymupdf.open(source_path) as pdf:
            if pdf.is_encrypted:
                raise ValueError(
                    "This PDF is encrypted and cannot be processed "
                    "without a valid password."
                )

            if len(pdf) == 0:
                raise ValueError("The PDF contains no pages.")

            raw_meta = pdf.metadata or {}
            for k, v in raw_meta.items():
                if isinstance(v, str) and v.strip():
                    pdf_metadata[k] = v.strip()

            for page_number, page in enumerate(pdf, start=1):
                text = page.get_text("text", sort=True).strip()

                pages.append(
                    {
                        "page_number": page_number,
                        "text": text,
                    }
                )

    except ValueError:
        raise
    except Exception as exc:
        logger.exception(
            "Failed to extract text from %s",
            source_path.name,
        )
        raise RuntimeError(
            f"Failed to extract text from {source_path.name}: {exc}"
        ) from exc

    # 5. Detect documents with no embedded text.
    if not any(page["text"] for page in pages):
        raise ValueError(
            "No extractable text was found. "
            "This PDF may be scanned and require OCR."
        )

    # 6. Build the extraction result.
    result: dict[str, Any] = {
        "document_id": doc_id,
        "source_file": source_path.name,
        "page_count": len(pages),
        "pdf_metadata": pdf_metadata,
        "pages": pages,
    }

    # 7. Save JSON safely.
    destination_dir.mkdir(parents=True, exist_ok=True)

    output_path = destination_dir / f"{doc_id}.json"
    temporary_path = destination_dir / f".{doc_id}.json.tmp"

    try:
        with temporary_path.open("w", encoding="utf-8") as file:
            json.dump(
                result,
                file,
                ensure_ascii=False,
                indent=2,
            )
            file.write("\n")

        temporary_path.replace(output_path)

    except Exception:
        logger.exception(
            "Failed to save extracted JSON for document %s",
            doc_id,
        )
        raise

    finally:
        if temporary_path.exists():
            temporary_path.unlink()

    # 8. Return the data and output path.
    return {
        **result,
        "output_json": str(output_path),
    }
