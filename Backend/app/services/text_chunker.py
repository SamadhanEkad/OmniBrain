"""
OmniBrain - Step 4: Text Cleaning and Chunking

Input:
    Backend/Data/processed/{document_id}.json

Output:
    Backend/Data/processed/{document_id}_chunks.json

Features:
    - PDF text cleaning
    - Recursive text splitting
    - Token counting with cl100k_base
    - Page number preservation
    - Conservative section-heading detection
    - Consistent metadata for downstream indexing
    - Original filename carried on every chunk (multi-PDF support)
"""

import json
import re
import unicodedata
from pathlib import Path
from typing import Any

import tiktoken
from langchain_text_splitters import RecursiveCharacterTextSplitter


# --------------------------------------------------
# 1. Configuration
# --------------------------------------------------

BACKEND_DIR = Path(__file__).resolve().parents[2]
PROCESSED_DIR = BACKEND_DIR / "Data" / "processed"

CHUNK_SIZE = 400
CHUNK_OVERLAP = 60
TOKEN_ENCODING = "cl100k_base"


# --------------------------------------------------
# 2. Clean extracted PDF text
# --------------------------------------------------

def clean_text(text: str) -> str:
    """Clean PDF extraction noise while preserving meaning."""

    text = unicodedata.normalize("NFC", text)

    # Standardize line endings.
    text = text.replace("\r\n", "\n").replace("\r", "\n")

    # Remove control characters but preserve tabs and newlines.
    text = "".join(
        char
        for char in text
        if unicodedata.category(char) != "Cc"
        or char in "\n\t"
    )

    # Normalize whitespace within each line.
    lines = [
        re.sub(r"[^\S\n]+", " ", line).strip()
        for line in text.split("\n")
    ]

    text = "\n".join(lines)

    # Remove excessive blank lines.
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()


# --------------------------------------------------
# 3. Identify likely section headings
# --------------------------------------------------

_STANDARD_HEADINGS = {
    "abstract", "background", "methods", "methodology", "materials and methods",
    "results", "discussion", "conclusion", "conclusions", "introduction",
    "overview", "summary", "executive summary", "literature review",
    "related work", "future work", "references", "bibliography", "appendix",
}


def is_likely_heading(line: str) -> bool:
    """
    Conservatively identify possible headings.

    This is a heuristic, not a semantic classifier. It cannot
    reliably identify every heading in every PDF.
    """

    line = line.strip()

    if not line or len(line) > 100:
        return False

    clean_line = line.strip("#*: \t").lower()
    if clean_line in _STANDARD_HEADINGS:
        return True

    # Ignore long sentences and lines ending like normal prose.
    if line.endswith((".", "?", "!", ";", ",")):
        return False

    words = line.split()

    if len(words) > 12:
        return False

    # Numbered headings such as "2. Applications" or "3.1 Methods".
    if re.match(
        r"^\d+(?:\.\d+)*[.)]?\s+\S+",
        line,
    ):
        return True

    # Short, uppercase headings such as "APPLICATIONS".
    letters = [char for char in line if char.isalpha()]

    if letters and len(letters) >= 3:
        if all(char.isupper() for char in letters):
            return True

    # Avoid guessing that ordinary short lines are headings.
    return False


def extract_page_heading(text: str) -> str | None:
    """Return the last likely heading found on a page, if any."""

    heading = None

    for line in text.splitlines():
        if is_likely_heading(line):
            heading = line.strip()

    return heading


def extract_chunk_heading(
    chunk_text: str,
    current_page_heading: str | None = None,
) -> str | None:
    """Extract section heading from chunk's initial lines or inherit page heading."""
    for line in chunk_text.splitlines()[:3]:
        line = line.strip()
        if is_likely_heading(line):
            return line.strip("#*: \t")
    return current_page_heading


# --------------------------------------------------
# 4. Split extracted JSON into chunks
# --------------------------------------------------

def chunk_extracted_json(
    input_json_path: str | Path,
    output_dir: str | Path | None = None,
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP,
    filename: str | None = None,
) -> dict[str, Any]:
    """
    Read page-wise extracted JSON, clean text, create chunks,
    preserve page metadata, and save the chunks as JSON.

    filename:
        The original uploaded PDF name (e.g. "report.pdf"). It is stored
        on every chunk so answers can name the file they came from.
    """

    input_path = Path(input_json_path).expanduser().resolve()

    if not input_path.is_file():
        raise FileNotFoundError(
            f"Extracted JSON file not found: {input_path}"
        )

    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive.")

    if chunk_overlap < 0 or chunk_overlap >= chunk_size:
        raise ValueError(
            "chunk_overlap must be non-negative and smaller "
            "than chunk_size."
        )

    with input_path.open("r", encoding="utf-8") as file:
        extracted = json.load(file)

    if not isinstance(extracted, dict):
        raise ValueError(
            "Input JSON must contain a JSON object."
        )

    document_id = extracted.get("document_id")
    pages = extracted.get("pages")

    if not isinstance(document_id, str) or not document_id.strip():
        raise ValueError("Missing or invalid document_id.")

    if (
        Path(document_id).name != document_id
        or document_id in {".", ".."}
        or "/" in document_id
        or "\\" in document_id
    ):
        raise ValueError("Invalid document_id.")

    if not isinstance(pages, list):
        raise ValueError(
            "Input JSON must contain a pages list."
        )

    # Prefer the explicit argument, then anything the extractor saved.
    source_filename = filename or extracted.get("filename")
    if not isinstance(source_filename, str) or not source_filename.strip():
        source_filename = None
    else:
        source_filename = source_filename.strip()

    # cl100k_base is an approximate token counter for this pipeline.
    # It is not necessarily the embedding model's native tokenizer.
    encoding = tiktoken.get_encoding(TOKEN_ENCODING)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=lambda text: len(encoding.encode(text)),
        separators=[
            "\n\n",
            "\n",
            ". ",
            "? ",
            "! ",
            "; ",
            " ",
            "",
        ],
        strip_whitespace=True,
    )

    chunks = []
    chunk_number = 0

    for page in pages:
        if not isinstance(page, dict):
            raise ValueError(
                "Each page must be a JSON object."
            )

        page_number = page.get("page_number")
        page_text = page.get("text", "")

        if (
            not isinstance(page_number, int)
            or isinstance(page_number, bool)
            or page_number < 1
        ):
            raise ValueError(
                "Each page needs a valid page_number."
            )

        if not isinstance(page_text, str):
            raise ValueError("Page text must be a string.")

        cleaned = clean_text(page_text)

        # Ignore pages without usable text.
        if not cleaned:
            continue

        # Keep chunks page-specific to preserve citations.
        section_heading = extract_page_heading(cleaned)
        page_chunks = splitter.split_text(cleaned)

        for chunk_text in page_chunks:
            chunk_text = chunk_text.strip()

            if not chunk_text:
                continue

            # Skip tiny running header / footer noise chunks (< 80 chars)
            if len(chunk_text) < 80:
                lower = chunk_text.lower()
                if (
                    re.search(r"page\s+\d+\s+of\s+\d+", lower)
                    or re.match(r"^page\s+\d+$", lower)
                    or "journal of" in lower
                    or "all rights reserved" in lower
                ):
                    continue

            chunk_number += 1
            chunk_heading = extract_chunk_heading(chunk_text, section_heading)

            chunk_record = {
                "chunk_id": f"chunk_{chunk_number:05d}",
                "document_id": document_id,
                "page_number": page_number,
                "section_heading": chunk_heading,
                "text": chunk_text,
                "token_count": len(
                    encoding.encode(chunk_text)
                ),
            }

            if source_filename:
                chunk_record["filename"] = source_filename

            chunks.append(chunk_record)

    if not chunks:
        raise ValueError(
            "No non-empty text chunks were generated."
        )

    result = {
        "document_id": document_id,
        "filename": source_filename,
        "source_file": extracted.get("source_file"),
        "chunk_count": len(chunks),
        "chunk_size_target": chunk_size,
        "chunk_overlap_target": chunk_overlap,
        "token_encoding": TOKEN_ENCODING,
        "chunks": chunks,
    }

    destination_dir = (
        Path(output_dir).expanduser().resolve()
        if output_dir is not None
        else PROCESSED_DIR
    )

    destination_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path = (
        destination_dir / f"{document_id}_chunks.json"
    )

    temporary_path = (
        destination_dir / f".{document_id}_chunks.json.tmp"
    )

    try:
        with temporary_path.open(
            "w",
            encoding="utf-8",
        ) as file:
            json.dump(
                result,
                file,
                ensure_ascii=False,
                indent=2,
            )

        temporary_path.replace(output_path)

    finally:
        if temporary_path.exists():
            temporary_path.unlink()

    return {
        **result,
        "output_json": str(output_path),
    }