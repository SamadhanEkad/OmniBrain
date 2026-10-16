"""
OmniBrain - Step 4: Text Cleaning and Chunking

  Input:
      Backend/Data/processed/{document_id}.json
  
  Output:
      Backend/Data/processed/{document_id}_chunks.json
  
  Chunking:
      Recursive splitting using cl100k_base token counting.
      Initial target: 700 tokens.
      Initial overlap: 100 tokens.
  """

import json
import re
import unicodedata
from pathlib import Path
from typing import Any

import tiktoken
from langchain_text_splitters import RecursiveCharacterTextSplitter

BACKEND_DIR = Path(__file__).resolve().parents[2]
PROCESSED_DIR = BACKEND_DIR / "Data" / "processed"

CHUNK_SIZE = 400
CHUNK_OVERLAP = 60
TOKEN_ENCODING = "cl100k_base"

def clean_text(text: str) -> str:
    """"Clean PDF extraction noise without removing meaningful content."""

    # Normalize equivalent Unicode representations.
    text = unicodedata.normalize("NFC",text)

    # Standardize Windows and Mac line ending.
    text = text.replace("\r\n","\n").replace("\r","\n")

    # Remove unwanted control characters while preserving tabs/newlines.
    text = "".join(
        char
        for char in text
        if unicodedata.category(char) != "Cc"
        or char in "\n\t"
    )

    # Normalize whitespace within each line.
    lines = [
        re.sub(r"[^\S\n]+"," ", line).strip()
        for line in text.split("\n")
    ]

    # Preserve paragraph boundries but remove excessive blank lines.
    text = "\n".join(lines)
    text = re.sub(r"\n{3,}","\n\n", text)

    return text.strip()

def chunk_extracted_json(
    input_json_path: str | Path,
    output_dir: str | Path | None = None,
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP,
) -> dict[str, Any]:
    """
    Read extracted page-wise JSON, clean text, create chunks,
    preserve page metadata, and save the result as JSON.
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
            "chunk_overlap must be >= 0 and smaller than chunk size."
        )

    # Load the output created by step 3
    with input_path.open("r", encoding="utf-8") as file:
        extracted = json.load(file)

    if not isinstance(extracted, dict):
        raise ValueError("Input JSON must contain a JSON object.")

    document_id = extracted.get("document_id")
    pages = extracted.get("pages")

    if not isinstance(document_id, str) or not document_id.strip():
        raise ValueError("Missing or  Invalid document_id.")

    if(
        Path(document_id).name != document_id
        or document_id in {".",".."}
        or "/" in document_id
        or "\\" in document_id
    ):
        raise ValueError("Invalid document_id.")

    if not isinstance(pages, list):
        raise ValueError("Input JSOn must contain a pages list.")

    # Token counting uses cl100k_base, an approximation for our
    # embedding model rather than its verified native tokenizer.
    encoding = tiktoken.get_encoding(TOKEN_ENCODING)

    
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=lambda text: len(encoding.encode(text)),
        separators=["\n\n", "\n", ". ", "? ", "! ", "; ", " ", ""],
        strip_whitespace=True,
    )


    chunks = []
    chunk_number = 0

    for page in pages:
        if not isinstance(page, dict):
            raise ValueError("Each page must be a JSON object.")

        page_number = page.get("page_number")
        page_text = page.get("text","")

        if (
            not isinstance(page_number, int)
            or isinstance(page_number, bool)
            or page_number < 1
        ):
            raise ValueError("Each page needs a valid page_number.")

        if not isinstance(page_text,str):
            raise ValueError("Page text must be a string.")
        
        cleaned = clean_text(page_text)

        # Ignore pages containing no meaningful text.
        if not cleaned:
            continue

        # Split each page separately to preserve page citations.
        page_chunks = splitter.split_text(cleaned)

        for chunk_text in page_chunks:
            if not chunk_text.strip(): 
                continue

            chunk_number += 1

            chunks.append(
                {
                    "chunk_id": f"chunk_{chunk_number:05d}",
                    "document_id": document_id,
                    "page_number": page_number,
                    "text": chunk_text,
                    "token_count": len(encoding.encode(chunk_text)),
                }
            )

        if not chunks:
            raise ValueError("No non-empty text chunks were generated.")

        result = {
            "document_id": document_id,
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
        destination_dir.mkdir(parents=True, exist_ok=True)

        output_path = destination_dir / f"{document_id}_chunks.json"
        temporary_path = destination_dir / f".{document_id}_chunks.json.tmp"

        try:
            with temporary_path.open("w", encoding="utf-8") as file:
                json.dump(result, file, ensure_ascii=False, indent=2)

            temporary_path.replace(output_path)

        finally:
            if temporary_path.exists():
                temporary_path.unlink()

        return {
            **result,
            "output_json": str(output_path),
        }