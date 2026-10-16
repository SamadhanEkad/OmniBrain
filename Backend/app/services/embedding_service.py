import json
import math
import sys
from pathlib import Path
from typing import Any

import ollama

# 1. Configuration

MODEL_NAME = "nomic-embed-text-v2-moe"

# The model supports upto 512 input tokens.
# We use safetly margin because token_count may be an estimate.
SAFE_TOKEN_LIMIT = 480

# Locate the Backend directory automatically.
BACKEND_DIR = Path(__file__).resolve().parents[2]

# Input and output files are stored here.
PROCESSED_DIR = BACKEND_DIR / "Data" / "processed"

# 2. Load chunks generated in step 4

def load_chunks(document_id: str) -> tuple[dict, list]:
    input_path = PROCESSED_DIR / f"{document_id}_chunks.json"

    if not input_path.is_file():
        raise FileNotFoundError(
            f"Chunked JSON not found: {input_path}"
        )

    with input_path.open("r", encoding="utf-8") as file:
        data = json.load(file)

    if not isinstance(data, dict):
        raise ValueError(
            "The chunked JSON must contain a JSON object."
        )

    chunks = data.get("chunks")

    if not isinstance(chunks, list) or not chunks:
        raise ValueError(
            "No chunks found. Expected a non-empty 'chunks' list."
        )

    return data, chunks

# 3. Generate embeddings

def generate_embeddings(
        document_id: str,
        save_output: bool = True,
) -> dict[str, Any]:
    """
    Generate embeddings for every chunk of a document.

    Preserves document ID, chunk ID, page number,
    text, and token count when available.
    """

    source_data, chunks = load_chunks(document_id)

    embedded_chunks = []
    embedding_dimension = None

    for index, chunk in enumerate(chunks, start=1):
        # Validate the chunk structure
        if not isinstance(chunk, dict):
            raise ValueError(
                f"chunk {index} must be a JSON object."
            )

        text = chunk.get("text")

        if not isinstance(text, str) or not text.strip():
            raise ValueError(
                f"Chunk {index} has missing or empty text."
            )

        chunk_id = chunk.get("chunk_id", f"chunk_{index:04d}")
        chunk_document_id = (
            chunk.get("document_id") or document_id
        )
        token_count = chunk.get("token_count")

        # Reject chunks that exceed our estimated safe limit.
        if (
            isinstance(token_count, int)
            and not isinstance(token_count, bool)
            and token_count > SAFE_TOKEN_LIMIT
        ):
            raise ValueError(
                f"chunk {chunk_id} has an estimated"
                f"{token_count} tokens. The safety limit is"
                f"{SAFE_TOKEN_LIMIT}. Reduce the chunk size"
                "in text_chunker.py and reenerate Step 4 output."
            )

        # Nomic retrieval prefix documnt content.
        embedding_input = f"search_document: {text}"

        # Request an embedding from Ollama.
        try:
            response = ollama.embed(
                model=MODEL_NAME,
                input=embedding_input,
            )
        except Exception as exc:
            raise RuntimeError(
                f"Embedding failed for chunk {chunk_id}."
                f"Check that ollama is running and model "
                f"'{MODEL_NAME}' is installed."
            ) from exc

        vectors = response.embeddings

        if not vectors or not vectors[0]:
            raise ValueError(
                f"Model returned an empty vector for {chunk_id}."
            )

        vector = vectors[0]

        # Validate vector values.
        if not all(
            isinstance(value, (int, float))
            and not isinstance(value, bool)
            and math.isfinite(value)
            for value in vector
        ):
            raise ValueError(
                f"Invalid numerical values in {chunk_id}."
            )

        # Ensure all chunks have the same vector dimension.
        if embedding_dimension is None:
            embedding_dimension = len(vector)
        elif len(vector) != embedding_dimension:
            raise ValueError(
                "Inconsistent embedding dimensions detected."
            )

        # Preserve original text and metadata.
        embedded_chunks.append(
            {
                "document_id": chunk_document_id,
                "chunk_id": chunk_id,
                "page_number": chunk.get("page_number"),
                "text": text,
                "token_count": token_count,
                "embedding": vector,
            }
        )

        print(
            f"Embedded chunk {index}/{len(chunks)}"
            f"({chunk_id})"
        )

    # Prepare the complete result.
    result = {
        "document_id": document_id,
        "source_file": source_data.get("source_file"),
        "model": MODEL_NAME,
        "embedding_dimension": embedding_dimension,
        "chunk_count": len(embedded_chunks),
        "chunks": embedded_chunks,
    }

    # Save the result for verification and debugging.
    if save_output:
        PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

        output_path = (
            PROCESSED_DIR
            / f"{document_id}_embeddings.json"
        )

        with output_path.open("w", encoding="utf-8") as file:
            json.dump(
                result,
                file,
                ensure_ascii=False,
                indent=2,
            )

        print(f"\nEnbeddings save to: {output_path}")

    print("\nEmbedding generation completed.")
    print(f"Documnet ID: {document_id}")
    print(f"Chunks embedded: {len(embedded_chunks)}")
    print(f"Vector Dimensions: {embedding_dimension}")

    return result

# 4.Allow excecution from the terminal

if __name__ == "__main__":

    if len(sys.argv) != 2:
        print(
            "Usage: python -m"
            "Backend.app.services.embedding_service"
            "<document_id"
        )
        sys.exit(1)
    
    document_id = sys.argv[1]

    try:
        generate_embeddings(document_id)
    except Exception as exc:
        print(f"\nERROR: {exc}")
        sys.exit(1)