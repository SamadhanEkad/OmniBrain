
import json
import logging
import math
import sys
from pathlib import Path
from typing import Any

import httpx
from google.genai import types
from google.genai.errors import APIError

from app.config import (
    EMBEDDING_FINGERPRINT,
    EMBEDDING_MODEL,
    EMBEDDING_PROVIDER,
    GEMINI_API_KEY,
    OLLAMA_URL,
    PROCESSED_DATA_DIR,
    VECTOR_SIZE,
)
from app.services.gemini_client import get_gemini_client

# --------------------------------------------------
# 1. Configuration
# --------------------------------------------------

MODEL_NAME = EMBEDDING_MODEL
MODEL_FINGERPRINT = EMBEDDING_FINGERPRINT
OLLAMA_CLIENT: Any | None = None
EMBEDDING_BATCH_SIZE = 32

# Nomic model input limit is 512 tokens.
# Keep a safety margin because our tokenizer is approximate.
SAFE_TOKEN_LIMIT = 480

BACKEND_DIR = Path(__file__).resolve().parents[2]
PROCESSED_DIR = PROCESSED_DATA_DIR

# Metadata fields that should survive embedding and indexing.
METADATA_FIELDS = (
    "section_heading",
    "domain",
    "key_concepts",
    "summary",
)
logger = logging.getLogger(__name__)

RETRIEVAL_DOCUMENT = "RETRIEVAL_DOCUMENT"
RETRIEVAL_QUERY = "RETRIEVAL_QUERY"


def _get_ollama_client() -> Any:
    global OLLAMA_CLIENT
    if OLLAMA_CLIENT is None:
        import ollama

        OLLAMA_CLIENT = ollama.Client(host=OLLAMA_URL, timeout=120)
    return OLLAMA_CLIENT


class EmbeddingProviderError(RuntimeError):
    """Raised when the configured embedding service cannot create vectors."""


class EmbeddingConfigurationError(RuntimeError):
    """Raised when the embedding provider or model is not configured."""


class EmbeddingAuthenticationError(EmbeddingProviderError):
    """Raised when the embedding provider rejects its credentials."""


class EmbeddingModelError(EmbeddingProviderError):
    """Raised when the configured embedding model is unavailable."""


class EmbeddingQuotaError(EmbeddingProviderError):
    """Raised when the embedding provider rate limit or quota is reached."""


class EmbeddingTimeoutError(EmbeddingProviderError):
    """Raised when the embedding provider times out."""


class EmbeddingUnavailableError(EmbeddingProviderError):
    """Raised when the embedding provider cannot be reached."""


def embed_texts(texts: list[str], task_type: str) -> list[list[float]]:
    """Embed text with the configured provider and validate vector dimensions."""
    if not texts or any(not isinstance(text, str) or not text.strip() for text in texts):
        raise ValueError("Embedding inputs must be non-empty strings.")
    if task_type not in {RETRIEVAL_DOCUMENT, RETRIEVAL_QUERY}:
        raise ValueError("Unsupported embedding task type.")
    if not MODEL_NAME:
        raise EmbeddingConfigurationError("EMBEDDING_MODEL is not configured.")
    if EMBEDDING_PROVIDER == "gemini" and not GEMINI_API_KEY:
        raise EmbeddingConfigurationError(
            "GEMINI_API_KEY is required for Gemini embeddings."
        )

    try:
        if EMBEDDING_PROVIDER == "gemini":
            response = get_gemini_client().models.embed_content(
                model=MODEL_NAME,
                contents=texts,
                config=types.EmbedContentConfig(
                    task_type=task_type,
                    output_dimensionality=VECTOR_SIZE,
                ),
            )
            vectors = [
                embedding.values
                for embedding in (response.embeddings or [])
                if embedding is not None
            ]
        elif EMBEDDING_PROVIDER == "ollama":
            prefix = (
                "search_document: "
                if task_type == RETRIEVAL_DOCUMENT
                else "search_query: "
            )
            response = _get_ollama_client().embed(
                model=MODEL_NAME,
                input=[prefix + text.strip() for text in texts],
            )
            vectors = response.embeddings
        else:
            raise ValueError("Unsupported embedding provider configuration.")
    except APIError as exc:
        status = getattr(exc, "code", None)
        if status in (401, 403):
            raise EmbeddingAuthenticationError(
                "Embedding provider authentication failed. Verify GEMINI_API_KEY."
            ) from exc
        if status == 404:
            raise EmbeddingModelError(
                "The configured embedding model is unavailable."
            ) from exc
        if status == 429:
            raise EmbeddingQuotaError(
                "Embedding provider quota or rate limit was reached."
            ) from exc
        if status in (408, 504):
            raise EmbeddingTimeoutError(
                "Embedding provider did not respond before the request timeout."
            ) from exc
        raise EmbeddingProviderError(
            "The configured embedding provider rejected the request."
        ) from exc
    except httpx.TimeoutException as exc:
        raise EmbeddingTimeoutError(
            "Embedding provider did not respond before the request timeout."
        ) from exc
    except httpx.RequestError as exc:
        raise EmbeddingUnavailableError(
            "The configured embedding provider could not be reached."
        ) from exc
    except Exception as exc:
        raise EmbeddingProviderError(
            f"The configured {EMBEDDING_PROVIDER} embedding provider failed."
        ) from exc

    if len(vectors) != len(texts):
        raise EmbeddingProviderError(
            "The embedding provider returned a different number of vectors "
            "than the number of submitted inputs."
        )

    for vector in vectors:
        if (
            not isinstance(vector, list)
            or len(vector) != VECTOR_SIZE
            or not all(
                isinstance(value, (int, float))
                and not isinstance(value, bool)
                and math.isfinite(value)
                for value in vector
            )
        ):
            raise EmbeddingProviderError(
                f"The embedding provider must return finite vectors with "
                f"exactly {VECTOR_SIZE} dimensions."
            )

    return vectors


# --------------------------------------------------
# 2. Load chunks generated by text_chunker.py
# --------------------------------------------------

def load_chunks(document_id: str) -> tuple[dict, list]:
    """Load the processed chunk JSON for one document."""

    if not isinstance(document_id, str) or not document_id.strip():
        raise ValueError("Document ID cannot be empty.")

    if (
        Path(document_id).name != document_id
        or document_id in {".", ".."}
        or "/" in document_id
        or "\\" in document_id
    ):
        raise ValueError("Invalid document_id.")

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

    if data.get("document_id") != document_id:
        raise ValueError(
            "The document ID in the chunk file does not match "
            "the requested document."
        )

    chunks = data.get("chunks")

    if not isinstance(chunks, list) or not chunks:
        raise ValueError(
            "No chunks found. Expected a non-empty 'chunks' list."
        )

    return data, chunks


# --------------------------------------------------
# 3. Generate embeddings
# --------------------------------------------------

def generate_embeddings(
    document_id: str,
    save_output: bool = True,
) -> dict[str, Any]:
    """
    Generate embeddings for all chunks of a document.

    Preserves:
        document_id, chunk_id, page_number, text, token_count,
        section_heading, domain, key_concepts, and summary.

    Semantic metadata is preserved when available. This function
    does not invent or generate missing semantic metadata.
    """

    source_data, chunks = load_chunks(document_id)

    prepared_chunks = []
    for index, chunk in enumerate(chunks, start=1):
        if not isinstance(chunk, dict):
            raise ValueError(
                f"Chunk {index} must be a JSON object."
            )

        text = chunk.get("text")

        if not isinstance(text, str) or not text.strip():
            raise ValueError(
                f"Chunk {index} has missing or empty text."
            )

        chunk_id = chunk.get(
            "chunk_id",
            f"chunk_{index:05d}",
        )

        chunk_document_id = (
            chunk.get("document_id") or document_id
        )

        if chunk_document_id != document_id:
            raise ValueError(
                f"Chunk {chunk_id} belongs to a different document."
            )

        token_count = chunk.get("token_count")

        # Reject chunks exceeding the estimated safe limit.
        if token_count is not None:
            if (
                not isinstance(token_count, int)
                or isinstance(token_count, bool)
                or token_count < 0
            ):
                raise ValueError(
                    f"Invalid token_count for chunk {chunk_id}."
                )

            if token_count > SAFE_TOKEN_LIMIT:
                raise ValueError(
                    f"Chunk {chunk_id} has {token_count} estimated "
                    f"tokens. The safety limit is {SAFE_TOKEN_LIMIT}. "
                    "Reduce CHUNK_SIZE in text_chunker.py and "
                    "regenerate the chunks."
                )

        clean_text = text.strip()
        prepared_chunks.append({
            "document_id": chunk_document_id,
            "chunk_id": str(chunk_id),
            "page_number": chunk.get("page_number"),
            "text": clean_text,
            "token_count": token_count,
            "metadata": {
                field: chunk[field]
                for field in METADATA_FIELDS
                if chunk.get(field) is not None
            },
        })
    vectors = _load_cached_vectors(document_id, prepared_chunks)
    if vectors is None:
        vectors = []
        for start in range(0, len(prepared_chunks), EMBEDDING_BATCH_SIZE):
            batch = prepared_chunks[start : start + EMBEDDING_BATCH_SIZE]
            batch_vectors = embed_texts(
                [chunk["text"] for chunk in batch],
                RETRIEVAL_DOCUMENT,
            )

            for chunk, vector in zip(batch, batch_vectors, strict=True):
                vectors.append(vector)
    else:
        logger.info("Reusing cached embeddings for document %s.", document_id)

    embedded_chunks = []
    for chunk, vector in zip(prepared_chunks, vectors, strict=True):
        embedded_chunk = {
            "document_id": chunk["document_id"],
            "chunk_id": chunk["chunk_id"],
            "page_number": chunk["page_number"],
            "text": chunk["text"],
            "token_count": chunk["token_count"],
            "embedding": vector,
            **chunk["metadata"],
        }
        embedded_chunks.append(embedded_chunk)

    # Prepare the complete embedding result.
    result = {
        "document_id": document_id,
        "source_file": source_data.get("source_file"),
        "model": MODEL_FINGERPRINT,
        "embedding_model": MODEL_FINGERPRINT,
        "embedding_dimension": VECTOR_SIZE,
        "chunk_count": len(embedded_chunks),
        "chunks": embedded_chunks,
    }

    # Save the result for debugging and vector indexing.
    if save_output:
        PROCESSED_DIR.mkdir(
            parents=True,
            exist_ok=True,
        )

        output_path = (
            PROCESSED_DIR
            / f"{document_id}_embeddings.json"
        )

        temporary_path = (
            PROCESSED_DIR
            / f".{document_id}_embeddings.json.tmp"
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

        logger.info("Saved %d embeddings for document %s.", len(embedded_chunks), document_id)

    return result


def _load_cached_vectors(
    document_id: str,
    chunks: list[dict],
) -> list[list[float]] | None:
    """Reuse a cache only when it exactly matches the current model and text."""
    cache_path = PROCESSED_DIR / f"{document_id}_embeddings.json"
    if not cache_path.is_file():
        return None

    try:
        with cache_path.open("r", encoding="utf-8") as file:
            cached = json.load(file)
    except (OSError, json.JSONDecodeError):
        logger.warning("Ignoring unreadable embedding cache for %s.", document_id)
        return None

    cached_chunks = cached.get("chunks") if isinstance(cached, dict) else None
    if (
        not isinstance(cached, dict)
        or cached.get("document_id") != document_id
        or cached.get("model") != MODEL_FINGERPRINT
        or cached.get("embedding_dimension") != VECTOR_SIZE
        or not isinstance(cached_chunks, list)
        or len(cached_chunks) != len(chunks)
    ):
        return None

    vectors = []
    for current, previous in zip(chunks, cached_chunks, strict=True):
        if (
            not isinstance(previous, dict)
            or previous.get("chunk_id") != current["chunk_id"]
            or previous.get("text") != current["text"]
        ):
            return None

        vector = previous.get("embedding")
        if (
            not isinstance(vector, list)
            or len(vector) != VECTOR_SIZE
            or not all(
                isinstance(value, (int, float))
                and not isinstance(value, bool)
                and math.isfinite(value)
                for value in vector
            )
        ):
            return None
        vectors.append(vector)

    return vectors


# --------------------------------------------------
# 4. Execute from the terminal
# --------------------------------------------------

if __name__ == "__main__":

    if len(sys.argv) != 2:
        print(
            "Usage: python -m app.services.embedding_service "
            "<document_id>"
        )
        sys.exit(1)

    document_id = sys.argv[1]

    try:
        generate_embeddings(document_id)

    except Exception as exc:
        print(f"\nERROR: {exc}")
        sys.exit(1)
