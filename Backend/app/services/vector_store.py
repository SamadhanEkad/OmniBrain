import argparse
import json
import logging
import math
import os
from pathlib import Path
from uuid import NAMESPACE_URL, uuid5

from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec

load_dotenv()

logger = logging.getLogger(__name__)

# --------------------------------------------------
# 1. Project paths and Pinecone configuration
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DIR = PROJECT_ROOT / "Data" / "processed"

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")

INDEX_NAME = os.getenv(
    "PINECONE_INDEX_NAME",
    "omnibrain-documents",
)
COLLECTION_NAME = INDEX_NAME

DEFAULT_EMBEDDING_MODEL = "nomic-embed-text-v2-moe"
DEFAULT_CLOUD = os.getenv("PINECONE_CLOUD", "aws")
DEFAULT_REGION = os.getenv("PINECONE_REGION", "us-east-1")

# Metadata fields that should be available during retrieval.
METADATA_FIELDS = (
    "section_heading",
    "domain",
    "key_concepts",
    "summary",
    "token_count",
    "start_char",
    "end_char",
)


# --------------------------------------------------
# 2. Small helpers
# --------------------------------------------------

def _validate_document_id(document_id: str) -> str:
    if not isinstance(document_id, str) or not document_id.strip():
        raise ValueError("Document ID cannot be empty.")

    if (
        Path(document_id).name != document_id
        or document_id in {".", ".."}
        or "/" in document_id
        or "\\" in document_id
    ):
        raise ValueError("Invalid document_id.")

    return document_id


def _point_id(document_id: str, chunk_id: str) -> str:
    """
    Deterministic Pinecone ID for one chunk.

    Because it is derived only from document_id + chunk_id, we can
    recompute every ID of a document later and delete exactly those
    vectors, without needing to list or filter anything.
    """
    return str(
        uuid5(
            NAMESPACE_URL,
            f"omnibrain:{document_id}:{chunk_id}",
        )
    )


# --------------------------------------------------
# 3. Connect to Pinecone
# --------------------------------------------------

def get_pinecone_client() -> Pinecone:
    """Connect to Pinecone and verify server availability."""
    api_key = os.getenv("PINECONE_API_KEY")
    if not api_key or not api_key.strip():
        raise RuntimeError(
            "PINECONE_API_KEY environment variable is not set. "
            "Please configure your Pinecone API key."
        )

    clean_key = api_key.strip().strip("'\"").strip()
    if clean_key.startswith("<") and clean_key.endswith(">"):
        raise RuntimeError(
            f"PINECONE_API_KEY is currently set to placeholder '{clean_key}'. "
            "Please replace it with your actual Pinecone API key (e.g. pcsk_...)."
        )

    client = Pinecone(api_key=clean_key)
    return client


# --------------------------------------------------
# 4. Create or validate the index
# --------------------------------------------------

def ensure_index(
    client: Pinecone,
    vector_size: int,
) -> None:
    """Create the Pinecone index if necessary and validate its configuration."""

    if (
        not isinstance(vector_size, int)
        or isinstance(vector_size, bool)
        or vector_size <= 0
    ):
        raise ValueError("vector_size must be a positive integer.")

    index_names = [idx.name for idx in client.list_indexes()]

    if INDEX_NAME not in index_names:
        client.create_index(
            name=INDEX_NAME,
            dimension=vector_size,
            metric="cosine",
            spec=ServerlessSpec(
                cloud=DEFAULT_CLOUD,
                region=DEFAULT_REGION,
            ),
        )
        print(
            f"Created Pinecone index: {INDEX_NAME} "
            f"({vector_size} dimensions)"
        )
        return

    description = client.describe_index(INDEX_NAME)
    existing_size = getattr(description, "dimension", None)
    existing_metric = getattr(description, "metric", None)

    if existing_size != vector_size:
        raise ValueError(
            f"Index dimension mismatch: expected "
            f"{vector_size}, found {existing_size}."
        )

    if existing_metric != "cosine":
        raise ValueError(
            f"The index must use cosine metric; found {existing_metric}."
        )

    print(f"Using existing Pinecone index: {INDEX_NAME}")


# Compatibility alias
ensure_collection = ensure_index


# --------------------------------------------------
# 5. Is this document already indexed?
# --------------------------------------------------

def is_document_indexed(client: Pinecone, document_id: str) -> bool:
    """Return True if at least one vector exists for this document."""
    _validate_document_id(document_id)

    index_names = [idx.name for idx in client.list_indexes()]
    if INDEX_NAME not in index_names:
        return False

    description = client.describe_index(INDEX_NAME)
    dimension = getattr(description, "dimension", None)
    if not isinstance(dimension, int) or dimension <= 0:
        return False

    index = client.Index(INDEX_NAME)
    result = index.query(
        vector=[0.1] * dimension,
        filter={"document_id": {"$eq": document_id}},
        top_k=1,
        include_metadata=False,
        include_values=False,
    )
    return bool(result.matches)


# --------------------------------------------------
# 6. Load and validate the embeddings JSON
# --------------------------------------------------

def load_embeddings(document_id: str):
    """Load and validate the embedding output for one document."""

    _validate_document_id(document_id)

    file_path = (
        PROCESSED_DIR / f"{document_id}_embeddings.json"
    )

    if not file_path.is_file():
        raise FileNotFoundError(
            f"Embeddings file not found: {file_path}"
        )

    with file_path.open("r", encoding="utf-8") as file:
        data = json.load(file)

    if not isinstance(data, dict):
        raise ValueError(
            "The embeddings file must contain a JSON object."
        )

    if data.get("document_id") != document_id:
        raise ValueError(
            "The document ID in the embeddings file does not match."
        )

    chunks = data.get("chunks")

    if not isinstance(chunks, list) or not chunks:
        raise ValueError(
            "The embeddings file contains no chunks."
        )

    vector_size = data.get("embedding_dimension")

    if vector_size is None:
        vector_size = data.get("vector_size")

    if vector_size is None:
        first_chunk = chunks[0]

        if not isinstance(first_chunk, dict):
            raise ValueError("The first chunk is invalid.")

        vector_size = len(first_chunk.get("embedding", []))

    if (
        not isinstance(vector_size, int)
        or isinstance(vector_size, bool)
        or vector_size <= 0
    ):
        raise ValueError("Invalid embedding vector dimension.")

    for index, chunk in enumerate(chunks, start=1):
        if not isinstance(chunk, dict):
            raise ValueError(
                f"Chunk {index} must be a JSON object."
            )

        vector = chunk.get("embedding")
        text = chunk.get("text") or chunk.get("content")
        chunk_id = chunk.get("chunk_id")

        if not isinstance(chunk_id, str) or not chunk_id.strip():
            raise ValueError(
                f"Chunk {index} has no valid chunk_id."
            )

        if not isinstance(text, str) or not text.strip():
            raise ValueError(
                f"Chunk {chunk_id} has missing or empty text."
            )

        if (
            not isinstance(vector, list)
            or len(vector) != vector_size
        ):
            raise ValueError(
                f"Chunk {chunk_id} has an invalid vector dimension."
            )

        if not all(
            isinstance(value, (int, float))
            and not isinstance(value, bool)
            and math.isfinite(value)
            for value in vector
        ):
            raise ValueError(
                f"Chunk {chunk_id} has invalid vector values."
            )

        if chunk.get("document_id", document_id) != document_id:
            raise ValueError(
                f"Chunk {chunk_id} belongs to another document."
            )

    print(f"Loaded {len(chunks)} chunks from: {file_path}")
    print(f"Vector dimension: {vector_size}")

    return data, chunks, vector_size


# --------------------------------------------------
# 7. Build and store vectors and metadata in Pinecone
# --------------------------------------------------

def store_embeddings(
    client: Pinecone,
    document_id: str,
    data: dict,
    chunks: list,
    vector_size: int,
    filename: str | None = None,
) -> None:
    """
    Store embedded chunks and their metadata in Pinecone.

    Every vector carries document_id, chunk_id, page_number and (when
    known) filename, so retrieval can filter by PDF and answers can cite
    the right file.
    """

    if not chunks:
        raise ValueError("Cannot store an empty chunk list.")

    resolved_filename = filename or data.get("filename")
    if not isinstance(resolved_filename, str) or not resolved_filename.strip():
        resolved_filename = None

    vectors = []

    for chunk in chunks:
        if chunk.get("document_id", document_id) != document_id:
            raise ValueError(
                f"Chunk {chunk.get('chunk_id')} belongs "
                "to another document."
            )

        vector = chunk["embedding"]
        text = (chunk.get("text") or chunk.get("content") or "").strip()
        chunk_id = str(chunk["chunk_id"])

        if len(vector) != vector_size:
            raise ValueError(
                f"Chunk {chunk_id} has an incorrect vector dimension."
            )

        payload = {
            "document_id": document_id,
            "chunk_id": chunk_id,
            "text": text,
            "embedding_model": (
                data.get("model")
                or data.get("embedding_model")
                or DEFAULT_EMBEDDING_MODEL
            ),
        }

        chunk_filename = resolved_filename or chunk.get("filename")
        if isinstance(chunk_filename, str) and chunk_filename.strip():
            payload["filename"] = chunk_filename.strip()

        page_number = chunk.get("page_number")
        if page_number is not None:
            payload["page_number"] = page_number

        # Preserve all supported optional metadata.
        for key in METADATA_FIELDS:
            value = chunk.get(key)

            if value is not None:
                if isinstance(value, (list, tuple)):
                    payload[key] = [str(item) for item in value]
                else:
                    payload[key] = value

        vectors.append({
            "id": _point_id(document_id, chunk_id),
            "values": vector,
            "metadata": payload,
        })

    index = client.Index(INDEX_NAME)
    batch_size = 100
    for i in range(0, len(vectors), batch_size):
        index.upsert(vectors=vectors[i : i + batch_size])

    print("\nStorage completed successfully.")
    print(f"Document ID: {document_id}")
    print(f"Filename: {resolved_filename}")
    print(f"Chunks stored: {len(vectors)}")
    print(f"Vector dimensions: {vector_size}")
    print(f"Index: {INDEX_NAME}")


# --------------------------------------------------
# 8. Delete one document's vectors
# --------------------------------------------------

def _load_chunk_ids(document_id: str) -> list[str]:
    """Read the chunk IDs of a document from its processed files."""
    for suffix in ("_chunks.json", "_embeddings.json"):
        path = PROCESSED_DIR / f"{document_id}{suffix}"
        if not path.is_file():
            continue

        try:
            with path.open("r", encoding="utf-8") as file:
                data = json.load(file)
        except (OSError, json.JSONDecodeError):
            logger.warning("Could not read %s while deleting vectors.", path)
            continue

        chunk_ids = [
            str(chunk["chunk_id"])
            for chunk in data.get("chunks", [])
            if isinstance(chunk, dict) and chunk.get("chunk_id")
        ]
        if chunk_ids:
            return chunk_ids

    return []


def delete_document_vectors(
    document_id: str,
    client: Pinecone | None = None,
) -> int:
    """
    Remove every vector belonging to one document.

    IDs are recomputed from the chunk list saved on disk, so this works
    on serverless indexes. If the chunk files are already gone it falls
    back to deleting by metadata filter. Returns the number of vectors
    deleted by ID (0 when the filter fallback was used).

    Call this BEFORE deleting the processed files.
    """
    _validate_document_id(document_id)

    own_client = client is None
    client = client or get_pinecone_client()

    try:
        index_names = [idx.name for idx in client.list_indexes()]
        if INDEX_NAME not in index_names:
            return 0

        index = client.Index(INDEX_NAME)
        chunk_ids = _load_chunk_ids(document_id)

        if chunk_ids:
            point_ids = [_point_id(document_id, cid) for cid in chunk_ids]
            for i in range(0, len(point_ids), 100):
                index.delete(ids=point_ids[i : i + 100])
            return len(point_ids)

        logger.warning(
            "No chunk list found for %s; deleting by metadata filter.",
            document_id,
        )
        index.delete(filter={"document_id": {"$eq": document_id}})
        return 0

    finally:
        if own_client and hasattr(client, "close"):
            client.close()


# --------------------------------------------------
# 9. Main execution function
# --------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Store OmniBrain embeddings in Pinecone."
    )

    parser.add_argument(
        "--document-id",
        required=True,
        help="Document ID generated during PDF upload.",
    )

    parser.add_argument(
        "--filename",
        default=None,
        help="Original PDF filename to store with each chunk.",
    )

    args = parser.parse_args()
    client = None

    try:
        # Load and validate embedding output.
        data, chunks, vector_size = load_embeddings(
            args.document_id
        )

        # Connect to Pinecone.
        client = get_pinecone_client()

        # Create or validate the index.
        ensure_index(client, vector_size)

        # Store vectors, text, and metadata.
        store_embeddings(
            client=client,
            document_id=args.document_id,
            data=data,
            chunks=chunks,
            vector_size=vector_size,
            filename=args.filename,
        )

        print("\nVector storage completed successfully.")

    except Exception as error:
        print(f"\nVector storage failed: {error}")
        raise SystemExit(1)

    finally:
        if client is not None and hasattr(client, "close"):
            client.close()


# --------------------------------------------------
# 10. Entry point
# --------------------------------------------------

if __name__ == "__main__":
    main()