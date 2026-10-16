import argparse
import json
import math
from pathlib import  Path
from uuid import NAMESPACE_URL, uuid5

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams

# 1. Project paths and Qdrant configuration

PROJECT_ROOT = Path(__file__).resolve().parents[2]

PROCESSED_DIR = PROJECT_ROOT / "Data" / "processed"

QDRANT_URL = "http://localhost:6333"

COLLECTION_NAME = "omnibrain_documents"

# 2. Connect to Qdrant

def get_qdrant_client():
    # Connect to the local Qdrant server

    client = QdrantClient(url=QDRANT_URL)

    # Confirm that Qdrant is reachable.
    client.get_collections()

    print(f"Connected to Qdrant at {QDRANT_URL}")

    return client

# 3. Create or validate the collection

def ensure_collection(client, vector_size: int):
    # Create the collection if needed and validate its size.
    if not client.collection_exists(COLLECTION_NAME):
        client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(
                size=vector_size,
                distance=Distance.COSINE,
            ),
        )

        print(
            f"Created collection: {COLLECTION_NAME}"
            f"({vector_size} dimensions)"
        )

    else:
        collection = client.get_collection(COLLECTION_NAME)
        vectors_config = collection.config.params.vectors

        # This project uses a single, unnamed vector per point.
        if not isinstance(vectors_config, VectorParams):
            existing_size = getattr(vectors_config, "size", None)
        else:
            existing_size = vectors_config.size

        if existing_size != vector_size:
            raise ValueError(
                f"Collection dimension mismatch: expected"
                f"{vector_size}, found {existing_size}."
                "Do not delete the collection automatically."
            )

        print(f"Using existing collection: {COLLECTION_NAME}")

# 4. Read abd validate the embeddings JSON

def load_embeddings(document_id: str):
    # Load embeddings generated in step 5

    file_path = (
        PROCESSED_DIR
        / f"{document_id}_embeddings.json"
    )

    if not file_path.is_file():
        raise FileNotFoundError(
            f"Embeddings file not found: {file_path}"
        )

    with file_path.open("r", encoding="utf-8") as file:
        data = json.load(file)

    if data.get("document_id") != document_id:
        raise ValueError(
            "The document ID in the JSON file does not exist."
        )

    chunks = data.get("chunks")

    if not isinstance(chunks, list) or not chunks:
        raise ValueError(
            "The embeddings file contains no chunks."
        )

    vector_size = data.get("vector_size")
    if vector_size is None:
        vector_size = data.get("embedding_dimension")

    if vector_size is None:
        vector_size = len(chunks[0].get("embedding", []))

    if not isinstance(vector_size, int) or vector_size <= 0:
        raise ValueError("Invalid embedding vector dimension.")

    for index, chunk in enumerate(chunks):
        vector = chunk.get("embedding")

        if not isinstance(vector, list) or len(vector) != vector_size:
            raise ValueError(
                f"Chunk {index} has an invalid vector dimension."
            )

        if not all(
            isinstance(value, (int, float))
            and not isinstance(value, bool)
            and math.isfinite(value)
            for value in vector
        ):
            raise ValueError(
                f"Chunk {index} has missing or empty text."
            )

    print(f"Loaded {len(chunks)} chunks from: {file_path}")
    print(f"vector dimension: {vector_size}")

    return data, chunks, vector_size

# 5.Store the vectors and metadata in Qdrant

def store_embeddings(
        client: QdrantClient,
        document_id: str,
        data: dict,
        chunks: list,
        vector_size: int,
):
    # Store embedded document chunks in Qdrant.

    points = []

    for chunk in chunks:
        vector = chunk["embedding"]
        text = chunk.get("text") or chunk.get("content")
        chunk_id = chunk["chunk_id"]

        # Generate a deterministic UUID for each document chunk.
        # Re-running Step 6 updates the same points instead of
        # creating duplicate points for the same document chunks.

        point_id = str(
            uuid5(
                NAMESPACE_URL,
                f"omnibrain:{document_id}:{chunk_id}",
            )
        )

        # Payload stores the original text and useful metadata.
        payload = {
            "document_id": document_id,
            "chunk_id": str(chunk_id),
            "text": text,
            "page_number": chunk.get("page_number"),
            "embedding_model": (
                data.get("embedding_model")
                or data.get("model")
                or "nomic-embed-text-v2-moe"
            ),
        }

        # Preserve additional metadata when available.
        for key in (
            "token_count",
            "start_char",
            "end_char",
        ):
            if chunk.get(key) is not None:
                payload[key] = chunk[key]

        # Create a Qdrant point containing the vector and playload.
        points.append(
            PointStruct(
                id=point_id,
                vector=vector,
                payload=payload,
            )
        )

    # Insert or update the points in Qdrant.
    client.upsert(
        collection_name = COLLECTION_NAME,
        points = points,
        wait = True,
    )

    print("\nStorage Completed successfully.")
    print(f"Document ID:{document_id}")
    print(f"Chunks stored: {len(points)}")
    print(f"Vector dimensions: {vector_size}")
    print(f"Collection: {COLLECTION_NAME}")

# 6. Main execution function

def main():
    parser = argparse.ArgumentParser(
        description="Store Omnibrain embedding in Qdrant."
    )

    parser.add_argument(
        "--document-id",
        required=True,
        help="Document ID generates during PDF upload.",
    )

    args = parser.parse_args()

    client = None

    try:
        # Load and validate step 5 output.
        data, chunks, vector_size = load_embeddings(
            args.document_id
        )

        # Connect to the Qdrant server.
        client = get_qdrant_client()

        # Create or validate the collection.
        ensure_collection(client, vector_size)

        # save vectors, text and metadata.
        store_embeddings(
            client = client,
            document_id=args.document_id,
            data=data,
            chunks=chunks,
            vector_size=vector_size,
        )

        print("\nStep Vector store completed successfully! ")

    except Exception as error:
        print(f"\nStep Vector store failed: {error}")
        raise SystemExit(1)

    finally:
        if client is not None:
            client.close()

# 7. Entry Point

if __name__ == "__main__":
    main()
