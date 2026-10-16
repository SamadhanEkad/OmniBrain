import argparse
import math

import requests
from qdrant_client import QdrantClient, models

# 1. Configuration

QDRANT_URL = "http://localhost:6333"
OLLAMA_URL = "http://localhost:11434"

COLLECTION_NAME = "omnibrain_documents"
EMBEDDING_MODEL = "nomic-embed-text-v2-moe"

VECTOR_SIZE = 768
DEFAULT_TOP_K = 3

# Keep this consistent with the Step 5 document embedding input.
# Use "search_query: " if Step 5 used "search_document: ".
QUERY_PREFIX = "search_query: "

# 2. Generate an embedding for the user's question

def embed_query(query:str) -> list[float]:
    # Convert user question into a numerical vector

    if not isinstance(query, str) or not query.strip():
        raise ValueError("The Search query cannot be empty.")

    response = requests.post(
        f"{OLLAMA_URL}/api/embed",
        json={
            "model": EMBEDDING_MODEL,
            "input": QUERY_PREFIX + query.strip(),
        },
        timeout=120,
    )

    response.raise_for_status()
    result = response.json()
    embeddings = result.get("embeddings")

    if not isinstance(embeddings, list) or not embeddings:
        raise ValueError(
            "Ollama returned no embeddings for the query."
        )

    vector = embeddings[0]

    if len(vector) != VECTOR_SIZE:
        raise ValueError(
            f"Expected {VECTOR_SIZE} dimensions, "
            f"received {len(vector)}."
        )

    if not all(
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(value)
        for value in vector
    ):
        raise ValueError(
            "The query embedding contains invalid values."
        )

    return vector

# 3. Connect to Qdrant

def get_qdrant_client() -> QdrantClient:
    """Connect to the local Qdrant collection."""

    client = QdrantClient(url=QDRANT_URL)

    try:
        collection = client.get_collection(COLLECTION_NAME)
        vector_config = collection.config.params.vectors

        # Support the single unnamed vector configuration.
        if isinstance(vector_config, dict):
            raise ValueError(
                "This service expects a single unnamed vector."
            )

        existing_size = getattr(vector_config, "size", None)
        existing_distance = getattr(vector_config, "distance", None)

        if existing_size != VECTOR_SIZE:
            raise ValueError(
                f"Expected {VECTOR_SIZE} dimensions, "
                f"but the collection uses {existing_size}."
            )

        if existing_distance != models.Distance.COSINE:
            raise ValueError(
                "The collection must use cosine distance."
            )

        print(f"Connected to collection: {COLLECTION_NAME}")
        return client

    except Exception:
        client.close()
        raise


# 4. Search for relavent document chunks

def search_documents(
        query: str,
        document_id: str | None = None,
        top_k: int = DEFAULT_TOP_K,
) -> list[dict]:
    # Retrive the most relevant chunks from Qdrant.

    if top_k < 1:
        raise ValueError("top_k must be at least 1.")
    
    # Step 7A: Embed the user's question.
    query_vector = embed_query(query)

    client = get_qdrant_client()

    try:
        # restrict the search to one document if requested.
        query_filter = None

        if document_id:
            query_filter = models.Filter(
                must = [
                    models.FieldCondition(
                        key="document_id",
                        match=models.MatchValue(
                            value=document_id
                        ),
                    )
                ]
            )   

        # Step 7B: Search using vector similarity.
        response = client.query_points(
            collection_name=COLLECTION_NAME,
            query=query_vector,
            query_filter=query_filter,
            limit=top_k,
            with_payload=True,
            with_vectors=False,
        )

        results = []

        for point in response.points:
            payload = point.payload or {}

            results.append(
                {
                    "score": point.score,
                    "document_id": payload.get("document_id"),
                    "chunk_id": payload.get("chunk_id"),
                    "page_number": payload.get("page_number"),
                    "text": payload.get("text", ""),
                }
            )

        return results

    finally:
        client.close()

# Display the search results

def main():
    # Run semantic search from the terminal.

    parser = argparse.ArgumentParser(
        description="Search omnibrain document embeddings."
    )

    parser.add_argument(
        "--query",
        required=True,
        help="Question or text to search for.",
    )

    parser.add_argument(
        "--document-id",
        default=None,
        help="Optional document ID to restrict the serach.",
    )

    parser.add_argument(
        "--top-k",
        type=int,
        default=DEFAULT_TOP_K,
        help="Number of relevant chunks to return."
    )

    args = parser.parse_args()

    try:
        results = search_documents(
            query=args.query,
            document_id=args.document_id,
            top_k=args.top_k,
        )

        if not results:
            print("No matching chunks were found.")
            return

        print(f"\nQuestion: {args.query}")
        print(f"Results found: {len(results)}")

        for index, result in enumerate(results, start=1):
            print(f"\n{'=' * 60}")
            print(f"Result {index}")
            print(f"Similarity score: {result['score']:.4f}")
            print(f"Document ID: {result['document_id']}")
            print(f"Chunk ID: {result['chunk_id']}")
            print(f"Page Number: {result['page_number']}")
            print(f"Text:\n{result['text']}")

    except Exception as error:
        print(f"\nSemantic search failed: {error}")
        raise SystemExit(1)

# 6. Entry point

if __name__ == "__main__":
    main()