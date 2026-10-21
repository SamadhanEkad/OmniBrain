import argparse
import json
import logging
import math
import os
import time
from pathlib import Path

import requests
from dotenv import load_dotenv
from pinecone import Pinecone

BACKEND_DIR = Path(__file__).resolve().parents[2]
load_dotenv(BACKEND_DIR / ".env")
load_dotenv()

logger = logging.getLogger(__name__)

# --------------------------------------------------
# 1. Configuration
# --------------------------------------------------

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")

INDEX_NAME = os.getenv(
    "PINECONE_INDEX_NAME",
    "omnibrain-documents",
)
COLLECTION_NAME = INDEX_NAME

OLLAMA_URL = os.getenv(
    "OLLAMA_URL",
    "http://localhost:11434",
).rstrip("/")

EMBEDDING_MODEL = os.getenv(
    "EMBEDDING_MODEL",
    "nomic-embed-text-v2-moe",
)

# Must match the embedding dimensions and Pinecone index.
VECTOR_SIZE = int(os.getenv("VECTOR_SIZE", "768"))

DEFAULT_TOP_K = 3
MAX_TOP_K = 20

# Must match the query/document embedding conventions used
# when the PDF chunks were originally indexed.
QUERY_PREFIX = "search_query: "

EMBEDDING_TIMEOUT = (5, 45)


# --------------------------------------------------
# 2. Generate an embedding for the user's question
# --------------------------------------------------

def embed_query(query: str) -> list[float]:
    """
    Convert a question into a validated embedding vector.

    This function embeds only the user's query.
    It does not extract, chunk, or re-embed the PDF.
    """

    if not isinstance(query, str) or not query.strip():
        raise ValueError("The search query cannot be empty.")

    if VECTOR_SIZE <= 0:
        raise ValueError("VECTOR_SIZE must be a positive integer.")

    try:
        response = requests.post(
            f"{OLLAMA_URL}/api/embed",
            json={
                "model": EMBEDDING_MODEL,
                "input": QUERY_PREFIX + query.strip(),
            },
            timeout=EMBEDDING_TIMEOUT,
        )

        response.raise_for_status()
        result = response.json()

    except requests.exceptions.Timeout as exc:
        raise RuntimeError(
            "Query embedding timed out. Check Ollama performance "
            "and ensure the embedding model is available."
        ) from exc

    except requests.exceptions.ConnectionError as exc:
        raise RuntimeError(
            f"Cannot connect to Ollama at {OLLAMA_URL}. "
            "Start Ollama and verify the service is running."
        ) from exc

    except requests.exceptions.HTTPError as exc:
        details = response.text[:1000]
        raise RuntimeError(
            "Ollama embedding request failed "
            f"(HTTP {response.status_code}): {details}"
        ) from exc

    except requests.exceptions.RequestException as exc:
        raise RuntimeError(
            f"Query embedding request failed: {exc}"
        ) from exc

    except ValueError as exc:
        raise RuntimeError(
            "Ollama returned an invalid JSON embedding response."
        ) from exc

    embeddings = result.get("embeddings")

    if not isinstance(embeddings, list) or not embeddings:
        raise RuntimeError(
            "Ollama returned no embeddings for the query."
        )

    vector = embeddings[0]

    if not isinstance(vector, list) or len(vector) != VECTOR_SIZE:
        actual_size = (
            len(vector) if isinstance(vector, list) else "invalid"
        )

        raise ValueError(
            f"Expected {VECTOR_SIZE} embedding dimensions, "
            f"but Ollama returned {actual_size}. "
            "Check the embedding model and VECTOR_SIZE."
        )

    if not all(
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(value)
        for value in vector
    ):
        raise ValueError(
            "The query embedding contains invalid numeric values."
        )

    return vector


# --------------------------------------------------
# 3. Connect to Pinecone
# --------------------------------------------------

def get_pinecone_client() -> Pinecone:
    """
    Connect to Pinecone and validate the index configuration.

    Does not create, delete, or re-index the collection.
    """
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

    try:
        index_names = [idx.name for idx in client.list_indexes()]

        if INDEX_NAME not in index_names:
            raise ValueError(
                f"Pinecone index '{INDEX_NAME}' does not exist. "
                "Ensure the index is created before querying."
            )

        description = client.describe_index(INDEX_NAME)
        existing_size = getattr(description, "dimension", None)
        existing_metric = getattr(description, "metric", None)

        if existing_size != VECTOR_SIZE:
            raise ValueError(
                f"Index '{INDEX_NAME}' uses {existing_size} dimensions, "
                f"but VECTOR_SIZE is {VECTOR_SIZE}. The embedding model, "
                "index, and configuration must agree."
            )

        if existing_metric != "cosine":
            raise ValueError(
                f"Index '{INDEX_NAME}' must use cosine distance; "
                f"found {existing_metric}."
            )

        logger.info(
            "Connected to Pinecone index '%s' "
            "(vector size=%s, metric=%s).",
            INDEX_NAME,
            existing_size,
            existing_metric,
        )

        return client

    except Exception as exc:
        if hasattr(client, "close"):
            client.close()
        if "UnauthorizedError" in type(exc).__name__ or "401" in str(exc) or "Invalid API key" in str(exc):
            raise RuntimeError(
                "Pinecone authentication failed: The provided PINECONE_API_KEY is invalid or unauthorized (HTTP 401). "
                "Please check your Pinecone API key and ensure it is active and correctly set in your environment."
            ) from exc
        raise


def load_all_document_chunks(document_id: str) -> list[dict]:
    """
    Load all processed chunks for a document from local processed files.
    Allows full-document coverage for small documents (e.g. resumes, notes).
    """
    if not isinstance(document_id, str) or not document_id.strip():
        return []

    chunks_file = BACKEND_DIR / "Data" / "processed" / f"{document_id.strip()}_chunks.json"
    if not chunks_file.is_file():
        return []

    try:
        with chunks_file.open("r", encoding="utf-8") as f:
            data = json.load(f)
            chunks = data.get("chunks", [])
            for c in chunks:
                if isinstance(c, dict):
                    c.setdefault("document_id", document_id)
            return chunks
    except Exception as exc:
        logger.warning("Could not load all document chunks for %s: %s", document_id, exc)
        return []


# --------------------------------------------------
# 4. Query helpers
# --------------------------------------------------

def _normalize_document_ids(
    document_id: str | None,
    document_ids: list[str] | None,
) -> list[str] | None:
    """Merge the single and multi document arguments into one clean list."""
    ids: list[str] = []

    if document_id is not None:
        if not isinstance(document_id, str) or not document_id.strip():
            raise ValueError(
                "document_id must be a non-empty string when provided."
            )
        ids.append(document_id.strip())

    if document_ids is not None:
        if not isinstance(document_ids, (list, tuple)):
            raise ValueError("document_ids must be a list of strings.")

        for item in document_ids:
            if not isinstance(item, str) or not item.strip():
                raise ValueError(
                    "Every document_ids entry must be a non-empty string."
                )
            ids.append(item.strip())

    # Remove duplicates but keep the caller's order.
    unique_ids = list(dict.fromkeys(ids))
    return unique_ids or None


def _build_filter(document_ids: list[str] | None) -> dict | None:
    """Pinecone metadata filter restricting results to chosen PDFs."""
    if not document_ids:
        return None

    if len(document_ids) == 1:
        return {"document_id": {"$eq": document_ids[0]}}

    return {"document_id": {"$in": list(document_ids)}}


def _query_once(
    index,
    query_vector: list[float],
    query_filter: dict | None,
    top_k: int,
) -> list[dict]:
    """Run one Pinecone query and convert matches to result dicts."""
    t_search_start = time.perf_counter()
    response = index.query(
        vector=query_vector,
        filter=query_filter,
        top_k=top_k,
        include_metadata=True,
        include_values=False,
    )
    logger.info(
        "Pinecone vector search completed in %.4f seconds.",
        time.perf_counter() - t_search_start,
    )

    results = []

    for match in response.matches:
        payload = match.metadata or {}

        text = payload.get("text")

        # Ignore records that cannot provide document content.
        if not isinstance(text, str) or not text.strip():
            logger.warning(
                "Skipping Pinecone match %s because its text "
                "payload is missing or empty.",
                match.id,
            )
            continue

        # Filter out tiny noise chunks (< 80 chars with running header / footer indicators)
        if len(text.strip()) < 80:
            lower_text = text.lower()
            if (
                "page " in lower_text
                or "journal of" in lower_text
                or "all rights reserved" in lower_text
            ):
                continue

        raw_page = payload.get("page_number")
        if raw_page is not None:
            try:
                page_number = int(raw_page)
            except (ValueError, TypeError):
                page_number = None
        else:
            page_number = None

        result = {
            "score": float(match.score) if match.score is not None else None,
            "document_id": payload.get("document_id"),
            "chunk_id": payload.get("chunk_id"),
            "page_number": page_number,
            "text": text.strip(),
        }

        # Preserve optional metadata if present in the index.
        for field in (
            "filename",
            "section_heading",
            "domain",
            "key_concepts",
            "summary",
        ):
            if payload.get(field) is not None:
                result[field] = payload[field]

        results.append(result)

    return results


# --------------------------------------------------
# 5. Search for relevant document chunks
# --------------------------------------------------

def search_documents(
    query: str,
    document_id: str | None = None,
    top_k: int = DEFAULT_TOP_K,
    document_ids: list[str] | None = None,
    per_document: bool = False,
) -> list[dict]:
    """
    Retrieve relevant existing chunks from Pinecone.

    Parameters
    ----------
    query:
        User's question.
    document_id:
        Optional ID used to restrict retrieval to one PDF.
    top_k:
        Number of chunks to retrieve (maximum MAX_TOP_K).
    document_ids:
        Optional list of IDs to restrict retrieval to several PDFs.
        Combined with document_id when both are given.
    per_document:
        When True and several documents are targeted, query each PDF
        separately so every PDF contributes chunks. Use this for
        comparison questions; otherwise one PDF could crowd out the rest.

    Returns
    -------
    list[dict]:
        score, document_id, chunk_id, page_number, text, and when
        available: filename, section_heading, domain, key_concepts, summary.

    Important
    ---------
    This function does not index PDFs or regenerate embeddings.
    It embeds the query once and searches the existing index.
    """

    # Validate the query.
    if not isinstance(query, str) or not query.strip():
        raise ValueError("The search query cannot be empty.")

    # Validate the number of results.
    if (
        not isinstance(top_k, int)
        or isinstance(top_k, bool)
        or top_k < 1
        or top_k > MAX_TOP_K
    ):
        raise ValueError(
            f"top_k must be an integer between 1 and {MAX_TOP_K}."
        )

    ids = _normalize_document_ids(document_id, document_ids)
    clean_query = query.strip()

    # Step A: Embed the question once.
    t_embed_start = time.perf_counter()
    query_vector = embed_query(clean_query)
    logger.info(
        "Query embedding completed in %.4f seconds.",
        time.perf_counter() - t_embed_start,
    )

    # Step B: Connect to the existing Pinecone index.
    client = get_pinecone_client()

    try:
        index = client.Index(INDEX_NAME)

        # Step C: Retrieve existing vectors.
        if per_document and ids and len(ids) > 1:
            per_doc_k = max(2, math.ceil(top_k / len(ids)))
            results = []

            for doc_id in ids:
                results.extend(
                    _query_once(
                        index,
                        query_vector,
                        _build_filter([doc_id]),
                        per_doc_k,
                    )
                )

            results.sort(
                key=lambda r: r["score"] if r["score"] is not None else 0.0,
                reverse=True,
            )
        else:
            results = _query_once(
                index,
                query_vector,
                _build_filter(ids),
                top_k,
            )

        for idx, r in enumerate(results, start=1):
            logger.info(
                "Retrieved match %d: doc=%s, chunk_id=%s, page=%s, score=%.4f, length=%d",
                idx,
                r.get("document_id"),
                r.get("chunk_id"),
                r.get("page_number"),
                r.get("score") if r.get("score") is not None else 0.0,
                len(r.get("text", "")),
            )

        logger.info(
            "Retrieved %d usable chunks (requested top_k=%d, "
            "documents=%s, per_document=%s).",
            len(results),
            top_k,
            ids or "all documents",
            per_document,
        )

        if not results:
            logger.warning(
                "No usable chunks were retrieved. Check whether the "
                "PDF has been indexed and whether document_id matches "
                "the stored Pinecone payload."
            )

        return results

    finally:
        if hasattr(client, "close"):
            client.close()


# --------------------------------------------------
# 6. Terminal testing
# --------------------------------------------------

def main():
    """Run semantic retrieval from the terminal."""

    parser = argparse.ArgumentParser(
        description="Search existing OmniBrain document embeddings."
    )

    parser.add_argument(
        "--query",
        required=True,
        help="Question or text to search for.",
    )

    parser.add_argument(
        "--document-id",
        action="append",
        default=None,
        help="Document ID to restrict the search. Repeat for several PDFs.",
    )

    parser.add_argument(
        "--per-document",
        action="store_true",
        help="Query each selected PDF separately (for comparisons).",
    )

    parser.add_argument(
        "--top-k",
        type=int,
        default=DEFAULT_TOP_K,
        help=f"Number of chunks to return (1-{MAX_TOP_K}).",
    )

    args = parser.parse_args()

    if not 1 <= args.top_k <= MAX_TOP_K:
        parser.error(f"--top-k must be between 1 and {MAX_TOP_K}")

    try:
        results = search_documents(
            query=args.query,
            document_ids=args.document_id,
            top_k=args.top_k,
            per_document=args.per_document,
        )

        if not results:
            print(
                "\nNo matching chunks were found. "
                "Confirm that the PDF is indexed in Pinecone and "
                "that the document ID is correct."
            )
            return

        print(f"\nQuery: {args.query}")
        print(f"Results found: {len(results)}")

        for index, result in enumerate(results, start=1):
            print(f"\n{'=' * 60}")
            print(f"Result {index}")
            print(f"Similarity score: {result['score']:.4f}")
            print(f"Document ID: {result['document_id']}")
            if result.get("filename"):
                print(f"Filename: {result['filename']}")
            print(f"Chunk ID: {result['chunk_id']}")
            print(f"Page number: {result['page_number']}")

            if result.get("section_heading"):
                print(f"Section: {result['section_heading']}")

            if result.get("domain"):
                print(f"Domain: {result['domain']}")

            if result.get("key_concepts"):
                print(f"Key concepts: {result['key_concepts']}")

            print(f"\nText:\n{result['text']}")

    except Exception as error:
        logger.exception("Semantic retrieval failed")
        parser.exit(
            status=1,
            message=f"\nSemantic retrieval failed: {error}\n",
        )


# --------------------------------------------------
# 7. Entry point
# --------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()