import argparse
import json
import logging
import math
import time

from app.config import (
    EMBEDDING_FINGERPRINT,
    PINECONE_INDEX_NAME,
    PROCESSED_DATA_DIR,
)
from app.services.embedding_service import (
    RETRIEVAL_QUERY,
    embed_texts,
)
from app.services.vector_store import get_pinecone_client

logger = logging.getLogger(__name__)

# --------------------------------------------------
# 1. Configuration
# --------------------------------------------------

INDEX_NAME = PINECONE_INDEX_NAME
COLLECTION_NAME = INDEX_NAME

DEFAULT_TOP_K = 3
MAX_TOP_K = 20

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

    return embed_texts([query.strip()], RETRIEVAL_QUERY)[0]


# --------------------------------------------------
# 3. Connect to Pinecone
# --------------------------------------------------

def load_all_document_chunks(document_id: str) -> list[dict]:
    """
    Load all processed chunks for a document from local processed files.
    Allows full-document coverage for small documents (e.g. resumes, notes).
    """
    if not isinstance(document_id, str) or not document_id.strip():
        return []

    chunks_file = PROCESSED_DATA_DIR / f"{document_id.strip()}_chunks.json"
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
    """Filter results to selected PDFs and the active embedding model."""
    model_filter = {
        "embedding_model": {"$eq": EMBEDDING_FINGERPRINT},
    }

    if not document_ids:
        return model_filter

    if len(document_ids) == 1:
        document_filter = {"document_id": {"$eq": document_ids[0]}}
    else:
        document_filter = {"document_id": {"$in": list(document_ids)}}

    return {"$and": [document_filter, model_filter]}


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