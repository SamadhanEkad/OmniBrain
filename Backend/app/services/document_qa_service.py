import logging
import re
import time
from threading import Lock
from typing import Any

from app.config import PROCESSED_DATA_DIR, RAW_DATA_DIR
from app.services.answer_generator import generate_answer
from app.services.embedding_service import generate_embeddings
from app.services.file_service import (
    delete_document_files,
    list_documents,
    load_document_info,
    save_document_info,
    update_processing_stage,
)
from app.services.metadata_generation import generate_metadata
from app.services.pdf_extractor import extract_pdf_text
from app.services.retrieval_service import (
    DEFAULT_TOP_K,
    MAX_TOP_K,
    load_all_document_chunks,
    search_documents,
)
from app.services.text_chunker import chunk_extracted_json
from app.services.vector_store import (
    delete_document_vectors,
    ensure_index,
    get_pinecone_client,
    is_document_indexed,
    load_embeddings,
    store_embeddings,
)

logger = logging.getLogger(__name__)

_document_locks: dict[str, Lock] = {}
_metadata_locks: dict[str, Lock] = {}
_document_locks_guard = Lock()

_FILENAME_QUESTION = (
    "filename",
    "file name",
    "original name",
    "name of the pdf",
    "name of the file",
    "name of this file",
    "name of the document",
    "name of this document",
    "what did i upload",
)
_PAGE_COUNT_QUESTION = (
    "how many pages",
    "number of pages",
    "page count",
    "how many page",
)
_OVERVIEW_QUESTION = (
    "summar",
    "overview",
    "purpose",
    "document type",
    "type of document",
    "what type",
    "what kind of document",
    "what is this pdf",
    "what is this document",
    "what is the document",
    "why was this document",
    "why is this document",
    "document for",
    "what's this document",
    "what's the document",
)

# Questions about the workspace itself (only meaningful with several PDFs).
_WORKSPACE_QUESTION = (
    "how many pdf",
    "how many documents",
    "how many files",
    "list of pdf",
    "list of documents",
    "list my",
    "what pdfs",
    "what documents have i",
    "uploaded files",
    "uploaded pdfs",
    "uploaded documents",
)

# Words that suggest the user wants every selected PDF to contribute.
_ACROSS_MARKERS = (
    "compare",
    "comparison",
    "difference",
    "differ",
    "both",
    "each document",
    "each pdf",
    "each file",
    "all documents",
    "all pdfs",
    "all files",
    "across",
    "every document",
    "every pdf",
    "common",
    "similar",
    "versus",
    " vs ",
)


# --------------------------------------------------
# 1. Document-level locking & metadata handling
# --------------------------------------------------

def _get_document_lock(document_id: str) -> Lock:
    """Return a lock to prevent concurrent indexing of one document."""
    with _document_locks_guard:
        return _document_locks.setdefault(document_id, Lock())


def _get_metadata_lock(document_id: str) -> Lock:
    """Return a lock to prevent duplicate metadata generation."""
    with _document_locks_guard:
        return _metadata_locks.setdefault(document_id, Lock())


def _validate_document_id(document_id: str) -> str:
    if (
        not isinstance(document_id, str)
        or not document_id.strip()
        or document_id in {".", ".."}
        or "/" in document_id
        or "\\" in document_id
    ):
        raise ValueError("Invalid document ID.")

    return document_id


def ensure_document_metadata(document_id: str) -> dict:
    """Generate document metadata at most once and persist its result."""
    with _get_metadata_lock(document_id):
        pdf_path = RAW_DATA_DIR / f"{document_id}.pdf"

        if not pdf_path.is_file():
            raise FileNotFoundError(
                f"Uploaded PDF not found for document {document_id}."
            )

        info = load_document_info(document_id)

        if info is None:
            info = save_document_info(
                document_id,
                filename=None,
                page_count=None,
                metadata_status="pending",
            )

        if isinstance(info.get("metadata"), dict):
            return info

        if info.get("metadata_status") == "failed":
            return info

        try:
            metadata = generate_metadata(
                str(pdf_path),
                document_id=document_id,
            )
        except Exception:
            logger.exception(
                "Metadata generation failed for document %s",
                document_id,
            )
            return save_document_info(
                document_id,
                metadata_status="failed",
            )

        return save_document_info(
            document_id,
            metadata=metadata,
            author=metadata.get("author") or info.get("author"),
            title=metadata.get("document_title") or info.get("title"),
            page_count=info.get("page_count") or metadata.get("total_pages"),
            metadata_status="complete",
        )


def _document_info_for_question(document_id: str) -> dict:
    pdf_path = RAW_DATA_DIR / f"{document_id}.pdf"

    if not pdf_path.is_file():
        raise FileNotFoundError(
            f"Uploaded PDF not found for document {document_id}."
        )

    info = load_document_info(document_id)

    if info is None:
        info = save_document_info(
            document_id,
            filename=None,
            page_count=None,
            metadata_status="pending",
        )

    return info


def _saved_document_answer(document_id: str, question: str) -> dict | None:
    normalized_question = " ".join(question.casefold().split())

    # Check filename question
    if any(term in normalized_question for term in _FILENAME_QUESTION):
        info = _document_info_for_question(document_id)
        filename = info.get("filename")
        answer = (
            f"The original filename is {filename}."
            if isinstance(filename, str) and filename.strip()
            else "The original uploaded filename is not available."
        )
        return {"answer": answer, "sources": []}

    # Check page count question
    if any(term in normalized_question for term in _PAGE_COUNT_QUESTION):
        info = _document_info_for_question(document_id)
        page_count = info.get("page_count")
        answer = (
            f"This PDF has {page_count} pages."
            if isinstance(page_count, int) and not isinstance(page_count, bool)
            else "The page count is not available."
        )
        return {"answer": answer, "sources": []}

    # Check overview questions
    if any(term in normalized_question for term in _OVERVIEW_QUESTION):
        info = ensure_document_metadata(document_id)
        if info.get("metadata_status") == "failed":
            return {
                "answer": (
                    "A document overview is unavailable because metadata "
                    "generation failed."
                ),
                "sources": [],
            }

        # Check if the question asks for a custom format/answer
        custom_markers = (
            "in 100 words", "in 50 words", "in 200 words", "words",
            "bullet", "points", "teach", "takeaway", "example", "simple",
            "beginner", "step", "compare", "conclusion", "finding", "question",
        )
        if any(marker in normalized_question for marker in custom_markers):
            return None

        metadata = info.get("metadata")
        if not isinstance(metadata, dict):
            return {
                "answer": (
                    "A document overview is unavailable because metadata "
                    "generation failed."
                ),
                "sources": [],
            }

        if any(
            term in normalized_question
            for term in ("purpose", "why was", "why is")
        ) or "document for" in normalized_question:
            field = "document_purpose"
        elif any(
            term in normalized_question
            for term in ("what type", "what kind", "document type", "type of document")
        ):
            field = "document_type"
        else:
            field = "document_summary"

        answer = metadata.get(field)
        if not isinstance(answer, str) or not answer.strip():
            answer = "The available metadata does not specify this information."

        return {"answer": answer.strip(), "sources": []}

    return None


# --------------------------------------------------
# 2. Ensure the PDF is indexed in Pinecone
# --------------------------------------------------

def ensure_document_indexed(document_id: str) -> None:
    """
    Extract, chunk, embed, and index a PDF only when it
    does not already have indexed chunks.
    """

    if not isinstance(document_id, str) or not document_id.strip():
        raise ValueError("Document ID cannot be empty.")

    pdf_path = RAW_DATA_DIR / f"{document_id}.pdf"

    if not pdf_path.is_file():
        raise FileNotFoundError(
            f"Uploaded PDF not found for document {document_id}."
        )

    with _get_document_lock(document_id):
        info = ensure_document_metadata(document_id)
        filename = info.get("filename")
        client = get_pinecone_client()

        try:
            # Already indexed? Then nothing to do.
            if is_document_indexed(client, document_id):
                update_processing_stage(document_id, "indexing", "complete")
                return

            # Extract text and page information from the PDF.
            extracted = extract_pdf_text(
                pdf_path,
                document_id=document_id,
            )

            # Split extracted content into searchable chunks.
            chunk_extracted_json(
                extracted["output_json"],
                filename=filename,
            )

            # Generate vector embeddings for the chunks.
            generate_embeddings(document_id)

            # Load embeddings and corresponding chunk data.
            data, chunks, vector_size = load_embeddings(
                document_id
            )

            if not data or not chunks:
                raise RuntimeError(
                    "Document indexing produced no embeddings or chunks."
                )

            # Create the index if needed and store the vectors.
            ensure_index(client, vector_size)

            store_embeddings(
                client=client,
                document_id=document_id,
                data=data,
                chunks=chunks,
                vector_size=vector_size,
                filename=filename,
            )
            update_processing_stage(document_id, "indexing", "complete")

        finally:
            if hasattr(client, "close"):
                client.close()


# --------------------------------------------------
# 3. Dynamic Question Routing & Retrieval
# --------------------------------------------------

def classify_question(question: str) -> str:
    """
    Classify the user's question into dynamic processing categories:
    - 'author': inquiries about document author, writer, publisher
    - 'title': official title of the document
    - 'document_overview': whole document summary, teachings, takeaways, conclusions
    - 'simple_explanation': request for beginner-friendly, simple language or examples
    - 'comparison_table': comparisons, pros/cons, tabular figures
    - 'specific_content': specific content inquiries
    """
    q = " ".join(question.casefold().split())

    if any(k in q for k in (
        "who wrote", "author of", "who is the author", "who published",
        "who created", "who prepared", "author's name", "authors",
    )):
        return "author"

    if any(k in q for k in (
        "official title", "document title", "title of the document",
        "title of this document", "title of this pdf",
    )):
        return "title"

    if any(k in q for k in (
        "summarize this document in", "summarize the entire document",
        "summarize in", "summary in", "what does this document teach",
        "what does this pdf teach", "main concepts covered", "key takeaways",
        "takeaways from this document", "takeaways from this pdf",
        "conclusion of this document", "conclusion and key takeaways",
        "generate possible questions", "generate questions and answers",
        "questions and answers from this", "what are the key findings",
        "what are the most important findings", "important points i should understand",
        "what are the important points",
    )):
        return "document_overview"

    if any(k in q for k in (
        "simple words", "simple language", "simple english",
        "for a beginner", "as if i am a beginner", "like a beginner",
        "layman", "with an example", "give an example", "give me an example",
        "step-by-step", "step by step",
    )):
        return "simple_explanation"

    if any(k in q for k in (
        "difference between", "compare ", "comparison between", "comparison of",
        "advantages and disadvantages", "pros and cons", "table of", "tabular",
    )):
        return "comparison_table"

    return "specific_content"


def _gather_overview_chunks(document_id: str, all_chunks: list[dict], metadata: dict | None) -> list[dict]:
    """Gather multi-section representative context for document-level understanding."""
    if all_chunks and len(all_chunks) <= 15:
        return all_chunks

    selected = []
    seen_ids = set()

    # Prepend document metadata summary if available
    if isinstance(metadata, dict):
        summary_text = metadata.get("document_summary", "")
        if summary_text:
            selected.append({
                "document_id": document_id,
                "chunk_id": "doc_summary",
                "page_number": 1,
                "text": f"Document Summary: {summary_text}",
            })

    # Add opening chunks (page 1, 2)
    for c in all_chunks:
        p = c.get("page_number", 1)
        cid = c.get("chunk_id")
        if p in (1, 2) and cid not in seen_ids:
            selected.append(c)
            seen_ids.add(cid)
            if len(selected) >= 5:
                break

    # Add ending chunks (conclusion/findings)
    if all_chunks:
        last_pages = {all_chunks[-1].get("page_number", 1)}
        for c in reversed(all_chunks):
            p = c.get("page_number", 1)
            cid = c.get("chunk_id")
            if p in last_pages and cid not in seen_ids:
                selected.append(c)
                seen_ids.add(cid)
                if len(selected) >= 8:
                    break

    return selected or all_chunks[:6]


_SECTION_PATTERNS = {
    "abstract": (
        r"\b(abstract|executive\s+summary)\b",
    ),
    "conclusion": (
        r"\b(conclusion|concluding\s+remarks?|what\s+does\s+the\s+paper\s+conclude)\b",
    ),
    "methods": (
        r"\b(methods?|methodology|materials\s+and\s+methods)\b",
    ),
    "results": (
        r"\b(results?|findings?)\b",
    ),
    "introduction": (
        r"\b(introduction|background)\b",
    ),
}


def detect_requested_section(question: str) -> str | None:
    """Detect if the user question is asking for a specific document section."""
    q = question.casefold()
    for section, patterns in _SECTION_PATTERNS.items():
        if any(re.search(pat, q) for pat in patterns):
            return section
    return None


def find_section_chunks(all_chunks: list[dict], section_name: str) -> list[dict]:
    """Find chunks specifically belonging to a requested section."""
    if not all_chunks:
        return []

    section_name = section_name.lower().strip()
    matches = []

    if section_name == "abstract":
        # Look for abstract in opening pages (1-3)
        for i, chunk in enumerate(all_chunks):
            page = chunk.get("page_number", 1)
            text = chunk.get("text", "")
            heading = (chunk.get("section_heading") or "").lower()

            if page <= 3:
                is_abstract = (
                    "abstract" in heading
                    or re.search(r"^\s*abstract\b", text, re.I | re.M)
                    or re.search(r"\babstract\b", text[:250], re.I)
                )
                if is_abstract:
                    matches.append(chunk)
                    # Include the next chunk on the same page to ensure complete
                    # coverage of Background, Methods, Results, and Conclusion
                    if i + 1 < len(all_chunks):
                        next_chunk = all_chunks[i + 1]
                        if next_chunk.get("page_number") == page:
                            next_text = next_chunk.get("text", "").lower()
                            if any(
                                k in next_text
                                for k in (
                                    "conclusion",
                                    "results",
                                    "keywords",
                                    "xgboost",
                                    "auc",
                                    "method",
                                )
                            ):
                                matches.append(next_chunk)
                    break
        return matches

    elif section_name == "conclusion":
        last_page = all_chunks[-1].get("page_number", 1)
        for chunk in all_chunks:
            page = chunk.get("page_number", 1)
            text = chunk.get("text", "")
            heading = (chunk.get("section_heading") or "").lower()
            if page >= max(1, last_page - 3):
                if "conclusion" in heading or re.search(
                    r"\bconclusions?\b", text[:200], re.I
                ):
                    matches.append(chunk)
        return matches

    elif section_name in ("methods", "methodology"):
        for chunk in all_chunks:
            heading = (chunk.get("section_heading") or "").lower()
            text = chunk.get("text", "")
            if any(k in heading for k in ("method", "materials")) or re.search(
                r"\b(methods?|methodology)\b", text[:150], re.I
            ):
                matches.append(chunk)
                if len(matches) >= 4:
                    break
        return matches

    elif section_name == "results":
        for chunk in all_chunks:
            heading = (chunk.get("section_heading") or "").lower()
            text = chunk.get("text", "")
            if "results" in heading or re.search(
                r"\bresults?\b", text[:150], re.I
            ):
                matches.append(chunk)
                if len(matches) >= 4:
                    break
        return matches

    return matches


def _attach_filenames(
    chunks: list[dict],
    filenames: dict[str, str | None],
) -> list[dict]:
    """
    Make sure every chunk knows which PDF it came from.

    PDFs indexed before multi-PDF support have no filename in Pinecone,
    so we fill it in from the saved document info.
    """
    for chunk in chunks:
        if not isinstance(chunk, dict) or chunk.get("filename"):
            continue

        name = filenames.get(chunk.get("document_id"))
        if name:
            chunk["filename"] = name

    return chunks


def _build_sources(chunks: list[dict]) -> list[dict]:
    """Deduplicated source list for citations."""
    sources = []
    seen_sources = set()

    for chunk in chunks:
        source_key = (
            chunk.get("document_id"),
            chunk.get("chunk_id"),
            chunk.get("page_number"),
        )

        if source_key in seen_sources:
            continue

        seen_sources.add(source_key)

        sources.append(
            {
                "document_id": chunk.get("document_id"),
                "filename": chunk.get("filename"),
                "chunk_id": chunk.get("chunk_id"),
                "page_number": chunk.get("page_number"),
                "score": chunk.get("score"),
                "text": chunk.get("text", ""),
            }
        )

    return sources


# --------------------------------------------------
# 4. Answer a question about ONE PDF
# --------------------------------------------------

def answer_document_question(
    document_id: str,
    question: str,
    top_k: int = DEFAULT_TOP_K,
) -> dict:
    """
    Retrieve relevant PDF passages and generate a clear,
    grounded answer using Gemini.
    """
    t_total_start = time.perf_counter()

    if not isinstance(question, str) or not question.strip():
        raise ValueError("Question cannot be empty.")

    if not isinstance(top_k, int) or isinstance(top_k, bool) or top_k < 1:
        raise ValueError("top_k must be a positive integer.")

    clean_question = question.strip()

    # 1. Check saved static answers (filename, page count, bare overview buttons)
    saved_answer = _saved_document_answer(document_id, clean_question)
    if saved_answer is not None:
        return saved_answer

    # 2. Make sure the document is available and indexed
    ensure_document_indexed(document_id)

    # 3. Classify question route & detect requested section
    category = classify_question(clean_question)
    requested_section = detect_requested_section(clean_question)
    all_chunks = load_all_document_chunks(document_id)
    doc_info = load_document_info(document_id) or {}
    metadata = doc_info.get("metadata")
    filenames = {document_id: doc_info.get("filename")}

    retrieved_chunks: list[dict] = []

    # If an explicit section like 'abstract' is requested, use targeted section chunks
    targeted_section_chunks = []
    if requested_section:
        targeted_section_chunks = find_section_chunks(all_chunks, requested_section)

    if requested_section == "abstract" and targeted_section_chunks:
        # Prepend targeted abstract chunks directly
        retrieved_chunks = list(targeted_section_chunks)
        # Also run search to complement with any specific question aspects
        try:
            extra = search_documents(
                query=clean_question,
                document_id=document_id,
                top_k=max(top_k, 3),
            )
            for c in extra:
                if c.get("chunk_id") not in {tc.get("chunk_id") for tc in retrieved_chunks}:
                    retrieved_chunks.append(c)
        except Exception:
            pass

    # Route: Document-level overview (100-word summaries, teachings, takeaways, conclusions)
    elif category == "document_overview":
        retrieved_chunks = _gather_overview_chunks(document_id, all_chunks, metadata)

    # Route: Author or Title
    elif category in ("author", "title"):
        # Opening page chunks contain author and title
        page1_chunks = [c for c in all_chunks if c.get("page_number") in (1, 2)]
        if not page1_chunks and all_chunks:
            page1_chunks = all_chunks[:3]

        retrieved_chunks = page1_chunks
        # Also enrich with any metadata
        if not retrieved_chunks and metadata:
            meta_snippet = f"Title: {metadata.get('document_title', 'Unknown')}. Author: {metadata.get('author', 'Not specified')}."
            retrieved_chunks = [{
                "document_id": document_id,
                "chunk_id": "meta_header",
                "page_number": 1,
                "text": meta_snippet,
            }]

    # Route: Simple explanation or Comparison/Table
    elif category in ("simple_explanation", "comparison_table"):
        if all_chunks and len(all_chunks) <= 15:
            retrieved_chunks = all_chunks
        else:
            try:
                retrieved_chunks = search_documents(
                    query=clean_question,
                    document_id=document_id,
                    top_k=max(top_k, 7),
                )
            except Exception as exc:
                logger.warning("Pinecone search failed for route %s: %s", category, exc)
                retrieved_chunks = all_chunks[:7]

    # Route: Specific content query
    else:
        # If document is very small (<= 15 chunks), all chunks can be considered if query is broad
        retrieved_chunks = search_documents(
            query=clean_question,
            document_id=document_id,
            top_k=top_k,
        )

        if not retrieved_chunks and all_chunks:
            # Fallback to initial chunks so Gemini can clarify information is not found
            retrieved_chunks = all_chunks[:top_k]

    # If still no chunks retrieved at all, formulate a clear response rather than 404
    if not retrieved_chunks:
        answer = (
            "The document does not contain information to answer this question. "
            "Please check that the document has extractable text or rephrase your question."
        )
        return {"answer": answer, "sources": []}

    _attach_filenames(retrieved_chunks, filenames)

    # Generate answer with Gemini
    answer = generate_answer(
        question=clean_question,
        retrieved_chunks=retrieved_chunks,
    )

    # Targeted retry: If abstract was requested, but answer says it was missing
    if requested_section == "abstract" and any(
        phrase in answer.lower()
        for phrase in (
            "does not contain the abstract",
            "cannot find the abstract",
            "abstract is unavailable",
            "abstract is not provided",
            "does not contain an abstract",
        )
    ):
        logger.info("Answer generator indicated abstract was missing. Retrying with targeted abstract chunks...")
        fallback_abstract = find_section_chunks(all_chunks, "abstract")
        if fallback_abstract:
            _attach_filenames(fallback_abstract, filenames)
            answer = generate_answer(
                question=clean_question,
                retrieved_chunks=fallback_abstract,
            )
            retrieved_chunks = fallback_abstract

    if not answer or not answer.strip():
        raise RuntimeError(
            "The answer generator returned an empty answer."
        )

    total_duration = time.perf_counter() - t_total_start
    logger.info(
        "Total document Q&A request for document %s (route: %s) completed in %.4f seconds.",
        document_id,
        category,
        total_duration,
    )

    return {
        "answer": answer.strip(),
        "sources": _build_sources(retrieved_chunks),
    }


# --------------------------------------------------
# 5. Answer a question across SEVERAL PDFs
# --------------------------------------------------

_ORDINALS = {
    "first": 0, "1st": 0,
    "second": 1, "2nd": 1,
    "third": 2, "3rd": 2,
    "fourth": 3, "4th": 3,
    "fifth": 4, "5th": 4,
    "last": -1,
}
_DOC_WORDS = r"(?:pdf|document|file|paper)"
_ORDINAL_RE = re.compile(
    rf"\b({'|'.join(_ORDINALS)})\s+{_DOC_WORDS}\b"
)
_NUMBERED_RE = re.compile(
    rf"\b{_DOC_WORDS}\s*(?:no\.?|number|#)?\s*([1-5])\b"
)


def _normalize_name(text: str) -> str:
    """Lowercase and collapse punctuation so names compare reliably."""
    return re.sub(r"[^a-z0-9]+", " ", text.casefold()).strip()


def _select_documents(
    workspace: list[dict],
    document_ids: list[str] | None,
) -> list[dict]:
    """Return the PDFs the user selected, or the whole workspace."""
    if not document_ids:
        return list(workspace)

    by_id = {doc["document_id"]: doc for doc in workspace}
    selected = []

    for doc_id in dict.fromkeys(document_ids):
        _validate_document_id(doc_id)

        if doc_id not in by_id:
            raise FileNotFoundError(
                f"Document {doc_id} was not found. It may have been deleted."
            )

        selected.append(by_id[doc_id])

    return selected


def _detect_mentioned_documents(
    question: str,
    workspace: list[dict],
) -> list[dict]:
    """
    Find PDFs the question refers to explicitly, either by file name /
    title ("in report_2024.pdf ...") or by position ("the second pdf").
    A mention overrides the checkbox selection, because it is the
    clearest signal of which PDF the user means.
    """
    mentioned: dict[str, dict] = {}
    padded_question = f" {_normalize_name(question)} "

    for doc in workspace:
        candidates = []

        filename = doc.get("filename")
        if isinstance(filename, str) and filename.strip():
            stem = re.sub(r"\.pdf$", "", filename.strip(), flags=re.I)
            normalized = _normalize_name(stem)
            candidates.append(normalized)
            # "1.Predicting mortality..." -> also try without the number.
            candidates.append(re.sub(r"^\d+\s+", "", normalized))

        info = load_document_info(doc["document_id"]) or {}
        title = info.get("title")
        if isinstance(title, str) and len(title.strip()) >= 8:
            candidates.append(_normalize_name(title))

        for name in candidates:
            if len(name) >= 4 and f" {name} " in padded_question:
                mentioned[doc["document_id"]] = doc
                break

    lowered = question.casefold()

    for match in _ORDINAL_RE.finditer(lowered):
        position = _ORDINALS[match.group(1)]
        try:
            doc = workspace[position]
        except IndexError:
            continue
        mentioned[doc["document_id"]] = doc

    for match in _NUMBERED_RE.finditer(lowered):
        position = int(match.group(1)) - 1
        if 0 <= position < len(workspace):
            doc = workspace[position]
            mentioned[doc["document_id"]] = doc

    # Keep upload order.
    return [doc for doc in workspace if doc["document_id"] in mentioned]


def _workspace_saved_answer(question: str, docs: list[dict]) -> dict | None:
    """Answer 'which files / how many pages' style questions directly."""
    normalized = " ".join(question.casefold().split())

    asks_filenames = any(
        term in normalized
        for term in _FILENAME_QUESTION + _WORKSPACE_QUESTION
    )
    asks_pages = any(term in normalized for term in _PAGE_COUNT_QUESTION)

    if not asks_filenames and not asks_pages:
        return None

    lines = []
    for number, doc in enumerate(docs, start=1):
        pages = doc.get("page_count")
        page_text = f" ({pages} pages)" if isinstance(pages, int) else ""
        lines.append(f"{number}. {doc.get('filename') or doc['document_id']}{page_text}")

    answer = f"You have {len(docs)} PDFs in this workspace:\n\n" + "\n".join(lines)
    return {"answer": answer, "sources": []}


def _answer_across_documents(
    question: str,
    docs: list[dict],
    top_k: int,
) -> dict:
    """Retrieve from several PDFs at once and answer with per-file attribution."""
    t_start = time.perf_counter()

    ids = [doc["document_id"] for doc in docs]
    filenames = {doc["document_id"]: doc.get("filename") for doc in docs}
    file_names = [name or doc_id for doc_id, name in filenames.items()]

    # Every PDF must be searchable before we query.
    for doc_id in ids:
        ensure_document_indexed(doc_id)

    category = classify_question(question)
    normalized = f" {' '.join(question.casefold().split())} "
    wants_every_doc = (
        category in ("comparison_table", "document_overview")
        or any(marker in normalized for marker in _ACROSS_MARKERS)
    )

    retrieved: list[dict] = []

    if category == "document_overview":
        # A few representative chunks from EACH PDF.
        for doc_id in ids:
            chunks = load_all_document_chunks(doc_id)
            info = load_document_info(doc_id) or {}
            retrieved.extend(
                _gather_overview_chunks(doc_id, chunks, info.get("metadata"))[:4]
            )

    elif category in ("author", "title"):
        # Opening pages of each PDF hold the title and authors.
        for doc_id in ids:
            chunks = load_all_document_chunks(doc_id)
            opening = [c for c in chunks if c.get("page_number") in (1, 2)]
            retrieved.extend((opening or chunks)[:2])

    else:
        k = min(max(top_k, 6), MAX_TOP_K)
        try:
            retrieved = search_documents(
                query=question,
                document_ids=ids,
                top_k=k,
                per_document=wants_every_doc,
            )
        except Exception as exc:
            logger.warning("Multi-document search failed: %s", exc)
            retrieved = []

        if not retrieved:
            for doc_id in ids:
                retrieved.extend(load_all_document_chunks(doc_id)[:2])

    if not retrieved:
        return {
            "answer": (
                "The selected documents do not contain information to answer "
                "this question. Check that they have extractable text or "
                "rephrase your question."
            ),
            "sources": [],
        }

    _attach_filenames(retrieved, filenames)

    answer = generate_answer(
        question=question,
        retrieved_chunks=retrieved,
        file_names=file_names,
    )

    if not answer or not answer.strip():
        raise RuntimeError(
            "The answer generator returned an empty answer."
        )

    logger.info(
        "Multi-document Q&A over %d PDFs (route: %s, every_doc=%s) took %.4f seconds.",
        len(ids),
        category,
        wants_every_doc,
        time.perf_counter() - t_start,
    )

    return {
        "answer": answer.strip(),
        "sources": _build_sources(retrieved),
    }


def answer_documents_question(
    question: str,
    document_ids: list[str] | None = None,
    top_k: int = DEFAULT_TOP_K,
) -> dict:
    """
    Main entry point for questions in a workspace with one or more PDFs.

    How the PDF(s) are chosen:
      1. The question names a file or position ("the second pdf") -> that PDF.
      2. Otherwise the PDFs the user ticked (document_ids).
      3. Otherwise every PDF in the workspace.

    One resulting PDF uses the original single-document pipeline. Several
    PDFs are searched together, and every excerpt keeps its file name so
    the answer cites the correct source.
    """
    if not isinstance(question, str) or not question.strip():
        raise ValueError("Question cannot be empty.")

    if not isinstance(top_k, int) or isinstance(top_k, bool) or top_k < 1:
        raise ValueError("top_k must be a positive integer.")

    clean_question = question.strip()

    workspace = list_documents()
    if not workspace:
        raise FileNotFoundError("No PDFs have been uploaded yet.")

    selected = _select_documents(workspace, document_ids)
    mentioned = _detect_mentioned_documents(clean_question, workspace)
    docs = mentioned or selected

    if len(docs) == 1:
        return answer_document_question(
            docs[0]["document_id"],
            clean_question,
            top_k=top_k,
        )

    saved = _workspace_saved_answer(clean_question, docs)
    if saved is not None:
        return saved

    return _answer_across_documents(clean_question, docs, top_k)


# --------------------------------------------------
# 6. Remove a PDF completely
# --------------------------------------------------

def remove_document(document_id: str) -> dict[str, Any]:
    """
    Delete a PDF everywhere: Pinecone vectors, processed files, raw PDF
    and info file. Frees a slot in the 5-PDF workspace limit.

    Vectors go first (their IDs are rebuilt from the processed chunk
    list). If that step fails nothing local is deleted, so you can retry.
    """
    _validate_document_id(document_id)

    with _get_document_lock(document_id):
        vectors_deleted = delete_document_vectors(document_id)

        processed_removed = 0
        for path in PROCESSED_DATA_DIR.glob(f"*{document_id}*"):
            if path.is_file():
                path.unlink(missing_ok=True)
                processed_removed += 1

        raw_removed = delete_document_files(document_id)

    with _document_locks_guard:
        _document_locks.pop(document_id, None)
        _metadata_locks.pop(document_id, None)

    return {
        "document_id": document_id,
        "deleted": bool(raw_removed or processed_removed or vectors_deleted),
        "vectors_deleted": vectors_deleted,
    }