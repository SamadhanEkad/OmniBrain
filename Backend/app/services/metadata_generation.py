
import json
import logging
import os
import re
from pathlib import Path
from typing import Any, Iterator

import pymupdf
import requests
from dotenv import load_dotenv

from app.config import GEMINI_API_KEY, GEMINI_MODEL

load_dotenv()

logger = logging.getLogger(__name__)

OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434").rstrip("/") + "/api/generate"
OLLAMA_LLM_MODEL = os.getenv("OLLAMA_LLM_MODEL", "llama3.2:1b")

# Metadata generation limits
METADATA_MAX_PAGES = 12
METADATA_CHARS_PER_PAGE = 1200
METADATA_TIMEOUT = (10, 120)

# The 10 suggested metadata questions to show in the UI.
METADATA_QUESTIONS = [
    "What is the name of this document?",
    "What type of document is this?",
    "What is the main purpose of this document?",
    "Summarize this document.",
    "What are the main topics covered?",
    "Who published or prepared this document?",
    "What period does this document cover?",
    "What are the key findings or important points?",
    "What are the main sections or chapters?",
    "What important data, figures or tables are included?",
]


class MetadataGenerationError(RuntimeError):
    """Raised when metadata cannot be produced."""


def iter_document_pages(
    pdf_path: str | Path,
) -> Iterator[dict[str, Any]]:
    """Yield text for each PDF page with one-based page references."""
    with pymupdf.open(pdf_path) as document:
        for page_number, page in enumerate(document, start=1):
            yield {
                "page_number": page_number,
                "text": page.get_text("text").strip(),
            }


def extract_document_text(
    pdf_path: str | Path,
) -> tuple[str, list[dict[str, Any]]]:
    """Return the extracted text and per-page text records."""
    pages = list(iter_document_pages(pdf_path))

    full_text = "\n\n".join(
        f"[page {page['page_number']}]\n{page['text']}"
        for page in pages
        if page["text"]
    )

    return full_text, pages


def _select_representative_pages(
    pdf_path: str | Path,
) -> tuple[int, list[dict[str, Any]], dict[str, Any]]:
    """
    Sample representative pages across a PDF for metadata generation.
    Always prioritize the opening pages (title, author, abstract, TOC)
    and closing pages (conclusion, summary).
    """
    with pymupdf.open(pdf_path) as document:
        total_pages = len(document)

        if total_pages == 0:
            raise ValueError("The PDF contains no pages.")

        raw_meta = document.metadata or {}
        pdf_metadata = {
            k: v.strip()
            for k, v in raw_meta.items()
            if isinstance(v, str) and v.strip()
        }

        # Build index set
        if total_pages <= METADATA_MAX_PAGES:
            indices = list(range(total_pages))
        else:
            # Prioritize pages 0, 1, 2 (first 3 pages) and last 2 pages
            selected_indices = {0, 1, min(2, total_pages - 1), max(0, total_pages - 2), total_pages - 1}
            # Fill the remaining slots across the body
            remaining_slots = METADATA_MAX_PAGES - len(selected_indices)
            if remaining_slots > 0:
                step = total_pages / (remaining_slots + 1)
                for i in range(1, remaining_slots + 1):
                    selected_indices.add(round(i * step))
            indices = sorted(selected_indices)

        samples = []
        for index in indices:
            if 0 <= index < total_pages:
                page = document.load_page(index)
                text = page.get_text("text").strip()
                if text:
                    samples.append({
                        "page_number": index + 1,
                        "text": text[:METADATA_CHARS_PER_PAGE],
                    })

    return total_pages, samples, pdf_metadata


def _clean_json_text(text: str) -> str:
    """Strip markdown code fence if present."""
    text = text.strip()
    match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", text)
    if match:
        return match.group(1).strip()
    return text


def _validate_metadata(
    value: Any,
    total_pages: int,
    pdf_metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Validate metadata fields and page references."""
    if not isinstance(value, dict):
        raise ValueError("Metadata must be a JSON object.")

    pdf_metadata = pdf_metadata or {}

    # Extract or infer title
    document_title = value.get("document_title") or pdf_metadata.get("title")
    if not isinstance(document_title, str) or not document_title.strip():
        document_title = "Document"

    # Extract or infer type
    document_type = value.get("document_type")
    if not isinstance(document_type, str) or not document_type.strip():
        document_type = "Informational Document"

    # Extract or infer purpose
    document_purpose = value.get("document_purpose")
    if not isinstance(document_purpose, str) or not document_purpose.strip():
        document_purpose = "To present document information and findings."

    # Extract or infer summary
    document_summary = value.get("document_summary")
    if not isinstance(document_summary, str) or not document_summary.strip():
        document_summary = "A document covering relevant topics and data."

    topics = value.get("key_topics", [])
    if not isinstance(topics, list) or not all(isinstance(topic, str) for topic in topics):
        topics = []

    sections = value.get("section_summaries", [])
    if not isinstance(sections, list):
        sections = []

    validated_sections = []
    for section in sections:
        if not isinstance(section, dict):
            continue

        title = section.get("section_title")
        summary = section.get("summary")
        references = section.get("page_references", [])

        if not isinstance(title, str) or not title.strip():
            continue
        if not isinstance(summary, str) or not summary.strip():
            continue

        if not isinstance(references, list):
            references = []

        valid_references = sorted({
            page
            for page in references
            if isinstance(page, int)
            and not isinstance(page, bool)
            and 1 <= page <= total_pages
        })

        validated_sections.append({
            "section_title": title.strip(),
            "summary": summary.strip(),
            "page_references": valid_references,
        })

    author = value.get("author") or pdf_metadata.get("author")
    if not isinstance(author, str) or not author.strip():
        author = None
    else:
        author = author.strip()

    result = {
        "document_title": document_title.strip(),
        "document_type": document_type.strip(),
        "document_purpose": document_purpose.strip(),
        "document_summary": document_summary.strip(),
        "key_topics": list(dict.fromkeys(
            topic.strip()
            for topic in topics
            if isinstance(topic, str) and topic.strip()
        )),
        "section_summaries": validated_sections,
    }
    if author:
        result["author"] = author

    return result


def _request_metadata_gemini(
    prompt: str,
    total_pages: int,
    pdf_metadata: dict[str, Any],
) -> dict[str, Any] | None:
    """Generate metadata using Google Gemini API."""
    api_key = GEMINI_API_KEY or os.getenv("GEMINI_API_KEY")
    if not api_key or not api_key.strip():
        return None

    clean_key = api_key.strip().strip("'\"").strip()
    if clean_key.startswith("<") and clean_key.endswith(">"):
        return None

    try:
        from google import genai
        from google.genai import types

        client = genai.Client(api_key=clean_key)
        target_model = (GEMINI_MODEL or os.getenv("GEMINI_MODEL", "gemini-3.8-flash")).strip()

        candidate_models = [target_model]
        for fallback in (
            "gemini-3.5-flash",
            "gemini-flash-lite-latest",
            "gemini-3.5-flash-lite",
            "gemini-3.1-flash-lite",
            "gemini-3.8-flash",
        ):
            if fallback not in candidate_models:
                candidate_models.append(fallback)

        config = types.GenerateContentConfig(
            temperature=0.1,
            response_mime_type="application/json",
            max_output_tokens=1500,
        )

        for model_name in candidate_models:
            try:
                response = client.models.generate_content(
                    model=model_name,
                    contents=prompt,
                    config=config,
                )
                if response and response.text:
                    cleaned_text = _clean_json_text(response.text)
                    parsed = json.loads(cleaned_text)
                    return _validate_metadata(parsed, total_pages, pdf_metadata)
            except Exception as exc:
                logger.warning("Gemini model '%s' metadata generation failed: %s", model_name, exc)
                continue

    except Exception as exc:
        logger.warning("Gemini metadata generation error: %s", exc)

    return None


def _request_metadata_ollama(
    prompt: str,
    total_pages: int,
    pdf_metadata: dict[str, Any],
) -> dict[str, Any] | None:
    """Fallback metadata generation via Ollama if available."""
    try:
        response = requests.post(
            OLLAMA_URL,
            json={
                "model": OLLAMA_LLM_MODEL,
                "prompt": prompt,
                "stream": False,
                "format": "json",
                "options": {
                    "temperature": 0.1,
                    "num_ctx": 4096,
                    "num_predict": 800,
                },
            },
            timeout=METADATA_TIMEOUT,
        )
        response.raise_for_status()
        result = response.json()
        generated = json.loads(_clean_json_text(result.get("response", "")))
        return _validate_metadata(generated, total_pages, pdf_metadata)
    except Exception as exc:
        logger.warning("Ollama metadata fallback failed: %s", exc)
        return None


def _build_heuristic_metadata(
    filename: str,
    total_pages: int,
    samples: list[dict[str, Any]],
    pdf_metadata: dict[str, Any],
) -> dict[str, Any]:
    """Deterministic fallback metadata when external APIs are unavailable."""
    title = pdf_metadata.get("title")
    if not title and samples:
        # Look at the first lines of page 1
        lines = [l.strip() for l in samples[0]["text"].splitlines() if l.strip()]
        if lines:
            title = lines[0][:120]
    if not title:
        title = Path(filename).stem.replace("_", " ").replace("-", " ").title()

    author = pdf_metadata.get("author")
    summary = f"Document '{title}' containing {total_pages} pages."
    if samples:
        first_page_snippet = " ".join(samples[0]["text"].split()[:60])
        if first_page_snippet:
            summary += f" Overview: {first_page_snippet}..."

    return _validate_metadata(
        {
            "document_title": title,
            "document_type": "PDF Document",
            "document_purpose": f"Information regarding {title}.",
            "document_summary": summary,
            "author": author,
            "key_topics": [title[:50]],
            "section_summaries": [],
        },
        total_pages,
        pdf_metadata,
    )


def generate_metadata(
    pdf_path: str | Path,
    document_id: str | None = None,
) -> dict[str, Any]:
    """
    Generate a compact document profile from representative PDF pages.
    Uses Gemini API as primary generator with safe fallbacks.
    """
    path = Path(pdf_path)

    if not path.is_file():
        raise FileNotFoundError(f"PDF file not found: {path}")

    total_pages, samples, pdf_metadata = _select_representative_pages(path)

    if not samples:
        raise ValueError(
            "No extractable text found. This PDF may be scanned "
            "and require OCR."
        )

    sample_page_numbers = [item["page_number"] for item in samples]

    evidence = "\n\n".join(
        f"[PDF page {item['page_number']}]\n{item['text']}"
        for item in samples
    )

    prompt = f"""You create an accurate, comprehensive metadata profile for an uploaded PDF document.

Return valid JSON only with exactly these fields:
{{
  "document_title": "string (the official title from the document content or cover)",
  "document_type": "string (e.g. Resume/CV, Research Paper, Lecture Notes, Financial Report, Manual, Guide, etc.)",
  "document_purpose": "string (the primary aim or objective of the document)",
  "document_summary": "string (a clear, cohesive summary of around 100-150 words capturing the core subject matter and findings)",
  "author": "string (name of author, organization, or presenter if stated, otherwise null)",
  "key_topics": ["string (up to 8 primary topics covered)"],
  "section_summaries": [
    {{
      "section_title": "string",
      "summary": "string",
      "page_references": [1]
    }}
  ]
}}

Rules:
- The PDF has {total_pages} pages.
- The supplied excerpts are from PDF pages {sample_page_numbers}.
- Base claims strictly on the supplied excerpts and PDF metadata.
- Do not invent facts, dates, organizations, or numbers.
- Filename: {path.name}
- Embedded PDF metadata: {json.dumps(pdf_metadata)}

Sampled excerpts:
{evidence}
"""

    # 1. Primary: Try Gemini API
    metadata = _request_metadata_gemini(prompt, total_pages, pdf_metadata)

    # 2. Secondary: Try Ollama fallback
    if metadata is None:
        metadata = _request_metadata_ollama(prompt, total_pages, pdf_metadata)

    # 3. Tertiary: Deterministic heuristic fallback
    if metadata is None:
        logger.info("Using heuristic metadata fallback for %s", path.name)
        metadata = _build_heuristic_metadata(path.name, total_pages, samples, pdf_metadata)

    metadata.update({
        "document_id": document_id,
        "source_file": path.name,
        "total_pages": total_pages,
        "pdf_metadata": pdf_metadata,
        "metadata_scope": (
            "sampled_pages"
            if total_pages > len(samples)
            else "all_pages"
        ),
        "sampled_page_references": sample_page_numbers,
        "suggested_metadata_questions": METADATA_QUESTIONS,
    })

    return metadata


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    pdf_file = input(
        "Enter the full path of the PDF: "
    ).strip().strip('"')

    try:
        result = generate_metadata(pdf_file)
        print(json.dumps(
            result,
            indent=2,
            ensure_ascii=False,
        ))
    except Exception:
        logger.exception(
            "Metadata generation failed for %s",
            pdf_file,
        )
