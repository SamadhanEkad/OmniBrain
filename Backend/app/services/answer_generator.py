import argparse
import logging
import os
import time

from dotenv import load_dotenv
from google import genai
from google.genai import types
from google.genai.errors import APIError, ClientError, ServerError

from app.config import BACKEND_ROOT, GEMINI_API_KEY, GEMINI_MODEL
from app.services.retrieval_service import (
    DEFAULT_TOP_K,
    search_documents,
)

load_dotenv(BACKEND_ROOT / ".env")
load_dotenv()

logger = logging.getLogger(__name__)

# --------------------------------------------------
# 1. Configuration
# --------------------------------------------------

DEFAULT_GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.8-flash")

MAX_CONTEXT_CHARS = 24000
# Larger budget used when several PDFs contribute excerpts.
MAX_MULTI_DOC_CONTEXT_CHARS = 40000
MAX_OUTPUT_TOKENS = 2048

# Balanced temperature for natural, beginner-friendly explanations while preserving facts.
TEMPERATURE = 0.2


# --------------------------------------------------
# 2. Gemini Client
# --------------------------------------------------

def get_gemini_client() -> genai.Client:
    """Connect to Google GenAI client using the configured API key."""
    api_key = os.getenv("GEMINI_API_KEY") or GEMINI_API_KEY
    if not api_key or not api_key.strip():
        raise RuntimeError(
            "GEMINI_API_KEY environment variable is not set. "
            "Please configure your Gemini API key."
        )
    clean_key = api_key.strip().strip("'\"").strip()
    if clean_key.startswith("<") and clean_key.endswith(">"):
        raise RuntimeError(
            f"GEMINI_API_KEY is currently set to placeholder '{clean_key}'. "
            "Please replace it with your actual Gemini API key."
        )
    return genai.Client(api_key=clean_key)


# --------------------------------------------------
# 3. Prepare retrieved document context
# --------------------------------------------------

def prepare_context(
    retrieved_chunks: list[dict],
    max_chars: int = MAX_CONTEXT_CHARS,
) -> str:
    """
    Prepare relevant PDF chunks for answer generation.

    Each excerpt header names its source file (when known) and page, so
    the model can attribute every fact to the right PDF.
    Duplicate chunks are skipped.
    """

    if not retrieved_chunks:
        raise ValueError(
            "No relevant document chunks were retrieved."
        )

    if (
        not isinstance(max_chars, int)
        or isinstance(max_chars, bool)
        or max_chars < 1
    ):
        raise ValueError("max_chars must be a positive integer.")

    context_parts = []
    seen_chunks = set()
    used_chars = 0

    for chunk in retrieved_chunks:
        if not isinstance(chunk, dict):
            continue

        text = chunk.get("text")

        if not isinstance(text, str) or not text.strip():
            continue

        text = text.strip()

        # Avoid duplicate chunks when chunk IDs are available.
        chunk_id = chunk.get("chunk_id")
        document_id = chunk.get("document_id")

        if chunk_id is not None:
            identity = (
                document_id,
                str(chunk_id),
            )

            if identity in seen_chunks:
                continue

            seen_chunks.add(identity)

        metadata = []

        filename = chunk.get("filename")

        if isinstance(filename, str) and filename.strip():
            metadata.append(f"Source: {filename.strip()}")
        elif document_id:
            metadata.append(f"Document: {document_id}")

        page_number = chunk.get("page_number")

        if page_number is not None:
            metadata.append(f"Page: {page_number}")

        optional_fields = (
            ("section_heading", "Section"),
            ("domain", "Domain"),
            ("key_concepts", "Key concepts"),
            ("summary", "Chunk summary"),
        )

        for field, label in optional_fields:
            value = chunk.get(field)

            if value is None or value == "":
                continue

            if isinstance(value, (list, tuple, set)):
                value = ", ".join(str(item) for item in value)

            metadata.append(f"{label}: {value}")

        label = ", ".join(metadata) or "Retrieved passage"
        prefix = f"\n\n[{label}]\n"

        remaining = max_chars - used_chars - len(prefix)

        if remaining <= 0:
            break

        text = text[:remaining]

        context_parts.append(prefix + text)
        used_chars += len(prefix) + len(text)

        if used_chars >= max_chars:
            break

    context = "".join(context_parts).strip()

    if not context:
        raise ValueError(
            "Retrieved chunks do not contain usable document text."
        )

    return context


# --------------------------------------------------
# 4. Build the answer-generation prompt
# --------------------------------------------------

def _multi_document_rules(file_names: list[str]) -> str:
    """Extra prompt rules used only when several PDFs are in play."""
    listing = ", ".join(f'"{name}"' for name in file_names)

    return f"""
7. MULTIPLE DOCUMENTS:
- The user is working with {len(file_names)} PDFs: {listing}.
- Every excerpt header starts with "Source: <file name>". Always know which file a fact came from.
- Cite as [file name, Page N], for example [report.pdf, Page 3].
- Never attribute a fact from one file to another. If files disagree, report each file's version separately.
- If the question names a specific file, answer only from that file's excerpts.
- If the answer appears in only some of the files, say which files contain it and do not claim the others do.
- For comparisons, organise the answer file by file (a Markdown table works well) before concluding.
"""


def build_prompt(
    question: str,
    context: str,
    question_type: str | None = None,
    file_names: list[str] | None = None,
) -> str:
    """
    Build a comprehensive, grounded prompt for OmniBrain answers.

    Follows strict factual grounding while supporting beginner-friendly
    explanations, clear examples, document summaries, comparisons,
    and distinct identification of external general knowledge.
    """

    if not isinstance(question, str) or not question.strip():
        raise ValueError("Question cannot be empty.")

    if not isinstance(context, str) or not context.strip():
        raise ValueError("Document context cannot be empty.")

    multi_document = bool(file_names) and len(file_names) > 1
    multi_rules = _multi_document_rules(file_names) if multi_document else ""

    citation_rule = (
        "- When a relevant source and page are supplied in the excerpt header, "
        "cite them as [file name, Page N] near the statement."
        if multi_document
        else "- When a relevant page number is supplied in the excerpt header [Page N], cite it as [Page N] near the statement."
    )

    return f"""You are OmniBrain, an intelligent AI-powered document assistant for understanding uploaded PDF documents.

YOUR TASK
Answer the user's question clearly, accurately, and naturally.
Use the supplied document excerpts as evidence for facts about the uploaded document(s).

CORE CAPABILITIES & RESPONSE MODES:

1. FACTUAL GROUNDING & FIDELITY:
- Base all statements about the document on the supplied document excerpts.
- Never invent facts, proper names, author names, degree titles, institutions, CGPA values, percentages, financial figures, dates, or conclusions that contradict or are absent from the document.
- Preserve exact numbers, dates, units, and names accurately.
{citation_rule}

2. SIMPLE EXPLANATIONS & BEGINNER-FRIENDLY CLARITY:
- When the user asks to explain a concept in simple words, explain like a beginner, explain step-by-step, or provide an example:
  - Explain the concept in clear, accessible, intuitive language.
  - Provide realistic, relatable examples or analogies to clarify difficult concepts.
  - Do not merely parrot raw technical text or refuse to explain simply.
  - Keep the core meaning strictly aligned with what the document states.

3. GENERAL KNOWLEDGE VS. DOCUMENT FACTS:
- If a question asks about a concept that is mentioned or touched upon in the PDF but not fully detailed, or asks for general context:
  - Explain what the document explicitly states.
  - You may supplement with helpful general knowledge, but clearly distinguish it (e.g. "Note: From general domain knowledge..." or "In general:") so the user knows what is in the document versus general background knowledge.
  - Never falsely claim external concepts came directly from the PDF.

4. SUMMARIZATION & DOCUMENT-LEVEL QUESTIONS:
- When asked to summarize (e.g. "Summarize this document in 100 words", "What are the key takeaways?", "What does this document teach us?"):
  - Provide a cohesive, well-structured synthesis.
  - Closely respect requested constraints (e.g. word count, bullet points).
  - Highlight the main objectives, findings, and conclusions.

5. COMPARISONS, TABLES & MULTI-PART QUESTIONS:
- When asked to compare concepts, pros and cons, or numerical values, synthesize across the excerpts.
- Use clean Markdown tables when comparing multiple attributes or structured data.
- Do not invent missing table cells.

6. MISSING INFORMATION & UNANSWERABLE QUESTIONS:
- If the requested fact or concept is not present in the supplied excerpts and cannot be answered from them, state clearly that the document does not contain this information.
- Do not guess or hallucinate missing facts.
{multi_rules}
DOCUMENT EXCERPTS:
<document_context>
{context}
</document_context>

USER QUESTION:
<user_question>
{question.strip()}
</user_question>

Format your response in clean Markdown. Return only the final answer.
""".strip()


# --------------------------------------------------
# 5. Generate an answer with Google Gemini API
# --------------------------------------------------

def generate_answer(
    question: str,
    retrieved_chunks: list[dict],
    model: str | None = None,
    question_type: str | None = None,
    file_names: list[str] | None = None,
    max_context_chars: int | None = None,
) -> str:
    """
    Generate an answer using retrieved PDF chunks and Google Gemini API.

    file_names:
        Names of the PDFs in play. With two or more, the prompt adds
        multi-document attribution rules.
    max_context_chars:
        Override the context budget (defaults depend on single vs multi).

    Returns a plain string to preserve compatibility with the
    existing OmniBrain question-answering API.
    """

    if not isinstance(question, str) or not question.strip():
        raise ValueError("Question cannot be empty.")

    if not isinstance(retrieved_chunks, list) or not retrieved_chunks:
        raise ValueError(
            "No relevant document chunks were retrieved. "
            "Check document indexing and retrieval."
        )

    multi_document = bool(file_names) and len(file_names) > 1

    if max_context_chars is None:
        max_context_chars = (
            MAX_MULTI_DOC_CONTEXT_CHARS if multi_document else MAX_CONTEXT_CHARS
        )

    t_ctx_start = time.perf_counter()
    context = prepare_context(retrieved_chunks, max_chars=max_context_chars)
    ctx_duration = time.perf_counter() - t_ctx_start
    logger.info("Context preparation completed in %.4f seconds.", ctx_duration)

    prompt = build_prompt(
        question,
        context,
        question_type=question_type,
        file_names=file_names,
    )

    target_model = (model or os.getenv("GEMINI_MODEL", DEFAULT_GEMINI_MODEL)).strip()
    client = get_gemini_client()

    logger.info(
        "Generating an answer with Gemini model '%s' (question: %d chars, context: %d chars).",
        target_model,
        len(question),
        len(context),
    )

    config = types.GenerateContentConfig(
        temperature=TEMPERATURE,
        max_output_tokens=MAX_OUTPUT_TOKENS,
    )

    t_gen_start = time.perf_counter()
    response = None

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

    last_error = None
    for current_model in candidate_models:
        for attempt in range(2):
            try:
                response = client.models.generate_content(
                    model=current_model,
                    contents=prompt,
                    config=config,
                )
                if response and response.text:
                    break
            except ClientError as exc:
                last_error = exc
                if exc.code in (404, 429):
                    logger.warning(
                        "Model '%s' returned %d (%s). Trying next model candidate.",
                        current_model,
                        exc.code,
                        getattr(exc, "message", str(exc)),
                    )
                    break
                logger.warning("Model '%s' client error (%s). Trying next model candidate.", current_model, exc)
                break
            except (ServerError, APIError) as exc:
                last_error = exc
                code = getattr(exc, "code", None)
                if code == 503 and attempt < 1:
                    logger.warning(
                        "Model '%s' returned 503. Retrying in 1s (attempt %d/2)...",
                        current_model,
                        attempt + 1,
                    )
                    time.sleep(1)
                    continue
                logger.warning("Model '%s' encountered error: %s", current_model, exc)
                break
            except Exception as exc:
                last_error = exc
                break
        if response and response.text:
            break

    if not response or not response.text:
        raise RuntimeError(f"Gemini answer generation failed: {last_error}")

    gen_duration = time.perf_counter() - t_gen_start
    logger.info("Gemini answer generation completed in %.4f seconds.", gen_duration)

    answer = response.text.strip()
    return answer


# --------------------------------------------------
# 6. Optional command-line testing
# --------------------------------------------------

def main():
    """Test document retrieval and answer generation from the terminal."""

    parser = argparse.ArgumentParser(
        description="OmniBrain PDF question answering with Gemini."
    )

    parser.add_argument(
        "--question",
        required=True,
        help="Question about an indexed PDF.",
    )

    parser.add_argument(
        "--document-id",
        action="append",
        default=None,
        help="Document ID to restrict retrieval. Repeat for several PDFs.",
    )

    parser.add_argument(
        "--top-k",
        type=int,
        default=DEFAULT_TOP_K,
        help="Number of relevant chunks to retrieve.",
    )

    parser.add_argument(
        "--model",
        default=DEFAULT_GEMINI_MODEL,
        help="Gemini model used for answer generation.",
    )

    args = parser.parse_args()

    try:
        chunks = search_documents(
            query=args.question,
            document_ids=args.document_id,
            top_k=args.top_k,
        )

        if not chunks:
            print(
                "\nNo relevant document passages were found. "
                "Check that the PDF has been indexed and that "
                "the document ID is correct."
            )
            return

        file_names = sorted(
            {c["filename"] for c in chunks if c.get("filename")}
        )

        answer = generate_answer(
            question=args.question,
            retrieved_chunks=chunks,
            model=args.model,
            file_names=file_names or None,
        )

        print("\n" + "=" * 60)
        print("OMNIBRAIN ANSWER (GEMINI)")
        print("=" * 60)
        print(answer)

        print("\nSOURCES")

        printed = set()

        for chunk in chunks:
            page = chunk.get("page_number")
            name = chunk.get("filename") or chunk.get("document_id")
            key = (name, page)

            if page is not None and key not in printed:
                printed.add(key)
                print(f"- {name}, Page {page}")

    except (ValueError, RuntimeError) as exc:
        parser.exit(
            status=1,
            message=f"\nERROR: {exc}\n",
        )

    except Exception as exc:
        logger.exception("Question answering failed.")
        parser.exit(
            status=1,
            message=f"\nUnexpected error: {exc}\n",
        )


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    )
    main()