import logging
import traceback
from uuid import UUID

import requests
from fastapi import APIRouter, HTTPException
from pinecone.exceptions import PineconeException
from pydantic import BaseModel, Field

from app.config import GEMINI_API_KEY, PINECONE_API_KEY
from app.services.document_qa_service import (
    answer_document_question,
    answer_documents_question,
)
from app.services.answer_generator import (
    GeminiAuthenticationError,
    GeminiModelError,
    GeminiQuotaError,
    GeminiTimeoutError,
    GeminiUnavailableError,
)
from app.services.embedding_service import (
    EmbeddingAuthenticationError,
    EmbeddingConfigurationError,
    EmbeddingModelError,
    EmbeddingProviderError,
    EmbeddingQuotaError,
    EmbeddingTimeoutError,
    EmbeddingUnavailableError,
)
from app.services.vector_store import PineconeConfigurationError

router = APIRouter(
    prefix="/api/documents",
    tags=["Questions"],
)
logger = logging.getLogger(__name__)


def log_redacted_traceback(context: str, exc: Exception) -> None:
    """Log an exception traceback while redacting configured service secrets."""
    formatted_traceback = "".join(
        traceback.format_exception(type(exc), exc, exc.__traceback__)
    )
    for secret in (GEMINI_API_KEY, PINECONE_API_KEY):
        if secret:
            formatted_traceback = formatted_traceback.replace(secret, "[REDACTED]")
    logger.error("%s\n%s", context, formatted_traceback.rstrip())


class QuestionRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)


class MultiDocumentQuestionRequest(QuestionRequest):
    document_ids: list[UUID] = Field(min_length=1)


class AnswerSource(BaseModel):
    document_id: str | None = None
    filename: str | None = None
    chunk_id: str | None = None
    page_number: int | None = None
    score: float | None = None
    text: str


class QuestionResponse(BaseModel):
    answer: str
    sources: list[AnswerSource]


@router.post("/{document_id}/answer", response_model=QuestionResponse)
def answer_question(document_id: UUID, request: QuestionRequest):
    question = request.question.strip()
    if not question:
        raise HTTPException(
            status_code=422,
            detail="Question cannot be empty.",
        )

    try:
        return answer_document_question(str(document_id), question)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except LookupError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except GeminiAuthenticationError as exc:
        log_redacted_traceback("Gemini authentication failed during document Q&A.", exc)
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except GeminiModelError as exc:
        log_redacted_traceback("Configured Gemini model is unavailable.", exc)
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except GeminiQuotaError as exc:
        log_redacted_traceback("Gemini quota/rate limit reached during document Q&A.", exc)
        raise HTTPException(status_code=429, detail=str(exc)) from exc
    except GeminiTimeoutError as exc:
        log_redacted_traceback("Gemini timed out during document Q&A.", exc)
        raise HTTPException(status_code=504, detail=str(exc)) from exc
    except GeminiUnavailableError as exc:
        log_redacted_traceback("Gemini is unavailable during document Q&A.", exc)
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except EmbeddingAuthenticationError as exc:
        log_redacted_traceback(
            "Embedding provider authentication failed during document Q&A.",
            exc,
        )
        raise HTTPException(
            status_code=503,
            detail="Embedding provider authentication failed.",
        ) from exc
    except EmbeddingModelError as exc:
        log_redacted_traceback(
            "Embedding model is unavailable during document Q&A.",
            exc,
        )
        raise HTTPException(
            status_code=503,
            detail="The configured embedding model is unavailable.",
        ) from exc
    except EmbeddingQuotaError as exc:
        log_redacted_traceback(
            "Embedding provider quota reached during document Q&A.",
            exc,
        )
        raise HTTPException(
            status_code=429,
            detail="The embedding provider rate limit or quota was reached.",
        ) from exc
    except EmbeddingTimeoutError as exc:
        log_redacted_traceback(
            "Embedding provider timed out during document Q&A.",
            exc,
        )
        raise HTTPException(
            status_code=504,
            detail="The embedding provider timed out.",
        ) from exc
    except EmbeddingUnavailableError as exc:
        log_redacted_traceback(
            "Embedding provider is unreachable during document Q&A.",
            exc,
        )
        raise HTTPException(
            status_code=503,
            detail="The embedding provider is temporarily unavailable.",
        ) from exc
    except EmbeddingProviderError as exc:
        log_redacted_traceback("Embedding provider failed during document Q&A.", exc)
        raise HTTPException(
            status_code=503,
            detail="Document embeddings are temporarily unavailable.",
        ) from exc
    except EmbeddingConfigurationError as exc:
        log_redacted_traceback(
            "Embedding provider is not configured for document Q&A.",
            exc,
        )
        raise HTTPException(
            status_code=503,
            detail="Document embeddings are not configured correctly.",
        ) from exc
    except PineconeConfigurationError as exc:
        log_redacted_traceback("Pinecone is not configured for document Q&A.", exc)
        raise HTTPException(
            status_code=503,
            detail="The vector database is not configured correctly.",
        ) from exc
    except PineconeException as exc:
        log_redacted_traceback("Pinecone failed during document Q&A.", exc)
        raise HTTPException(
            status_code=503,
            detail="The vector database is temporarily unavailable.",
        ) from exc
    except (
        requests.exceptions.RequestException,
        RuntimeError,
    ) as exc:
        log_redacted_traceback("Document question request failed.", exc)
        raise HTTPException(
            status_code=503,
            detail=(
                "Document question answering is temporarily unavailable. "
                "Check the backend readiness endpoint and service logs."
            ),
        ) from exc
    except Exception as exc:
        log_redacted_traceback("Unexpected document question failure.", exc)
        raise HTTPException(
            status_code=500,
            detail="An unexpected error occurred while answering the question.",
        ) from exc


@router.post("/answer", response_model=QuestionResponse)
def answer_selected_documents(request: MultiDocumentQuestionRequest):
    question = request.question.strip()
    if not question:
        raise HTTPException(
            status_code=422,
            detail="Question cannot be empty.",
        )

    try:
        return answer_documents_question(
            question,
            document_ids=[str(document_id) for document_id in request.document_ids],
        )
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except LookupError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except GeminiAuthenticationError as exc:
        log_redacted_traceback("Gemini authentication failed during multi-document Q&A.", exc)
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except GeminiModelError as exc:
        log_redacted_traceback("Configured Gemini model is unavailable.", exc)
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except GeminiQuotaError as exc:
        log_redacted_traceback("Gemini quota/rate limit reached during multi-document Q&A.", exc)
        raise HTTPException(status_code=429, detail=str(exc)) from exc
    except GeminiTimeoutError as exc:
        log_redacted_traceback("Gemini timed out during multi-document Q&A.", exc)
        raise HTTPException(status_code=504, detail=str(exc)) from exc
    except GeminiUnavailableError as exc:
        log_redacted_traceback("Gemini is unavailable during multi-document Q&A.", exc)
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except EmbeddingAuthenticationError as exc:
        log_redacted_traceback(
            "Embedding provider authentication failed during multi-document Q&A.",
            exc,
        )
        raise HTTPException(
            status_code=503,
            detail="Embedding provider authentication failed.",
        ) from exc
    except EmbeddingModelError as exc:
        log_redacted_traceback(
            "Embedding model is unavailable during multi-document Q&A.",
            exc,
        )
        raise HTTPException(
            status_code=503,
            detail="The configured embedding model is unavailable.",
        ) from exc
    except EmbeddingQuotaError as exc:
        log_redacted_traceback(
            "Embedding provider quota reached during multi-document Q&A.",
            exc,
        )
        raise HTTPException(
            status_code=429,
            detail="The embedding provider rate limit or quota was reached.",
        ) from exc
    except EmbeddingTimeoutError as exc:
        log_redacted_traceback(
            "Embedding provider timed out during multi-document Q&A.",
            exc,
        )
        raise HTTPException(
            status_code=504,
            detail="The embedding provider timed out.",
        ) from exc
    except EmbeddingUnavailableError as exc:
        log_redacted_traceback(
            "Embedding provider is unreachable during multi-document Q&A.",
            exc,
        )
        raise HTTPException(
            status_code=503,
            detail="The embedding provider is temporarily unavailable.",
        ) from exc
    except EmbeddingProviderError as exc:
        log_redacted_traceback(
            "Embedding provider failed during multi-document Q&A.",
            exc,
        )
        raise HTTPException(
            status_code=503,
            detail="Document embeddings are temporarily unavailable.",
        ) from exc
    except EmbeddingConfigurationError as exc:
        log_redacted_traceback(
            "Embedding provider is not configured for multi-document Q&A.",
            exc,
        )
        raise HTTPException(
            status_code=503,
            detail="Document embeddings are not configured correctly.",
        ) from exc
    except PineconeConfigurationError as exc:
        log_redacted_traceback(
            "Pinecone is not configured for multi-document Q&A.",
            exc,
        )
        raise HTTPException(
            status_code=503,
            detail="The vector database is not configured correctly.",
        ) from exc
    except PineconeException as exc:
        log_redacted_traceback("Pinecone failed during multi-document Q&A.", exc)
        raise HTTPException(
            status_code=503,
            detail="The vector database is temporarily unavailable.",
        ) from exc
    except (
        requests.exceptions.RequestException,
        RuntimeError,
    ) as exc:
        log_redacted_traceback("Multi-document question request failed.", exc)
        raise HTTPException(
            status_code=503,
            detail=(
                "Document question answering is temporarily unavailable. "
                "Check the backend readiness endpoint and service logs."
            ),
        ) from exc
    except Exception as exc:
        log_redacted_traceback("Unexpected multi-document question failure.", exc)
        raise HTTPException(
            status_code=500,
            detail="An unexpected error occurred while answering the question.",
        ) from exc
