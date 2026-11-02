import logging
import requests
from uuid import UUID

from fastapi import APIRouter, HTTPException
from pinecone.exceptions import PineconeException
from pydantic import BaseModel, Field

from app.services.document_qa_service import (
    answer_document_question,
    answer_documents_question,
)
from app.services.answer_generator import GeminiAuthenticationError

router = APIRouter(
    prefix="/api/documents",
    tags=["Questions"],
)
logger = logging.getLogger(__name__)


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
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except (
        requests.exceptions.RequestException,
        RuntimeError,
        PineconeException,
    ) as exc:
        logger.error(
            "Document question request failed (%s).",
            type(exc).__name__,
        )
        raise HTTPException(
            status_code=503,
            detail=(
                "Document question answering is temporarily unavailable. "
                "Check the backend readiness endpoint and service logs."
            ),
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
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except (
        requests.exceptions.RequestException,
        RuntimeError,
        PineconeException,
    ) as exc:
        logger.error(
            "Multi-document question request failed (%s).",
            type(exc).__name__,
        )
        raise HTTPException(
            status_code=503,
            detail=(
                "Document question answering is temporarily unavailable. "
                "Check the backend readiness endpoint and service logs."
            ),
        ) from exc
