import requests
from uuid import UUID

from fastapi import APIRouter, HTTPException
from pinecone.exceptions import PineconeException
from pydantic import BaseModel, Field

from app.services.document_qa_service import answer_document_question

router = APIRouter(
    prefix="/api/documents",
    tags=["Questions"],
)


class QuestionRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)


class AnswerSource(BaseModel):
    document_id: str | None = None
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
    except (
        requests.exceptions.RequestException,
        RuntimeError,
        PineconeException,
    ) as exc:
        raise HTTPException(
            status_code=503,
            detail=f"Document question answering is unavailable: {exc}",
        ) from exc
