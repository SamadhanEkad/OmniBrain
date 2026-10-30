import logging
from typing import List

from fastapi import APIRouter, File, HTTPException, UploadFile
from pinecone.exceptions import PineconeException

from app.services.document_qa_service import remove_document
from app.services.file_service import save_pdfs

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/api/documents",
    tags=["Documents"]
)


@router.post("/upload")
async def upload_documents(files: List[UploadFile] = File(...)):
    """Upload 1 to MAX_FILES_PER_UPLOAD PDFs in one request.

    Returns: {"uploaded": [...], "failed": [...]}
    """
    return await save_pdfs(files)


@router.delete("/{document_id}")
def delete_document(document_id: str):
    """Delete one uploaded PDF, its processed data, and its vectors."""
    try:
        result = remove_document(document_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid document ID.")
    except (RuntimeError, PineconeException) as exc:
        logger.exception("Could not fully delete document %s.", document_id)
        raise HTTPException(
            status_code=503,
            detail="Document deletion is temporarily unavailable.",
        ) from exc

    return {
        "document_id": document_id,
        "deleted": result["deleted"],
    }