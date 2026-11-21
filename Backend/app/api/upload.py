import logging
from threading import Lock
from typing import List
from uuid import UUID

from fastapi import APIRouter, BackgroundTasks, File, HTTPException, UploadFile
from pinecone.exceptions import PineconeException

from app.api.questions import log_redacted_traceback
from app.services.document_qa_service import (
    ensure_document_indexed,
    remove_document,
)
from app.services.file_service import (
    PROCESS_INSTANCE_ID,
    load_document_info,
    save_pdfs,
    update_processing_stage,
)

logger = logging.getLogger(__name__)
_scheduled_documents: set[str] = set()
_scheduled_documents_lock = Lock()

router = APIRouter(
    prefix="/api/documents",
    tags=["Documents"]
)


def process_uploaded_document(document_id: str) -> None:
    try:
        ensure_document_indexed(document_id)
    except Exception as exc:
        log_redacted_traceback(
            f"Background processing failed for document {document_id}.",
            exc,
        )
    finally:
        with _scheduled_documents_lock:
            _scheduled_documents.discard(document_id)


def schedule_document_processing(
    background_tasks: BackgroundTasks,
    document_id: str,
) -> None:
    with _scheduled_documents_lock:
        if document_id in _scheduled_documents:
            return
        _scheduled_documents.add(document_id)

    try:
        background_tasks.add_task(process_uploaded_document, document_id)
    except Exception:
        with _scheduled_documents_lock:
            _scheduled_documents.discard(document_id)
        raise


@router.post("/upload")
async def upload_documents(
    background_tasks: BackgroundTasks,
    files: List[UploadFile] = File(...),
):
    """Upload 1 to MAX_FILES_PER_UPLOAD PDFs in one request.

    Returns: {"uploaded": [...], "failed": [...]}
    """
    result = await save_pdfs(files)
    for document in result["uploaded"]:
        schedule_document_processing(
            background_tasks,
            document["document_id"],
        )
    return result


@router.get("/{document_id}/status")
def document_processing_status(
    document_id: str,
    background_tasks: BackgroundTasks,
):
    """Return the persisted status of one uploaded document's processing."""
    try:
        document_id = str(UUID(document_id))
    except ValueError as exc:
        raise HTTPException(status_code=400, detail="Invalid document ID.") from exc

    try:
        info = load_document_info(document_id)
    except RuntimeError as exc:
        logger.exception("Could not read processing status for %s.", document_id)
        raise HTTPException(
            status_code=500,
            detail="Document processing status is temporarily unavailable.",
        ) from exc

    if info is None:
        raise HTTPException(status_code=404, detail="Document not found.")

    stages = info.get("processing_status", {})
    if not isinstance(stages, dict):
        stages = {}
    required_stages = ("extraction", "chunking", "embedding", "indexing")
    stale_stages = [
        stage
        for stage in required_stages
        if isinstance(stages.get(stage), dict)
        and stages[stage].get("status") == "running"
        and stages[stage].get("process_instance") != PROCESS_INSTANCE_ID
    ]
    for stage in stale_stages:
        update_processing_stage(document_id, stage, "pending")
    if stale_stages:
        info = load_document_info(document_id) or info
        stages = info.get("processing_status", {})
        if not isinstance(stages, dict):
            stages = {}

    statuses = [
        stages.get(stage, {}).get("status")
        for stage in required_stages
        if isinstance(stages.get(stage), dict)
    ]

    if "failed" in statuses:
        processing_state = "failed"
    elif (
        isinstance(stages.get("indexing"), dict)
        and stages["indexing"].get("status") == "complete"
    ) or (
        len(statuses) == len(required_stages)
        and all(status == "complete" for status in statuses)
    ):
        processing_state = "ready"
    elif "running" in statuses:
        processing_state = "processing"
    else:
        processing_state = "queued"

    if processing_state == "queued":
        schedule_document_processing(background_tasks, document_id)

    return {
        "document_id": document_id,
        "processing_state": processing_state,
        "processing_status": stages,
    }


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