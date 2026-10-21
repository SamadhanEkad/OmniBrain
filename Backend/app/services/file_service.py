import asyncio
import json
import logging
from pathlib import Path
from typing import Any
from uuid import uuid4

import pymupdf
from fastapi import HTTPException, UploadFile

from app.config import (
    RAW_DATA_DIR,
    MAX_UPLOAD_SIZE_BYTES,
    MAX_FILES_PER_UPLOAD,
)

logger = logging.getLogger(__name__)

# Serializes uploads so two simultaneous requests don't write at the same time.
_upload_lock = asyncio.Lock()


def _document_info_path(document_id: str) -> Path:
    if (
        not isinstance(document_id, str)
        or not document_id.strip()
        or document_id in {".", ".."}
        or "/" in document_id
        or "\\" in document_id
    ):
        raise ValueError("Invalid document ID.")

    return RAW_DATA_DIR / f"{document_id}.json"


def load_document_info(document_id: str) -> dict[str, Any] | None:
    """Load durable filename, page-count, and generated-metadata details."""
    info_path = _document_info_path(document_id)

    if not info_path.is_file():
        return None

    try:
        with info_path.open("r", encoding="utf-8") as info_file:
            info = json.load(info_file)
    except (OSError, json.JSONDecodeError) as exc:
        logger.exception("Could not read document information for %s", document_id)
        raise RuntimeError(
            f"Could not read saved information for document {document_id}."
        ) from exc

    if not isinstance(info, dict) or info.get("document_id") != document_id:
        raise RuntimeError(
            f"Saved information for document {document_id} is invalid."
        )

    return info


def save_document_info(
    document_id: str,
    **updates: Any,
) -> dict[str, Any]:
    """Atomically update the durable record associated with a PDF."""
    info_path = _document_info_path(document_id)
    info = load_document_info(document_id) or {"document_id": document_id}
    info.update(updates)
    temporary_path = info_path.with_name(f".{info_path.name}.tmp")

    try:
        with temporary_path.open("w", encoding="utf-8") as info_file:
            json.dump(info, info_file, ensure_ascii=False, indent=2)
            info_file.write("\n")

        temporary_path.replace(info_path)
    finally:
        temporary_path.unlink(missing_ok=True)

    return info


def update_processing_stage(
    document_id: str,
    stage: str,
    status: str,
    error: str | None = None,
) -> dict[str, Any]:
    """Persist the status of one independent document-processing stage."""
    if status not in {"pending", "running", "complete", "failed"}:
        raise ValueError(f"Invalid processing status: {status}")

    info = load_document_info(document_id) or {"document_id": document_id}
    stages = info.get("processing_status")
    if not isinstance(stages, dict):
        stages = {}

    stage_info: dict[str, Any] = {"status": status}
    if error:
        stage_info["error"] = error
    stages[stage] = stage_info
    return save_document_info(
        document_id,
        processing_status=stages,
    )


# --------------------------------------------------
# Workspace document listing helpers
# --------------------------------------------------

def count_documents() -> int:
    """Number of PDFs currently stored in the workspace."""
    # Temporary uploads are named ".<id>.tmp", so they are never counted.
    return sum(1 for _ in RAW_DATA_DIR.glob("*.pdf"))


def list_documents() -> list[dict[str, Any]]:
    """Return basic info for every stored PDF, oldest upload first."""
    documents = []

    for pdf_path in sorted(
        RAW_DATA_DIR.glob("*.pdf"),
        key=lambda p: p.stat().st_mtime,
    ):
        document_id = pdf_path.stem
        try:
            info = load_document_info(document_id)
        except RuntimeError:
            logger.warning("Skipping unreadable info for %s", document_id)
            continue

        if not info:
            continue

        documents.append(
            {
                "document_id": document_id,
                "filename": info.get("filename", pdf_path.name),
                "page_count": info.get("page_count"),
            }
        )

    return documents


def _ensure_upload_limit(new_files: int) -> None:
    """Raise 400 if ONE upload request contains too many files.

    This checks only the files in the current request. It does not count
    PDFs that were uploaded earlier.
    """
    if new_files > MAX_FILES_PER_UPLOAD:
        raise HTTPException(
            status_code=400,
            detail=(
                f"You can upload at most {MAX_FILES_PER_UPLOAD} PDFs at a time. "
                f"You selected {new_files}."
            ),
        )


def delete_document_files(document_id: str) -> bool:
    """Delete the PDF and its info file from disk.

    This only removes local files. The caller must also remove the
    document's vectors from Pinecone (see vector_store.py).
    """
    info_path = _document_info_path(document_id)
    pdf_path = RAW_DATA_DIR / f"{document_id}.pdf"

    existed = pdf_path.is_file() or info_path.is_file()
    pdf_path.unlink(missing_ok=True)
    info_path.unlink(missing_ok=True)
    return existed


# --------------------------------------------------
# Saving uploads
# --------------------------------------------------

async def _save_single_pdf(file: UploadFile) -> dict:
    """Validate and store one PDF. Does NOT check the per-upload limit."""
    # 1. Check the file extension
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(
            status_code=400,
            detail="Please upload a PDF file."
        )

    # 2. Generate a unique document ID
    document_id = str(uuid4())

    # 3. Define permanent and temporary file paths
    file_path = RAW_DATA_DIR / f"{document_id}.pdf"
    temp_path = RAW_DATA_DIR / f".{document_id}.tmp"

    size = 0

    try:
        # 4. Save the upload temporarily in small chunks
        with temp_path.open("wb") as output:
            while True:
                chunk = await file.read(1024 * 1024)

                if not chunk:
                    break

                size += len(chunk)

                if size > MAX_UPLOAD_SIZE_BYTES:
                    limit_mb = MAX_UPLOAD_SIZE_BYTES // (1024 * 1024)
                    raise HTTPException(
                        status_code=413,
                        detail=f"File exceeds the {limit_mb} MB limit."
                    )

                output.write(chunk)

        # 5. Reject empty files
        if size == 0:
            raise HTTPException(
                status_code=400,
                detail="The uploaded file is empty."
            )

        # 6. Validate the PDF structure
        try:
            with pymupdf.open(temp_path) as pdf:
                if pdf.is_encrypted:
                    raise HTTPException(
                        status_code=400,
                        detail="Password-protected PDFs are not supported."
                    )

                if pdf.page_count == 0:
                    raise HTTPException(
                        status_code=400,
                        detail="The PDF contains no pages."
                    )

                page_count = pdf.page_count
                raw_meta = pdf.metadata or {}
                pdf_metadata = {
                    k: v.strip()
                    for k, v in raw_meta.items()
                    if isinstance(v, str) and v.strip()
                }

        except HTTPException:
            raise
        except Exception as exc:
            raise HTTPException(
                status_code=400,
                detail="Invalid or unreadable PDF."
            ) from exc

        # 7. Move the validated PDF to permanent storage
        temp_path.replace(file_path)

        save_document_info(
            document_id,
            filename=file.filename,
            page_count=page_count,
            metadata_status="pending",
            processing_status={
                "metadata": {"status": "pending"},
                "extraction": {"status": "pending"},
                "chunking": {"status": "pending"},
                "embedding": {"status": "pending"},
                "indexing": {"status": "pending"},
            },
        )

        # 8. Return upload details
        return {
            "document_id": document_id,
            "filename": file.filename,
            "status": "uploaded",
            "page_count": page_count,
            "size_bytes": size
        }

    finally:
        await file.close()
        temp_path.unlink(missing_ok=True)


async def save_pdf(file: UploadFile) -> dict:
    """Save one PDF."""
    async with _upload_lock:
        return await _save_single_pdf(file)


async def save_pdfs(files: list[UploadFile]) -> dict:
    """Save several PDFs in one request (at most MAX_FILES_PER_UPLOAD).

    The whole request is rejected up front if it contains too many files.
    After that, each file is handled independently, so one bad PDF does
    not stop the others from uploading.
    """
    if not files:
        raise HTTPException(
            status_code=400,
            detail="No files were uploaded."
        )

    _ensure_upload_limit(len(files))

    async with _upload_lock:
        uploaded: list[dict] = []
        failed: list[dict] = []

        for file in files:
            try:
                uploaded.append(await _save_single_pdf(file))
            except HTTPException as exc:
                failed.append(
                    {
                        "filename": file.filename,
                        "status": "failed",
                        "error": exc.detail,
                    }
                )

    return {"uploaded": uploaded, "failed": failed}