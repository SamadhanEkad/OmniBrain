from typing import List

from fastapi import APIRouter, File, HTTPException, UploadFile

from app.services.file_service import delete_document_files, save_pdfs

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
    """Delete one uploaded PDF and its saved info file."""
    try:
        existed = delete_document_files(document_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid document ID.")

    # If the files are already gone, still report success so the
    # frontend can remove the PDF from its list.
    return {"document_id": document_id, "deleted": existed}