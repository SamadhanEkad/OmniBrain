from fastapi import APIRouter, File, UploadFile

from Backend.app.services.file_service import save_pdf

router = APIRouter(
    prefix="/api/documents",
    tags=["Documents"]
)

@router.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    result = await save_pdf(file)
    return result