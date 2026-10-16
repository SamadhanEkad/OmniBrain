from uuid import uuid4

import pymupdf
from fastapi import HTTPException, UploadFile

from Backend.app.config import RAW_DATA_DIR, MAX_UPLOAD_SIZE_BYTES

async def save_pdf(file: UploadFile) -> dict:
    # 1. Check the file extension
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(
            status_code=400,
            detail="Please upload s PDF file."
        )

    # 2. Generate a unique document ID
    document_id = str(uuid4())

    # 3. Define permenent and temporary file paths
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
                    raise HTTPException(
                        status_code=413,
                        detail="File exceeds the 20 MiB limit."
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

        except HTTPException:
            raise
        except Exception as exc:
            raise HTTPException(
                status_code=400,
                detail="Invalid or unreadable PDF."
            ) from exc

        # 7. Move the validated PDF to permenent storage
        temp_path.replace(file_path)

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

    