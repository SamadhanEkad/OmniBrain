
import pymupdf
import pytest
from fastapi import HTTPException, UploadFile
from io import BytesIO
from unittest.mock import patch

from Backend.app.services.file_service import save_pdf


def create_test_pdf():
    pdf = pymupdf.open()
    pdf.new_page()
    content = pdf.tobytes()
    pdf.close()
    return content


@pytest.mark.asyncio
async def test_valid_pdf_is_saved(tmp_path):
    pdf_content = create_test_pdf()

    with (
        patch(
            "Backend.app.services.file_service.RAW_DATA_DIR",
            tmp_path
        ),
        patch(
            "Backend.app.services.file_service.MAX_UPLOAD_SIZE_BYTES",
            20 * 1024 * 1024
        ),
    ):
        upload = UploadFile(
            filename="sample.pdf",
            file=BytesIO(pdf_content)
        )

        result = await save_pdf(upload)

        saved_file = tmp_path / f"{result['document_id']}.pdf"

        assert saved_file.exists()
        assert result["status"] == "uploaded"
        assert result["page_count"] == 1
        assert saved_file.read_bytes() == pdf_content


@pytest.mark.asyncio
async def test_non_pdf_is_rejected():
    upload = UploadFile(
        filename="notes.txt",
        file=BytesIO(b"Hello")
    )

    with pytest.raises(HTTPException) as exc:
        await save_pdf(upload)

    assert exc.value.status_code == 400
