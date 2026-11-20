import os
import time
from pathlib import Path

import pymupdf
import requests


BASE_URL = os.getenv("SMOKE_TEST_BASE_URL", "http://127.0.0.1:8000").rstrip("/")
PDF_PATH = Path(__file__).with_name("omnibrain_smoke.pdf")


def create_pdf() -> None:
    with pymupdf.open() as pdf:
        page = pdf.new_page()
        page.insert_text(
            (72, 72),
            "OmniBrain smoke test document.\n"
            "The smoke test filename is omnibrain_smoke.pdf.\n"
            "The unique answer token is BLUE-HERON-451.",
        )
        pdf.save(PDF_PATH)


def main() -> None:
    create_pdf()
    try:
        with PDF_PATH.open("rb") as pdf_file:
            upload = requests.post(
                f"{BASE_URL}/api/documents/upload",
                files={"files": (PDF_PATH.name, pdf_file, "application/pdf")},
                timeout=120,
            )
        upload.raise_for_status()
        uploaded = upload.json()["uploaded"]
        if not uploaded:
            raise RuntimeError(f"No document was uploaded: {upload.json()}")

        document_id = uploaded[0]["document_id"]
        status_url = f"{BASE_URL}/api/documents/{document_id}/status"
        for _ in range(60):
            status_response = requests.get(status_url, timeout=30)
            status_response.raise_for_status()
            state = status_response.json()["processing_state"]
            if state == "ready":
                break
            if state == "failed":
                raise RuntimeError(status_response.json())
            time.sleep(2)
        else:
            raise RuntimeError("Document indexing timed out.")

        answer = requests.post(
            f"{BASE_URL}/api/documents/answer",
            json={
                "question": "What unique answer token appears in this document?",
                "document_ids": [document_id],
            },
            timeout=120,
        )
        answer.raise_for_status()
        answer_body = answer.json()
        if "BLUE-HERON-451" not in answer_body["answer"]:
            raise RuntimeError(answer_body)
        if not answer_body["sources"]:
            raise RuntimeError("Answer returned no sources.")
        if not all(
            "page_number" in source
            for source in answer_body["sources"]
        ):
            raise RuntimeError("A source omitted page_number.")

        filename_answer = requests.post(
            f"{BASE_URL}/api/documents/answer",
            json={
                "question": "What is the filename of this document?",
                "document_ids": [document_id],
            },
            timeout=120,
        )
        filename_answer.raise_for_status()
        if "omnibrain_smoke.pdf" not in filename_answer.text:
            raise RuntimeError(filename_answer.json())

        deleted = requests.delete(
            f"{BASE_URL}/api/documents/{document_id}",
            timeout=30,
        )
        deleted.raise_for_status()

        missing = requests.post(
            f"{BASE_URL}/api/documents/answer",
            json={"question": "What is in this document?", "document_ids": [document_id]},
            timeout=30,
        )
        if missing.status_code != 404:
            raise RuntimeError(
                f"Expected 404 after delete, got {missing.status_code}: {missing.text}"
            )
        if missing.json().get("detail") != (
            "This document is no longer available on the server. "
            "Please upload it again."
        ):
            raise RuntimeError(missing.json())

        print("smoke test passed")
    finally:
        PDF_PATH.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
