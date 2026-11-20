import os
import time
from pathlib import Path

import pymupdf
import requests


BASE_URL = os.getenv("OMNIBRAIN_API_URL", "http://127.0.0.1:8000").rstrip("/")
PDF_PATH = Path(__file__).with_name("debug_ask_repro.pdf")


def create_pdf() -> None:
    with pymupdf.open() as pdf:
        page = pdf.new_page()
        page.insert_textbox(
            pymupdf.Rect(72, 72, 540, 240),
            "The first distinct fact is that the Atlas observatory is in Northport.\n"
            "The second distinct fact is that its telescope opened in 1998.\n"
            "The third distinct fact is that the annual report is called Blue Lantern.",
            fontsize=12,
        )
        pdf.save(PDF_PATH)


def main() -> None:
    document_id = None
    create_pdf()
    try:
        with PDF_PATH.open("rb") as pdf_file:
            upload_response = requests.post(
                f"{BASE_URL}/api/documents/upload",
                files={"files": (PDF_PATH.name, pdf_file, "application/pdf")},
                timeout=120,
            )
        print(f"UPLOAD HTTP {upload_response.status_code}")
        print(upload_response.text)
        if not upload_response.ok:
            return

        uploaded = upload_response.json().get("uploaded", [])
        if not uploaded:
            print("UPLOAD_ERROR response contained no uploaded documents")
            return
        document_id = uploaded[0]["document_id"]

        status_url = f"{BASE_URL}/api/documents/{document_id}/status"
        for _ in range(90):
            status_response = requests.get(status_url, timeout=30)
            print(
                f"STATUS HTTP {status_response.status_code} "
                f"{status_response.text}"
            )
            if not status_response.ok:
                return
            state = status_response.json().get("processing_state")
            if state == "ready":
                break
            if state == "failed":
                return
            time.sleep(2)
        else:
            print("STATUS_ERROR document indexing timed out")
            return

        answer_response = requests.post(
            f"{BASE_URL}/api/documents/answer",
            json={
                "question": "What year did the Atlas observatory telescope open?",
                "document_ids": [document_id],
            },
            timeout=660,
        )
        print(f"ANSWER HTTP {answer_response.status_code}")
        print(answer_response.text)
    finally:
        if document_id:
            try:
                delete_response = requests.delete(
                    f"{BASE_URL}/api/documents/{document_id}",
                    timeout=30,
                )
                print(f"DELETE HTTP {delete_response.status_code}")
                print(delete_response.text)
            except requests.RequestException as exc:
                print(f"DELETE_ERROR {type(exc).__name__}: {exc}")
        PDF_PATH.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
