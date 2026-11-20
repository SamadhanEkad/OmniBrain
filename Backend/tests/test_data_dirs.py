import json
from types import SimpleNamespace
from unittest.mock import Mock

from app.services import embedding_service, pdf_extractor, text_chunker


def test_processing_services_use_monkeypatched_processed_data_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(pdf_extractor, "PROCESSED_DATA_DIR", tmp_path)
    monkeypatch.setattr(text_chunker, "PROCESSED_DATA_DIR", tmp_path)
    monkeypatch.setattr(embedding_service, "PROCESSED_DATA_DIR", tmp_path)

    pdf_path = tmp_path / "source.pdf"
    pdf_path.write_bytes(b"%PDF-1.4")

    # Exercise the output-path helpers by creating a one-page PDF with PyMuPDF.
    import pymupdf

    with pymupdf.open() as pdf:
        pdf.new_page()
        pdf[0].insert_text((72, 72), "OmniBrain data directory test.")
        pdf.save(pdf_path)

    extracted = pdf_extractor.extract_pdf_text(
        pdf_path,
        document_id="data-dir-test",
    )
    assert (tmp_path / "data-dir-test.json").is_file()

    chunked = text_chunker.chunk_extracted_json(
        extracted["output_json"],
        output_dir=tmp_path,
    )
    assert (tmp_path / "data-dir-test_chunks.json").is_file()

    with (tmp_path / "data-dir-test_chunks.json").open("r", encoding="utf-8") as file:
        data = json.load(file)
    assert data["document_id"] == "data-dir-test"

    client = Mock()
    client.models.embed_content.return_value = SimpleNamespace(
        embeddings=[SimpleNamespace(values=[0.25] * 768)],
    )
    monkeypatch.setattr(embedding_service, "get_gemini_client", Mock(return_value=client))
    monkeypatch.setattr(embedding_service, "EMBEDDING_PROVIDER", "gemini")
    monkeypatch.setattr(embedding_service, "MODEL_NAME", "gemini-embedding-001")
    monkeypatch.setattr(embedding_service, "GEMINI_API_KEY", "test-key")
    result = embedding_service.generate_embeddings("data-dir-test")

    assert (tmp_path / "data-dir-test_embeddings.json").is_file()
    assert result["document_id"] == "data-dir-test"

