import json
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.services import answer_generator
from app.services.answer_generator import build_prompt
from app.services import document_qa_service
from app.services import file_service
from app.services.text_chunker import chunk_extracted_json


def test_answer_prompt_uses_the_supplied_evidence_and_question():
    prompt = build_prompt(
        "How does the document describe the project?",
        "EDUCATION\nData Science\nPROJECTS\nCustomer analytics",
    )
    normalized_prompt = " ".join(prompt.split())

    assert "Use the supplied document excerpts as evidence for facts" in normalized_prompt
    assert "EDUCATION Data Science PROJECTS Customer analytics" in normalized_prompt
    assert "<user_question> How does the document describe the project?" in normalized_prompt


def test_answer_generation_uses_bounded_context_and_output(monkeypatch):
    mock_response = SimpleNamespace(text="A grounded answer.")
    mock_client = Mock()
    mock_client.models.generate_content.return_value = mock_response
    monkeypatch.setattr(
        answer_generator,
        "get_gemini_client",
        Mock(return_value=mock_client),
    )

    result = answer_generator.generate_answer(
        "What does the document explain?",
        [{"text": "evidence " * 2000}],
    )

    assert result == "A grounded answer."
    mock_client.models.generate_content.assert_called_once()
    call_kwargs = mock_client.models.generate_content.call_args.kwargs
    assert call_kwargs["config"].max_output_tokens == answer_generator.MAX_OUTPUT_TOKENS
    assert call_kwargs["config"].temperature == answer_generator.TEMPERATURE
    assert len(answer_generator.prepare_context([{"text": "evidence " * 2000}])) <= answer_generator.MAX_CONTEXT_CHARS


def test_answer_question_endpoint_returns_answer_and_sources(monkeypatch):
    result = {
        "answer": "The report contains two sections.",
        "sources": [
            {
                "document_id": "123e4567-e89b-12d3-a456-426614174000",
                "chunk_id": "chunk_00001",
                "page_number": 2,
                "score": 0.91,
                "text": "Two sections are described.",
            }
        ],
    }
    answer = Mock(return_value=result)
    monkeypatch.setattr(
        "app.api.questions.answer_document_question",
        answer,
    )

    response = TestClient(app).post(
        "/api/documents/123e4567-e89b-12d3-a456-426614174000/answer",
        json={"question": " How many sections are there? "},
    )

    assert response.status_code == 200
    assert response.json() == result
    answer.assert_called_once_with(
        "123e4567-e89b-12d3-a456-426614174000",
        "How many sections are there?",
    )


def test_answer_question_endpoint_rejects_invalid_document_id():
    response = TestClient(app).post(
        "/api/documents/not-a-uuid/answer",
        json={"question": "What is in this document?"},
    )

    assert response.status_code == 422


def test_answer_question_endpoint_reports_vector_db_unavailable(monkeypatch):
    monkeypatch.setattr(
        "app.api.questions.answer_document_question",
        Mock(side_effect=RuntimeError("Pinecone is not reachable.")),
    )

    response = TestClient(app).post(
        "/api/documents/123e4567-e89b-12d3-a456-426614174000/answer",
        json={"question": "What is in this document?"},
    )

    assert response.status_code == 503
    assert "Pinecone is not reachable." in response.json()["detail"]


def test_delete_document_endpoint_also_removes_vectors(monkeypatch):
    document_id = "123e4567-e89b-12d3-a456-426614174000"
    remove = Mock(
        return_value={
            "document_id": document_id,
            "deleted": True,
            "vectors_deleted": 4,
        }
    )
    monkeypatch.setattr("app.api.upload.remove_document", remove)

    response = TestClient(app).delete(f"/api/documents/{document_id}")

    assert response.status_code == 200
    assert response.json() == {
        "document_id": document_id,
        "deleted": True,
    }
    remove.assert_called_once_with(document_id)


def test_document_is_indexed_once_and_existing_index_is_reused(
    monkeypatch,
    tmp_path,
):
    document_id = "123e4567-e89b-12d3-a456-426614174000"
    (tmp_path / f"{document_id}.pdf").touch()
    monkeypatch.setattr(document_qa_service, "RAW_DATA_DIR", tmp_path)
    monkeypatch.setattr(file_service, "RAW_DATA_DIR", tmp_path)

    client = Mock()
    index_mock = Mock()
    client.list_indexes.return_value = []
    client.Index.return_value = index_mock
    monkeypatch.setattr(
        document_qa_service,
        "get_pinecone_client",
        Mock(return_value=client),
    )
    extract = Mock(return_value={"output_json": "extracted.json"})
    chunk = Mock()
    embed = Mock()
    load = Mock(return_value=({"document_id": document_id}, [{"text": "text"}], 768))
    ensure = Mock()
    store = Mock()
    monkeypatch.setattr(document_qa_service, "extract_pdf_text", extract)
    monkeypatch.setattr(document_qa_service, "chunk_extracted_json", chunk)
    monkeypatch.setattr(document_qa_service, "generate_embeddings", embed)
    generate_metadata = Mock(
        return_value={"document_summary": "Test document", "total_pages": 1}
    )
    monkeypatch.setattr(document_qa_service, "generate_metadata", generate_metadata)
    monkeypatch.setattr(document_qa_service, "load_embeddings", load)
    monkeypatch.setattr(document_qa_service, "ensure_index", ensure)
    monkeypatch.setattr(document_qa_service, "store_embeddings", store)

    document_qa_service.ensure_document_indexed(document_id)

    extract.assert_called_once()
    chunk.assert_called_once_with("extracted.json", filename=None)
    embed.assert_called_once_with(document_id)
    ensure.assert_called_once_with(client, 768)
    store.assert_called_once()
    client.close.assert_called_once()

    client.reset_mock()
    index_mock.reset_mock()
    client.list_indexes.return_value = [SimpleNamespace(name="omnibrain-documents")]
    client.describe_index.return_value = SimpleNamespace(
        dimension=768,
        metric="cosine",
    )
    index_mock.query.return_value = SimpleNamespace(matches=[SimpleNamespace(id="match-1")])
    extract.reset_mock()

    document_qa_service.ensure_document_indexed(document_id)

    extract.assert_not_called()
    client.close.assert_called_once()
    generate_metadata.assert_called_once()


def test_answer_document_question_returns_retrieved_citations(monkeypatch):
    document_id = "123e4567-e89b-12d3-a456-426614174000"
    chunks = [
        {
            "document_id": document_id,
            "chunk_id": "chunk_00001",
            "page_number": 3,
            "score": 0.87,
            "text": "The project began in 2024.",
        }
    ]
    index = Mock()
    search = Mock(return_value=chunks)
    generate = Mock(return_value="The project began in 2024.")
    monkeypatch.setattr(document_qa_service, "ensure_document_indexed", index)
    monkeypatch.setattr(document_qa_service, "search_documents", search)
    monkeypatch.setattr(document_qa_service, "generate_answer", generate)

    result = document_qa_service.answer_document_question(
        document_id,
        "When did the project begin?",
    )

    index.assert_called_once_with(document_id)
    search.assert_called_once_with(
        query="When did the project begin?",
        document_id=document_id,
        top_k=3,
    )
    generate.assert_called_once_with(
        question="When did the project begin?",
        retrieved_chunks=chunks,
    )
    assert result == {
        "answer": "The project began in 2024.",
        "sources": [
            {
                **chunks[0],
                "filename": None,
            }
        ],
    }


def test_filename_and_page_count_questions_use_saved_upload_information(
    monkeypatch,
    tmp_path,
):
    document_id = "123e4567-e89b-12d3-a456-426614174000"
    (tmp_path / f"{document_id}.pdf").touch()
    monkeypatch.setattr(document_qa_service, "RAW_DATA_DIR", tmp_path)
    monkeypatch.setattr(file_service, "RAW_DATA_DIR", tmp_path)
    file_service.save_document_info(
        document_id,
        filename="original-report.pdf",
        page_count=12,
        metadata_status="pending",
    )
    search = Mock()
    monkeypatch.setattr(document_qa_service, "search_documents", search)

    filename_result = document_qa_service.answer_document_question(
        document_id,
        "What is the uploaded PDF's filename?",
    )
    page_count_result = document_qa_service.answer_document_question(
        document_id,
        "How many pages does this document have?",
    )

    assert filename_result == {
        "answer": "The original filename is original-report.pdf.",
        "sources": [],
    }
    assert page_count_result == {
        "answer": "This PDF has 12 pages.",
        "sources": [],
    }
    search.assert_not_called()


def test_overview_questions_reuse_persisted_metadata(monkeypatch, tmp_path):
    document_id = "123e4567-e89b-12d3-a456-426614174000"
    (tmp_path / f"{document_id}.pdf").touch()
    monkeypatch.setattr(document_qa_service, "RAW_DATA_DIR", tmp_path)
    monkeypatch.setattr(file_service, "RAW_DATA_DIR", tmp_path)
    file_service.save_document_info(
        document_id,
        filename="original-report.pdf",
        page_count=12,
        metadata_status="pending",
    )
    generate_metadata = Mock(
        return_value={
            "document_summary": "The report describes project outcomes.",
            "document_purpose": "To evaluate the project.",
            "document_type": "Project report",
            "total_pages": 12,
        }
    )
    monkeypatch.setattr(
        document_qa_service,
        "generate_metadata",
        generate_metadata,
    )
    monkeypatch.setattr(
        document_qa_service,
        "search_documents",
        Mock(side_effect=AssertionError("Overview must not use retrieval")),
    )

    summary = document_qa_service.answer_document_question(
        document_id,
        "Summarize this document.",
    )
    purpose = document_qa_service.answer_document_question(
        document_id,
        "What is the purpose of this document?",
    )
    document_type = document_qa_service.answer_document_question(
        document_id,
        "What type of document is this?",
    )

    assert summary["answer"] == "The report describes project outcomes."
    assert purpose["answer"] == "To evaluate the project."
    assert document_type["answer"] == "Project report"
    assert summary["sources"] == purpose["sources"] == document_type["sources"] == []
    generate_metadata.assert_called_once_with(
        str(tmp_path / f"{document_id}.pdf"),
        document_id=document_id,
    )
    assert file_service.load_document_info(document_id)["metadata_status"] == "complete"


def test_missing_document_is_reported_for_saved_information_questions(
    monkeypatch,
    tmp_path,
):
    monkeypatch.setattr(document_qa_service, "RAW_DATA_DIR", tmp_path)

    with pytest.raises(FileNotFoundError, match="Uploaded PDF not found"):
        document_qa_service.answer_document_question(
            "123e4567-e89b-12d3-a456-426614174000",
            "What is the filename?",
        )


def test_metadata_generation_failure_is_saved_and_reported_gracefully(
    monkeypatch,
    tmp_path,
):
    document_id = "123e4567-e89b-12d3-a456-426614174000"
    (tmp_path / f"{document_id}.pdf").touch()
    monkeypatch.setattr(document_qa_service, "RAW_DATA_DIR", tmp_path)
    monkeypatch.setattr(file_service, "RAW_DATA_DIR", tmp_path)
    file_service.save_document_info(
        document_id,
        filename="original-report.pdf",
        page_count=12,
        metadata_status="pending",
    )
    generate_metadata = Mock(side_effect=RuntimeError("Ollama unavailable"))
    monkeypatch.setattr(
        document_qa_service,
        "generate_metadata",
        generate_metadata,
    )

    first_result = document_qa_service.answer_document_question(
        document_id,
        "Summarize this document.",
    )
    second_result = document_qa_service.answer_document_question(
        document_id,
        "What is its purpose?",
    )

    assert "metadata generation failed" in first_result["answer"]
    assert second_result == first_result
    assert first_result["sources"] == []
    assert file_service.load_document_info(document_id)["metadata_status"] == "failed"
    generate_metadata.assert_called_once()


def test_chunker_keeps_chunks_from_every_page(tmp_path):
    document_id = "123e4567-e89b-12d3-a456-426614174000"
    source = tmp_path / "extracted.json"
    source.write_text(
        json.dumps(
            {
                "document_id": document_id,
                "source_file": "sample.pdf",
                "pages": [
                    {"page_number": 1, "text": "First page has unique text."},
                    {"page_number": 2, "text": "Second page has other text."},
                ],
            }
        ),
        encoding="utf-8",
    )

    result = chunk_extracted_json(
        source,
        output_dir=tmp_path / "processed",
    )

    assert {chunk["page_number"] for chunk in result["chunks"]} == {1, 2}
    assert result["chunk_count"] == 2


def test_question_classification():
    from app.services.document_qa_service import classify_question

    assert classify_question("Who is the author of this document?") == "author"
    assert classify_question("Who wrote this?") == "author"
    assert classify_question("What is the official title of the document?") == "title"
    assert classify_question("Summarize this document in 100 words") == "document_overview"
    assert classify_question("What does this document teach us?") == "document_overview"
    assert classify_question("What are the key takeaways from this document?") == "document_overview"
    assert classify_question("Explain this topic in simple words.") == "simple_explanation"
    assert classify_question("Explain this concept as if I am a beginner") == "simple_explanation"
    assert classify_question("Explain this topic with an example") == "simple_explanation"
    assert classify_question("What is the difference between these two concepts?") == "comparison_table"
    assert classify_question("What are the advantages and disadvantages?") == "comparison_table"
    assert classify_question("When was the model trained?") == "specific_content"


def test_summarize_in_100_words_calls_answer_generator(monkeypatch, tmp_path):
    document_id = "123e4567-e89b-12d3-a456-426614174000"
    (tmp_path / f"{document_id}.pdf").touch()
    monkeypatch.setattr(document_qa_service, "RAW_DATA_DIR", tmp_path)
    monkeypatch.setattr(file_service, "RAW_DATA_DIR", tmp_path)
    file_service.save_document_info(
        document_id,
        filename="report.pdf",
        page_count=5,
        metadata={"document_summary": "Static summary"},
        metadata_status="complete",
    )

    index = Mock()
    generate = Mock(return_value="A custom 100-word generated summary.")
    monkeypatch.setattr(document_qa_service, "ensure_document_indexed", index)
    monkeypatch.setattr(document_qa_service, "generate_answer", generate)

    result = document_qa_service.answer_document_question(
        document_id,
        "Summarize this document in 100 words.",
    )

    assert result["answer"] == "A custom 100-word generated summary."
    generate.assert_called_once()
    assert generate.call_args.kwargs["question"] == "Summarize this document in 100 words."


def test_simple_explanation_routes_to_generator(monkeypatch, tmp_path):
    document_id = "123e4567-e89b-12d3-a456-426614174000"
    (tmp_path / f"{document_id}.pdf").touch()
    monkeypatch.setattr(document_qa_service, "RAW_DATA_DIR", tmp_path)
    monkeypatch.setattr(file_service, "RAW_DATA_DIR", tmp_path)
    file_service.save_document_info(
        document_id,
        filename="report.pdf",
        page_count=5,
        metadata_status="complete",
    )

    chunks = [{"document_id": document_id, "chunk_id": "c1", "page_number": 1, "text": "Gradient boosting works by...", "score": 0.9}]
    monkeypatch.setattr(document_qa_service, "ensure_document_indexed", Mock())
    monkeypatch.setattr(document_qa_service, "search_documents", Mock(return_value=chunks))
    generate = Mock(return_value="Imagine you have a team of helpers...")
    monkeypatch.setattr(document_qa_service, "generate_answer", generate)

    result = document_qa_service.answer_document_question(
        document_id,
        "Explain this topic in simple words.",
    )

    assert "Imagine you have a team" in result["answer"]
    generate.assert_called_once()


def test_comparison_routes_to_generator(monkeypatch, tmp_path):
    document_id = "123e4567-e89b-12d3-a456-426614174000"
    (tmp_path / f"{document_id}.pdf").touch()
    monkeypatch.setattr(document_qa_service, "RAW_DATA_DIR", tmp_path)
    monkeypatch.setattr(file_service, "RAW_DATA_DIR", tmp_path)
    file_service.save_document_info(
        document_id,
        filename="report.pdf",
        page_count=5,
        metadata_status="complete",
    )

    chunks = [{"document_id": document_id, "chunk_id": "c1", "page_number": 1, "text": "Model A vs Model B", "score": 0.9}]
    monkeypatch.setattr(document_qa_service, "ensure_document_indexed", Mock())
    monkeypatch.setattr(document_qa_service, "search_documents", Mock(return_value=chunks))
    generate = Mock(return_value="| Feature | Model A | Model B |")
    monkeypatch.setattr(document_qa_service, "generate_answer", generate)

    result = document_qa_service.answer_document_question(
        document_id,
        "What is the difference between Model A and Model B?",
    )

    assert "| Feature |" in result["answer"]
    generate.assert_called_once()
