import json
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from app.services import embedding_service


def test_document_embeddings_use_the_local_ollama_provider(
    monkeypatch,
    tmp_path,
):
    document_id = "123e4567-e89b-12d3-a456-426614174000"
    chunks_path = tmp_path / f"{document_id}_chunks.json"
    chunks_path.write_text(
        json.dumps(
            {
                "document_id": document_id,
                "source_file": "notes.pdf",
                "chunks": [
                    {
                        "document_id": document_id,
                        "chunk_id": "chunk_00001",
                        "page_number": 1,
                        "text": "The project uses a configured embedding model.",
                        "token_count": 8,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    client = Mock()
    client.embed.return_value = SimpleNamespace(
        embeddings=[[0.25] * 768],
    )
    monkeypatch.setattr(embedding_service, "PROCESSED_DIR", tmp_path)
    monkeypatch.setattr(embedding_service, "OLLAMA_CLIENT", client)
    monkeypatch.setattr(embedding_service, "EMBEDDING_PROVIDER", "ollama")
    monkeypatch.setattr(embedding_service, "MODEL_NAME", "nomic-local")
    monkeypatch.setattr(
        embedding_service,
        "MODEL_FINGERPRINT",
        "ollama:nomic-local:768",
    )

    result = embedding_service.generate_embeddings(
        document_id,
        save_output=False,
    )

    client.embed.assert_called_once_with(
        model="nomic-local",
        input=["search_document: The project uses a configured embedding model."],
    )
    assert result["model"] == "ollama:nomic-local:768"
    assert result["embedding_dimension"] == 768

    client.embed.return_value = SimpleNamespace(
        embeddings=[[0.25] * 767],
    )
    with pytest.raises(
        embedding_service.EmbeddingProviderError,
        match="exactly 768 dimensions",
    ):
        embedding_service.generate_embeddings(document_id, save_output=False)


def test_embeddings_are_batched_and_reused_from_the_valid_cache(
    monkeypatch,
    tmp_path,
):
    document_id = "123e4567-e89b-12d3-a456-426614174000"
    chunks_path = tmp_path / f"{document_id}_chunks.json"
    chunks_path.write_text(
        json.dumps(
            {
                "document_id": document_id,
                "source_file": "notes.pdf",
                "chunks": [
                    {
                        "document_id": document_id,
                        "chunk_id": f"chunk_{index:05d}",
                        "page_number": index,
                        "text": f"Page {index} evidence.",
                    }
                    for index in range(1, 4)
                ],
            }
        ),
        encoding="utf-8",
    )
    client = Mock()
    vector = [0.25] * 768
    client.embed.side_effect = [
        SimpleNamespace(embeddings=[vector, vector]),
        SimpleNamespace(embeddings=[vector]),
    ]
    monkeypatch.setattr(embedding_service, "PROCESSED_DIR", tmp_path)
    monkeypatch.setattr(embedding_service, "OLLAMA_CLIENT", client)
    monkeypatch.setattr(embedding_service, "EMBEDDING_BATCH_SIZE", 2)
    monkeypatch.setattr(embedding_service, "EMBEDDING_PROVIDER", "ollama")
    monkeypatch.setattr(embedding_service, "MODEL_NAME", "nomic-local")
    monkeypatch.setattr(
        embedding_service,
        "MODEL_FINGERPRINT",
        "ollama:nomic-local:768",
    )

    first_result = embedding_service.generate_embeddings(document_id)
    second_result = embedding_service.generate_embeddings(document_id)

    assert len(first_result["chunks"]) == 3
    assert second_result == first_result
    assert client.embed.call_count == 2
    assert client.embed.call_args_list[0].kwargs["input"] == [
        "search_document: Page 1 evidence.",
        "search_document: Page 2 evidence.",
    ]
    assert client.embed.call_args_list[1].kwargs["input"] == [
        "search_document: Page 3 evidence.",
    ]


def test_gemini_document_embeddings_use_retrieval_document_task(
    monkeypatch,
    tmp_path,
):
    document_id = "123e4567-e89b-12d3-a456-426614174000"
    (tmp_path / f"{document_id}_chunks.json").write_text(
        json.dumps(
            {
                "document_id": document_id,
                "source_file": "notes.pdf",
                "chunks": [
                    {
                        "document_id": document_id,
                        "chunk_id": "chunk_00001",
                        "page_number": 1,
                        "text": "The project has a Gemini embedding.",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    response = SimpleNamespace(
        embeddings=[SimpleNamespace(values=[0.25] * 768)],
    )
    client = Mock()
    client.models.embed_content.return_value = response
    monkeypatch.setattr(embedding_service, "PROCESSED_DIR", tmp_path)
    monkeypatch.setattr(embedding_service, "get_gemini_client", Mock(return_value=client))
    monkeypatch.setattr(embedding_service, "EMBEDDING_PROVIDER", "gemini")
    monkeypatch.setattr(embedding_service, "MODEL_NAME", "gemini-embedding-001")
    monkeypatch.setattr(
        embedding_service,
        "MODEL_FINGERPRINT",
        "gemini:gemini-embedding-001:768",
    )

    result = embedding_service.generate_embeddings(document_id, save_output=False)

    client.models.embed_content.assert_called_once()
    call = client.models.embed_content.call_args.kwargs
    assert call["model"] == "gemini-embedding-001"
    assert call["contents"] == ["The project has a Gemini embedding."]
    assert call["config"].task_type == "RETRIEVAL_DOCUMENT"
    assert call["config"].output_dimensionality == 768
    assert result["model"] == "gemini:gemini-embedding-001:768"
    assert result["chunks"][0]["embedding"] == [0.25] * 768


def test_query_embedding_uses_the_same_gemini_model_and_query_task(monkeypatch):
    from app.services import retrieval_service

    client = Mock()
    client.models.embed_content.return_value = SimpleNamespace(
        embeddings=[SimpleNamespace(values=[0.5] * 768)],
    )
    monkeypatch.setattr(embedding_service, "get_gemini_client", Mock(return_value=client))
    monkeypatch.setattr(embedding_service, "EMBEDDING_PROVIDER", "gemini")
    monkeypatch.setattr(embedding_service, "MODEL_NAME", "gemini-embedding-001")

    vector = retrieval_service.embed_query("  What does the PDF say?  ")

    call = client.models.embed_content.call_args.kwargs
    assert vector == [0.5] * 768
    assert call["model"] == "gemini-embedding-001"
    assert call["contents"] == ["What does the PDF say?"]
    assert call["config"].task_type == "RETRIEVAL_QUERY"
    assert call["config"].output_dimensionality == 768


@pytest.mark.parametrize(
    ("status_code", "exception_type"),
    [
        (401, embedding_service.EmbeddingAuthenticationError),
        (403, embedding_service.EmbeddingAuthenticationError),
        (404, embedding_service.EmbeddingModelError),
        (429, embedding_service.EmbeddingQuotaError),
        (504, embedding_service.EmbeddingTimeoutError),
        (500, embedding_service.EmbeddingProviderError),
    ],
)
def test_gemini_embedding_errors_are_classified_without_provider_details(
    monkeypatch,
    status_code,
    exception_type,
):
    from google.genai.errors import ClientError

    client = Mock()
    client.models.embed_content.side_effect = ClientError(
        code=status_code,
        response_json={"message": "private provider detail"},
    )
    monkeypatch.setattr(embedding_service, "get_gemini_client", Mock(return_value=client))
    monkeypatch.setattr(embedding_service, "EMBEDDING_PROVIDER", "gemini")
    monkeypatch.setattr(embedding_service, "MODEL_NAME", "test-embedding-model")
    monkeypatch.setattr(embedding_service, "GEMINI_API_KEY", "test-key")

    with pytest.raises(exception_type) as exc_info:
        embedding_service.embed_texts(
            ["test input"],
            embedding_service.RETRIEVAL_QUERY,
        )

    assert "private provider detail" not in str(exc_info.value)
