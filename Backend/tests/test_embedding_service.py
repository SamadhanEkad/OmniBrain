import json
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from app.services import embedding_service


def test_document_embeddings_use_configured_ollama_model(
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

    result = embedding_service.generate_embeddings(
        document_id,
        save_output=False,
    )

    client.embed.assert_called_once_with(
        model=embedding_service.MODEL_NAME,
        input="search_document: The project uses a configured embedding model.",
    )
    assert result["model"] == embedding_service.MODEL_NAME
    assert result["embedding_dimension"] == 768

    client.embed.return_value = SimpleNamespace(
        embeddings=[[0.25] * 767],
    )
    with pytest.raises(ValueError, match="Expected 768 embedding dimensions"):
        embedding_service.generate_embeddings(document_id, save_output=False)
