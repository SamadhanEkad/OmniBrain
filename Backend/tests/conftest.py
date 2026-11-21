import os

import pytest


@pytest.fixture(autouse=True)
def isolate_embedding_environment(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("PINECONE_API_KEY", raising=False)
    monkeypatch.delenv("OMNIBRAIN_DATA_DIR", raising=False)
