import importlib
from fastapi.testclient import TestClient


def make_client(monkeypatch, cors_origins):
    monkeypatch.setenv("ENVIRONMENT", "production")
    monkeypatch.setenv("CORS_ORIGINS", cors_origins)
    import app.main as main

    main = importlib.reload(main)
    return TestClient(main.app)


def assert_answer_preflight_allowed(client, origin):
    response = client.options(
        "/api/documents/answer",
        headers={
            "Origin": origin,
            "Access-Control-Request-Method": "POST",
            "Access-Control-Request-Headers": "content-type",
        },
    )
    assert response.status_code == 200
    assert response.headers["access-control-allow-origin"] == origin


def test_answer_preflight_allowed_for_origin(monkeypatch):
    origin = "https://frontend.onrender.com"
    assert_answer_preflight_allowed(make_client(monkeypatch, origin), origin)


def test_answer_preflight_allowed_for_origin_without_trailing_slash(monkeypatch):
    origin = "https://frontend.onrender.com"
    assert_answer_preflight_allowed(
        make_client(monkeypatch, f"{origin}/"),
        origin,
    )


def test_answer_preflight_rejects_another_origin(monkeypatch):
    origin = "https://frontend.onrender.com"
    client = make_client(monkeypatch, origin)
    response = client.options(
        "/api/documents/answer",
        headers={
            "Origin": "https://another.example.com",
            "Access-Control-Request-Method": "POST",
            "Access-Control-Request-Headers": "content-type",
        },
    )
    assert "access-control-allow-origin" not in response.headers
