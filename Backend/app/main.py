from contextlib import asynccontextmanager
import logging
import os

import requests
from fastapi import FastAPI, Response
from fastapi.middleware.cors import CORSMiddleware
from google.genai.errors import APIError

from app.api.questions import router as questions_router
from app.api.upload import router as upload_router
from app.config import (
    EMBEDDING_MODEL,
    GEMINI_API_KEY,
    GEMINI_MODEL,
    OLLAMA_URL,
    PINECONE_API_KEY,
    PINECONE_INDEX_NAME,
    VECTOR_SIZE,
)
from app.services.answer_generator import get_gemini_client
from app.services.retrieval_service import get_pinecone_client


# --------------------------------------------------
# 1. Configuration
# --------------------------------------------------

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO").upper(),
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
)

logger = logging.getLogger(__name__)

APP_VERSION = "1.0.0"

DEFAULT_CORS_ORIGINS = (
    "http://localhost:5173,"
    "http://127.0.0.1:5173,"
    "http://localhost:5174,"
    "http://127.0.0.1:5174,"
    "http://localhost:5175,"
    "http://127.0.0.1:5175,"
    "http://localhost:3000,"
    "http://127.0.0.1:3000"
)


def parse_cors_origins(origins: str) -> list[str]:
    """Parse comma-separated origins and normalize optional trailing slashes."""
    return list(
        dict.fromkeys(
            origin.strip().rstrip("/")
            for origin in origins.split(",")
            if origin.strip()
        )
    )


CORS_ORIGINS = parse_cors_origins(
    os.getenv("CORS_ORIGINS", DEFAULT_CORS_ORIGINS)
)

# --------------------------------------------------
# 2. Create FastAPI application
# --------------------------------------------------

@asynccontextmanager
async def lifespan(_: FastAPI):
    logger.info("OmniBrain API is starting.")
    logger.info("API version: %s", APP_VERSION)
    logger.info("Allowed frontend origins: %s", CORS_ORIGINS)
    yield
    logger.info("OmniBrain API is shutting down.")


app = FastAPI(
    title="OmniBrain API",
    description=(
        "Backend API for PDF upload, document processing, "
        "metadata generation, and document question answering."
    ),
    version=APP_VERSION,
    lifespan=lifespan,
)


# --------------------------------------------------
# 3. Configure CORS for the React/Vite frontend
# --------------------------------------------------

app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_origin_regex=r"^https?://(localhost|127\.0\.0\.1)(:\d+)?$",
    allow_credentials=False,
    allow_methods=["GET", "POST", "DELETE", "OPTIONS"],
    allow_headers=["Content-Type"],
)


# --------------------------------------------------
# 4. Register existing API routers
# --------------------------------------------------

app.include_router(upload_router)
app.include_router(questions_router)


# --------------------------------------------------
# 5. Root endpoint
# --------------------------------------------------

@app.get("/", tags=["System"])
def home():
    """Return basic API information."""

    return {
        "message": "Welcome to OmniBrain API",
        "status": "running",
        "version": APP_VERSION,
    }


# --------------------------------------------------
# 6. Health check
# --------------------------------------------------

@app.get("/health", tags=["System"])
def health_check():
    """
    Confirm that the FastAPI application is responding.

    This checks the API process only. It does not verify that
    Ollama, Pinecone, or document indexing is working.
    """

    return {
        "status": "healthy",
        "service": "OmniBrain API",
        "version": APP_VERSION,
    }


@app.get("/ready", tags=["System"])
def readiness_check(response: Response):
    """Check that the configured AI and vector services are reachable."""
    checks: dict[str, str] = {}

    if not GEMINI_API_KEY or not GEMINI_API_KEY.strip():
        checks["gemini"] = "not_configured"
    elif not GEMINI_MODEL:
        checks["gemini"] = "model_not_configured"
    else:
        try:
            get_gemini_client().models.get(model=GEMINI_MODEL)
            checks["gemini"] = "ready"
        except APIError as exc:
            checks["gemini"] = (
                "authentication_failed"
                if getattr(exc, "code", None) in (401, 403)
                else "unavailable"
            )
            logger.warning(
                "Gemini readiness check failed (HTTP %s).",
                getattr(exc, "code", "unknown"),
            )
        except Exception as exc:
            checks["gemini"] = "unavailable"
            logger.warning(
                "Gemini readiness check failed (%s).",
                type(exc).__name__,
            )

    if not PINECONE_API_KEY or not PINECONE_API_KEY.strip():
        checks["pinecone"] = "not_configured"
    else:
        client = None
        try:
            client = get_pinecone_client()
            index_names = [index.name for index in client.list_indexes()]
            if PINECONE_INDEX_NAME not in index_names:
                checks["pinecone"] = "index_unavailable"
            else:
                description = client.describe_index(PINECONE_INDEX_NAME)
                checks["pinecone"] = (
                    "ready"
                    if getattr(description, "dimension", None) == VECTOR_SIZE
                    and getattr(description, "metric", None) == "cosine"
                    else "index_misconfigured"
                )
        except Exception as exc:
            checks["pinecone"] = (
                "authentication_failed"
                if getattr(exc, "status", None) == 401
                or "unauthorized" in type(exc).__name__.casefold()
                else "unavailable"
            )
            logger.warning(
                "Pinecone readiness check failed (%s).",
                type(exc).__name__,
            )
        finally:
            if client is not None and hasattr(client, "close"):
                client.close()

    try:
        ollama_response = requests.get(
            f"{OLLAMA_URL}/api/tags",
            timeout=(3, 8),
        )
        ollama_response.raise_for_status()
        models = ollama_response.json().get("models", [])
        model_names = {
            model["name"].split(":")[0]
            for model in models
            if isinstance(model, dict)
            and isinstance(model.get("name"), str)
        }
        checks["ollama"] = (
            "ready" if EMBEDDING_MODEL in model_names else "model_unavailable"
        )
    except (requests.exceptions.RequestException, ValueError, AttributeError):
        checks["ollama"] = "unavailable"

    is_ready = all(status == "ready" for status in checks.values())
    response.status_code = 200 if is_ready else 503
    return {
        "status": "ready" if is_ready else "not_ready",
        "checks": checks,
    }
