from contextlib import asynccontextmanager
import logging
import os
from urllib.parse import urlsplit

from fastapi import FastAPI, Response
from fastapi.middleware.cors import CORSMiddleware
from google.genai.errors import APIError

from app.api.questions import (
    log_redacted_traceback,
    router as questions_router,
)
from app.api.upload import router as upload_router
from app.config import (
    EMBEDDING_MODEL,
    EMBEDDING_PROVIDER,
    GEMINI_API_KEY,
    GEMINI_MODEL,
    PINECONE_API_KEY,
    PINECONE_INDEX_NAME,
    VECTOR_SIZE,
)
from app.services.answer_generator import get_gemini_client
from app.services.embedding_service import (
    EmbeddingAuthenticationError,
    EmbeddingConfigurationError,
    EmbeddingModelError,
    EmbeddingQuotaError,
    EmbeddingTimeoutError,
    EmbeddingUnavailableError,
    RETRIEVAL_QUERY,
    embed_texts,
)
from app.services.vector_store import get_pinecone_client


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
    "http://127.0.0.1:5173"
)


def parse_cors_origins(origins: str) -> list[str]:
    """Parse comma-separated URLs into normalized HTTP(S) origins."""
    normalized = []
    for value in origins.split(","):
        value = value.strip()
        if not value:
            continue

        parsed = urlsplit(value)
        scheme = parsed.scheme.lower()
        if (
            scheme not in {"http", "https"}
            or not parsed.hostname
            or parsed.username is not None
            or parsed.password is not None
        ):
            raise ValueError(f"Invalid CORS origin: {value!r}")

        try:
            port = parsed.port
        except ValueError as exc:
            raise ValueError(f"Invalid CORS origin: {value!r}") from exc

        hostname = parsed.hostname.lower()
        if ":" in hostname:
            hostname = f"[{hostname}]"
        default_port = 80 if scheme == "http" else 443
        authority = hostname if port is None or port == default_port else f"{hostname}:{port}"
        origin = f"{scheme}://{authority}"
        if origin not in normalized:
            normalized.append(origin)

    return normalized


def get_cors_origins(origins: str | None, environment: str) -> list[str]:
    is_production = environment.strip().casefold() == "production"
    configured = origins if origins is not None else ""
    parsed_origins = parse_cors_origins(configured)
    if not is_production:
        for local_origin in ("http://localhost:5173", "http://127.0.0.1:5173"):
            if local_origin not in parsed_origins:
                parsed_origins.append(local_origin)
    if is_production:
        return [
            origin
            for origin in parsed_origins
            if urlsplit(origin).hostname not in {"localhost", "127.0.0.1", "::1"}
        ]
    return parsed_origins


ENVIRONMENT = os.getenv("ENVIRONMENT", "development").strip().casefold()
CORS_ORIGINS = get_cors_origins(os.getenv("CORS_ORIGINS"), ENVIRONMENT)

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
    allow_credentials=False,
    allow_methods=["GET", "POST", "DELETE", "OPTIONS"],
    allow_headers=["Accept", "Content-Type"],
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

    This checks the API process only; /ready checks the configured
    providers and vector index.
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
        logger.error("Gemini readiness failed: GEMINI_API_KEY is not configured.")
    elif not GEMINI_MODEL:
        checks["gemini"] = "model_not_configured"
        logger.error("Gemini readiness failed: GEMINI_MODEL is not configured.")
    else:
        try:
            get_gemini_client().models.get(model=GEMINI_MODEL)
            checks["gemini"] = "ready"
        except APIError as exc:
            status = getattr(exc, "code", None)
            if status in (401, 403):
                checks["gemini"] = "authentication_failed"
            elif status == 404:
                checks["gemini"] = "model_unavailable"
            elif status == 429:
                checks["gemini"] = "quota_exceeded"
            else:
                checks["gemini"] = "unavailable"
            log_redacted_traceback(
                f"Gemini readiness check failed (HTTP {status or 'unknown'}).",
                exc,
            )
        except Exception as exc:
            checks["gemini"] = "unavailable"
            log_redacted_traceback(
                f"Gemini readiness check failed ({type(exc).__name__}).",
                exc,
            )

    if not PINECONE_API_KEY or not PINECONE_API_KEY.strip():
        checks["pinecone"] = "not_configured"
        logger.error("Pinecone readiness failed: PINECONE_API_KEY is not configured.")
    else:
        client = None
        try:
            client = get_pinecone_client()
            index_names = [index.name for index in client.list_indexes()]
            if PINECONE_INDEX_NAME not in index_names:
                checks["pinecone"] = "index_unavailable"
                logger.error(
                    "Pinecone readiness failed: index '%s' was not found.",
                    PINECONE_INDEX_NAME,
                )
            else:
                description = client.describe_index(PINECONE_INDEX_NAME)
                checks["pinecone"] = (
                    "ready"
                    if getattr(description, "dimension", None) == VECTOR_SIZE
                    and getattr(description, "metric", None) == "cosine"
                    else "index_misconfigured"
                )
                if checks["pinecone"] == "index_misconfigured":
                    logger.error(
                        "Pinecone readiness failed: index '%s' must have "
                        "dimension %d and cosine metric.",
                        PINECONE_INDEX_NAME,
                        VECTOR_SIZE,
                    )
        except Exception as exc:
            checks["pinecone"] = (
                "authentication_failed"
                if getattr(exc, "status", None) == 401
                or "unauthorized" in type(exc).__name__.casefold()
                else "unavailable"
            )
            log_redacted_traceback(
                f"Pinecone readiness check failed ({type(exc).__name__}).",
                exc,
            )
        finally:
            if client is not None and hasattr(client, "close"):
                client.close()

    try:
        if not EMBEDDING_MODEL or (
            EMBEDDING_PROVIDER == "gemini"
            and (not GEMINI_API_KEY or not GEMINI_API_KEY.strip())
        ):
            checks["embedding"] = "not_configured"
            logger.error(
                "Embedding readiness failed: model or credentials are not configured."
            )
        else:
            vectors = embed_texts(
                ["OmniBrain readiness probe."],
                RETRIEVAL_QUERY,
            )
            checks["embedding"] = (
                "ready"
                if len(vectors) == 1 and len(vectors[0]) == VECTOR_SIZE
                else "dimension_mismatch"
            )
    except EmbeddingConfigurationError as exc:
        checks["embedding"] = "not_configured"
        logger.error("Embedding readiness failed: %s", exc)
    except EmbeddingAuthenticationError as exc:
        checks["embedding"] = "authentication_failed"
        log_redacted_traceback("Embedding readiness authentication failed.", exc)
    except EmbeddingModelError as exc:
        checks["embedding"] = "model_unavailable"
        log_redacted_traceback("Embedding readiness model check failed.", exc)
    except EmbeddingQuotaError as exc:
        checks["embedding"] = "quota_exceeded"
        log_redacted_traceback("Embedding readiness quota check failed.", exc)
    except EmbeddingTimeoutError as exc:
        checks["embedding"] = "timeout"
        log_redacted_traceback("Embedding readiness timed out.", exc)
    except EmbeddingUnavailableError as exc:
        checks["embedding"] = "unavailable"
        log_redacted_traceback("Embedding provider is unreachable.", exc)
    except Exception as exc:
        checks["embedding"] = "unavailable"
        log_redacted_traceback(
            (
                f"{EMBEDDING_PROVIDER} embedding readiness check failed "
                f"({type(exc).__name__})."
            ),
            exc,
        )

    is_ready = all(status == "ready" for status in checks.values())
    response.status_code = 200 if is_ready else 503
    return {
        "status": "ready" if is_ready else "not_ready",
        "checks": checks,
    }
