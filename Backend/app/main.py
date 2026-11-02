from contextlib import asynccontextmanager
import logging
import os

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
