import logging
import os

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.questions import router as questions_router
from app.api.upload import router as upload_router


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
DEPLOYED_FRONTEND_ORIGIN = "https://omnibrain-frontend-be3o.onrender.com"

CORS_ORIGINS = [
    origin.strip()
    for origin in os.getenv(
        "CORS_ORIGINS",
        DEFAULT_CORS_ORIGINS,
    ).split(",")
    if origin.strip()
]
if DEPLOYED_FRONTEND_ORIGIN not in CORS_ORIGINS:
    CORS_ORIGINS.append(DEPLOYED_FRONTEND_ORIGIN)


# --------------------------------------------------
# 2. Create FastAPI application
# --------------------------------------------------

app = FastAPI(
    title="OmniBrain API",
    description=(
        "Backend API for PDF upload, document processing, "
        "metadata generation, and document question answering."
    ),
    version=APP_VERSION,
)


# --------------------------------------------------
# 3. Configure CORS for the React/Vite frontend
# --------------------------------------------------

app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_origin_regex=r"^https?://(localhost|127\.0\.0\.1)(:\d+)?$",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
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


# --------------------------------------------------
# 7. Application startup and shutdown logging
# --------------------------------------------------

@app.on_event("startup")
async def startup_event():
    logger.info("OmniBrain API is starting.")
    logger.info("API version: %s", APP_VERSION)
    logger.info("Allowed frontend origins: %s", CORS_ORIGINS)


@app.on_event("shutdown")
async def shutdown_event():
    logger.info("OmniBrain API is shutting down.")
