import os
from pathlib import Path
from dotenv import load_dotenv

# Project root: OmniBrain/
BACKEND_ROOT = Path(__file__).resolve().parents[1]

# Load environment variables from Backend/.env or current working directory
load_dotenv(BACKEND_ROOT / ".env")
load_dotenv()

# --------------------------------------------------
# 1. Project directories
# --------------------------------------------------

# Directory containing original uploaded PDFs.
RAW_DATA_DIR = BACKEND_ROOT / "Data" / "raw"

# Directory containing processed extraction, chunking, and embedding files.
PROCESSED_DATA_DIR = BACKEND_ROOT / "Data" / "processed"


# --------------------------------------------------
# 2. Upload configuration
# --------------------------------------------------

# Maximum size of each PDF: 20 MB by default.
MAX_UPLOAD_SIZE_BYTES = int(
    os.getenv(
        "MAX_UPLOAD_SIZE_BYTES",
        str(20 * 1024 * 1024),
    )
)

if MAX_UPLOAD_SIZE_BYTES <= 0:
    raise ValueError(
        "MAX_UPLOAD_SIZE_BYTES must be a positive integer."
    )

# Maximum number of PDFs allowed in ONE upload request.
MAX_FILES_PER_UPLOAD = int(
    os.getenv(
        "MAX_FILES_PER_UPLOAD",
        "5",
    )
)

if MAX_FILES_PER_UPLOAD <= 0:
    raise ValueError(
        "MAX_FILES_PER_UPLOAD must be a positive integer."
    )


# --------------------------------------------------
# 3. Pinecone configuration
# --------------------------------------------------

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "omnibrain-documents")
PINECONE_CLOUD = os.getenv("PINECONE_CLOUD", "aws")
PINECONE_REGION = os.getenv("PINECONE_REGION", "us-east-1")

# Ollama configuration shared by document and query embedding.
OLLAMA_URL = os.getenv(
    "OLLAMA_URL",
    "http://localhost:11434",
).rstrip("/")
EMBEDDING_MODEL = os.getenv(
    "EMBEDDING_MODEL",
    "nomic-embed-text-v2-moe",
)
VECTOR_SIZE = int(os.getenv("VECTOR_SIZE", "768"))
OLLAMA_LLM_MODEL = os.getenv("OLLAMA_LLM_MODEL", "llama3.2:1b")

if VECTOR_SIZE <= 0:
    raise ValueError("VECTOR_SIZE must be a positive integer.")


# --------------------------------------------------
# 4. Gemini configuration
# --------------------------------------------------

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.8-flash")


# --------------------------------------------------
# 5. Ensure required directories exist
# --------------------------------------------------

RAW_DATA_DIR.mkdir(
    parents=True,
    exist_ok=True,
)
PROCESSED_DATA_DIR.mkdir(
    parents=True,
    exist_ok=True,
)