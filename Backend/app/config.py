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

# All chunks from all PDFs live in one namespace.
# Each chunk's metadata holds doc_id, filename and page,
# and retrieval filters on doc_id to pick the right PDF.
PINECONE_NAMESPACE = os.getenv("PINECONE_NAMESPACE", "default")

# Number of chunks fetched per question.
RETRIEVAL_TOP_K = int(
    os.getenv(
        "RETRIEVAL_TOP_K",
        "6",
    )
)


# --------------------------------------------------
# 4. Gemini configuration
# --------------------------------------------------

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.5-flash")


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