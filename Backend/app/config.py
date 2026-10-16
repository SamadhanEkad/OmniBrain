from pathlib import Path

# Project root: OmniBrain/
BACKEND_ROOT = Path(__file__).resolve().parents[1]

# Original uploaded PDFs
RAW_DATA_DIR = BACKEND_ROOT / "Data" / "raw"

# Maximum PDF upload size: 20 MB
MAX_UPLOAD_SIZE_BYTES = 20 * 1024 * 1024

# Ensure the raw-data directory exists
RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)