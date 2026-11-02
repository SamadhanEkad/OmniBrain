from functools import lru_cache

from google import genai
from google.genai import types

from app.config import GEMINI_API_KEY

GEMINI_TIMEOUT_MS = 60_000


@lru_cache(maxsize=1)
def get_gemini_client() -> genai.Client:
    """Create the shared Gemini client with bounded timeouts and retries."""
    api_key = GEMINI_API_KEY
    if not api_key or not api_key.strip():
        raise RuntimeError(
            "GEMINI_API_KEY environment variable is not set. "
            "Please configure your Gemini API key."
        )

    clean_key = api_key.strip().strip("'\"").strip()
    if clean_key.startswith("<") and clean_key.endswith(">"):
        raise RuntimeError(
            "GEMINI_API_KEY is set to a placeholder. "
            "Set a valid Gemini API key in the backend environment."
        )
    return genai.Client(
        api_key=clean_key,
        http_options=types.HttpOptions(
            timeout=GEMINI_TIMEOUT_MS,
            retry_options=types.HttpRetryOptions(
                attempts=2,
                initial_delay=1,
                max_delay=4,
                jitter=1,
                http_status_codes=[408, 429, 500, 502, 503, 504],
            ),
        ),
    )
