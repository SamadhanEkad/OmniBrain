# Ask-flow diagnosis

## Root cause

Gemini answer generation intermittently returned HTTP 503 for the configured
model; the frontend then replaced the backend's diagnostic message with an
unrelated generic document-processing error.

## Evidence

- `GET /health` returned HTTP 200:
  `{"status":"healthy","service":"OmniBrain API","version":"1.0.0"}`.
- `GET /ready` returned HTTP 200:
  `{"status":"ready","checks":{"gemini":"ready","pinecone":"ready","embedding":"ready"}}`.
- `Backend/scripts/debug_ask.py` uploaded a generated one-page PDF (HTTP 200),
  observed processing reach `ready`, and asked the question:
  - Answer HTTP 200:
    `{"answer":"The Atlas observatory telescope opened in **1998** [Page 1].", ...}`
  - Returned source contained `page_number: 1` and the complete source passage.
  - Delete returned HTTP 200 and `{"deleted":true}`.
- Browser flow uploaded `browser_debug.pdf`; the document progressed to
  `processing_state: ready`, and the textarea became enabled. The browser then
  sent the question request.
- Browser answer request: HTTP 503, body:
  `{"detail":"Gemini answer generation is temporarily unavailable."}`
- Browser console:
  - `Failed to load resource: the server responded with a status of 503 (Service Unavailable)`
  - `[OmniBrain API] /api/documents/answer failed with HTTP 503.`
- The chat bubble displayed
  `Document processing is temporarily unavailable. Please try again shortly.`
  instead of the backend's response detail.
- `client.models.list()` returned `models/gemini-3.5-flash` with
  `generateContent` in supported actions. The configured
  `GEMINI_MODEL=gemini-3.5-flash` therefore exists for the configured API key.
- The embedding model configured in the existing local `Backend/.env` is
  `BAAI/bge-base-en-v1.5`; Pinecone and embedding readiness passed, and the
  successful API request completed retrieval and generation. Neither is the
  observed failing layer.
- The browser's `Frontend/.env.development` points to
  `http://127.0.0.1:8000`, matching the running backend. The request reached
  that API, so this is not a CORS, URL, or request-shape failure.

## Fault layer

Primary: transient Gemini answer-generation unavailability (HTTP 503), not a
missing model. Secondary: frontend error mapping obscures the backend's 503
detail. The API path and response shape worked on the successful repro.
