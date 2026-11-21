# CHANGES

## Files changed or created

- `AUDIT.md` - Phase 1 repository audit and baseline verification.
- `Backend/.env.example` - Gemini and Gemini-index defaults.
- `Backend/app/api/questions.py` - Fixed missing-document 404 response.
- `Backend/app/config.py` - `OMNIBRAIN_DATA_DIR`, Gemini default, Gemini index default.
- `Backend/app/services/embedding_service.py` - Canonical processed directory,
  Gemini quota retry/backoff, and embedding isolation fixes.
- `Backend/app/services/answer_generator.py` - Transient Gemini 429/503 retry,
  friendly exhausted-retry errors, and optional fallback model.
- `Backend/app/services/file_service.py` - Processed-directory config export.
- `Backend/app/services/metadata_generation.py` - Legacy-helper comment.
- `Backend/app/services/pdf_extractor.py` - Canonical processed directory.
- `Backend/app/services/text_chunker.py` - Canonical processed directory and
  cached tiktoken helper.
- `Backend/app/services/vector_store.py` - Canonical processed directory.
- `Backend/requirements.txt` - Removed `sentence-transformers`.
- `Backend/requirements-local-embeddings.txt` - Optional local embedding extras.
- `Backend/scripts/smoke_test.py` - Local API end-to-end smoke test.
- `Backend/scripts/debug_ask.py`, `DEBUG_ASK.md` - Evidence-based PDF ask
  reproduction and root-cause findings.
- `Backend/tests/conftest.py` - Autouse environment isolation.
- `Backend/tests/test_data_dirs.py` - Data-directory test coverage.
- `Backend/tests/test_document_qa.py` - Index/test fixes and chunker isolation.
- `Backend/tests/test_embedding_service.py` - Gemini test isolation and retry test.
- `Frontend/src/App.jsx` - Missing-document state handling.
- `Frontend/src/components/DocumentPanel.jsx` - Missing-document status label.
- `Frontend/src/services/api.js` - Preserves backend answer errors.
- `Frontend/tests/App.test.jsx` - Retry, answer-error, preparation, and
  post-delete history coverage.
- `README.md` - Gemini and Render documentation updates.
- `render.yaml` - Backend and frontend Render Blueprint.
- `DEPLOY.md` - Render dashboard deployment instructions.

## Verification

- Backend pytest without `.env`: `68 passed`.
- Backend app import: `OmniBrain API`.
- Frontend: `8` Vitest tests passed.
- Frontend production build: passed.
- Clean runtime install: passed; pip list had no `torch` or
  `sentence-transformers`.
- Health: `/health` returned healthy.
- `/ready` after Gemini configuration: Gemini and embedding ready; the Gemini
  Pinecone index was initially absent and was successfully auto-created during
  smoke indexing.
- Smoke indexing: document extraction, chunking, Gemini embeddings, and Pinecone
  upsert passed. Final answer generation was blocked by Gemini 503
  "high demand" responses from the configured answer model.
- Secret scan: real credentials were found only in the ignored, untracked
  `Backend/.env`; values were withheld. No secret-pattern matches were found
  in `git log --all`.
- Tracked-file inventory: no `.env`, PDF, or `Backend/Data` files are tracked;
  allowed environment examples and public frontend environment files remain.

## WHAT I MUST DO MANUALLY

- No required `Backend/.env` model change: `gemini-3.5-flash` was verified for
  the configured key. Optionally set `GEMINI_FALLBACK_MODEL` only after
  confirming that model is available to the same key; do not overwrite `.env`.
- In Render, set `PINECONE_API_KEY`, `GEMINI_API_KEY`, and `CORS_ORIGINS` to
  the real frontend origin. Confirm the Pinecone index is 768-dimensional with
  cosine metric.
- Set the frontend Render `VITE_API_URL` to the real backend URL.
- Merge and push `stabilize-render`, redeploy the Blueprint, and verify
  `/health` and `/ready`.
- Re-upload PDFs after deployment or an embedding-provider/model change. The
  free Render plan can lose uploaded files on restart; use persistent storage
  if they must survive.

## Ask flow stabilization

- `Backend/scripts/debug_ask.py` creates, uploads, asks about, and deletes a
  generated PDF without exposing credentials.
- `Backend/app/services/answer_generator.py` retries Gemini 429/503 responses
  up to three attempts and supports an optional fallback model after primary
  model 503 failures.
- `Frontend/src/services/api.js` retains backend error details in chat bubbles;
  the chat welcome state shows when uploaded PDFs are still being prepared.
- The diagnostic API flow returned HTTP 200 with a grounded answer and
  `page_number: 1`; deletion returned HTTP 200.
- Messages remain visible in browser history after deleting their PDF and
  refreshing; the chat no longer replaces an answered conversation with its
  welcome screen when no documents remain.
- Targeted Gemini tests: 16 passed. Full backend tests: 71 passed. Frontend
  tests: 11 passed.

## Deployment-readiness verification

- `render.yaml`, `DEPLOY.md`, `Backend/.env.example`, and `README.md` use the
  same defaults: `GEMINI_MODEL=gemini-3.5-flash`,
  `EMBEDDING_PROVIDER=gemini`, `EMBEDDING_MODEL=gemini-embedding-001`,
  `VECTOR_SIZE=768`, and `PINECONE_INDEX_NAME=omnibrain-documents-gemini`.
- `Backend/requirements.txt` contains neither `torch` nor
  `sentence-transformers`.
- Production-style backend on port `10000`: `/health` HTTP 200; `/ready` HTTP
  200 with Gemini, Pinecone, and embedding all `ready`.
- Production frontend build and preview responded HTTP 200 and connected to
  the local backend. Browser upload displayed the preparation status; the
  successful summary and follow-up question both returned HTTP 200 with
  `page_number: 1` sources. Deleting the PDF made its status endpoint return
  404; after refresh, the conversation messages and citations remained visible
  in browser history.
- During browser testing, one initial answer request returned HTTP 500 with
  `An unexpected error occurred while answering the question.` A subsequent
  UI retry and a direct API retry both succeeded. Running two backend
  processes against the same OneDrive-backed data directory also produced
  `PermissionError [WinError 5]` during metadata replacement; the extra test
  process was stopped, and the single-backend flow completed.
