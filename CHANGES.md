# CHANGES

## Files changed or created

- `AUDIT.md` - Phase 1 repository audit and baseline verification.
- `Backend/.env.example` - Gemini and Gemini-index defaults.
- `Backend/app/api/questions.py` - Fixed missing-document 404 response.
- `Backend/app/config.py` - `OMNIBRAIN_DATA_DIR`, Gemini default, Gemini index default.
- `Backend/app/services/embedding_service.py` - Canonical processed directory,
  Gemini quota retry/backoff, and embedding isolation fixes.
- `Backend/app/services/file_service.py` - Processed-directory config export.
- `Backend/app/services/metadata_generation.py` - Legacy-helper comment.
- `Backend/app/services/pdf_extractor.py` - Canonical processed directory.
- `Backend/app/services/text_chunker.py` - Canonical processed directory and
  cached tiktoken helper.
- `Backend/app/services/vector_store.py` - Canonical processed directory.
- `Backend/requirements.txt` - Removed `sentence-transformers`.
- `Backend/requirements-local-embeddings.txt` - Optional local embedding extras.
- `Backend/scripts/smoke_test.py` - Local API end-to-end smoke test.
- `Backend/tests/conftest.py` - Autouse environment isolation.
- `Backend/tests/test_data_dirs.py` - Data-directory test coverage.
- `Backend/tests/test_document_qa.py` - Index/test fixes and chunker isolation.
- `Backend/tests/test_embedding_service.py` - Gemini test isolation and retry test.
- `Frontend/src/App.jsx` - Missing-document state handling.
- `Frontend/src/components/DocumentPanel.jsx` - Missing-document status label.
- `Frontend/src/services/api.js` - Network error includes configured API URL.
- `Frontend/tests/App.test.jsx` - Missing-document state test.
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
- Secret scan: 0 actual secret matches in tracked working files and 0 in
  `git log -p --all`.
- Tracked-file inventory: no `.env`, PDF, or `Backend/Data` files are tracked;
  allowed environment examples and public frontend environment files remain.

## MANUAL STEPS FOR ME

- Confirm or create the `omnibrain-documents-gemini` Pinecone index with 768
  dimensions and cosine metric.
- Set Render secrets: `PINECONE_API_KEY`, `GEMINI_API_KEY`, and
  `CORS_ORIGINS`.
- Set the frontend `VITE_API_URL` to the real `omnibrain-backend` Render URL.
- Set `GEMINI_MODEL` to a model available to the configured API key.
- Deploy the Blueprint, verify `/health` and `/ready`, then trigger the frontend
  deploy.
- Re-upload existing PDFs after deployment because old local files and BGE
  vectors are incompatible with the Gemini embedding fingerprint.
- On the free Render plan, expect uploaded PDFs to disappear after restarts;
  use a paid persistent disk or `OMNIBRAIN_DATA_DIR` for durable storage.
