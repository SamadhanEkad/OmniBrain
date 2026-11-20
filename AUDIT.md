# OmniBrain Stabilization Audit

## (a) Failing tests and root causes

- Baseline Backend: `65 passed, 1 failed`.
- `tests/test_document_qa.py::test_document_is_indexed_once_and_existing_index_is_reused`
  failed because the test monkeypatches `file_service.PROCESSED_DATA_DIR`, but
  `Backend/app/services/file_service.py` does not currently export that
  attribute.
- The exact three-failure baseline described in the task was not observed.
  Gemini-specific tests and the tiktoken chunker test passed in the current
  pre-audit tree. Pre-existing uncommitted changes appear to have altered the
  test state since the original report.

## (b) Config inconsistencies

- `config.py` defaults `EMBEDDING_PROVIDER` to `local` and
  `PINECONE_INDEX_NAME` to `omnibrain-documents`.
- `pdf_extractor.py` and `text_chunker.py` define separate
  `PROCESSED_DIR` values instead of importing `config.PROCESSED_DATA_DIR`.
- `vector_store.py` defines `PROCESSED_DIR` from `PROJECT_ROOT / "Data" /
  "processed"` rather than using `config.PROCESSED_DATA_DIR`.
- `embedding_service.py` aliases `PROCESSED_DIR = PROCESSED_DATA_DIR`, but tests
  and callers may depend on the alias instead of the canonical config value.
- `OMNIBRAIN_DATA_DIR` is not implemented.
- Pinecone and embedding model/index fingerprints are otherwise consistently
  derived from provider, model, and vector size.
- Gemini answer and embedding model defaults differ intentionally:
  `GEMINI_MODEL` defaults to `gemini-3.5-flash`, while embedding model defaults
  to `gemini-embedding-001`.

## (c) Every place that touches local disk

- `Backend/app/config.py`: creates raw and processed data directories.
- `Backend/app/services/file_service.py`: writes uploaded PDFs, document JSON,
  temporary files, processing status, and deletes PDF/info files.
- `Backend/app/services/pdf_extractor.py`: reads uploaded PDFs and writes
  `<document_id>.json` under processed data.
- `Backend/app/services/text_chunker.py`: reads extracted JSON and writes
  `<document_id>_chunks.json`.
- `Backend/app/services/embedding_service.py`: reads chunk JSON and writes
  `<document_id>_embeddings.json`.
- `Backend/app/services/retrieval_service.py`: reads chunks JSON.
- `Backend/app/services/vector_store.py`: reads embedding JSON and delete
  metadata from processed files.
- `Backend/app/services/document_qa_service.py`: reads raw PDFs/info/chunks,
  locates local files, and deletes raw/processed document artifacts.
- `Backend/app/services/metadata_generation.py`: reads PDFs and writes/updates
  metadata as a legacy helper, but is not called by API routes.
- Backend tests use `tmp_path` for PDF, extraction, chunk, embedding, cache,
  and document-info files.

## (d) torch, sentence-transformers, and Ollama imports

- `Backend/requirements.txt` currently includes `sentence-transformers==6.2.0`
  and transitively installs its local torch stack.
- `embedding_service._get_local_embedding_model()` lazily imports
  `sentence_transformers`.
- `_get_ollama_client()` lazily imports `ollama`.
- `metadata_generation.py` contains Ollama HTTP/helper code but is unused by
  API routes.
- `requirements-dev.txt` includes `ollama==0.6.3` and depends on runtime
  requirements.
- No eager top-level `torch` import was found.

## (e) Frontend findings

- API paths match backend upload, status, delete, answer, and health routes.
- `api.js` already fails clearly in production when `VITE_API_URL` is missing
  or loopback.
- Network failures mention the failure generally but do not include the
  configured API URL.
- Answer 404 handling currently surfaces the backend detail but does not mark
  documents missing or remove stale active selections.
- `App.jsx` stores the complete workspace and conversations in localStorage.
  It performs JSON parsing but may contain stale documents after backend
  restarts/deletions.
- Frontend tests pass, but there is no focused missing-document 404 test.
- `Frontend/.env.production` contains a real public Render URL, which is not
  secret.
- Frontend upload/status code has been partially hardened in the working tree,
  including polling and document state handling.

## (f) Verified-facts re-check

- Config currently reads environment variables at import time and creates
  `Backend/Data/raw` and `Backend/Data/processed`; this fact is correct.
- `EMBEDDING_PROVIDER` currently defaults to `local`, and
  `gemini-embedding-001` with `output_dimensionality=VECTOR_SIZE` is implemented.
- The duplicate processed-directory definitions are present in
  `pdf_extractor.py` and `text_chunker.py`.
- `vector_store.py` additionally duplicates the directory definition, so the
  first fact can be expanded: it is not the only downstream inconsistency.
- The file-based pipeline and `FileNotFoundError` behavior are correct.
- Questions routes return exception text directly for `FileNotFoundError` and
  `LookupError`; a fixed safe message is still needed.
- Metadata-generation helpers are unused by routes and are dead API code.
- Ollama is only lazily used when configured; metadata-generation is separate.
- Baseline test count differs from the supplied verified-failure list; the
  current tree has one backend failure, not three.
- Pinecone and retrieval filter by the embedding fingerprint.
- Frontend API URL and localStorage facts are correct.
- Frontend scripts and environment files are correct.
- `render.yaml` is absent.
- `.python-version` contains `3.12`.

## Verification results

- `cd Backend; .\.venv\Scripts\python.exe -m pytest -q` - FAIL
  (`65 passed, 1 failed`; one test attribute/setup failure).
- `cd Frontend; npm ci` - PASS (`257 packages installed`).
- `cd Frontend; npm test` - PASS (`1 file, 7 tests passed`).
- `cd Frontend; npm run build` - PASS (`Vite production build completed`).
