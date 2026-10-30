# OmniBrain

OmniBrain is a web app for asking questions about information in PDF documents. Upload PDFs, ask a question in the chat, and receive an answer grounded in the document text, with source passages and page references where available.

The app accepts up to **5 PDFs in one upload**. This limit applies to each upload request, not to the total number of PDFs in a conversation.

## Key features

- Upload and manage PDF documents, with a 20 MB limit per file.
- Extract and index PDF text when a document is first queried.
- Search document content using vector similarity.
- Ask questions about one PDF or select several PDFs. For several PDFs, the frontend requests an answer from each separately and combines the results.
- View answers with supporting source text and page references.
- Keep chat conversations and theme preferences in the browser's local storage.

## Architecture

Uploading a PDF saves and validates it on the backend. The more expensive indexing steps happen when the PDF is first queried: OmniBrain extracts text, splits it into chunks, embeds the chunks, and stores them in Pinecone. For each question, it embeds the query, retrieves relevant chunks, and sends that context to Gemini to generate an answer.

1. The React frontend sends PDF files to the FastAPI backend.
2. The backend validates and saves each PDF on the local filesystem.
3. On the first question about a PDF, PyMuPDF extracts its embedded text and page numbers.
4. The backend cleans the text and splits it into overlapping chunks.
5. Ollama creates chunk embeddings, which are stored in Pinecone.
6. For a question, Ollama creates a query embedding and Pinecone returns matching chunks.
7. Google Gemini generates an answer from the retrieved text.
8. The backend returns the answer and source details to the frontend.

```mermaid
flowchart TD
    U[User] --> FE[React and Vite frontend]
    FE -->|PDF upload| API[FastAPI backend]
    API -->|Validate and save| PDF[Local PDF storage]
    FE -->|Question for a selected PDF| API
    API -->|First question triggers indexing| EX[PyMuPDF text extraction]
    PDF --> EX
    EX --> CH[Text cleaning and chunking]
    CH --> EMB[Ollama embedding model]
    EMB -->|Document vectors| DB[(Pinecone)]
    API -->|Question text| QE[Ollama query embedding]
    QE -->|Similarity search| DB
    DB -->|Relevant text chunks| GEN[Google Gemini]
    GEN -->|Answer and sources| API
    API --> FE
```

Document metadata generation tries Gemini first, then the local Ollama model `llama3.2:1b`, and finally a deterministic fallback. The `llama3.2:1b` model is **not** the question-answering model.

## Technology stack

| Area | Technologies in the current implementation |
|---|---|
| Frontend | React 18, JavaScript, Vite 6, CSS |
| Frontend libraries | Lucide React, React Markdown, KaTeX, Remark GFM, Remark Math, Rehype KaTeX |
| Backend | Python, FastAPI, Uvicorn, Pydantic |
| AI and LLM | Google Gemini through `google-genai` for answers; Ollama for embeddings and an optional local metadata fallback |
| Embedding model | `nomic-embed-text-v2-moe` (Ollama) |
| Document processing | PyMuPDF for PDF text extraction; LangChain text splitters and `tiktoken` for chunking and token counting |
| Vector database | Pinecone, using a cosine-distance index |
| Configuration | `python-dotenv` and environment variables |

The answer model is selected with `GEMINI_MODEL` (default: `gemini-3.8-flash`). The intended `llama3.2:1b` model is only an optional metadata-generation fallback; it is not the question-answering model.

## Local development URLs

| Component | URL | Purpose |
|---|---|---|
| Frontend | `http://localhost:5173` | OmniBrain user interface (Vite's default port) |
| Backend | `http://127.0.0.1:8000` | FastAPI backend |
| API documentation | `http://127.0.0.1:8000/docs` | Interactive Swagger UI |
| Alternative API documentation | `http://127.0.0.1:8000/redoc` | ReDoc API reference |
| Health check | `http://127.0.0.1:8000/health` | Checks whether the API process is responding |

These are local development addresses, not public deployment URLs. If port 5173 is occupied, Vite may start on a different port and print that address in the terminal.

## Important services and links

| Service | URL | Purpose |
|---|---|---|
| OmniBrain repository | [github.com/SamadhanEkad/OmniBrain](https://github.com/SamadhanEkad/OmniBrain) | Source repository, verified from the Git remote |
| Ollama | [ollama.com](https://ollama.com/) | Runs the local embedding model and optional metadata fallback |
| Ollama API documentation | [docs.ollama.com](https://docs.ollama.com/) | Ollama installation and API reference |
| Pinecone | [pinecone.io](https://www.pinecone.io/) | Hosted vector database used to store and search document embeddings |
| Pinecone documentation | [docs.pinecone.io](https://docs.pinecone.io/) | Pinecone index and API reference |
| Google Gemini API | [ai.google.dev/gemini-api/docs](https://ai.google.dev/gemini-api/docs) | Cloud API used to generate answers |

Ollama's default local API address is `http://localhost:11434`. Pinecone is a cloud service; the configured index defaults to `omnibrain-documents` in AWS `us-east-1`.

## Project structure

```text
OmniBrain/
├── .python-version            # Python runtime selected for Render
├── Backend/
│   ├── app/
│   │   ├── api/                 # PDF upload, deletion, and question routes
│   │   ├── services/            # PDF processing, embeddings, retrieval, and answers
│   │   ├── config.py            # Backend settings and data paths
│   │   └── main.py              # FastAPI application
│   ├── Data/
│   │   ├── raw/                 # Uploaded PDFs and document information
│   │   └── processed/           # Extracted text, chunks, and embeddings
│   ├── tests/                   # Backend tests
│   ├── requirements.txt         # Runtime Python dependencies
│   └── requirements-dev.txt     # Backend test dependencies
├── Frontend/
│   ├── src/
│   │   ├── components/          # Chat, upload, document, and history UI
│   │   ├── services/            # Backend API client and request helpers
│   │   ├── App.jsx              # Main application
│   │   └── index.css            # Application styles
│   ├── package.json             # Frontend dependencies and scripts
│   ├── package-lock.json        # Locked frontend dependency versions
│   └── vite.config.js           # Vite configuration
└── .gitignore
```

## Prerequisites

- Python 3.12 (the repository's tested and Render-selected runtime; the source uses Python 3.10+ syntax).
- Node.js 18.x, 20.x, or 22+ and npm (the frontend uses Vite 6).
- [Ollama](https://ollama.com/) running locally, with the embedding model `nomic-embed-text-v2-moe` available.
- A Pinecone account and API key.
- A Google Gemini API key for answer generation.

The local `llama3.2:1b` model is optional: it is used only if Gemini-based metadata generation is unavailable. OmniBrain can also use deterministic metadata as a final fallback.

## Installation and setup

### 1. Clone the repository

```powershell
git clone https://github.com/SamadhanEkad/OmniBrain.git
cd OmniBrain
```

### 2. Set up the backend

```powershell
cd Backend
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

If PowerShell prevents virtual-environment activation, use the environment's Python executable directly: `.\.venv\Scripts\python.exe`.

To run backend tests, install `requirements-dev.txt` instead of `requirements.txt`:

```powershell
python -m pip install -r requirements-dev.txt
python -m pytest
```

### 3. Configure AI and database services

Start Ollama using the installed Ollama application or its normal service command, then download the embedding model:

```powershell
ollama pull nomic-embed-text-v2-moe
```

To enable the optional local metadata fallback as well:

```powershell
ollama pull llama3.2:1b
```

Set the Pinecone and Gemini credentials in the same PowerShell session that will run the backend. Replace the placeholders locally; do not commit real credentials:

```powershell
$env:PINECONE_API_KEY = "<your-pinecone-api-key>"
$env:GEMINI_API_KEY = "<your-gemini-api-key>"
$env:GEMINI_MODEL = "<available-gemini-model>"
```

The code creates the Pinecone index if it does not exist. Its defaults are index `omnibrain-documents`, AWS cloud, region `us-east-1`, and 768 vector dimensions. The current implementation uses Pinecone's default namespace; no configurable namespace is active. The embedding model, vector size, and Pinecone index dimensions must agree.

The backend loads configuration from environment variables and, when present, `Backend/.env` (via `python-dotenv`). `Backend/.env.example` contains local-development placeholders; do not overwrite an existing `.env` or commit real credentials.

### 4. Start the backend

From the `Backend` directory, with the virtual environment active and required environment variables set:

```powershell
python -m uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
```

### 5. Start the frontend

Open another terminal:

```powershell
cd OmniBrain\Frontend
npm ci
npm run dev
```

The frontend uses `VITE_API_URL` to select the backend address; it defaults to `http://127.0.0.1:8000`. Set that variable before starting Vite only if the backend is at a different address. `VITE_API_BASE_URL` remains supported as a legacy alias.

### 6. Open OmniBrain

Open the frontend URL printed by Vite (normally `http://localhost:5173`). Attach one or more PDFs, upload them, and ask a question. The first question about a PDF may take longer because the backend extracts, chunks, embeds, and indexes it before answering.

## Main features

- **PDF uploads:** Upload up to five PDFs in one request. Each file must be a readable, unencrypted PDF no larger than 20 MB.
- **Text extraction and indexing:** PyMuPDF extracts page text; the backend cleans and chunks it, then creates embeddings and stores them in Pinecone when first needed.
- **Document question answering:** Ask about a selected PDF. The backend retrieves matching passages and uses Gemini to generate the response.
- **Multiple selected PDFs:** The interface sends a separate question request for each selected PDF and combines successful answers under their file names. It is not a single joint retrieval request.
- **Sources and page references:** Responses include retrieved text and available document, chunk, page, and similarity-score information.
- **Chat history:** Conversations and theme preference are stored in browser local storage.
- **Document removal:** The API removes the uploaded PDF, its local processing files, and its Pinecone vectors.

## API endpoints

| Method | Endpoint | Purpose |
|---|---|---|
| `GET` | `/` | Returns basic API status and version information |
| `GET` | `/health` | Checks that the API process is responding; it does not check AI or database services |
| `POST` | `/api/documents/upload` | Accepts multipart form data with one or more files under the repeated `files` field; returns `uploaded` and `failed` lists |
| `DELETE` | `/api/documents/{document_id}` | Removes the local document files and its Pinecone vectors |
| `POST` | `/api/documents/{document_id}/answer` | Accepts JSON `{"question": "..."}` and returns an `answer` and `sources` list |

The question must contain 1–2000 characters. Each source may include `document_id`, `chunk_id`, `page_number`, `score`, and `text`. The interactive API reference is available at `/docs` while the backend is running.

## Configuration and environment variables

| Variable | Purpose | Default / requirement |
|---|---|---|
| `PINECONE_API_KEY` | Authenticates with Pinecone | Required for indexing and retrieval |
| `PINECONE_INDEX_NAME` | Pinecone index name | `omnibrain-documents` |
| `PINECONE_CLOUD` | Cloud used when creating the index | `aws` |
| `PINECONE_REGION` | Region used when creating the index | `us-east-1` |
| `GEMINI_API_KEY` | Authenticates Gemini answer generation | Required for generated answers |
| `GEMINI_MODEL` | Gemini model name | `gemini-3.8-flash`; configure a model available to your API key |
| `OLLAMA_URL` | Ollama API base address used for document/query embeddings and the metadata fallback | `http://localhost:11434` locally; set a reachable service URL on Render |
| `EMBEDDING_MODEL` | Ollama model used for both document and query embeddings | `nomic-embed-text-v2-moe` |
| `OLLAMA_LLM_MODEL` | Local Ollama model used as a metadata fallback | `llama3.2:1b` |
| `VECTOR_SIZE` | Expected Pinecone and embedding vector dimension | `768` |
| `MAX_FILES_PER_UPLOAD` | Maximum PDFs in one upload request | `5` |
| `MAX_UPLOAD_SIZE_BYTES` | Maximum size of each PDF | `20971520` bytes (20 MiB) |
| `CORS_ORIGINS` | Comma-separated allowed frontend origins | Localhost origins, including Vite's default port |
| `LOG_LEVEL` | Backend log level | `INFO` |
| `PORT` | HTTP port for the backend | Supplied by Render; do not hardcode it |
| `VITE_API_URL` | Backend base URL used by the frontend | Local development defaults to `http://127.0.0.1:8000`; required for production builds |

Backend variables are read from the process environment and `Backend/.env` when present. The frontend reads `VITE_API_URL` from its Vite environment, with `VITE_API_BASE_URL` retained as a legacy alias; `Frontend/.env.example` shows the local default. Production frontend code rejects a missing or loopback API URL. Never put credentials in source control or any `VITE_*` variable.

## Render deployment

Deploy the backend as a **Web Service** and the frontend as a separate **Static Site**. The project has no client-side router, so a static-site rewrite is not needed for the current single-page UI.

### Backend Web Service

| Render setting | Value |
|---|---|
| Root Directory | `Backend` |
| Build Command | `pip install -r requirements.txt` |
| Start Command | `uvicorn app.main:app --host 0.0.0.0 --port $PORT` |
| Health Check Path | `/health` |
| Python | `.python-version` selects the latest available Python 3.12 patch |

Set these variables in the Render dashboard; do not put credentials in `render.yaml` or source code:

| Variable | Requirement |
|---|---|
| `PINECONE_API_KEY` | Required; Pinecone index/vector storage |
| `GEMINI_API_KEY` | Required for generated answers |
| `OLLAMA_URL` | Required; URL of an Ollama API reachable from Render |
| `CORS_ORIGINS` | Required after frontend creation; the exact frontend origin, without a trailing slash |
| `GEMINI_MODEL` | Optional override; set to a model available to the configured Gemini API key |
| `PINECONE_INDEX_NAME` | Optional; defaults to `omnibrain-documents` |
| `PINECONE_CLOUD`, `PINECONE_REGION` | Optional; used if the index needs to be created |
| `EMBEDDING_MODEL`, `VECTOR_SIZE` | Keep at `nomic-embed-text-v2-moe` and `768` unless all stored vectors and the Pinecone index are deliberately migrated |
| `OLLAMA_LLM_MODEL` | Optional metadata fallback only; defaults to `llama3.2:1b` |

The default `OLLAMA_URL` (`localhost:11434`) is only for local development. On Render, localhost refers to the backend container, not a developer's computer. Provide an always-available Ollama service accessible from the backend and make sure it has `nomic-embed-text-v2-moe` available. Document metadata generation can fall back if Ollama is unavailable, but document indexing and retrieval cannot. No Ollama service or hosted embedding provider is provisioned by this repository.

The backend writes uploaded PDFs, document information, extracted text, chunks, and locally cached embeddings under `Backend/Data`. Render's ordinary filesystem is ephemeral. To preserve uploaded documents across restarts and deploys, use a paid Web Service with a persistent disk mounted at:

```text
/opt/render/project/src/Backend/Data
```

Only files under the mount are durable. Without it, PDFs and local processing data are lost on restart even though vectors already written to Pinecone may remain. Those local files are currently required to answer questions and manage documents. Keep the disk private and size it for the uploaded PDFs and generated JSON.

### Frontend Static Site

| Render setting | Value |
|---|---|
| Root Directory | `Frontend` |
| Build Command | `npm ci && npm run build` |
| Publish Directory | `dist` |
| Environment Variable | `VITE_API_URL=https://<your-backend-service>.onrender.com` |

Create the backend and frontend services first, then replace the placeholder with the actual backend URL and set `CORS_ORIGINS` on the backend to the actual frontend origin, for example `https://<your-frontend-site>.onrender.com`. These are placeholders, not deployed service URLs. Local Vite origins remain allowed for development; production CORS allows the deployed OmniBrain frontend and any additionally configured origins.

### Operational and privacy limitations

- The backend has no authentication or per-user data isolation. Anyone who obtains a document ID can use the public API to ask questions about or delete that document. Do not upload confidential PDFs to a public deployment without adding an appropriate access-control layer.
- Gemini and Pinecone credentials and usage are required for their respective services; their quotas and costs depend on the accounts you configure. Keep the keys in Render's backend environment variables only.
- Pinecone stores vectors and text metadata remotely; the PDF and processing JSON remain local to the backend disk. The API health check only confirms that FastAPI is responding, not that Ollama, Gemini, or Pinecone is available.
- The first question triggers synchronous extraction, chunking, and per-chunk embedding. A large document or a slow external Ollama service can exceed the hosting platform's request limits. External Ollama, Gemini, and Pinecone connectivity was not exercised during local validation.
- PDF support is embedded-text extraction only. OCR, image/chart understanding, and a SQL agent are not implemented.

## Limitations

- PDF extraction reads embedded text only. Scanned PDFs without selectable text are rejected; OCR and visual chart/table analysis are not implemented.
- Upload limits are five PDFs per request and 20 MiB per file. The limit is per request, not a total conversation cap.
- The first question about a PDF performs indexing and may take longer than subsequent questions.
- Retrieval and generated answers can be incomplete or incorrect; check cited source passages against the original PDF.
- Questions across several selected PDFs are handled as separate per-document requests and combined in the frontend.
- Deleting a PDF removes its local files and its Pinecone vectors.

## Troubleshooting

- **Backend does not start:** Confirm Python 3.12 is selected, activate the virtual environment, and install `Backend/requirements.txt`.
- **Frontend dependencies are missing:** Run `npm ci` from `Frontend/`, then start it with `npm run dev`.
- **Frontend cannot reach the backend:** Start the backend on port 8000, check that `VITE_API_URL` points to it, and restart Vite after changing its environment. The backend allows local development origins by default.
- **Question answering fails:** Check `GEMINI_API_KEY` and `GEMINI_MODEL`, and verify that the configured Gemini model is available to the API key.
- **Embedding or indexing fails:** Make sure the configured Ollama service is reachable and `nomic-embed-text-v2-moe` is installed. Check that `PINECONE_API_KEY` is valid and the index dimension matches `VECTOR_SIZE` (768 by default).
- **PDF upload or extraction fails:** Check that the file is a valid, unencrypted PDF under 20 MiB and contains selectable text. Scanned PDFs need OCR, which is not currently provided.

## Future scope

Potential future improvements, not currently implemented, include OCR for scanned PDFs, visual analysis of charts and tables, joint retrieval across multiple PDFs, and authentication with per-user document isolation.

## Author and license

The GitHub repository is owned by [SamadhanEkad](https://github.com/SamadhanEkad). No license file is present in the repository, so the project's license terms are not specified here.

## Application URLs

### Frontend

- **URL:** [http://localhost:5173](http://localhost:5173/)
- **Purpose:** Open the OmniBrain web application to upload PDF documents and ask questions.

### Backend

- **URL:** [http://127.0.0.1:8000](http://127.0.0.1:8000/)
- **Purpose:** FastAPI backend that handles document uploads, document processing, information retrieval, and answer generation.

### Backend API Documentation

- **Swagger UI:** [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)
- **ReDoc:** [http://127.0.0.1:8000/redoc](http://127.0.0.1:8000/redoc)
- **Purpose:** Explore and test the available backend API endpoints.

### Important Notes

1. These URLs are for local development.
2. Start the backend and frontend servers before opening the application.
3. The frontend uses Vite's default port `5173`, and the backend is configured for port `8000`. Check the server output if the configured ports differ.
4. This repository does not specify deployed service URLs.
