# Deploy OmniBrain to Render

## Connecting frontend and backend

1. Deploy the backend service first.
2. Copy the public backend URL from the Render Dashboard.
3. Set `VITE_API_URL` on the static site to that exact backend URL.
4. Deploy or redeploy the static site so Vite builds the URL into the bundle.
5. Copy the frontend static-site URL from the Render Dashboard.
6. Set `CORS_ORIGINS` on the backend to that exact frontend origin:
   `https://your-frontend.onrender.com` with no trailing slash or path.
7. Redeploy or restart the backend after changing `CORS_ORIGINS`.
8. Run:
   `python scripts/check_deployment.py --backend https://your-backend.onrender.com --frontend https://your-frontend.onrender.com`
9. Open the frontend and wait up to 60 seconds for the status badge to show
   `Online`.

### Common failures

| Failure | Fix |
|---|---|
| CORS error in the browser console | Set `CORS_ORIGINS` to the exact frontend HTTPS origin, with no trailing slash, and redeploy/restart the backend. |
| "Failed to fetch" or stale API URL | Set `VITE_API_URL` on the static site and redeploy it; Vite values are baked in at build time. |
| Mixed-content warning | Use an HTTPS backend URL. An HTTP backend cannot be called from an HTTPS frontend. |
| Backend shows offline at first | Wait up to 60 seconds while the Render free-plan backend wakes. The deployment check waits up to 90 seconds. |
| Wrong Gemini model or quota | Check `/ready`; set a model available to the API key and configure a valid `GEMINI_API_KEY`. |

## Render Dashboard

1. Push the `render.yaml` Blueprint to the repository.
2. Open the Render Dashboard and select **New > Blueprint**.
3. Choose the OmniBrain repository and approve the Blueprint.
4. In the deployed `omnibrain-backend` service, set `PINECONE_API_KEY`,
   `GEMINI_API_KEY`, and `CORS_ORIGINS` in Environment settings.
5. In the deployed `omnibrain-frontend` static site, set `VITE_API_URL` to the
   real backend URL, for example `https://omnibrain-backend.onrender.com`.
6. Trigger a frontend deploy after setting `VITE_API_URL`.
7. Open `/health`, then `/ready`, on the backend URL.

## Environment variables

| Variable | Source |
|---|---|
| `PINECONE_API_KEY` | Pinecone Dashboard: API Keys |
| `PINECONE_INDEX_NAME` | Blueprint default: `omnibrain-documents-gemini` |
| `PINECONE_CLOUD` | Pinecone region setting; Blueprint: `aws` |
| `PINECONE_REGION` | Pinecone region setting; Blueprint: `us-east-1` |
| `GEMINI_API_KEY` | Google AI Studio API key |
| `GEMINI_MODEL` | A model available to the Gemini API key; Phase 4 used `gemini-3.5-flash` |
| `GEMINI_FALLBACK_MODEL` | Optional; a model available to the same key, tried after three primary-model 503 responses |
| `EMBEDDING_PROVIDER` | Blueprint: `gemini` |
| `EMBEDDING_MODEL` | Blueprint: `gemini-embedding-001` |
| `VECTOR_SIZE` | Blueprint: `768` |
| `CORS_ORIGINS` | Exact frontend Render origin, with no trailing slash |
| `VITE_API_URL` | Render URL printed for `omnibrain-backend` |

## Finding the real URLs

Open **Dashboard > omnibrain-backend** and copy the public URL shown at the top
of the service. Open **Dashboard > omnibrain-frontend** for the static-site URL.
Set `CORS_ORIGINS` to the frontend URL and `VITE_API_URL` to the backend URL.

## Free-plan data behavior

The Render free plan is stateless: uploaded PDFs and processed local JSON may
disappear on restart or deploy. The free service also sleeps after inactivity.
Users must re-upload documents after a restart.

For persistent local PDFs and processed files, use a paid Render plan and mount
a persistent disk at `/opt/render/project/src/Backend/Data`, or set
`OMNIBRAIN_DATA_DIR` to that mount path. A persistent disk does not make old
BGE/Gemini model vectors compatible automatically; re-indexing uses the
configured embedding fingerprint.
