# Deployed connection-path audit

## a. Production API base URL

`Frontend/src/services/api.js` selects `VITE_API_URL`, then the legacy
`VITE_API_BASE_URL`, trimming whitespace and trailing slashes. If neither is
set, production builds deliberately report a configuration error. Vite embeds
these variables at build time.

`Frontend/.env.production` currently contains
`VITE_API_URL=https://omnibrain-backend-1tdn.onrender.com`. A Render service
environment variable named `VITE_API_URL` overrides this file during the
static-site build. I did not contact Render, so I cannot verify whether the
file's current URL is still the deployed backend.

## b. URL shape and target

The checked-in production value uses HTTPS and has no trailing slash or path.
Its hostname appears to be a backend URL, but its current ownership and
availability are unverified. It is left unchanged pending manual confirmation.

## c. Backend CORS behavior

`Backend/app/main.py` reads comma-separated `CORS_ORIGINS`. The current parser
strips whitespace and trailing slashes but does not remove URL paths. The
middleware allows configured origins and also uses a regex that permits any
localhost/127.0.0.1 port, regardless of environment. Configured origins are
otherwise exact matches.

Starlette's CORS middleware handles OPTIONS preflights. The current allowlist
contains GET, POST, DELETE, and OPTIONS. `Content-Type` is allowed explicitly;
`Accept` is a CORS-safelisted request header. The API does not send an
Authorization header. CORS has `allow_credentials=False`; frontend fetch calls
do not set credentials, so cross-origin requests do not send cookies.

The deployed frontend must be configured as the exact HTTPS origin only, for
example `https://your-frontend.onrender.com` (no trailing slash or path).

## d. Sleeping backend behavior

The initial `/health` request is attempted once with a 10-second timeout. A
network failure sets the badge to Offline immediately. Health is checked again
every 30 seconds, when the page becomes visible, when the browser comes online,
or when the badge is clicked. It is not permanently offline, but it does not
retry every five seconds or explain that a free service may be waking.

## e. Cookies and credentials

`api.js` does not set `credentials`; fetch therefore uses its default
same-origin credential mode. Render's frontend and backend are cross-origin,
so cookies are not sent in the deployed setup. Explicitly using
`credentials: "omit"` would make this guarantee clear for all configurations.

## f. Localhost references

`Frontend/src/services/api.js` uses its localhost fallback only in
non-production builds. `Backend/app/main.py` currently includes local origins
in its defaults and regex in every environment; these are not gated to
development yet. `Backend/app/config.py` also defaults the optional local
Ollama provider URL to `http://localhost:11434`. No other localhost/127.0.0.1
API base URL appears in the application source. The word "localhost" in the
frontend settings text is explanatory text, not a URL.

## g. Render and deployment documentation

`render.yaml` defines the backend as a web service rooted at `Backend`, starts
Uvicorn on `$PORT`, and uses `/health` as its health check. `CORS_ORIGINS` is
`sync: false`. The frontend is a static site rooted at `Frontend`, publishes
`dist`, and has `VITE_API_URL` set to `sync: false`. The backend does not yet
set `ENVIRONMENT=production`. `DEPLOY.md` says to redeploy the frontend after
changing `VITE_API_URL`.

## h. Static-site rewrite

The frontend does not use a client-side router or URL path routes. Its views
are selected in React state, so a `/* -> /index.html` rewrite is not needed.

## Verified known facts

- The local development API URL is `http://127.0.0.1:8000`.
- `/health`, `/ready`, upload, answer, and document-delete routes exist.
- The backend currently allows the expected POST preflight method, but the
  configured headers and origin normalization need hardening.
- The Render Blueprint uses a free backend plan. Free instances can sleep and
  their local disk is ephemeral.
- The health badge is derived from the `/health` response; after one failed
  check it shows Offline and relies on later periodic checks.

## Phase 1 conclusion

The source identifies concrete connection-path fixes; no value is needed to
continue. The production URL remains unverified and unchanged. Confirm it in
the Render dashboard before redeploying; do not substitute a guessed URL.
