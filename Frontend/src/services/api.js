const configuredApiBaseUrlRaw =
  import.meta.env.VITE_API_URL?.trim() ||
  import.meta.env.VITE_API_BASE_URL?.trim() ||
  "";
const configuredApiBaseUrl = configuredApiBaseUrlRaw.replace(/\/+$/, "");
const isProductionBuild = import.meta.env.MODE === "production";
const isLoopbackUrl = /^https?:\/\/(?:localhost|127\.0\.0\.1|\[::1\])(?::\d+)?(?:\/|$)/i.test(
  configuredApiBaseUrl,
);
function getApiConfigurationError() {
  if (configuredApiBaseUrl && !/^https?:\/\//i.test(configuredApiBaseUrl)) {
    return "VITE_API_URL must start with https://";
  }
  if (isProductionBuild && (!configuredApiBaseUrl || isLoopbackUrl)) {
    return "Set VITE_API_URL to the deployed backend URL. Production builds cannot use a localhost API.";
  }
  if (!configuredApiBaseUrl) return null;

  let url;
  try {
    url = new URL(configuredApiBaseUrl);
  } catch {
    return "VITE_API_URL must be an absolute HTTP or HTTPS backend URL.";
  }
  if (
    !["http:", "https:"].includes(url.protocol) ||
    url.username ||
    url.password ||
    url.search ||
    url.hash
  ) {
    return "VITE_API_URL must be an HTTP or HTTPS backend URL without credentials, query, or fragment.";
  }
  if (
    isProductionBuild &&
    url.protocol !== "https:" &&
    !isLoopbackUrl
  ) {
    return "VITE_API_URL must use HTTPS in production to avoid browser mixed-content blocking.";
  }
  if (
    typeof window !== "undefined" &&
    window.location.protocol === "https:" &&
    url.protocol === "http:" &&
    !isLoopbackUrl
  ) {
    return "This secure page cannot connect to an HTTP backend. Set VITE_API_URL to an HTTPS URL.";
  }
  return null;
}
const apiConfigurationError = getApiConfigurationError();
const API_BASE_URL = (
  configuredApiBaseUrl ||
  (isProductionBuild ? "" : "http://127.0.0.1:8000")
).replace(/\/+$/, "");

const BACKEND_UNAVAILABLE_EVENT = "omnibrain:backend-unavailable";
const BACKEND_REACHABLE_EVENT = "omnibrain:backend-reachable";

const DEFAULT_REQUEST_TIMEOUT = 15_000;
const UPLOAD_TIMEOUT = 90_000;
const ANSWER_TIMEOUT = 90_000;
const MAX_UPLOAD_SIZE_BYTES = 20 * 1024 * 1024;
const MAX_UPLOAD_FILES = 5;

// --------------------------------------------------
// 1. Backend connection status events
// --------------------------------------------------

function signalBackendUnavailable() {
  window.dispatchEvent(new CustomEvent(BACKEND_UNAVAILABLE_EVENT));
}

function signalBackendReachable(status = "online") {
  window.dispatchEvent(
    new CustomEvent(BACKEND_REACHABLE_EVENT, { detail: { status } }),
  );
}

// --------------------------------------------------
// 2. API error class
// --------------------------------------------------

export class ApiError extends Error {
  constructor(message, status = null, code = null) {
    super(message);
    this.name = "ApiError";
    this.status = status;
    this.code = code;
  }
}

// --------------------------------------------------
// 3. Response parsing and error messages
// --------------------------------------------------

function getErrorMessage(body, status) {
  if (typeof body === "string" && body.trim()) {
    return body;
  }

  if (body && typeof body === "object") {
    const detail = body.detail ?? body.message;

    if (typeof detail === "string" && detail.trim()) {
      return detail;
    }

    if (Array.isArray(detail)) {
      return detail
        .map((item) => {
          if (typeof item === "string") {
            return item;
          }

          return item?.msg || JSON.stringify(item);
        })
        .join("; ");
    }
  }

  return `Request failed (${status}).`;
}

async function parseResponse(response) {
  const rawBody = await response.text();

  if (!rawBody.trim()) {
    return null;
  }

  try {
    return JSON.parse(rawBody);
  } catch {
    return rawBody;
  }
}

// --------------------------------------------------
// 4. Common HTTP request handler
// --------------------------------------------------

async function request(
  path,
  options = {},
  timeoutMs = DEFAULT_REQUEST_TIMEOUT,
  interruptionMessages = {},
) {
  if (apiConfigurationError) {
    throw new ApiError(apiConfigurationError, null, "API_CONFIG_ERROR");
  }

  const controller = new AbortController();
  const externalSignal = options.signal;
  const abortFromCaller = () => controller.abort();
  if (externalSignal?.aborted) {
    controller.abort();
  } else {
    externalSignal?.addEventListener("abort", abortFromCaller, { once: true });
  }

  let timedOut = false;

  const timeoutId = window.setTimeout(() => {
    timedOut = true;
    controller.abort();
  }, timeoutMs);

  try {
    const response = await fetch(`${API_BASE_URL}${path}`, {
      ...options,
      credentials: "omit",
      signal: controller.signal,
    });

    signalBackendReachable(
      response.status === 429 || response.status >= 500
        ? "degraded"
        : "online",
    );
    const body = await parseResponse(response);

    if (!response.ok) {
      if (response.status >= 500) {
        console.error(
          `[OmniBrain API] ${path} failed with HTTP ${response.status}.`,
        );
      }

      const hasBackendDetail =
        body &&
        typeof body === "object" &&
        (typeof body.detail === "string" ||
          typeof body.message === "string" ||
          Array.isArray(body.detail));
      const message = hasBackendDetail
        ? getErrorMessage(body, response.status)
        : response.status === 429
          ? "The AI service is busy. Please wait a moment before trying again."
          : response.status === 503
            ? "The service is temporarily unavailable. Please try again shortly."
            : response.status === 504
              ? "This request took too long to complete. Please try again shortly."
              : response.status >= 500
                ? "OmniBrain could not complete this request. Please try again shortly."
                : getErrorMessage(body, response.status);

      throw new ApiError(
        message,
        response.status,
        "HTTP_ERROR",
      );
    }

    return body;
  } catch (error) {
    if (error instanceof ApiError) {
      throw error;
    }

    if (externalSignal?.aborted) {
      throw new ApiError("Request cancelled.", null, "CANCELLED");
    }

    if (
      timedOut ||
      (error instanceof DOMException && error.name === "AbortError")
    ) {
      console.error(`[OmniBrain API] ${path} timed out.`, error);
      signalBackendReachable(path === "/health" ? "offline" : "degraded");
      throw new ApiError(
        interruptionMessages.timeout ||
          "This request is taking longer than expected. Try again shortly.",
        null,
        "TIMEOUT",
      );
    }

    if (error instanceof TypeError) {
      console.error(`[OmniBrain API] Could not reach ${API_BASE_URL}.`, error);
      signalBackendUnavailable();

      throw new ApiError(
        interruptionMessages.network ||
          `OmniBrain is temporarily unreachable at ${API_BASE_URL || "the configured backend URL"}. The connection will be rechecked automatically.`,
        null,
        "NETWORK_ERROR",
      );
    }

    console.error(`[OmniBrain API] ${path} request failed.`, error);
    throw new ApiError(
      "OmniBrain could not complete this request. Please try again shortly.",
      null,
      "REQUEST_ERROR",
    );
  } finally {
    window.clearTimeout(timeoutId);
    externalSignal?.removeEventListener("abort", abortFromCaller);
  }
}

// --------------------------------------------------
// 5. Public API functions
// --------------------------------------------------

export const api = {
  baseUrl: API_BASE_URL || "Not configured",

  // Check whether the FastAPI backend is responding.
  checkHealth(timeoutMs = 10_000) {
    return request("/health", {}, timeoutMs);
  },

  getDocumentStatus(documentId) {
    return request(
      `/api/documents/${encodeURIComponent(documentId)}/status`,
      {},
      DEFAULT_REQUEST_TIMEOUT,
    );
  },

  // Upload 1 to 5 PDFs in one request.
  uploadDocuments(files, conversationId = null) {
    // Accept an array, a FileList, or a single File.
    const fileList =
      files instanceof File ? [files] : Array.from(files || []);

    if (fileList.length === 0) {
      return Promise.reject(
        new ApiError(
          "Please select at least one PDF file.",
          null,
          "INVALID_FILE",
        ),
      );
    }

    if (fileList.length > MAX_UPLOAD_FILES) {
      return Promise.reject(
        new ApiError(
          `You can upload at most ${MAX_UPLOAD_FILES} PDFs at a time.`,
          null,
          "TOO_MANY_FILES",
        ),
      );
    }

    for (const file of fileList) {
      if (typeof File === "undefined" || !(file instanceof File)) {
        return Promise.reject(
          new ApiError(
            "Please select a valid PDF file.",
            null,
            "INVALID_FILE",
          ),
        );
      }

      const isPdf =
        file.type === "application/pdf" ||
        file.name.toLowerCase().endsWith(".pdf");

      if (!isPdf) {
        return Promise.reject(
          new ApiError(
            `${file.name} is not a PDF. Only PDF files are supported.`,
            null,
            "INVALID_FILE_TYPE",
          ),
        );
      }

      if (file.size === 0) {
        return Promise.reject(
          new ApiError(`${file.name} is empty.`, null, "EMPTY_FILE"),
        );
      }

      if (file.size > MAX_UPLOAD_SIZE_BYTES) {
        return Promise.reject(
          new ApiError(
            `${file.name} exceeds the 20 MB upload limit.`,
            null,
            "FILE_TOO_LARGE",
          ),
        );
      }
    }

    const formData = new FormData();
    // Same field name for every file; must match the backend parameter.
    fileList.forEach((file) => formData.append("files", file));

    if (conversationId) {
      formData.append("conversation_id", conversationId);
    }

    return request(
      "/api/documents/upload",
      {
        method: "POST",
        body: formData,
      },
      UPLOAD_TIMEOUT,
      {
        timeout:
          "Upload confirmation is taking longer than expected. Check the document status before retrying.",
        network:
          "Could not connect to the backend. The server may be waking up. Wait a minute and try again.",
      },
    );
  },

  // Delete one uploaded PDF: DELETE /api/documents/{document_id}
  deleteDocument(documentId) {
    if (typeof documentId !== "string" || !documentId.trim()) {
      return Promise.reject(
        new ApiError("Missing document ID.", null, "MISSING_DOCUMENT_ID"),
      );
    }

    return request(
      `/api/documents/${encodeURIComponent(documentId)}`,
      { method: "DELETE" },
      DEFAULT_REQUEST_TIMEOUT,
    );
  },

  // Ask one grounded question across the selected PDFs.
  async answerQuestion(documentIds, question, signal) {
    const ids = Array.isArray(documentIds) ? documentIds : [documentIds];

    if (
      ids.length === 0 ||
      ids.some((id) => typeof id !== "string" || !id.trim())
    ) {
      throw new ApiError(
        "Upload or select a document before asking a question.",
        null,
        "MISSING_DOCUMENT_ID",
      );
    }

    if (typeof question !== "string" || !question.trim()) {
      throw new ApiError("Please enter a question.", null, "EMPTY_QUESTION");
    }

    const text = question.trim();

    return request(
      "/api/documents/answer",
      {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ question: text, document_ids: ids }),
        signal,
      },
      ANSWER_TIMEOUT,
      {
        timeout:
          "This answer is taking longer than expected. Please try again shortly.",
        network:
          "Could not connect to the backend. The server may be waking up. Wait a minute and try again.",
      },
    );
  },
};