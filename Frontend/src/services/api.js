const configuredApiBaseUrl = import.meta.env.VITE_API_BASE_URL?.trim() || "";
const isProductionBuild = import.meta.env.MODE === "production";
const isLoopbackUrl = /^https?:\/\/(?:localhost|127\.0\.0\.1|\[::1\])(?::\d+)?(?:\/|$)/i.test(
  configuredApiBaseUrl,
);
const apiConfigurationError =
  isProductionBuild && (!configuredApiBaseUrl || isLoopbackUrl)
    ? "Set VITE_API_BASE_URL to the deployed backend URL. Production builds cannot use a localhost API."
    : null;
const API_BASE_URL = (
  configuredApiBaseUrl ||
  (isProductionBuild ? "" : "http://127.0.0.1:8000")
).replace(/\/+$/, "");

const BACKEND_UNAVAILABLE_EVENT = "omnibrain:backend-unavailable";
const BACKEND_REACHABLE_EVENT = "omnibrain:backend-reachable";

const DEFAULT_REQUEST_TIMEOUT = 15_000;
const UPLOAD_TIMEOUT = 120_000;
const ANSWER_TIMEOUT = 660_000;
const MAX_UPLOAD_SIZE_BYTES = 20 * 1024 * 1024;
const MAX_UPLOAD_FILES = 5;

// --------------------------------------------------
// 1. Backend connection status events
// --------------------------------------------------

function signalBackendUnavailable() {
  window.dispatchEvent(new CustomEvent(BACKEND_UNAVAILABLE_EVENT));
}

function signalBackendReachable() {
  window.dispatchEvent(new CustomEvent(BACKEND_REACHABLE_EVENT));
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

  let timedOut = false;

  const timeoutId = window.setTimeout(() => {
    timedOut = true;
    controller.abort();
  }, timeoutMs);

  try {
    const response = await fetch(`${API_BASE_URL}${path}`, {
      ...options,
      signal: controller.signal,
    });

    const body = await parseResponse(response);

    if (!response.ok) {
      if (response.status >= 500) {
        signalBackendUnavailable();
      }

      throw new ApiError(
        getErrorMessage(body, response.status),
        response.status,
        "HTTP_ERROR",
      );
    }

    signalBackendReachable();

    return body;
  } catch (error) {
    if (error instanceof ApiError) {
      throw error;
    }

    if (
      timedOut ||
      (error instanceof DOMException && error.name === "AbortError")
    ) {
      throw new ApiError(
        interruptionMessages.timeout ||
          "The request timed out. Please try again.",
        null,
        "TIMEOUT",
      );
    }

    if (error instanceof TypeError) {
      signalBackendUnavailable();

      throw new ApiError(
        interruptionMessages.network ||
          `Could not reach OmniBrain at ${API_BASE_URL}. ` +
            "Check that the backend is running and its CORS settings " +
            "allow this frontend.",
        null,
        "NETWORK_ERROR",
      );
    }

    throw new ApiError(
      error instanceof Error ? error.message : "The request failed.",
      null,
      "REQUEST_ERROR",
    );
  } finally {
    window.clearTimeout(timeoutId);
  }
}

// --------------------------------------------------
// 5. Public API functions
// --------------------------------------------------

export const api = {
  baseUrl: API_BASE_URL || "Not configured",

  // Check whether the FastAPI backend is responding.
  checkHealth() {
    return request("/health", {}, 10_000);
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
          "The upload timed out. The backend may already have received " +
          "or saved the PDFs, but the result could not be confirmed. " +
          "Check the backend before retrying.",
        network:
          "The upload connection failed. The backend may already have " +
          "received the PDFs, but the result could not be confirmed. " +
          "Check the backend before retrying.",
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

  // Ask a question about one or more PDFs.
  // The backend answers one PDF per request:
  //   POST /api/documents/{document_id}/answer
  // With several PDFs, each one is asked separately and the answers
  // are combined under the PDF names.
  async answerQuestion(documentIds, question) {
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

    const askOne = (id) =>
      request(
        `/api/documents/${encodeURIComponent(id)}/answer`,
        {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ question: text }),
        },
        ANSWER_TIMEOUT,
        {
          timeout:
            "Answer generation took too long. Check the backend logs " +
            "before submitting the question again.",
          network:
            "The connection to OmniBrain was interrupted. Check the " +
            "backend status before retrying.",
        },
      );

    // One PDF: use the backend response as it is.
    if (ids.length === 1) {
      return askOne(ids[0]);
    }

    // Several PDFs: ask each one, then combine the results.
    const settled = await Promise.allSettled(ids.map(askOne));

    const parts = [];
    const sources = [];
    const failures = [];

    settled.forEach((outcome, index) => {
      if (
        outcome.status === "fulfilled" &&
        typeof outcome.value?.answer === "string"
      ) {
        const value = outcome.value;
        const label =
          value.sources?.find((source) => source?.filename)?.filename ||
          `PDF ${index + 1}`;

        parts.push(`**${label}**\n\n${value.answer}`);

        if (Array.isArray(value.sources)) {
          sources.push(...value.sources);
        }
      } else {
        failures.push(
          outcome.status === "rejected" && outcome.reason instanceof Error
            ? outcome.reason.message
            : "unexpected response",
        );
      }
    });

    if (parts.length === 0) {
      throw new ApiError(
        failures[0] || "The question could not be answered.",
      );
    }

    if (failures.length > 0) {
      parts.push(
        `_${failures.length} PDF${failures.length === 1 ? "" : "s"} could not be searched: ${failures[0]}_`,
      );
    }

    return { answer: parts.join("\n\n---\n\n"), sources };
  },
};