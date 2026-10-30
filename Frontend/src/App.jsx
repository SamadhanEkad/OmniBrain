import { useCallback, useEffect, useRef, useState } from "react";
import { ArrowUpRight, FileText, Info, Trash2 } from "lucide-react";
import ConversationToolbar, {
  copyText,
  getConversationText,
} from "./components/ConversationToolbar.jsx";
import FileUploader from "./components/FileUploader.jsx";
import HistoryPanel from "./components/HistoryPanel.jsx";
import MessageBubble from "./components/MessageBubble.jsx";
import Sidebar from "./components/Sidebar.jsx";
import { ApiError, api } from "./services/api.js";
import { createRequestSequence } from "./services/requestSequence.js";

const WORKSPACE_KEY = "omnibrain.workspace.v1";
const THEME_KEY = "omnibrain.theme";
// Maximum number of PDFs in ONE upload (not a limit on the whole conversation).
const MAX_FILES = 5;

function createId() {
  return globalThis.crypto?.randomUUID?.() || `${Date.now()}-${Math.random().toString(36).slice(2)}`;
}

function createConversation() {
  return {
    id: createId(),
    title: "New chat",
    documents: [],
    selectedIds: [],
    messages: [],
    saved: false,
    updatedAt: Date.now(),
  };
}

function loadTheme() {
  try {
    const saved = localStorage.getItem(THEME_KEY);
    if (saved === "dark" || saved === "light") return saved;
  } catch {
    return "dark";
  }
  return globalThis.document?.documentElement.dataset.theme === "light" ? "light" : "dark";
}

// Older saved conversations stored a single `document`; newer ones store `documents`.
function normalizeDocuments(item) {
  const list = Array.isArray(item.documents)
    ? item.documents
    : item.document
      ? [item.document]
      : [];

  const seen = new Set();

  return list.filter((doc) => {
    if (
      !doc ||
      typeof doc.document_id !== "string" ||
      seen.has(doc.document_id)
    ) {
      return false;
    }

    seen.add(doc.document_id);
    return true;
  });
}

function loadWorkspace() {
  try {
    const saved = JSON.parse(localStorage.getItem(WORKSPACE_KEY) || "null");

    if (!saved || !Array.isArray(saved.conversations)) {
      const conversation = createConversation();
      return { activeId: conversation.id, conversations: [conversation] };
    }

    const conversations = saved.conversations
      .filter((item) => item && typeof item.id === "string" && Array.isArray(item.messages))
      .map((item) => {
        const lastQuestion = [...item.messages]
          .reverse()
          .find((message) => message.role === "user")?.content;

        const documents = normalizeDocuments(item);
        const knownIds = new Set(documents.map((doc) => doc.document_id));

        return {
          id: item.id,
          title: typeof item.title === "string" ? item.title : "New chat",
          documents,
          selectedIds: Array.isArray(item.selectedIds)
            ? item.selectedIds.filter((id) => knownIds.has(id))
            : [],
          saved: item.saved === true,
          updatedAt: Number(item.updatedAt) || Date.now(),
          messages: item.messages
            .filter(
              (message) =>
                message &&
                (message.role === "user" || message.role === "assistant"),
            )
            .map((message) =>
              message.pending
                ? {
                    id: message.id || createId(),
                    role: "assistant",
                    content: "No answer was saved.",
                    error: "This request was interrupted when the workspace was closed.",
                    retryQuestion: lastQuestion,
                  }
                : {
                    id: message.id || createId(),
                    role: message.role,
                    content: String(message.content || ""),
                    ...(Array.isArray(message.sources)
                      ? { sources: message.sources }
                      : {}),
                    ...(message.error
                      ? { error: String(message.error) }
                      : {}),
                    ...(message.retryQuestion
                      ? { retryQuestion: String(message.retryQuestion) }
                      : {}),
                  },
            ),
        };
      });

    if (!conversations.length) {
      const conversation = createConversation();
      return { activeId: conversation.id, conversations: [conversation] };
    }

    const activeId = conversations.some((item) => item.id === saved.activeId)
      ? saved.activeId
      : conversations[0].id;

    return { activeId, conversations };
  } catch {
    const conversation = createConversation();
    return { activeId: conversation.id, conversations: [conversation] };
  }
}

function messageTitle(question) {
  const singleLine = question.replace(/\s+/g, " ").trim();
  return singleLine.length > 54
    ? `${singleLine.slice(0, 51)}…`
    : singleLine || "New chat";
}

function formatBytes(bytes) {
  if (typeof bytes !== "number") return "Not available";

  return bytes < 1024 * 1024
    ? `${Math.max(1, Math.round(bytes / 1024))} KB`
    : `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
}

function pickDocument(result) {
  return {
    document_id: result.document_id,
    filename: result.filename,
    status: result.status,
    page_count: result.page_count,
    size_bytes: result.size_bytes,
  };
}

function Welcome({ documents }) {
  const count = documents.length;

  return (
    <div className="welcome-state">
      <div className="welcome-anim" aria-hidden="true">
        <span className="wa-glow" />
        <span className="wa-ring wa-ring--1"><i /></span>
        <span className="wa-ring wa-ring--2"><i /></span>
        <span className="wa-ring wa-ring--3"><i /></span>
        <span className="wa-core" />
      </div>

      <h1>
        {count === 0
          ? "Master Your Documents in Seconds."
          : count === 1
            ? "Your document is ready."
            : `Your ${count} documents are ready.`}
      </h1>

      <p className="welcome-description">
        {count === 0
          ? `Upload up to ${MAX_FILES} PDFs at a time, ask what you need, and get answers backed by real page references.`
          : count === 1
            ? "Ask anything about it below. Answers are based on passages retrieved from your PDF and may not cover every page."
            : "Ask anything below. Answers name the PDF and page they came from. Pick specific PDFs to narrow the search."}
      </p>
    </div>
  );
}

function ChatWorkspace({
  conversation,
  onAsk,
  isAnswering,
  isUploading,
  uploadError,
  onUpload,
  onRetryAnswer,
  onToggleSelect,
  onDeleteDocument,
  deletingId,
}) {
  const [question, setQuestion] = useState("");
  const conversationRef = useRef(null);
  const inputRef = useRef(null);

  const documents = conversation.documents;
  const selectedIds = conversation.selectedIds || [];
  const messages = conversation.messages;
  const hasDocuments = documents.length > 0;

  const canAsk =
    hasDocuments &&
    Boolean(question.trim()) &&
    !isAnswering &&
    !isUploading;

  let scopeHint = "Attach a PDF to begin";
  if (documents.length === 1) {
    scopeHint = "Answers use the active PDF";
  } else if (documents.length > 1 && selectedIds.length > 0) {
    scopeHint = `Answers use ${selectedIds.length} selected PDF${selectedIds.length === 1 ? "" : "s"}`;
  } else if (documents.length > 1) {
    scopeHint = `Answers search all ${documents.length} PDFs`;
  }

  useEffect(() => {
    if (conversationRef.current) {
      conversationRef.current.scrollTop =
        conversationRef.current.scrollHeight;
    }
  }, [messages]);

  function submit(event) {
    event.preventDefault();

    if (!canAsk) return;

    const submittedQuestion = question.trim();
    setQuestion("");
    onAsk(submittedQuestion);
    inputRef.current?.focus();
  }

  function handleKeyDown(event) {
    if (
      event.key === "Enter" &&
      !event.shiftKey &&
      !event.nativeEvent.isComposing
    ) {
      event.preventDefault();

      if (canAsk) {
        submit(event);
      }
    }
  }

  return (
    <main className="chat-workspace">
      <div className="conversation" ref={conversationRef} aria-live="polite">
        {!hasDocuments ? (
          <Welcome documents={[]} />
        ) : messages.length === 0 ? (
          <Welcome documents={documents} />
        ) : (
          <div className="message-list">
            {messages.map((message) => (
              <MessageBubble
                message={message}
                key={message.id}
                onRetry={
                  message.retryQuestion
                    ? () =>
                        onRetryAnswer(
                          message.retryQuestion,
                          message.id,
                        )
                    : undefined
                }
              />
            ))}
          </div>
        )}
      </div>

      <div className="composer-wrap">
        {uploadError && (
          <div className="composer-error" role="alert">
            <Info size={15} />
            <span>{uploadError}</span>
          </div>
        )}

        <form
          className={`composer ${!hasDocuments ? "composer--disabled" : ""}`}
          onSubmit={submit}
        >
          {documents.length > 1 && (
            <div
              className="scope-chips"
              role="group"
              aria-label="PDFs to search"
            >
              <span className="scope-chips-label">Search in:</span>

              {documents.map((doc) => {
                const active = selectedIds.includes(doc.document_id);

                return (
                  <button
                    key={doc.document_id}
                    type="button"
                    className={`scope-chip ${active ? "scope-chip--active" : ""}`}
                    aria-pressed={active}
                    title={doc.filename}
                    onClick={() => onToggleSelect(doc.document_id)}
                  >
                    {doc.filename}
                  </button>
                );
              })}
            </div>
          )}

          <FileUploader
            documents={documents}
            maxFiles={MAX_FILES}
            onUpload={onUpload}
            uploading={isUploading}
            onDeleteDocument={onDeleteDocument}
            deletingId={deletingId}
          />

          <label className="visually-hidden" htmlFor="question-input">
            Ask about your documents
          </label>

          <textarea
            id="question-input"
            ref={inputRef}
            rows={1}
            placeholder={
              hasDocuments
                ? documents.length > 1
                  ? "Ask anything about your documents…"
                  : "Ask anything about your document…"
                : "Upload a document to start asking questions"
            }
            value={question}
            disabled={!hasDocuments || isAnswering || isUploading}
            onChange={(event) => setQuestion(event.target.value)}
            onKeyDown={handleKeyDown}
          />

          <div className="composer-bottom">
            <span className="composer-hint">
              {isAnswering ? (
                <>
                  <span className="spinner spinner--small" />
                  Searching and preparing an answer…
                </>
              ) : hasDocuments ? (
                <>
                  {scopeHint}
                  <span className="hint-separator">·</span>
                  <kbd>Enter</kbd> to send
                </>
              ) : (
                <>
                  {scopeHint}
                  <span className="hint-separator">·</span>
                  <kbd>Shift+Enter</kbd> for a new line
                </>
              )}
            </span>

            <button
              className="send-button"
              type="submit"
              aria-label="Send question"
              disabled={!canAsk}
            >
              <ArrowUpRight size={17} />
            </button>
          </div>
        </form>

        <p className="composer-disclaimer">
          Answers can be imperfect. Check the cited passages against your document.
        </p>
      </div>
    </main>
  );
}

function WorkspaceDialog({ dialog, onClose, onConfirm, onRenameChange }) {
  if (!dialog) return null;

  const isRename = dialog.type === "rename";

  return (
    <div
      className="dialog-backdrop"
      onMouseDown={(event) => {
        if (event.target === event.currentTarget) onClose();
      }}
    >
      <section
        className="workspace-dialog"
        role={isRename ? "dialog" : "alertdialog"}
        aria-modal="true"
        aria-labelledby="workspace-dialog-title"
        aria-describedby="workspace-dialog-description"
        onKeyDown={(event) => {
          if (event.key === "Escape") onClose();
        }}
      >
        <div className="eyebrow">
          {isRename ? "CONVERSATION" : "CONFIRM ACTION"}
        </div>

        <h2 id="workspace-dialog-title">{dialog.title}</h2>
        <p id="workspace-dialog-description">{dialog.message}</p>

        {isRename ? (
          <form
            onSubmit={(event) => {
              event.preventDefault();
              onConfirm();
            }}
          >
            <label
              className="visually-hidden"
              htmlFor="rename-conversation-input"
            >
              Conversation title
            </label>

            <input
              id="rename-conversation-input"
              autoFocus
              maxLength={80}
              value={dialog.value}
              onChange={(event) => onRenameChange(event.target.value)}
            />

            <div className="dialog-actions">
              <button
                className="button button--subtle"
                type="button"
                onClick={onClose}
              >
                Cancel
              </button>

              <button className="button button--primary" type="submit">
                Save title
              </button>
            </div>
          </form>
        ) : (
          <div className="dialog-actions">
            <button
              className="button button--subtle"
              type="button"
              onClick={onClose}
            >
              Cancel
            </button>

            <button
              className={`button ${dialog.danger ? "button--danger" : "button--primary"}`}
              type="button"
              onClick={onConfirm}
            >
              {dialog.confirmLabel}
            </button>
          </div>
        )}
      </section>
    </div>
  );
}

function DocumentsView({
  documents,
  selectedIds,
  deletingId,
  onToggleSelect,
  onSelectAll,
  onClearSelection,
  onDelete,
}) {
  const allSelected =
    documents.length > 0 && selectedIds.length === documents.length;

  return (
    <main className="secondary-view">
      <div className="secondary-heading">
        <div className="eyebrow">WORKSPACE</div>
        <h1>Documents</h1>
        <p>
          The PDFs attached to the selected conversation ({documents.length}).
        </p>
      </div>

      {documents.length > 0 ? (
        <>
          {documents.length > 1 && (
            <div className="document-scope">
              <span>
                {selectedIds.length > 0
                  ? `Questions search ${selectedIds.length} selected PDF${selectedIds.length === 1 ? "" : "s"}`
                  : "Questions search all PDFs"}
              </span>

              <button
                className="button button--subtle"
                type="button"
                onClick={allSelected ? onClearSelection : onSelectAll}
              >
                {allSelected ? "Clear selection" : "Select all"}
              </button>
            </div>
          )}

          {documents.map((doc) => (
            <section className="managed-document" key={doc.document_id}>
              {documents.length > 1 && (
                <input
                  type="checkbox"
                  checked={selectedIds.includes(doc.document_id)}
                  disabled={deletingId === doc.document_id}
                  aria-label={`Search ${doc.filename}`}
                  onChange={() => onToggleSelect(doc.document_id)}
                />
              )}

              <span className="managed-document-icon">
                <FileText size={21} />
              </span>

              <div className="managed-document-details">
                <strong>{doc.filename}</strong>

                <span>
                  {typeof doc.page_count === "number" && (
                    <>
                      {doc.page_count} pages <i />
                    </>
                  )}

                  {typeof doc.size_bytes === "number" && (
                    <>
                      {formatBytes(doc.size_bytes)} <i />
                    </>
                  )}

                  {doc.status || "Status unavailable"}
                </span>

                <small>Document ID: {doc.document_id}</small>
              </div>

              <span className="document-status">
                <i /> {doc.status || "Unknown"}
              </span>

              <button
                className="icon-button"
                type="button"
                disabled={deletingId === doc.document_id}
                aria-label={`Delete ${doc.filename}`}
                title="Delete this PDF"
                onClick={() => onDelete(doc)}
              >
                <Trash2 size={16} />
              </button>
            </section>
          ))}

          <p className="documents-note">
            <Info size={15} /> Deleting a PDF removes it from the backend.
          </p>
        </>
      ) : (
        <div className="documents-empty">
          <span className="empty-document-icon">
            <FileText size={22} />
          </span>

          <h2>No documents in this conversation</h2>

          <p>
            Attach up to {MAX_FILES} PDFs at a time from the chat composer.
            Documents stay scoped to the conversation where they were
            uploaded.
          </p>
        </div>
      )}
    </main>
  );
}

function HistoryView({
  conversations,
  activeConversationId,
  onSelectConversation,
  onDeleteConversation,
  onDeleteHistory,
}) {
  return (
    <main className="secondary-view history-view">
      <HistoryPanel
        conversations={conversations}
        activeConversationId={activeConversationId}
        onSelect={onSelectConversation}
        onDeleteConversation={onDeleteConversation}
        onDeleteHistory={onDeleteHistory}
      />
    </main>
  );
}

function SettingsView({ onRefreshHealth }) {
  return (
    <main className="secondary-view settings-view">
      <div className="secondary-heading">
        <div className="eyebrow">PREFERENCES</div>
        <h1>Settings</h1>
        <p>Connection details for this OmniBrain workspace.</p>
      </div>

      <section className="settings-card">
        <div className="settings-card-title">
          <div>
            <h2>Backend connection</h2>
            <p>API endpoint configured for this frontend.</p>
          </div>

          <button
            className="button button--subtle"
            type="button"
            onClick={onRefreshHealth}
          >
            Run health check
          </button>
        </div>

        <div className="settings-field">
          <span>API base URL</span>
          <code>{api.baseUrl}</code>
        </div>

        <div className="settings-field">
          <span>Health check</span>
          <code>GET /health</code>
        </div>

        <div className="settings-field">
          <span>Document upload (up to {MAX_FILES} PDFs at a time)</span>
          <code>POST /api/documents/upload</code>
        </div>

        <div className="settings-field">
          <span>Question answering</span>
          <code>POST /api/documents/&#123;document_id&#125;/answer</code>
        </div>

        <div className="settings-field">
          <span>Delete a PDF</span>
          <code>DELETE /api/documents/&#123;document_id&#125;</code>
        </div>

        <p className="settings-note">
          <Info size={15} /> Set <code>VITE_API_BASE_URL</code> in the frontend
          environment. For production, use the deployed backend URL rather than
          a localhost address.
        </p>
      </section>

      <p className="settings-note settings-limits">
        <Info size={15} /> This app has no authentication, server-side
        conversation history, or public conversation links. Removing a
        conversation, deleting history, or resetting the workspace also deletes
        the PDFs uploaded in them from the backend.
      </p>
    </main>
  );
}

export default function App() {
  const [workspace, setWorkspace] = useState(loadWorkspace);
  const workspaceRef = useRef(workspace);
  const [activeView, setActiveView] = useState("chat");
  const [backendStatus, setBackendStatus] = useState("checking");
  const [isUploading, setIsUploading] = useState(false);
  const [isAnswering, setIsAnswering] = useState(false);
  const [deletingId, setDeletingId] = useState(null);
  const [uploadError, setUploadError] = useState(null);
  const [storageError, setStorageError] = useState("");
  const [notice, setNotice] = useState("");
  const [dialog, setDialog] = useState(null);
  const [theme, setTheme] = useState(loadTheme);

  const answerRequest = useRef(null);
  const answerSequence = useRef(createRequestSequence());
  const uploadRequest = useRef(false);
  const healthRequest = useRef(null);
  const noticeTimer = useRef(null);
  const mounted = useRef(false);

  const activeConversation =
    workspace.conversations.find((item) => item.id === workspace.activeId) ||
    workspace.conversations[0];

  const documents = activeConversation?.documents || [];
  const selectedIds = activeConversation?.selectedIds || [];

  const commitWorkspace = useCallback((update) => {
    const current = workspaceRef.current;
    const next = typeof update === "function" ? update(current) : update;

    workspaceRef.current = next;
    setWorkspace(next);

    try {
      localStorage.setItem(WORKSPACE_KEY, JSON.stringify(next));
      setStorageError("");
    } catch {
      setStorageError(
        "Browser storage is unavailable; this conversation may not persist after closing this tab.",
      );
    }

    return next;
  }, []);

  const updateConversation = useCallback((conversationId, update) => {
    return commitWorkspace((current) => ({
      ...current,
      conversations: current.conversations.map((conversation) =>
        conversation.id === conversationId
          ? { ...conversation, ...update(conversation), updatedAt: Date.now() }
          : conversation,
      ),
    }));
  }, [commitWorkspace]);

  const showNotice = useCallback((message) => {
    setNotice(message);
    window.clearTimeout(noticeTimer.current);
    noticeTimer.current = window.setTimeout(() => setNotice(""), 3200);
  }, []);

  const refreshHealth = useCallback(() => {
    if (healthRequest.current) return healthRequest.current;

    if (mounted.current) setBackendStatus("checking");

    healthRequest.current = api.checkHealth()
      .then((result) => {
        const status = result?.status === "healthy" ? "online" : "offline";

        if (mounted.current) setBackendStatus(status);

        return status;
      })
      .catch(() => {
        if (mounted.current) setBackendStatus("offline");

        return "offline";
      })
      .finally(() => {
        healthRequest.current = null;
      });

    return healthRequest.current;
  }, []);

  useEffect(() => {
    mounted.current = true;

    const markOffline = () => setBackendStatus("offline");
    const markOnline = () => setBackendStatus("online");

    window.addEventListener("omnibrain:backend-unavailable", markOffline);
    window.addEventListener("omnibrain:backend-reachable", markOnline);

    refreshHealth();

    const interval = window.setInterval(refreshHealth, 30_000);

    return () => {
      mounted.current = false;

      window.removeEventListener("omnibrain:backend-unavailable", markOffline);
      window.removeEventListener("omnibrain:backend-reachable", markOnline);

      window.clearInterval(interval);
    };
  }, [refreshHealth]);

  useEffect(() => {
    try {
      localStorage.setItem(THEME_KEY, theme);
    } catch {
      setStorageError("Theme preference could not be saved in browser storage.");
    }

    documentElementTheme(theme);
  }, [theme]);

  useEffect(() => () => window.clearTimeout(noticeTimer.current), []);

  function cancelAnswer(conversationId) {
    const pending = answerRequest.current;

    if (
      !pending ||
      (conversationId && pending.conversationId !== conversationId)
    ) {
      return;
    }

    answerSequence.current.invalidate();
    answerRequest.current = null;
    setIsAnswering(false);

    updateConversation(pending.conversationId, (conversation) => ({
      messages: conversation.messages.map((message) =>
        message.id === pending.assistantId && message.pending
          ? {
              ...message,
              content: "Answer generation stopped.",
              pending: false,
              error: "This answer was stopped after switching conversations.",
              retryQuestion: pending.question,
            }
          : message,
      ),
    }));
  }

  // Delete PDFs from the backend (best effort) and tell the user how it went.
  async function purgeDocuments(docs, baseMessage) {
    if (!docs.length) {
      showNotice(baseMessage);
      return;
    }

    const results = await Promise.allSettled(
      docs.map((doc) => api.deleteDocument(doc.document_id)),
    );

    const failed = results.filter(
      (result) => result.status === "rejected",
    ).length;

    showNotice(
      failed
        ? `${baseMessage} ${failed} PDF${failed === 1 ? "" : "s"} could not be deleted from the backend.`
        : `${baseMessage} Its PDFs were deleted from the backend.`,
    );
  }

  function startNewChat() {
    cancelAnswer();

    const current = workspaceRef.current;
    const existing = current.conversations.find(
      (item) => item.id === current.activeId,
    );

    const conversation = createConversation();

    commitWorkspace((latest) => ({
      activeId: conversation.id,
      conversations: [
        conversation,
        ...latest.conversations.filter((item) => (
          item.id !== existing?.id ||
          item.messages.length > 0 ||
          item.documents.length > 0 ||
          item.saved
        )),
      ],
    }));

    setActiveView("chat");
  }

  function switchConversation(conversationId) {
    if (conversationId === workspaceRef.current.activeId) {
      setActiveView("chat");
      return;
    }

    cancelAnswer();

    commitWorkspace((current) => ({
      ...current,
      activeId: conversationId,
    }));

    setActiveView("chat");
  }

  async function uploadDocuments(
    files,
    conversationId = workspaceRef.current.activeId,
  ) {
    if (uploadRequest.current) return false;

    if (!Array.isArray(files) || files.length === 0) return false;

    // The limit applies to one upload, not to the whole conversation.
    if (files.length > MAX_FILES) {
      setUploadError({
        conversationId,
        message: `You can upload at most ${MAX_FILES} PDFs at a time. You selected ${files.length}.`,
      });

      return false;
    }

    uploadRequest.current = true;
    setUploadError(null);
    setIsUploading(true);

    try {
      const result = await api.uploadDocuments(files);

      const uploaded = (Array.isArray(result?.uploaded) ? result.uploaded : [])
        .filter(
          (item) =>
            item?.document_id &&
            item?.filename &&
            item.status === "uploaded",
        )
        .map(pickDocument);

      const failed = Array.isArray(result?.failed) ? result.failed : [];

      if (uploaded.length === 0) {
        const first = failed[0];

        throw new ApiError(
          first?.error
            ? `${first.filename || "PDF"}: ${first.error}`
            : "The backend response did not include valid upload details.",
        );
      }

      updateConversation(conversationId, (item) => {
        const known = new Set(item.documents.map((doc) => doc.document_id));

        return {
          documents: [
            ...item.documents,
            ...uploaded.filter((doc) => !known.has(doc.document_id)),
          ],
        };
      });

      // Some PDFs may have been accepted while others were rejected.
      if (failed.length > 0) {
        setUploadError({
          conversationId,
          message: failed
            .map(
              (item) =>
                `${item.filename || "PDF"}: ${item.error || "upload failed"}`,
            )
            .join(" · "),
        });
      }

      if (workspaceRef.current.activeId === conversationId) {
        setActiveView("chat");
      }

      return true;
    } catch (error) {
      const message =
        error instanceof Error ? error.message : "The upload failed.";

      setUploadError({ conversationId, message });

      return false;
    } finally {
      uploadRequest.current = false;
      setIsUploading(false);
    }
  }

  function toggleSelect(documentId) {
    if (!activeConversation) return;

    updateConversation(activeConversation.id, (item) => {
      const current = item.selectedIds || [];

      return {
        selectedIds: current.includes(documentId)
          ? current.filter((id) => id !== documentId)
          : [...current, documentId],
      };
    });
  }

  function selectAllDocuments() {
    if (!activeConversation) return;

    updateConversation(activeConversation.id, (item) => ({
      selectedIds: item.documents.map((doc) => doc.document_id),
    }));
  }

  function clearDocumentSelection() {
    if (!activeConversation) return;

    updateConversation(activeConversation.id, () => ({ selectedIds: [] }));
  }

  function requestDeleteDocument(doc) {
    if (!activeConversation) return;

    const conversationId = activeConversation.id;

    setDialog({
      type: "confirm",
      title: "Delete this PDF?",
      message: `"${doc.filename}" will be removed from the backend and can no longer be searched.`,
      confirmLabel: "Delete PDF",
      danger: true,
      onConfirm: async () => {
        cancelAnswer(conversationId);
        setDeletingId(doc.document_id);

        try {
          await api.deleteDocument(doc.document_id);

          updateConversation(conversationId, (item) => ({
            documents: item.documents.filter(
              (entry) => entry.document_id !== doc.document_id,
            ),
            selectedIds: (item.selectedIds || []).filter(
              (id) => id !== doc.document_id,
            ),
          }));

          showNotice(`Deleted ${doc.filename}.`);
        } catch (error) {
          showNotice(
            error instanceof Error
              ? error.message
              : "The PDF could not be deleted.",
          );
        } finally {
          setDeletingId(null);
        }
      },
    });
  }

  async function runQuestion(
    question,
    conversationId = workspaceRef.current.activeId,
    retryMessageId = null,
  ) {
    if (answerRequest.current || uploadRequest.current) return;

    const conversation = workspaceRef.current.conversations.find(
      (item) => item.id === conversationId,
    );

    if (!conversation?.documents.length) return;

    const assistantId = retryMessageId || createId();
    const generation = answerSequence.current.begin();

    // Always send explicit IDs: the ticked PDFs, or every PDF in THIS
    // conversation. Never leave it empty, or the backend could search PDFs
    // that belong to other conversations.
    const knownIds = conversation.documents.map((doc) => doc.document_id);
    const ticked = (conversation.selectedIds || []).filter((id) =>
      knownIds.includes(id),
    );
    const documentIds = ticked.length > 0 ? ticked : knownIds;

    answerRequest.current = { assistantId, conversationId, question };

    setIsAnswering(true);
    setActiveView("chat");

    updateConversation(conversationId, (item) => {
      const messagesBefore = retryMessageId
        ? item.messages.filter((message) => message.id !== retryMessageId)
        : item.messages;

      const questionMessage = retryMessageId
        ? messagesBefore
        : [
            ...messagesBefore,
            { id: createId(), role: "user", content: question },
          ];

      const withAssistant = [
        ...questionMessage,
        {
          id: assistantId,
          role: "assistant",
          content: "Searching the documents…",
          pending: true,
        },
      ];

      const hasPriorQuestions = item.messages.some(
        (message) => message.role === "user",
      );

      return {
        title: hasPriorQuestions ? item.title : messageTitle(question),
        messages: withAssistant,
      };
    });

    try {
      const result = await api.answerQuestion(documentIds, question);

      if (!answerSequence.current.isCurrent(generation)) return;

      if (
        typeof result?.answer !== "string" ||
        !result.answer.trim() ||
        !Array.isArray(result.sources)
      ) {
        throw new ApiError(
          "The backend returned an unexpected answer format.",
        );
      }

      const allowed = new Set(documentIds);

      if (
        result.sources.some(
          (source) =>
            source?.document_id && !allowed.has(source.document_id),
        )
      ) {
        throw new ApiError(
          "The backend returned passages from a PDF outside this search. The answer was hidden to prevent mixing documents.",
        );
      }

      updateConversation(conversationId, (item) => ({
        messages: item.messages.map((message) =>
          message.id === assistantId
            ? {
                ...message,
                content: result.answer,
                sources: result.sources,
                pending: false,
              }
            : message,
        ),
      }));
    } catch (error) {
      if (!answerSequence.current.isCurrent(generation)) return;

      const message =
        error instanceof Error
          ? error.message
          : "The question could not be answered.";

      updateConversation(conversationId, (item) => ({
        messages: item.messages.map((current) =>
          current.id === assistantId
            ? {
                ...current,
                content: "I couldn’t complete that answer.",
                error: message,
                pending: false,
                retryQuestion: question,
              }
            : current,
        ),
      }));
    } finally {
      if (answerSequence.current.isCurrent(generation)) {
        answerRequest.current = null;
        setIsAnswering(false);
      }
    }
  }

  function toggleSaved() {
    if (!activeConversation) return;

    const saved = !activeConversation.saved;

    updateConversation(activeConversation.id, () => ({ saved }));

    showNotice(
      saved ? "Bookmarked in this browser only." : "Bookmark removed.",
    );
  }

  async function shareConversation() {
    if (!activeConversation) return;

    const text = getConversationText(activeConversation);
    const title = activeConversation.title || "OmniBrain conversation";

    if (navigator.share) {
      try {
        await navigator.share({ title, text });
        showNotice("Conversation shared.");
        return;
      } catch (error) {
        if (error instanceof DOMException && error.name === "AbortError") {
          return;
        }
      }
    }

    try {
      await copyText(text);
      showNotice("Conversation copied to clipboard. No public link was created.");
    } catch (error) {
      showNotice(
        error instanceof Error
          ? error.message
          : "Could not share this conversation.",
      );
    }
  }

  async function copyConversation() {
    try {
      await copyText(getConversationText(activeConversation));
      showNotice("Conversation copied to clipboard.");
    } catch (error) {
      showNotice(
        error instanceof Error
          ? error.message
          : "Could not copy this conversation.",
      );
    }
  }

  function renameConversation() {
    if (!activeConversation) return;

    setDialog({
      type: "rename",
      conversationId: activeConversation.id,
      title: "Rename conversation",
      message: "Choose a name that will be saved in this browser.",
      value: activeConversation.title,
    });
  }

  function saveConversationName() {
    if (dialog?.type !== "rename") return;

    const trimmed = dialog.value.trim();

    if (!trimmed) {
      showNotice("Conversation title cannot be empty.");
      return;
    }

    updateConversation(dialog.conversationId, () => ({
      title: trimmed.slice(0, 80),
    }));

    setDialog(null);
  }

  function clearCurrentConversation() {
    if (!activeConversation) return;

    const conversationId = activeConversation.id;

    setDialog({
      type: "confirm",
      title: "Clear this conversation?",
      message:
        "Its messages will be removed from this browser. The attached PDFs will stay.",
      confirmLabel: "Clear messages",
      danger: true,
      onConfirm: () => {
        cancelAnswer(conversationId);

        updateConversation(conversationId, () => ({
          title: "New chat",
          messages: [],
        }));

        setUploadError(null);
        showNotice("Conversation messages cleared.");
      },
    });
  }

  function deleteConversation(conversationId) {
    const target = workspaceRef.current.conversations.find(
      (item) => item.id === conversationId,
    );

    if (!target) return;

    setDialog({
      type: "confirm",
      title: "Remove this conversation?",
      message:
        "This removes it from local browser history and deletes its uploaded PDFs from the backend.",
      confirmLabel: "Remove conversation",
      danger: true,
      onConfirm: () => {
        const wasActive =
          conversationId === workspaceRef.current.activeId;

        const docsToPurge = target.documents || [];

        cancelAnswer(conversationId);

        const remaining = workspaceRef.current.conversations.filter(
          (item) => item.id !== conversationId,
        );

        const nextActive = wasActive ? createConversation() : null;

        commitWorkspace({
          activeId: nextActive?.id || workspaceRef.current.activeId,
          conversations: nextActive
            ? [nextActive, ...remaining]
            : remaining,
        });

        if (wasActive) setActiveView("chat");

        purgeDocuments(docsToPurge, "Conversation removed.");
      },
    });
  }

  function deleteHistory() {
    const history = workspaceRef.current.conversations.filter(
      (item) => item.messages.length || item.documents.length || item.saved,
    );

    if (!history.length) return;

    setDialog({
      type: "confirm",
      title: "Delete local history?",
      message:
        "All conversations will be removed from this browser, and every PDF uploaded in them will be deleted from the backend.",
      confirmLabel: "Delete history",
      danger: true,
      onConfirm: () => {
        const docsToPurge = workspaceRef.current.conversations.flatMap(
          (item) => item.documents || [],
        );

        cancelAnswer();

        const fresh = createConversation();

        commitWorkspace({
          activeId: fresh.id,
          conversations: [fresh],
        });

        setUploadError(null);
        setActiveView("chat");

        purgeDocuments(docsToPurge, "Local conversation history cleared.");
      },
    });
  }

  function resetWorkspace() {
    setDialog({
      type: "confirm",
      title: "Reset this workspace?",
      message:
        "Local conversations will be cleared and every uploaded PDF will be deleted from the backend.",
      confirmLabel: "Reset workspace",
      danger: true,
      onConfirm: () => {
        const fresh = createConversation();
        const docsToPurge = workspaceRef.current.conversations.flatMap(
          (item) => item.documents || [],
        );

        try {
          cancelAnswer();
          localStorage.removeItem(WORKSPACE_KEY);

          workspaceRef.current = {
            activeId: fresh.id,
            conversations: [fresh],
          };

          setWorkspace(workspaceRef.current);
          setStorageError("");
          setUploadError(null);
          setActiveView("chat");

          purgeDocuments(docsToPurge, "Browser workspace reset.");
        } catch {
          setStorageError(
            "Could not clear locally saved conversations from browser storage.",
          );
        }
      },
    });
  }

  return (
    <div className="app-shell">
      <Sidebar
        activeView={activeView}
        document={documents[0] || null}
        documents={documents}
        onNewChat={startNewChat}
        onOpenDocuments={() => setActiveView("documents")}
        onOpenHistory={() => setActiveView("history")}
        onOpenSettings={() => setActiveView("settings")}
        onResetWorkspace={resetWorkspace}
      />

      <div className="main-column">
        <ConversationToolbar
          conversation={activeConversation}
          theme={theme}
          backendStatus={backendStatus}
          onToggleTheme={() =>
            setTheme((value) => (value === "dark" ? "light" : "dark"))
          }
          onToggleSaved={toggleSaved}
          onShare={shareConversation}
          onCopy={copyConversation}
          onRename={renameConversation}
          onClear={clearCurrentConversation}
          onDelete={() => deleteConversation(activeConversation.id)}
          onRefreshHealth={refreshHealth}
        />

        {activeView === "chat" ? (
          <ChatWorkspace
            key={activeConversation.id}
            conversation={activeConversation}
            onAsk={(question) =>
              runQuestion(question, activeConversation.id)
            }
            onRetryAnswer={(question, messageId) =>
              runQuestion(
                question,
                activeConversation.id,
                messageId,
              )
            }
            isAnswering={isAnswering}
            isUploading={isUploading}
            uploadError={
              uploadError?.conversationId === activeConversation.id
                ? uploadError.message
                : ""
            }
            onUpload={(files) =>
              uploadDocuments(files, activeConversation.id)
            }
            onToggleSelect={toggleSelect}
            onDeleteDocument={requestDeleteDocument}
            deletingId={deletingId}
          />
        ) : activeView === "documents" ? (
          <DocumentsView
            documents={documents}
            selectedIds={selectedIds}
            deletingId={deletingId}
            onToggleSelect={toggleSelect}
            onSelectAll={selectAllDocuments}
            onClearSelection={clearDocumentSelection}
            onDelete={requestDeleteDocument}
          />
        ) : activeView === "history" ? (
          <HistoryView
            conversations={workspace.conversations}
            activeConversationId={workspace.activeId}
            onSelectConversation={switchConversation}
            onDeleteConversation={deleteConversation}
            onDeleteHistory={deleteHistory}
          />
        ) : (
          <SettingsView onRefreshHealth={refreshHealth} />
        )}

        <footer className="app-footer">
          <span>
            OmniBrain <span className="footer-dot">·</span> Document intelligence
          </span>

          <span className="footer-links">
            <span>PDF-grounded answers</span>
            <span>Conversation history stays in this browser</span>
          </span>
        </footer>
      </div>

      {(notice || storageError) && (
        <div
          className={`toast ${storageError ? "toast--error" : ""}`}
          role="status"
          aria-live="polite"
        >
          <Info size={15} />
          <span>{storageError || notice}</span>

          {storageError && (
            <button
              className="text-button"
              type="button"
              onClick={() => setStorageError("")}
            >
              Dismiss
            </button>
          )}
        </div>
      )}

      <WorkspaceDialog
        dialog={dialog}
        onClose={() => setDialog(null)}
        onConfirm={() => {
          if (dialog?.type === "rename") {
            saveConversationName();
          } else {
            dialog?.onConfirm?.();
          }

          setDialog(null);
        }}
        onRenameChange={(value) =>
          setDialog((current) => ({ ...current, value }))
        }
      />
    </div>
  );
}

function documentElementTheme(theme) {
  document.documentElement.dataset.theme = theme;

  document
    .querySelector('meta[name="theme-color"]')
    ?.setAttribute(
      "content",
      theme === "dark" ? "#0B0F14" : "#F5F7FA",
    );
}