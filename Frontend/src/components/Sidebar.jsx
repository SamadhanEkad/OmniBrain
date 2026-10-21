import {
  FileText,
  FolderOpen,
  History,
  LogOut,
  MessageSquarePlus,
  Settings2,
} from "lucide-react";

function formatFileSize(size) {
  if (size === undefined || size === null || size === "") return "-";
  if (typeof size === "string") return size;
  if (size < 1024) return `${size} B`;
  if (size < 1024 * 1024) return `${Math.round(size / 1024)} KB`;
  return `${(size / (1024 * 1024)).toFixed(1)} MB`;
}

export default function Sidebar({
  activeView,
  document,
  onNewChat,
  onOpenDocuments,
  onOpenHistory,
  onOpenSettings,
  onResetWorkspace,
}) {
  const docName =
    document?.name ||
    document?.filename ||
    document?.fileName ||
    document?.file_name ||
    document?.title ||
    "Untitled document";
  const docId =
    document?.id ||
    document?.document_id ||
    document?.documentId ||
    "-";
  const docPages =
    document?.pages ??
    document?.pageCount ??
    document?.page_count ??
    document?.num_pages ??
    "-";
  const docSize = formatFileSize(
    document?.size ??
      document?.fileSize ??
      document?.file_size ??
      document?.size_bytes
  );

  return (
    <aside className="sidebar">
      <div className="brand">
  <span className="brand-mark">
    <svg viewBox="0 0 48 48" fill="none" aria-hidden="true">
      <defs>
        <linearGradient id="ob-tile" x1="0" y1="0" x2="48" y2="48" gradientUnits="userSpaceOnUse">
          <stop offset="0" stopColor="#0ea5e9" />
          <stop offset="1" stopColor="#6366f1" />
        </linearGradient>
        <linearGradient id="ob-spark" x1="17" y1="19" x2="31" y2="33" gradientUnits="userSpaceOnUse">
          <stop offset="0" stopColor="#0ea5e9" />
          <stop offset="1" stopColor="#6366f1" />
        </linearGradient>
      </defs>

      {/* tile */}
      <rect width="48" height="48" rx="13" fill="url(#ob-tile)" />

      {/* document page with folded corner */}
      <path
        d="M16 9h11l8 8v19a3 3 0 0 1-3 3H16a3 3 0 0 1-3-3V12a3 3 0 0 1 3-3z"
        fill="#ffffff"
      />
      <path
        d="M27 9v5.5a2.5 2.5 0 0 0 2.5 2.5H35"
        stroke="#c7d2fe"
        strokeWidth="1.6"
        strokeLinecap="round"
        strokeLinejoin="round"
      />

      {/* AI sparkle */}
      <path
        d="M24 20c.7 4.2 2.8 6.3 7 7-4.2.7-6.3 2.8-7 7-.7-4.2-2.8-6.3-7-7 4.2-.7 6.3-2.8 7-7z"
        fill="url(#ob-spark)"
      />
      <circle cx="31.5" cy="32.5" r="1.4" fill="url(#ob-spark)" />
    </svg>
  </span>

  <span className="brand-name">Omni<span>Brain</span></span>
</div>

      <button className="new-chat-button" type="button" aria-label="New chat (Ctrl K)" title="New chat (Ctrl K)" onClick={onNewChat}>
        <MessageSquarePlus size={17} />
        <span>New chat</span>
        <span className="shortcut">Ctrl K</span>
      </button>

      <div className="nav-section-label">WORKSPACE</div>
      <nav className="side-navigation" aria-label="Main navigation">
        <button
          className={`nav-item ${activeView === "documents" ? "is-active" : ""}`}
          type="button"
          title="Documents"
          onClick={onOpenDocuments}
        >
          <FolderOpen size={17} />
          <span>Documents</span>
        </button>
        <button
          className={`nav-item ${activeView === "history" ? "is-active" : ""}`}
          type="button"
          title="History"
          onClick={onOpenHistory}
        >
          <History size={17} />
          <span>History</span>
        </button>
      </nav>

      {document && (
        <section className="sidebar-document" aria-label="Document information">
          <div className="sidebar-document-header">
            <span className="sidebar-document-icon">
              <FileText size={18} />
            </span>
            <span className="sidebar-document-status">
              <i /> uploaded
            </span>
          </div>

          <strong className="sidebar-document-name" title={docName}>
            {docName}
          </strong>
          <small className="sidebar-document-id" title={String(docId)}>
            ID · {docId}
          </small>

          <div className="sidebar-document-meta">
            <div>
              <small>Pages</small>
              <strong>{docPages}</strong>
            </div>
            <div>
              <small>File size</small>
              <strong>{docSize}</strong>
            </div>
          </div>
        </section>
      )}

      <div className="sidebar-spacer" />
      <div className="sidebar-footer">
        <button
          className={`nav-item settings-link ${activeView === "settings" ? "is-active" : ""}`}
          type="button"
          title="Settings"
          onClick={onOpenSettings}
        >
          <Settings2 size={17} />
          <span>Settings</span>
        </button>
        <div className="sidebar-profile">
          <span className="profile-avatar">OB</span>
          <span className="profile-copy">
            <strong>OmniBrain</strong>
            <small>Document workspace</small>
          </span>
          <span className="profile-menu" aria-hidden="true">···</span>
        </div>
        <button className="nav-item reset-workspace-button" type="button" onClick={onResetWorkspace} title="Clear this browser's local conversations">
          <LogOut size={16} />
          <span>Reset workspace</span>
        </button>
      </div>
    </aside>
  );
}