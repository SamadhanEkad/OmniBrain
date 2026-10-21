import { FileText, Sparkles, Trash2 } from "lucide-react";

const MAX_FILES = 5;

function formatBytes(bytes) {
  if (typeof bytes !== "number") return "Not available";

  return bytes < 1024 * 1024
    ? `${Math.max(1, Math.round(bytes / 1024))} KB`
    : `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
}

/**
 * Props
 * -----
 * documents          Uploaded PDFs: [{ document_id, filename, status,
 *                    page_count, size_bytes }]
 * selectedIds        document_ids the user ticked. Empty = search all PDFs.
 * onToggleSelect(id) Tick / untick one PDF.
 * onSelectAll()      Tick every PDF.
 * onClearSelection() Untick every PDF.
 * onDelete(id)       Delete one PDF (App should call the delete API).
 * deletingId         document_id currently being deleted, if any.
 * maxFiles           Workspace limit (default 5).
 */
export default function DocumentPanel({
  documents = [],
  selectedIds = [],
  onToggleSelect,
  onSelectAll,
  onClearSelection,
  onDelete,
  deletingId = null,
  maxFiles = MAX_FILES,
}) {
  const selectedCount = selectedIds.length;
  const allSelected =
    documents.length > 0 && selectedCount === documents.length;

  let scopeLabel = "Questions search all PDFs";
  if (documents.length === 1) {
    scopeLabel = "Questions search this PDF";
  } else if (selectedCount > 0) {
    scopeLabel = `Questions search ${selectedCount} selected PDF${
      selectedCount === 1 ? "" : "s"
    }`;
  }

  function handleDelete(doc) {
    if (typeof onDelete !== "function") return;

    const confirmed = window.confirm(
      `Delete "${doc.filename || "this PDF"}"? It will be removed from your workspace and cannot be searched anymore.`,
    );

    if (confirmed) {
      onDelete(doc.document_id);
    }
  }

  return (
    <aside
      className="document-panel"
      aria-label="Document intelligence"
    >
      <div className="panel-heading">
        <div>
          <div className="eyebrow">CONTEXT</div>
          <h2>Document intelligence</h2>
        </div>

        <span className="panel-heading-icon">
          <Sparkles size={17} />
        </span>
      </div>

      {documents.length > 0 ? (
        <>
          <div className="document-scope">
            <div>
              <strong>
                {documents.length}/{maxFiles} PDFs
              </strong>
              <p>{scopeLabel}</p>
            </div>

            {documents.length > 1 && (
              <button
                className="button"
                type="button"
                onClick={allSelected ? onClearSelection : onSelectAll}
              >
                {allSelected ? "Clear selection" : "Select all"}
              </button>
            )}
          </div>

          {documents.map((doc) => {
            const isSelected = selectedIds.includes(doc.document_id);
            const isDeleting = deletingId === doc.document_id;

            return (
              <section
                key={doc.document_id}
                className={
                  isSelected
                    ? "document-card document-card--selected"
                    : "document-card"
                }
              >
                <div className="document-card-top">
                  {documents.length > 1 && (
                    <input
                      type="checkbox"
                      checked={isSelected}
                      disabled={isDeleting}
                      aria-label={`Search ${doc.filename || "this PDF"}`}
                      onChange={() => onToggleSelect?.(doc.document_id)}
                    />
                  )}

                  <span className="document-icon">
                    <FileText size={19} />
                  </span>

                  <span className="document-status">
                    <i />
                    {doc.status || "Status unavailable"}
                  </span>

                  <button
                    className="icon-button"
                    type="button"
                    disabled={isDeleting}
                    aria-label={`Delete ${doc.filename || "PDF"}`}
                    title="Delete this PDF"
                    onClick={() => handleDelete(doc)}
                  >
                    <Trash2 size={15} />
                  </button>
                </div>

                <h3 title={doc.filename}>{doc.filename}</h3>

                <p className="document-id">
                  ID · {doc.document_id}
                </p>

                {(typeof doc.page_count === "number" ||
                  typeof doc.size_bytes === "number") && (
                  <div className="document-facts">
                    {typeof doc.page_count === "number" && (
                      <div>
                        <span>Pages</span>
                        <strong>{doc.page_count}</strong>
                      </div>
                    )}

                    {typeof doc.size_bytes === "number" && (
                      <div>
                        <span>File size</span>
                        <strong>{formatBytes(doc.size_bytes)}</strong>
                      </div>
                    )}
                  </div>
                )}

                {isDeleting && (
                  <p className="upload-note" role="status">
                    Deleting…
                  </p>
                )}
              </section>
            );
          })}
        </>
      ) : (
        <div className="panel-empty panel-empty--large">
          <span className="empty-document-icon">
            <FileText size={22} />
          </span>

          <h3>No documents yet</h3>

          <p>
            Attach up to {maxFiles} PDFs in the chat composer to see
            their details here.
          </p>
        </div>
      )}
    </aside>
  );
}