import { useRef, useState } from "react";
import {
  AlertCircle,
  FileUp,
  FileText,
  Paperclip,
  X,
} from "lucide-react";

const MAX_FILE_SIZE = 20 * 1024 * 1024;
const MAX_FILES = 5;

function formatFileSize(bytes) {
  if (bytes < 1024 * 1024) {
    return `${Math.max(1, Math.round(bytes / 1024))} KB`;
  }

  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
}

function isSameFile(a, b) {
  return a.name === b.name && a.size === b.size;
}

/**
 * Props
 * -----
 * onUpload(files)       Receives an ARRAY of File objects. Return (or resolve
 *                       to) true when the upload succeeded, false otherwise.
 * uploading             True while an upload is in progress.
 * documents             PDFs already uploaded to this conversation.
 * maxFiles              Maximum number of PDFs in ONE upload (default 5).
 * onDeleteDocument(doc) Called when the X on an uploaded PDF is clicked.
 * deletingId            document_id currently being deleted (or null).
 */
export default function FileUploader({
  onUpload,
  uploading = false,
  documents = [],
  maxFiles = MAX_FILES,
  onDeleteDocument,
  deletingId = null,
}) {
  const inputRef = useRef(null);

  const [files, setFiles] = useState([]);
  const [errors, setErrors] = useState([]);

  // The limit applies to the files selected for this one upload only.
  const slotsAfterSelection = Math.max(0, maxFiles - files.length);

  function selectFiles(fileList) {
    const incoming = Array.from(fileList || []);

    if (incoming.length === 0) {
      return;
    }

    const nextErrors = [];
    const accepted = [];
    let room = maxFiles - files.length;

    for (const candidate of incoming) {
      const isPdf =
        candidate.type === "application/pdf" ||
        candidate.name.toLowerCase().endsWith(".pdf");

      if (!isPdf) {
        nextErrors.push(`${candidate.name}: not a PDF file.`);
        continue;
      }

      if (candidate.size === 0) {
        nextErrors.push(`${candidate.name}: the PDF is empty.`);
        continue;
      }

      if (candidate.size > MAX_FILE_SIZE) {
        nextErrors.push(
          `${candidate.name}: exceeds the 20 MiB upload limit.`,
        );
        continue;
      }

      const duplicate =
        files.some((existing) => isSameFile(existing, candidate)) ||
        accepted.some((existing) => isSameFile(existing, candidate)) ||
        documents.some((doc) => doc.filename === candidate.name);

      if (duplicate) {
        nextErrors.push(`${candidate.name}: already added.`);
        continue;
      }

      if (room <= 0) {
        nextErrors.push(
          `${candidate.name}: not added. You can upload at most ${maxFiles} PDFs at a time.`,
        );
        continue;
      }

      accepted.push(candidate);
      room -= 1;
    }

    if (accepted.length > 0) {
      setFiles((current) => [...current, ...accepted]);
    }

    setErrors(nextErrors);
  }

  function removeSelectedFile(index) {
    if (uploading) {
      return;
    }

    setFiles((current) => current.filter((_, i) => i !== index));
    setErrors([]);

    if (inputRef.current) {
      inputRef.current.value = "";
    }
  }

  function clearSelection() {
    setFiles([]);
    setErrors([]);

    if (inputRef.current) {
      inputRef.current.value = "";
    }
  }

  async function submitUpload() {
    if (
      files.length === 0 ||
      uploading ||
      typeof onUpload !== "function"
    ) {
      return;
    }

    setErrors([]);

    try {
      const succeeded = await onUpload(files);

      if (succeeded) {
        clearSelection();
      } else {
        setErrors([
          "The PDFs could not be uploaded. Check the error message and try again.",
        ]);
      }
    } catch (error) {
      setErrors([
        error instanceof Error
          ? error.message
          : "The upload failed. Please try again.",
      ]);
    }
  }

  const attachTitle =
    slotsAfterSelection === 0
      ? `You can upload at most ${maxFiles} PDFs at a time. Upload these first.`
      : `Attach PDFs (up to ${slotsAfterSelection} more in this upload, 20 MiB each)`;

  return (
    <div className="composer-attachment">
      <input
        ref={inputRef}
        className="visually-hidden"
        type="file"
        accept=".pdf,application/pdf"
        multiple
        aria-label="Choose PDF documents"
        disabled={uploading || slotsAfterSelection === 0}
        onChange={(event) => {
          selectFiles(event.target.files);

          // Allow selecting the same file again after an error.
          event.target.value = "";
        }}
      />

      {/* PDFs already uploaded to this conversation, each with an X */}
      {documents.length > 0 && files.length === 0 && (
        <div className="active-attachments">
          <span className="active-attachment-label">
            {documents.length} {documents.length === 1 ? "PDF" : "PDFs"}
          </span>

          {documents.map((doc) => (
            <div
              key={doc.document_id}
              className="active-attachment"
              title={doc.filename || "Uploaded PDF"}
            >
              <FileText size={14} aria-hidden="true" />

              <strong>{doc.filename || "Uploaded document"}</strong>

              {typeof onDeleteDocument === "function" && (
                <button
                  className="active-attachment-remove"
                  type="button"
                  disabled={uploading || deletingId === doc.document_id}
                  aria-label={`Remove ${doc.filename || "this PDF"}`}
                  title="Remove this PDF"
                  onClick={() => onDeleteDocument(doc)}
                >
                  <X size={14} aria-hidden="true" />
                </button>
              )}
            </div>
          ))}
        </div>
      )}

      {/* Files chosen but not uploaded yet */}
      {files.length > 0 ? (
        <div className="selected-attachments">
          {files.map((file, index) => (
            <div
              key={`${file.name}-${file.size}-${index}`}
              className="selected-attachment"
            >
              <FileText size={15} aria-hidden="true" />

              <span
                className="selected-attachment-name"
                title={file.name}
              >
                {file.name}
              </span>

              <span className="selected-attachment-size">
                {formatFileSize(file.size)}
              </span>

              <button
                className="icon-button"
                type="button"
                disabled={uploading}
                aria-label={`Remove ${file.name}`}
                title={`Remove ${file.name}`}
                onClick={() => removeSelectedFile(index)}
              >
                <X size={15} aria-hidden="true" />
              </button>
            </div>
          ))}

          <div className="selected-attachments-actions">
            {slotsAfterSelection > 0 && (
              <button
                className="attachment-button"
                type="button"
                disabled={uploading}
                aria-label="Add more PDFs"
                title={attachTitle}
                onClick={() => inputRef.current?.click()}
              >
                <Paperclip size={16} aria-hidden="true" />
              </button>
            )}

            <button
              className="button button--primary attachment-upload"
              type="button"
              disabled={uploading || !onUpload}
              onClick={submitUpload}
            >
              {uploading ? (
                <>
                  <span className="spinner" aria-hidden="true" />
                  Uploading…
                </>
              ) : (
                <>
                  <FileUp size={14} aria-hidden="true" />
                  {files.length === 1
                    ? "Upload PDF"
                    : `Upload ${files.length} PDFs`}
                </>
              )}
            </button>
          </div>
        </div>
      ) : (
        <button
          className="attachment-button"
          type="button"
          disabled={uploading}
          aria-label={
            documents.length > 0 ? "Attach more PDFs" : "Attach PDFs"
          }
          title={attachTitle}
          onClick={() => inputRef.current?.click()}
        >
          <Paperclip size={16} aria-hidden="true" />
        </button>
      )}

      {errors.length > 0 && (
        <div className="inline-error attachment-error" role="alert">
          <AlertCircle size={15} aria-hidden="true" />

          <div>
            {errors.map((message) => (
              <p key={message}>{message}</p>
            ))}
          </div>
        </div>
      )}

      {uploading && (
        <p className="upload-note" role="status" aria-live="polite">
          Uploading and processing your PDFs. Large documents may take
          longer to process.
        </p>
      )}
    </div>
  );
}