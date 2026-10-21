import { BookmarkCheck, Clock3, FileText, MessageSquare, Trash2 } from "lucide-react";

function previewConversation(conversation) {
  const lastMessage = [...(conversation.messages || [])]
    .reverse()
    .find((message) => message.role === "user");
  return lastMessage?.content || conversation.document?.filename || "No messages yet";
}

function formatDate(timestamp) {
  const date = new Date(timestamp);
  if (Number.isNaN(date.getTime())) return "Recently";
  const today = new Date();
  if (date.toDateString() === today.toDateString()) {
    return date.toLocaleTimeString([], { hour: "numeric", minute: "2-digit" });
  }
  return date.toLocaleDateString([], { month: "short", day: "numeric" });
}

export default function HistoryPanel({
  conversations,
  activeConversationId,
  onSelect,
  onDeleteConversation,
  onDeleteHistory,
}) {
  const visibleConversations = [...conversations]
    .filter((conversation) => conversation.messages?.length || conversation.document || conversation.saved)
    .sort((first, second) => (second.updatedAt || 0) - (first.updatedAt || 0));

  return (
    <div className="history-panel">
      <div className="history-heading">
        <div>
          <div className="eyebrow">THIS BROWSER ONLY</div>
          <h2>Recent conversations</h2>
        </div>
        <button className="history-delete-all" type="button" onClick={onDeleteHistory} disabled={!visibleConversations.length} title="Delete locally saved conversation history">
          <Trash2 size={14} />
          <span>Clear all</span>
        </button>
      </div>
      <p className="history-note">Saved in this browser. This API has no conversation history service.</p>
      {visibleConversations.length ? (
        <div className="history-list">
          {visibleConversations.map((conversation) => (
            <article className={`history-card ${conversation.id === activeConversationId ? "is-active" : ""}`} key={conversation.id}>
              <button className="history-select" type="button" onClick={() => onSelect(conversation.id)} aria-current={conversation.id === activeConversationId ? "page" : undefined}>
                <span className="history-card-icon">
                  {conversation.document ? <FileText size={15} /> : <MessageSquare size={15} />}
                </span>
                <span className="history-card-content">
                  <strong>{conversation.title || "New chat"}</strong>
                  <span>{previewConversation(conversation)}</span>
                  <small>
                    <Clock3 size={11} />
                    {formatDate(conversation.updatedAt)}
                    {conversation.saved && <><span className="history-dot">·</span><BookmarkCheck size={11} /> Bookmarked</>}
                  </small>
                </span>
              </button>
              <button className="history-delete-one" type="button" onClick={() => onDeleteConversation(conversation.id)} aria-label={`Delete ${conversation.title || "conversation"} from local history`} title="Delete from local history">
                <Trash2 size={13} />
              </button>
            </article>
          ))}
        </div>
      ) : (
        <div className="history-empty">
          <MessageSquare size={20} />
          <strong>No recent conversations</strong>
          <span>Start a chat and it will appear here.</span>
        </div>
      )}
    </div>
  );
}
