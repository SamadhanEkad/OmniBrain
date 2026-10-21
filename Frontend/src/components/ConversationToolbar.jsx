import { useEffect, useRef, useState } from "react";
import {
  Bookmark,
  BookmarkCheck,
  Copy,
  Ellipsis,
  PanelRightClose,
  PanelRightOpen,
  Pencil,
  Share2,
  Sun,
  Moon,
  Trash2,
} from "lucide-react";

async function copyText(text) {
  if (navigator.clipboard?.writeText) {
    await navigator.clipboard.writeText(text);
    return;
  }

  const field = document.createElement("textarea");
  field.value = text;
  field.setAttribute("readonly", "");
  field.style.position = "fixed";
  field.style.opacity = "0";
  document.body.append(field);
  field.select();
  const copied = document.execCommand("copy");
  field.remove();
  if (!copied) throw new Error("Clipboard access is unavailable in this browser.");
}

export function getConversationText(conversation) {
  const lines = [conversation.title || "OmniBrain conversation", ""];
  for (const message of conversation.messages || []) {
    if (message.role === "user" || (message.role === "assistant" && !message.pending && !message.error)) {
      lines.push(`${message.role === "user" ? "You" : "OmniBrain"}: ${message.content}`);
    }
  }
  return lines.join("\n\n");
}

export default function ConversationToolbar({
  conversation,
  theme,
  panelOpen,
  onToggleTheme,
  onTogglePanel,
  onToggleSaved,
  onShare,
  onCopy,
  onRename,
  onClear,
  onDelete,
  onRefreshHealth,
  backendStatus,
}) {
  const [menuOpen, setMenuOpen] = useState(false);
  const menuRef = useRef(null);

  useEffect(() => {
    if (!menuOpen) return undefined;
    function handlePointerDown(event) {
      if (!menuRef.current?.contains(event.target)) setMenuOpen(false);
    }
    function handleKeyDown(event) {
      if (event.key === "Escape") setMenuOpen(false);
    }
    document.addEventListener("pointerdown", handlePointerDown);
    document.addEventListener("keydown", handleKeyDown);
    return () => {
      document.removeEventListener("pointerdown", handlePointerDown);
      document.removeEventListener("keydown", handleKeyDown);
    };
  }, [menuOpen]);

  function runAction(action) {
    setMenuOpen(false);
    action();
  }

  return (
    <header className="workspace-header">
      <div className="conversation-title">
        <span className="eyebrow">OMNIBRAIN WORKSPACE</span>
        <strong>{conversation?.document ? "Document chat" : "New conversation"}</strong>
      </div>
      <div className="header-actions">
        <button className={`header-health header-health--${backendStatus}`} type="button" onClick={onRefreshHealth} title="Check backend connection again">
          <i />
          {backendStatus === "online" ? "Connected" : backendStatus === "checking" ? "Connecting" : "Offline · recheck"}
        </button>
        <span className="header-divider" />
        <button className="toolbar-icon" type="button" onClick={onShare} aria-label="Share conversation" title="Share conversation">
          <Share2 size={16} />
        </button>
        <button className={`toolbar-icon ${conversation?.saved ? "toolbar-icon--saved" : ""}`} type="button" onClick={onToggleSaved} aria-label={conversation?.saved ? "Remove bookmark" : "Bookmark conversation"} title={conversation?.saved ? "Bookmarked locally" : "Bookmark locally"}>
          {conversation?.saved ? <BookmarkCheck size={16} /> : <Bookmark size={16} />}
        </button>
        <div className="toolbar-menu-wrap" ref={menuRef}>
          <button className={`toolbar-icon ${menuOpen ? "is-open" : ""}`} type="button" onClick={() => setMenuOpen((open) => !open)} aria-label="More conversation actions" aria-haspopup="menu" aria-expanded={menuOpen}>
            <Ellipsis size={18} />
          </button>
          {menuOpen && (
            <div className="toolbar-menu" role="menu" aria-label="Conversation actions">
              <button type="button" role="menuitem" onClick={() => runAction(onRename)}><Pencil size={14} /> Rename conversation</button>
              <button type="button" role="menuitem" onClick={() => runAction(onCopy)}><Copy size={14} /> Copy conversation</button>
              <div className="menu-separator" />
              <button type="button" role="menuitem" onClick={() => runAction(onClear)}><Trash2 size={14} /> Clear messages</button>
              <button className="menu-danger" type="button" role="menuitem" onClick={() => runAction(onDelete)}><Trash2 size={14} /> Delete local conversation</button>
            </div>
          )}
        </div>
        <span className="header-divider header-theme-divider" />
        <button className="toolbar-icon theme-toggle" type="button" onClick={onToggleTheme} aria-label={`Switch to ${theme === "dark" ? "light" : "dark"} theme`} title={`Switch to ${theme === "dark" ? "light" : "dark"} theme`}>
          {theme === "dark" ? <Sun size={16} /> : <Moon size={16} />}
        </button>
        <button className="toolbar-icon panel-toggle" type="button" onClick={onTogglePanel} aria-label={panelOpen ? "Collapse side panel" : "Open side panel"} title={panelOpen ? "Collapse side panel" : "Open side panel"}>
          {panelOpen ? <PanelRightClose size={16} /> : <PanelRightOpen size={16} />}
        </button>
      </div>
    </header>
  );
}

export { copyText };
