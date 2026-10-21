import DocumentPanel from "./DocumentPanel.jsx";

export default function RightPanel({
  document,
  latestSources,
}) {
  return (
    <aside className="right-panel">
      <div className="right-panel-content">
        <DocumentPanel
          document={document}
          latestSources={latestSources}
        />
      </div>
    </aside>
  );
}