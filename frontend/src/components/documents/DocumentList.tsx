import type { Document } from "../../types/document"

interface Props {
  documents:        Document[]
  selectedId?:      string
  onSelect:         (doc: Document) => void
  onDelete:         (id: string) => void
  loadError?:       string | null
}

type Status = Document["status"]

const STATUS_CONFIG: Record<Status, { label: string; bg: string; color: string; dot: string }> = {
  uploaded:   { label: "Uploaded",   bg: "#f3f4f6", color: "#6b7280", dot: "#9ca3af" },
  processing: { label: "Processing", bg: "#fefce8", color: "#ca8a04", dot: "#facc15" },
  ready:      { label: "Ready",      bg: "#f0fdf4", color: "#16a34a", dot: "#4ade80" },
  error:      { label: "Error",      bg: "#fef2f2", color: "#dc2626", dot: "#f87171" },
}

function StatusBadge({ status }: { status: Status }) {
  const cfg = STATUS_CONFIG[status] ?? STATUS_CONFIG.error
  return (
    <span style={{
      display: "inline-flex", alignItems: "center", gap: 4,
      fontSize: 10, fontWeight: 600, padding: "2px 7px", borderRadius: 10,
      background: cfg.bg, color: cfg.color,
    }}>
      {/* Animated dot for processing */}
      <span style={{
        width: 6, height: 6, borderRadius: "50%", background: cfg.dot,
        display: "inline-block",
        animation: status === "processing" ? "pulse 1.5s ease-in-out infinite" : "none",
      }} />
      {cfg.label}
      <style>{`
        @keyframes pulse {
          0%, 100% { opacity: 1; }
          50%       { opacity: 0.3; }
        }
      `}</style>
    </span>
  )
}

function truncate(s: string, n: number) {
  return s.length > n ? s.slice(0, n - 1) + "…" : s
}

export function DocumentList({ documents, selectedId, onSelect, onDelete, loadError }: Props) {
  if (loadError) {
    return (
      <div style={{ fontSize: 12, color: "#dc2626", padding: "8px 4px" }}>
        ⚠ {loadError}
      </div>
    )
  }

  if (documents.length === 0) {
    return (
      <p style={{ fontSize: 12, color: "#9ca3af", margin: "8px 0" }}>
        Chưa có tài liệu nào.
      </p>
    )
  }

  return (
    <ul style={{ listStyle: "none", margin: 0, padding: 0, display: "flex", flexDirection: "column", gap: 4 }}>
      {documents.map(doc => {
        const isSelected = doc.id === selectedId
        const isReady    = doc.status === "ready"

        return (
          <li
            key={doc.id}
            style={{
              borderRadius: 8,
              border:       `1px solid ${isSelected ? "#93c5fd" : "#e5e7eb"}`,
              background:   isSelected ? "#eff6ff" : "#fff",
              overflow:     "hidden",
            }}
          >
            {/* Main row */}
            <div style={{ display: "flex", alignItems: "center", padding: "7px 10px", gap: 8 }}>
              {/* PDF icon */}
              <span style={{ fontSize: 18, flexShrink: 0 }}>📄</span>

              {/* Info */}
              <div style={{ flex: 1, minWidth: 0 }}>
                <p style={{
                  margin: 0, fontSize: 12, fontWeight: 500,
                  color: "#111827", overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap",
                }}>
                  {truncate(doc.filename, 28)}
                </p>
                <div style={{ marginTop: 3, display: "flex", alignItems: "center", gap: 6 }}>
                  <StatusBadge status={doc.status} />
                </div>
              </div>

              {/* Action buttons */}
              <div style={{ display: "flex", gap: 4, flexShrink: 0 }}>
                <button
                  onClick={() => isReady && onSelect(doc)}
                  disabled={!isReady}
                  title={isReady ? "Chọn tài liệu này" : "Đang xử lý…"}
                  style={{
                    border:     "none",
                    borderRadius: 6,
                    padding:    "3px 8px",
                    fontSize:   11,
                    cursor:     isReady ? "pointer" : "not-allowed",
                    background: isSelected ? "#2563eb" : isReady ? "#e0f2fe" : "#f3f4f6",
                    color:      isSelected ? "#fff"    : isReady ? "#0369a1" : "#9ca3af",
                    fontWeight: 500,
                    transition: "background 0.15s",
                  }}
                >
                  {isSelected ? "✓ Đang dùng" : "Chọn"}
                </button>

                <button
                  onClick={() => {
                    if (window.confirm(`Xóa "${doc.filename}"?`)) onDelete(doc.id)
                  }}
                  title="Xóa tài liệu"
                  style={{
                    border:       "none",
                    borderRadius: 6,
                    width:        24,
                    height:       24,
                    display:      "flex",
                    alignItems:   "center",
                    justifyContent: "center",
                    cursor:       "pointer",
                    background:   "transparent",
                    color:        "#9ca3af",
                    fontSize:     14,
                    transition:   "background 0.15s, color 0.15s",
                  }}
                  onMouseEnter={e => {
                    (e.currentTarget as HTMLButtonElement).style.background = "#fef2f2"
                    ;(e.currentTarget as HTMLButtonElement).style.color = "#dc2626"
                  }}
                  onMouseLeave={e => {
                    (e.currentTarget as HTMLButtonElement).style.background = "transparent"
                    ;(e.currentTarget as HTMLButtonElement).style.color = "#9ca3af"
                  }}
                >
                  ×
                </button>
              </div>
            </div>
          </li>
        )
      })}
    </ul>
  )
}
