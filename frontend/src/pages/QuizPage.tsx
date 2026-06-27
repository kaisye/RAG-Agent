import { useState } from "react"
import { QuizPanel } from "../components/quiz/QuizPanel"
import { useDocuments } from "../hooks/useDocuments"
import type { Document } from "../types/document"

export function QuizPage() {
  const { documents } = useDocuments()
  const ready         = documents.filter(d => d.status === "ready")
  const [selected, setSelected] = useState<Document | null>(null)
  const [jumpPage, setJumpPage] = useState(1)

  return (
    <div style={{ display: "flex", height: "100%", overflow: "hidden" }}>

      {/* ── Left: document picker ── */}
      <aside style={{
        width: 220, borderRight: "1px solid #e5e7eb",
        background: "#fafafa", display: "flex", flexDirection: "column",
        flexShrink: 0, overflow: "hidden",
      }}>
        <div style={{ padding: "12px", borderBottom: "1px solid #e5e7eb" }}>
          <p style={{ margin: 0, fontSize: 11, fontWeight: 700, color: "#374151",
            textTransform: "uppercase", letterSpacing: "0.06em" }}>
            Tài liệu ({ready.length})
          </p>
        </div>
        <div style={{ flex: 1, overflowY: "auto", padding: "8px" }}>
          {ready.length === 0 ? (
            <p style={{ fontSize: 12, color: "#9ca3af", padding: "8px 4px" }}>
              Chưa có tài liệu sẵn sàng. Tải lên ở trang Chat.
            </p>
          ) : (
            <ul style={{ listStyle: "none", margin: 0, padding: 0,
              display: "flex", flexDirection: "column", gap: 4 }}>
              {ready.map(doc => (
                <li key={doc.id}>
                  <button
                    onClick={() => { setSelected(doc); setJumpPage(1) }}
                    style={{
                      width: "100%", textAlign: "left", padding: "7px 10px",
                      borderRadius: 7, border: "1px solid",
                      borderColor:  selected?.id === doc.id ? "#93c5fd" : "#e5e7eb",
                      background:   selected?.id === doc.id ? "#eff6ff" : "#fff",
                      cursor:       "pointer", fontSize: 12, color: "#111827",
                      fontWeight:   selected?.id === doc.id ? 600 : 400,
                    }}
                  >
                    📄 {doc.filename.length > 22
                      ? doc.filename.slice(0, 21) + "…"
                      : doc.filename}
                  </button>
                </li>
              ))}
            </ul>
          )}
        </div>
      </aside>

      {/* ── Center: quiz ── */}
      <main style={{ flex: 1, overflowY: "auto", padding: "20px 24px", background: "#f3f4f6" }}>
        <div style={{ maxWidth: 720, margin: "0 auto" }}>
          <div style={{ marginBottom: 16 }}>
            <h1 style={{ margin: 0, fontSize: 18, fontWeight: 800, color: "#111827" }}>
              Quiz Generator
            </h1>
            <p style={{ margin: "4px 0 0", fontSize: 13, color: "#6b7280" }}>
              {selected
                ? `Tài liệu: ${selected.filename}`
                : "Chọn tài liệu ở bên trái để bắt đầu."}
            </p>
          </div>

          <div style={{
            background: "#fff", border: "1px solid #e5e7eb",
            borderRadius: 12, padding: "20px",
          }}>
            <QuizPanel
              documentId={selected?.id ?? ""}
              onJumpToPage={setJumpPage}
            />
          </div>

          {/* Page jump hint */}
          {jumpPage > 1 && (
            <div style={{
              marginTop: 12, padding: "8px 12px", borderRadius: 8,
              background: "#eff6ff", border: "1px solid #bfdbfe",
              fontSize: 12, color: "#1d4ed8",
            }}>
              Nguồn tham khảo: trang {jumpPage} — mở trang Chat để xem trong PDF viewer.
            </div>
          )}
        </div>
      </main>
    </div>
  )
}
