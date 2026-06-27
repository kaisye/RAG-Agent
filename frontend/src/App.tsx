import { useState } from "react"
import { ChatPage } from "./pages/ChatPage"
import { EvaluationPage } from "./pages/EvaluationPage"

type Page = "chat" | "evaluation"

export default function App() {
  const [page, setPage] = useState<Page>("chat")

  return (
    <div style={{ display: "flex", flexDirection: "column", height: "100vh", fontFamily: "sans-serif" }}>
      {/* Nav */}
      <nav style={{
        display: "flex", alignItems: "center", gap: 0,
        borderBottom: "1px solid #e5e7eb", padding: "0 1rem", background: "#fff",
      }}>
        <span style={{ fontWeight: 700, fontSize: 15, marginRight: "auto", color: "#111" }}>
          RAG Insight — AIO2026
        </span>
        {(["chat", "evaluation"] as Page[]).map(p => (
          <button
            key={p}
            onClick={() => setPage(p)}
            style={{
              padding: "0.75rem 1.25rem",
              border: "none", background: "none", cursor: "pointer",
              fontWeight: page === p ? 600 : 400,
              color: page === p ? "#2563eb" : "#6b7280",
              borderBottom: page === p ? "2px solid #2563eb" : "2px solid transparent",
              textTransform: "capitalize",
            }}
          >
            {p === "chat" ? "Chat" : "Evaluation"}
          </button>
        ))}
      </nav>

      {/* Page content */}
      <main style={{ flex: 1, overflow: "hidden" }}>
        {page === "chat" ? <ChatPage /> : <EvaluationPage />}
      </main>
    </div>
  )
}
