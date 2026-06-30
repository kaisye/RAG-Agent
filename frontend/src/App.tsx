import { useState } from "react"
import { ChatPage } from "./pages/ChatPage"
import { EvaluationPage } from "./pages/EvaluationPage"
import { QuizPage } from "./pages/QuizPage"

type Page = "chat" | "quiz" | "evaluation"

const NAV_ITEMS: { key: Page; label: string }[] = [
  { key: "chat",       label: "Chat" },
  { key: "quiz",       label: "Quiz" },
  { key: "evaluation", label: "Evaluation" },
]

export default function App() {
  const [page, setPage] = useState<Page>("chat")

  return (
    <div style={{ display: "flex", flexDirection: "column", height: "100vh", fontFamily: "sans-serif" }}>
      <nav style={{
        display: "flex", alignItems: "center", gap: 0,
        borderBottom: "1px solid #e5e7eb", padding: "0 1rem",
        background: "#fff", flexShrink: 0,
      }}>
        <span style={{ fontWeight: 700, fontSize: 15, marginRight: "auto", color: "#111" }}>
          RAG Agent — AIO2026
        </span>
        {NAV_ITEMS.map(({ key, label }) => (
          <button
            key={key}
            onClick={() => setPage(key)}
            style={{
              padding:       "0.75rem 1.25rem",
              border:        "none",
              background:    "none",
              cursor:        "pointer",
              fontWeight:    page === key ? 600 : 400,
              color:         page === key ? "#2563eb" : "#6b7280",
              borderBottom:  page === key ? "2px solid #2563eb" : "2px solid transparent",
              fontSize:      14,
            }}
          >
            {label}
          </button>
        ))}
      </nav>

      <main style={{ flex: 1, overflow: "hidden" }}>
        {page === "chat"       && <ChatPage />}
        {page === "quiz"       && <QuizPage />}
        {page === "evaluation" && <EvaluationPage />}
      </main>
    </div>
  )
}
