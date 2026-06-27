import { useEffect, useRef } from "react"
import type { Message } from "../../hooks/useChat"
import { CitationChip } from "./CitationChip"

interface Props {
  messages:      Message[]
  isStreaming:   boolean
  onJumpToPage?: (page: number) => void
}

export function MessageList({ messages, isStreaming, onJumpToPage }: Props) {
  const bottomRef = useRef<HTMLDivElement>(null)

  // Auto-scroll to bottom on new tokens
  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" })
  }, [messages])

  if (messages.length === 0) {
    return (
      <div style={{
        flex: 1, display: "flex", alignItems: "center", justifyContent: "center",
        color: "#9ca3af", flexDirection: "column", gap: 8,
      }}>
        <span style={{ fontSize: 32 }}>💬</span>
        <p style={{ margin: 0, fontSize: 14 }}>Chọn tài liệu và đặt câu hỏi để bắt đầu</p>
      </div>
    )
  }

  return (
    <div style={{ flex: 1, overflowY: "auto", padding: "1rem", display: "flex", flexDirection: "column", gap: 12 }}>
      {messages.map((msg, i) => (
        <div
          key={i}
          style={{
            display:       "flex",
            justifyContent: msg.role === "user" ? "flex-end" : "flex-start",
          }}
        >
          <div
            style={{
              maxWidth:     "80%",
              borderRadius: msg.role === "user" ? "18px 18px 4px 18px" : "18px 18px 18px 4px",
              padding:      "10px 14px",
              background:   msg.role === "user" ? "#2563eb" : "#f3f4f6",
              color:        msg.role === "user" ? "#fff" : "#111827",
              fontSize:     14,
              lineHeight:   1.55,
              whiteSpace:   "pre-wrap",
              wordBreak:    "break-word",
            }}
          >
            {/* Content */}
            {msg.content}

            {/* Streaming cursor */}
            {isStreaming && i === messages.length - 1 && msg.role === "assistant" && (
              <span style={{ display: "inline-block", width: 8, height: 14,
                background: "#6b7280", borderRadius: 2, marginLeft: 3,
                animation: "blink 1s step-end infinite",
                verticalAlign: "text-bottom" }}
              />
            )}

            {/* Citations row */}
            {msg.citations.length > 0 && (
              <div style={{ marginTop: 8, borderTop: "1px solid rgba(0,0,0,0.08)", paddingTop: 6 }}>
                {msg.citations.map((c, ci) => (
                  <CitationChip key={ci} citation={c} onJumpToPage={onJumpToPage} />
                ))}
              </div>
            )}
          </div>
        </div>
      ))}
      <div ref={bottomRef} />

      <style>{`
        @keyframes blink {
          0%, 100% { opacity: 1; }
          50%       { opacity: 0; }
        }
      `}</style>
    </div>
  )
}
