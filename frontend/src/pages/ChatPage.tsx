import { useState } from "react"
import { useChat } from "../hooks/useChat"
import { MessageList } from "../components/chat/MessageList"
import { MessageInput } from "../components/chat/MessageInput"
import { DEFAULT_CONFIG } from "../types/pipeline"
import type { PipelineConfig } from "../types/pipeline"

// Placeholder — sẽ thay bằng react-pdf PdfViewerPanel trong 11.3
function PdfPlaceholder({ documentId, currentPage }: { documentId: string; currentPage: number }) {
  if (!documentId) {
    return (
      <div style={{ display: "flex", flexDirection: "column", alignItems: "center",
        justifyContent: "center", height: "100%", color: "#9ca3af", gap: 8 }}>
        <span style={{ fontSize: 40 }}>📄</span>
        <p style={{ margin: 0, fontSize: 13 }}>Chọn tài liệu để xem PDF</p>
      </div>
    )
  }
  return (
    <div style={{ display: "flex", flexDirection: "column", alignItems: "center",
      justifyContent: "center", height: "100%", color: "#6b7280", gap: 4 }}>
      <span style={{ fontSize: 24 }}>📑</span>
      <p style={{ margin: 0, fontSize: 13 }}>Trang {currentPage}</p>
      <p style={{ margin: 0, fontSize: 11, opacity: 0.7 }}>{documentId}</p>
    </div>
  )
}

export function ChatPage() {
  const [documentId, setDocumentId] = useState("")
  const [config, setConfig]         = useState<PipelineConfig>(DEFAULT_CONFIG)
  const [currentPage, setCurrentPage] = useState(1)
  const [docInput, setDocInput]       = useState("")

  const { messages, sendMessage, isStreaming, error, clearHistory } = useChat(documentId, config)

  const handleSelectDoc = () => {
    const id = docInput.trim()
    if (!id) return
    setDocumentId(id)
    clearHistory()
    setCurrentPage(1)
  }

  return (
    <div style={{ display: "flex", height: "100%", overflow: "hidden" }}>

      {/* ── Left sidebar ── */}
      <aside style={{
        width: 280, borderRight: "1px solid #e5e7eb",
        display: "flex", flexDirection: "column", gap: 0,
        background: "#fafafa", flexShrink: 0,
      }}>
        {/* Document selector */}
        <div style={{ padding: "16px 12px", borderBottom: "1px solid #e5e7eb" }}>
          <p style={{ margin: "0 0 8px", fontSize: 12, fontWeight: 600, color: "#374151" }}>
            TÀI LIỆU
          </p>
          <div style={{ display: "flex", gap: 6 }}>
            <input
              value={docInput}
              onChange={e => setDocInput(e.target.value)}
              onKeyDown={e => e.key === "Enter" && handleSelectDoc()}
              placeholder="Document ID…"
              style={{
                flex: 1, fontSize: 12, border: "1px solid #d1d5db",
                borderRadius: 6, padding: "5px 8px", outline: "none",
              }}
            />
            <button
              onClick={handleSelectDoc}
              style={{
                fontSize: 11, padding: "5px 10px", borderRadius: 6,
                border: "none", background: "#2563eb", color: "#fff",
                cursor: "pointer", whiteSpace: "nowrap",
              }}
            >
              Chọn
            </button>
          </div>
          {documentId && (
            <p style={{ margin: "6px 0 0", fontSize: 11, color: "#059669" }}>
              ✓ {documentId}
            </p>
          )}
        </div>

        {/* Pipeline config */}
        <div style={{ padding: "12px", flex: 1, overflowY: "auto" }}>
          <p style={{ margin: "0 0 10px", fontSize: 12, fontWeight: 600, color: "#374151" }}>
            PIPELINE CONFIG
          </p>
          {(
            [
              { key: "chunking_strategy",  label: "Chunking",
                options: [["recursive","Recursive"],["semantic","Semantic ★"]] },
              { key: "retrieval_strategy", label: "Retrieval",
                options: [["vector","Vector"],["hybrid_rrf","Hybrid RRF ★"],
                          ["hybrid_interleaving","Hybrid Xen kẽ"],["bm25","BM25"]] },
              { key: "query_transform",    label: "Query Transform",
                options: [["none","Không"],["decomposition","Decomposition ★"],["hyde","HyDE"]] },
              { key: "rerank_strategy",    label: "Reranking",
                options: [["none","Không"],["cross_encoder","Cross-Encoder"],["mmr","MMR"]] },
            ] as { key: keyof PipelineConfig; label: string; options: [string,string][] }[]
          ).map(({ key, label, options }) => (
            <div key={key} style={{ marginBottom: 10 }}>
              <label style={{ fontSize: 11, color: "#6b7280", display: "block", marginBottom: 3 }}>
                {label}
              </label>
              <select
                value={config[key] as string}
                onChange={e => setConfig(c => ({ ...c, [key]: e.target.value }))}
                disabled={isStreaming}
                style={{
                  width: "100%", fontSize: 12, border: "1px solid #d1d5db",
                  borderRadius: 6, padding: "4px 6px", background: "#fff",
                  cursor: isStreaming ? "not-allowed" : "pointer",
                }}
              >
                {options.map(([v, l]) => <option key={v} value={v}>{l}</option>)}
              </select>
            </div>
          ))}
        </div>

        {/* Clear button */}
        {messages.length > 0 && (
          <div style={{ padding: "10px 12px", borderTop: "1px solid #e5e7eb" }}>
            <button
              onClick={clearHistory}
              style={{
                width: "100%", fontSize: 12, padding: "6px", borderRadius: 6,
                border: "1px solid #e5e7eb", background: "#fff", cursor: "pointer",
                color: "#6b7280",
              }}
            >
              Xóa lịch sử
            </button>
          </div>
        )}
      </aside>

      {/* ── Center: chat ── */}
      <main style={{ flex: 1, display: "flex", flexDirection: "column", minWidth: 0 }}>
        {error && (
          <div style={{
            padding: "8px 16px", background: "#fef2f2", borderBottom: "1px solid #fecaca",
            fontSize: 13, color: "#dc2626",
          }}>
            ⚠ {error}
          </div>
        )}

        <MessageList
          messages={messages}
          isStreaming={isStreaming}
          onJumpToPage={setCurrentPage}
        />

        <MessageInput
          onSend={sendMessage}
          disabled={isStreaming || !documentId}
          placeholder={
            !documentId
              ? "Chọn tài liệu trước…"
              : "Đặt câu hỏi về tài liệu… (Enter gửi)"
          }
        />
      </main>

      {/* ── Right: PDF viewer placeholder ── */}
      <aside style={{
        width: 360, borderLeft: "1px solid #e5e7eb",
        flexShrink: 0, background: "#f9fafb",
      }}>
        <div style={{
          padding: "10px 12px", borderBottom: "1px solid #e5e7eb",
          fontSize: 12, fontWeight: 600, color: "#374151",
        }}>
          PDF VIEWER
        </div>
        <div style={{ height: "calc(100% - 41px)" }}>
          <PdfPlaceholder documentId={documentId} currentPage={currentPage} />
        </div>
      </aside>
    </div>
  )
}
