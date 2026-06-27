import { useState } from "react"
import { useChat } from "../hooks/useChat"
import { useDocuments } from "../hooks/useDocuments"
import { MessageList } from "../components/chat/MessageList"
import { MessageInput } from "../components/chat/MessageInput"
import { FileUploadPanel } from "../components/documents/FileUploadPanel"
import { DocumentList } from "../components/documents/DocumentList"
import { DEFAULT_CONFIG } from "../types/pipeline"
import type { PipelineConfig } from "../types/pipeline"
import type { Document } from "../types/document"

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
      <span style={{ fontSize: 32 }}>📑</span>
      <p style={{ margin: 0, fontSize: 13, fontWeight: 500 }}>Trang {currentPage}</p>
      <p style={{ margin: "2px 0 0", fontSize: 11, opacity: 0.6, wordBreak: "break-all",
        maxWidth: 200, textAlign: "center" }}>{documentId}</p>
    </div>
  )
}

export function ChatPage() {
  const [selectedDoc, setSelectedDoc]   = useState<Document | null>(null)
  const [config, setConfig]             = useState<PipelineConfig>(DEFAULT_CONFIG)
  const [currentPage, setCurrentPage]   = useState(1)

  const { documents, uploading, uploadError, loadError,
          uploadDocument, deleteDocument, clearUploadError } = useDocuments()

  const documentId = selectedDoc?.id ?? ""
  const { messages, sendMessage, isStreaming, error: chatError, clearHistory } = useChat(documentId, config)

  const handleSelect = (doc: Document) => {
    setSelectedDoc(doc)
    clearHistory()
    setCurrentPage(1)
  }

  const handleDelete = async (id: string) => {
    await deleteDocument(id)
    if (selectedDoc?.id === id) {
      setSelectedDoc(null)
      clearHistory()
    }
  }

  return (
    <div style={{ display: "flex", height: "100%", overflow: "hidden" }}>

      {/* ── Left sidebar ── */}
      <aside style={{
        width: 280, borderRight: "1px solid #e5e7eb",
        display: "flex", flexDirection: "column",
        background: "#fafafa", flexShrink: 0, overflow: "hidden",
      }}>
        {/* Upload area */}
        <div style={{ padding: "12px", borderBottom: "1px solid #e5e7eb" }}>
          <p style={{ margin: "0 0 8px", fontSize: 12, fontWeight: 600, color: "#374151" }}>
            TẢI LÊN PDF
          </p>
          <FileUploadPanel
            onUpload={uploadDocument}
            uploading={uploading}
            error={uploadError}
            onClearError={clearUploadError}
          />
        </div>

        {/* Document list */}
        <div style={{ flex: 1, overflowY: "auto", padding: "12px" }}>
          <p style={{ margin: "0 0 8px", fontSize: 12, fontWeight: 600, color: "#374151" }}>
            TÀI LIỆU ({documents.length})
          </p>
          <DocumentList
            documents={documents}
            selectedId={selectedDoc?.id}
            onSelect={handleSelect}
            onDelete={handleDelete}
            loadError={loadError}
          />
        </div>

        {/* Pipeline config */}
        <div style={{ padding: "12px", borderTop: "1px solid #e5e7eb", overflowY: "auto", maxHeight: 280 }}>
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
            <div key={key} style={{ marginBottom: 8 }}>
              <label style={{ fontSize: 11, color: "#6b7280", display: "block", marginBottom: 2 }}>
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

          {messages.length > 0 && (
            <button
              onClick={clearHistory}
              style={{
                marginTop: 8, width: "100%", fontSize: 12, padding: "5px",
                borderRadius: 6, border: "1px solid #e5e7eb",
                background: "#fff", cursor: "pointer", color: "#6b7280",
              }}
            >
              Xóa lịch sử
            </button>
          )}
        </div>
      </aside>

      {/* ── Center: chat ── */}
      <main style={{ flex: 1, display: "flex", flexDirection: "column", minWidth: 0 }}>
        {chatError && (
          <div style={{
            padding: "8px 16px", background: "#fef2f2", borderBottom: "1px solid #fecaca",
            fontSize: 13, color: "#dc2626", flexShrink: 0,
          }}>
            ⚠ {chatError}
          </div>
        )}

        {/* Selected doc indicator */}
        {selectedDoc && (
          <div style={{
            padding: "6px 16px", borderBottom: "1px solid #e5e7eb",
            fontSize: 12, color: "#059669", background: "#f0fdf4", flexShrink: 0,
          }}>
            📄 {selectedDoc.filename}
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
              ? "Chọn tài liệu ở bên trái trước…"
              : "Đặt câu hỏi về tài liệu… (Enter gửi)"
          }
        />
      </main>

      {/* ── Right: PDF viewer placeholder ── */}
      <aside style={{
        width: 360, borderLeft: "1px solid #e5e7eb",
        flexShrink: 0, background: "#f9fafb", display: "flex", flexDirection: "column",
      }}>
        <div style={{
          padding: "10px 12px", borderBottom: "1px solid #e5e7eb",
          fontSize: 12, fontWeight: 600, color: "#374151", flexShrink: 0,
        }}>
          PDF VIEWER {currentPage > 1 && `— Trang ${currentPage}`}
        </div>
        <div style={{ flex: 1 }}>
          <PdfPlaceholder documentId={documentId} currentPage={currentPage} />
        </div>
      </aside>
    </div>
  )
}
