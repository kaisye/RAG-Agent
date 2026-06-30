import { useState } from "react"
import { useChat } from "../hooks/useChat"
import { useDocuments } from "../hooks/useDocuments"
import { MessageList } from "../components/chat/MessageList"
import { MessageInput } from "../components/chat/MessageInput"
import { FileUploadPanel } from "../components/documents/FileUploadPanel"
import { DocumentList } from "../components/documents/DocumentList"
import { PdfViewerPanel } from "../components/pdf/PdfViewerPanel"
import { PipelineConfigPanel } from "../components/config/PipelineConfigPanel"
import { DEFAULT_CONFIG } from "../types/pipeline"
import type { PipelineConfig } from "../types/pipeline"
import type { Document } from "../types/document"

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

        {/* Pipeline config — component with RAGAS numbers */}
        <div style={{ overflowY: "auto", maxHeight: 320, flexShrink: 0 }}>
          <PipelineConfigPanel
            config={config}
            onChange={setConfig}
            disabled={isStreaming}
          />
        </div>

        {/* Clear history */}
        {messages.length > 0 && (
          <div style={{ padding: "8px 12px", borderTop: "1px solid #e5e7eb", flexShrink: 0 }}>
            <button
              onClick={clearHistory}
              style={{
                width: "100%", fontSize: 12, padding: "5px",
                borderRadius: 6, border: "1px solid #e5e7eb",
                background: "#fff", cursor: "pointer", color: "#6b7280",
              }}
            >
              Xóa lịch sử
            </button>
          </div>
        )}
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

      {/* ── Right: PDF viewer ── */}
      <aside style={{
        width: 400, borderLeft: "1px solid #e5e7eb",
        flexShrink: 0, display: "flex", flexDirection: "column", overflow: "hidden",
      }}>
        <div style={{
          padding: "10px 12px", borderBottom: "1px solid #e5e7eb",
          fontSize: 12, fontWeight: 600, color: "#374151", flexShrink: 0,
          background: "#f9fafb",
        }}>
          PDF VIEWER
        </div>
        <div style={{ flex: 1, overflow: "hidden" }}>
          <PdfViewerPanel
            fileUrl={documentId ? `/static/uploads/${documentId}.pdf` : ""}
            currentPage={currentPage}
            onPageChange={setCurrentPage}
          />
        </div>
      </aside>
    </div>
  )
}
