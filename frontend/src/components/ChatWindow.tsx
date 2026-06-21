import { useState } from 'react'
import { useDocuments } from '../hooks/useDocuments'
import { useChat } from '../hooks/useChat'
import { FileUploadPanel } from './FileUploadPanel'
import { MessageList } from './MessageList'
import { MessageInput } from './MessageInput'
import { PdfViewerPanel } from './PdfViewerPanel'

export function ChatWindow() {
  const [selectedDocId, setSelectedDocId] = useState<string>('')
  const [currentPage, setCurrentPage] = useState<number>(1)   // 1-based, for react-pdf

  const { documents, uploading, error: uploadError, upload, remove } = useDocuments()
  const { messages, streaming, error: chatError, sendMessage, stop, clear } = useChat(selectedDocId)

  function handleSelectDoc(id: string) {
    setSelectedDocId(id)
    setCurrentPage(1)
    clear()
  }

  function handleJumpToPage(page: number) {
    setCurrentPage(page)
  }

  const selectedDoc = documents.find(d => d.id === selectedDocId)
  const pdfReady = selectedDoc?.status === 'ready'

  return (
    <div style={{ display: 'flex', height: '100vh', fontFamily: 'system-ui, sans-serif', overflow: 'hidden' }}>

      {/* ── Left sidebar: document list ── */}
      <FileUploadPanel
        documents={documents}
        uploading={uploading}
        error={uploadError}
        selectedId={selectedDocId}
        onUpload={upload}
        onSelect={handleSelectDoc}
        onDelete={remove}
      />

      {/* ── Center: chat ── */}
      <div style={{ flex: '0 0 420px', display: 'flex', flexDirection: 'column', borderRight: '1px solid #e0e0e0', overflow: 'hidden' }}>
        {/* Header */}
        <div style={{ padding: '12px 16px', borderBottom: '1px solid #e0e0e0', display: 'flex', alignItems: 'center', gap: '12px', flexShrink: 0 }}>
          <h2 style={{ margin: 0, fontSize: '15px', fontWeight: 600, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
            {selectedDoc ? selectedDoc.filename : 'RAG PDF Chat'}
          </h2>
          {selectedDocId && (
            <button
              onClick={clear}
              style={{ marginLeft: 'auto', flexShrink: 0, padding: '4px 10px', borderRadius: '6px', border: '1px solid #ccc', background: '#fff', cursor: 'pointer', fontSize: '12px' }}
            >
              Clear
            </button>
          )}
        </div>

        {chatError && (
          <div style={{ padding: '8px 16px', background: '#fdecea', color: '#c62828', fontSize: '13px', flexShrink: 0 }}>
            Lỗi: {chatError}
          </div>
        )}

        <MessageList
          messages={messages}
          streaming={streaming}
          onJumpToPage={handleJumpToPage}
        />

        <MessageInput
          onSend={sendMessage}
          onStop={stop}
          disabled={!selectedDocId}
          streaming={streaming}
        />
      </div>

      {/* ── Right: PDF viewer ── */}
      <div style={{ flex: 1, display: 'flex', flexDirection: 'column', overflow: 'hidden' }}>
        {/* Toolbar */}
        <div style={{ padding: '8px 16px', borderBottom: '1px solid #e0e0e0', display: 'flex', alignItems: 'center', gap: '8px', flexShrink: 0, background: '#fff' }}>
          <span style={{ fontSize: '13px', color: '#555' }}>
            {pdfReady ? `Trang ${currentPage}` : selectedDocId ? 'Đang xử lý tài liệu…' : 'Chưa chọn tài liệu'}
          </span>
        </div>

        <div style={{ flex: 1, overflow: 'hidden' }}>
          {pdfReady ? (
            <PdfViewerPanel
              documentId={selectedDocId}
              currentPage={currentPage}
            />
          ) : (
            <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', height: '100%', color: '#aaa', fontSize: '14px' }}>
              {selectedDocId ? 'Đợi tài liệu sẵn sàng (ready)…' : 'Chọn hoặc upload một tài liệu PDF để xem.'}
            </div>
          )}
        </div>
      </div>
    </div>
  )
}
