import { useState } from 'react'
import { useDocuments } from '../hooks/useDocuments'
import { useChat } from '../hooks/useChat'
import { FileUploadPanel } from './FileUploadPanel'
import { MessageList } from './MessageList'
import { MessageInput } from './MessageInput'

export function ChatWindow() {
  const [selectedDocId, setSelectedDocId] = useState<string>('')
  const { documents, uploading, error: uploadError, upload, remove } = useDocuments()
  const { messages, streaming, error: chatError, sendMessage, stop, clear } = useChat(selectedDocId)

  function handleSelectDoc(id: string) {
    setSelectedDocId(id)
    clear()
  }

  return (
    <div style={{ display: 'flex', height: '100vh', fontFamily: 'system-ui, sans-serif' }}>
      {/* Left sidebar */}
      <FileUploadPanel
        documents={documents}
        uploading={uploading}
        error={uploadError}
        selectedId={selectedDocId}
        onUpload={upload}
        onSelect={handleSelectDoc}
        onDelete={remove}
      />

      {/* Chat area */}
      <div style={{ flex: 1, display: 'flex', flexDirection: 'column', overflow: 'hidden' }}>
        {/* Header */}
        <div style={{ padding: '12px 16px', borderBottom: '1px solid #e0e0e0', display: 'flex', alignItems: 'center', gap: '12px' }}>
          <h2 style={{ margin: 0, fontSize: '16px', fontWeight: 600 }}>
            {selectedDocId
              ? documents.find(d => d.id === selectedDocId)?.filename ?? 'Chat'
              : 'RAG PDF Chat'}
          </h2>
          {selectedDocId && (
            <button
              onClick={clear}
              style={{ marginLeft: 'auto', padding: '4px 10px', borderRadius: '6px', border: '1px solid #ccc', background: '#fff', cursor: 'pointer', fontSize: '12px' }}
            >
              Clear chat
            </button>
          )}
        </div>

        {chatError && (
          <div style={{ padding: '8px 16px', background: '#fdecea', color: '#c62828', fontSize: '13px' }}>
            Error: {chatError}
          </div>
        )}

        <MessageList messages={messages} streaming={streaming} />

        <MessageInput
          onSend={sendMessage}
          onStop={stop}
          disabled={!selectedDocId}
          streaming={streaming}
        />
      </div>
    </div>
  )
}
