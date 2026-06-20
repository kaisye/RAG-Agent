import { useRef } from 'react'
import type { Document } from '../types'

interface Props {
  documents: Document[]
  uploading: boolean
  error: string | null
  selectedId: string
  onUpload: (file: File) => void
  onSelect: (id: string) => void
  onDelete: (id: string) => void
}

const STATUS_LABEL: Record<string, string> = {
  pending:   '⏳ pending',
  uploaded:  '⏳ uploaded',
  parsing:   '⏳ parsing…',
  chunking:  '⏳ chunking…',
  embedding: '⏳ embedding…',
  ready:     '✅ ready',
  failed:    '❌ failed',
  error:     '❌ error',
}

export function FileUploadPanel({ documents, uploading, error, selectedId, onUpload, onSelect, onDelete }: Props) {
  const inputRef = useRef<HTMLInputElement>(null)

  function handleFile(e: React.ChangeEvent<HTMLInputElement>) {
    const file = e.target.files?.[0]
    if (file) onUpload(file)
    e.target.value = ''
  }

  return (
    <div style={{ padding: '12px', borderRight: '1px solid #e0e0e0', width: '260px', display: 'flex', flexDirection: 'column', gap: '8px' }}>
      <h3 style={{ margin: 0, fontSize: '14px', fontWeight: 600 }}>Documents</h3>

      <button
        onClick={() => inputRef.current?.click()}
        disabled={uploading}
        style={{ padding: '8px', cursor: 'pointer', borderRadius: '6px', border: '1px dashed #999', background: '#fafafa' }}
      >
        {uploading ? 'Uploading…' : '+ Upload PDF'}
      </button>
      <input ref={inputRef} type="file" accept=".pdf" style={{ display: 'none' }} onChange={handleFile} />

      {error && <p style={{ color: 'red', fontSize: '12px', margin: 0 }}>{error}</p>}

      <ul style={{ listStyle: 'none', padding: 0, margin: 0, flex: 1, overflowY: 'auto' }}>
        {documents.map(doc => (
          <li
            key={doc.id}
            onClick={() => doc.status === 'ready' && onSelect(doc.id)}
            style={{
              padding: '8px',
              borderRadius: '6px',
              cursor: doc.status === 'ready' ? 'pointer' : 'default',
              background: selectedId === doc.id ? '#e8f0fe' : 'transparent',
              marginBottom: '4px',
              display: 'flex',
              justifyContent: 'space-between',
              alignItems: 'center',
            }}
          >
            <div style={{ overflow: 'hidden' }}>
              <div style={{ fontSize: '13px', fontWeight: 500, whiteSpace: 'nowrap', overflow: 'hidden', textOverflow: 'ellipsis', maxWidth: '170px' }}>
                {doc.filename}
              </div>
              <div style={{ fontSize: '11px', color: '#666' }}>{STATUS_LABEL[doc.status] ?? doc.status}</div>
            </div>
            <button
              onClick={e => { e.stopPropagation(); onDelete(doc.id) }}
              style={{ background: 'none', border: 'none', cursor: 'pointer', color: '#999', fontSize: '16px', lineHeight: 1 }}
              title="Delete"
            >×</button>
          </li>
        ))}
        {documents.length === 0 && (
          <li style={{ color: '#aaa', fontSize: '13px', textAlign: 'center', padding: '16px 0' }}>
            No documents yet
          </li>
        )}
      </ul>
    </div>
  )
}
