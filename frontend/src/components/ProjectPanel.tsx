import { useState } from 'react'
import type { Document, Project } from '../types'

interface Props {
  projects: Project[]
  documents: Document[]
  selectedId: string
  previewDocId: string
  loading: boolean
  error: string | null
  onSelect: (id: string) => void
  onCreate: (name: string, description: string) => void
  onDelete: (id: string) => void
  onAddDoc: (projectId: string, documentId: string) => void
  onRemoveDoc: (projectId: string, documentId: string) => void
  onPreviewDoc: (docId: string) => void
}

const STATUS_DOT: Record<string, string> = {
  ready:     '🟢',
  embedding: '🟡',
  chunking:  '🟡',
  parsing:   '🟡',
  uploaded:  '🟡',
  pending:   '🟡',
  failed:    '🔴',
  error:     '🔴',
}

export function ProjectPanel({
  projects, documents, selectedId, previewDocId, loading, error,
  onSelect, onCreate, onDelete, onAddDoc, onRemoveDoc, onPreviewDoc,
}: Props) {
  const [newName, setNewName] = useState('')
  const [expandedId, setExpandedId] = useState<string | null>(null)

  function handleCreate(e: React.FormEvent) {
    e.preventDefault()
    if (!newName.trim()) return
    onCreate(newName.trim(), '')
    setNewName('')
  }

  const readyDocs = documents.filter(d => d.status === 'ready')

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '8px', flex: 1, overflow: 'hidden' }}>
      <h3 style={{ margin: 0, fontSize: '14px', fontWeight: 600 }}>Projects</h3>

      {/* Create project */}
      <form onSubmit={handleCreate} style={{ display: 'flex', gap: '4px' }}>
        <input
          value={newName}
          onChange={e => setNewName(e.target.value)}
          placeholder="Project name…"
          style={{ flex: 1, padding: '5px 7px', fontSize: '12px', borderRadius: '5px', border: '1px solid #ccc' }}
        />
        <button
          type="submit"
          disabled={loading || !newName.trim()}
          style={{
            padding: '5px 8px', fontSize: '12px', borderRadius: '5px',
            border: '1px solid #1a73e8', background: '#1a73e8', color: '#fff',
            cursor: 'pointer', flexShrink: 0,
          }}
        >+</button>
      </form>

      {error && <p style={{ color: 'red', fontSize: '12px', margin: 0 }}>{error}</p>}

      {/* Project list */}
      <ul style={{ listStyle: 'none', padding: 0, margin: 0, flex: 1, overflowY: 'auto' }}>
        {projects.map(project => {
          const isExpanded = expandedId === project.id
          const isSelected = selectedId === project.id

          return (
            <li key={project.id} style={{ marginBottom: '4px' }}>
              {/* Project row */}
              <div
                onClick={() => onSelect(project.id)}
                style={{
                  padding: '7px 8px', borderRadius: '6px', cursor: 'pointer',
                  background: isSelected ? '#e8f0fe' : '#f5f5f5',
                  display: 'flex', alignItems: 'center', gap: '4px',
                }}
              >
                <span
                  onClick={e => { e.stopPropagation(); setExpandedId(isExpanded ? null : project.id) }}
                  style={{ fontSize: '10px', color: '#666', width: '14px', flexShrink: 0, userSelect: 'none' }}
                >
                  {isExpanded ? '▼' : '▶'}
                </span>
                <div style={{ flex: 1, overflow: 'hidden' }}>
                  <div style={{ fontSize: '13px', fontWeight: 500, whiteSpace: 'nowrap', overflow: 'hidden', textOverflow: 'ellipsis' }}>
                    {project.name}
                  </div>
                  <div style={{ fontSize: '11px', color: '#888' }}>
                    {project.document_ids.length} doc{project.document_ids.length !== 1 ? 's' : ''}
                  </div>
                </div>
                <button
                  onClick={e => { e.stopPropagation(); onDelete(project.id) }}
                  title="Delete project"
                  style={{ background: 'none', border: 'none', cursor: 'pointer', color: '#bbb', fontSize: '16px', lineHeight: 1, flexShrink: 0 }}
                >×</button>
              </div>

              {/* Expanded: manage documents */}
              {isExpanded && (
                <div style={{ margin: '4px 0 4px 14px', padding: '8px', background: '#fafafa', borderRadius: '6px', border: '1px solid #eee' }}>

                  {/* Docs already in project */}
                  <p style={{ margin: '0 0 5px', fontSize: '11px', fontWeight: 600, color: '#555' }}>
                    Tài liệu trong project
                  </p>

                  {project.document_ids.length === 0 && (
                    <p style={{ margin: '0 0 6px', fontSize: '11px', color: '#aaa' }}>Chưa có tài liệu nào</p>
                  )}

                  {project.document_ids.map(docId => {
                    const doc = documents.find(d => d.id === docId)
                    const isReady = doc?.status === 'ready'
                    const isPreviewing = previewDocId === docId && isSelected

                    return (
                      <div
                        key={docId}
                        style={{
                          display: 'flex', alignItems: 'center', gap: '4px', marginBottom: '3px',
                          padding: '4px 6px', borderRadius: '5px',
                          background: isPreviewing ? '#e8f0fe' : 'transparent',
                          border: isPreviewing ? '1px solid #c5d3f7' : '1px solid transparent',
                        }}
                      >
                        <span style={{ fontSize: '10px', flexShrink: 0 }}>
                          {STATUS_DOT[doc?.status ?? 'pending'] ?? '⚪'}
                        </span>

                        {/* Filename — click to preview */}
                        <span
                          onClick={e => { e.stopPropagation(); if (isReady) onPreviewDoc(docId) }}
                          title={isReady ? `Xem PDF: ${doc?.filename}` : doc?.status}
                          style={{
                            flex: 1, fontSize: '11px', overflow: 'hidden',
                            textOverflow: 'ellipsis', whiteSpace: 'nowrap',
                            color: isReady ? '#1a73e8' : '#aaa',
                            cursor: isReady ? 'pointer' : 'default',
                            textDecoration: isPreviewing ? 'underline' : isReady ? 'underline dotted' : 'none',
                            textUnderlineOffset: '2px',
                          }}
                        >
                          {doc?.filename ?? docId.slice(0, 8) + '…'}
                        </span>

                        {/* Remove */}
                        <button
                          onClick={e => { e.stopPropagation(); onRemoveDoc(project.id, docId) }}
                          title="Xóa khỏi project"
                          style={{ background: 'none', border: 'none', cursor: 'pointer', color: '#ddd', fontSize: '14px', lineHeight: 1, flexShrink: 0 }}
                          onMouseEnter={e => (e.currentTarget.style.color = '#e57373')}
                          onMouseLeave={e => (e.currentTarget.style.color = '#ddd')}
                        >−</button>
                      </div>
                    )
                  })}

                  {/* Add ready docs not yet in project */}
                  {readyDocs.filter(d => !project.document_ids.includes(d.id)).length > 0 && (
                    <>
                      <p style={{ margin: '8px 0 4px', fontSize: '11px', fontWeight: 600, color: '#555' }}>
                        Thêm tài liệu:
                      </p>
                      {readyDocs
                        .filter(d => !project.document_ids.includes(d.id))
                        .map(doc => (
                          <div key={doc.id} style={{ display: 'flex', alignItems: 'center', gap: '4px', marginBottom: '3px' }}>
                            <span style={{ flex: 1, fontSize: '11px', overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
                              {doc.filename}
                            </span>
                            <button
                              onClick={() => onAddDoc(project.id, doc.id)}
                              title="Thêm vào project"
                              style={{ background: 'none', border: 'none', cursor: 'pointer', color: '#43a047', fontSize: '15px', lineHeight: 1, flexShrink: 0 }}
                            >+</button>
                          </div>
                        ))
                      }
                    </>
                  )}
                </div>
              )}
            </li>
          )
        })}

        {projects.length === 0 && (
          <li style={{ color: '#aaa', fontSize: '13px', textAlign: 'center', padding: '16px 0' }}>
            No projects yet
          </li>
        )}
      </ul>
    </div>
  )
}
