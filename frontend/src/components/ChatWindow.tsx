import { useState, useCallback, useRef } from 'react'
import { useDocuments } from '../hooks/useDocuments'
import { useProjects } from '../hooks/useProjects'
import { useChat } from '../hooks/useChat'
import { FileUploadPanel } from './FileUploadPanel'
import { ProjectPanel } from './ProjectPanel'
import { MessageList } from './MessageList'
import { MessageInput } from './MessageInput'
import { PdfViewerPanel } from './PdfViewerPanel'
import { ContextWindowPanel } from './ContextWindowPanel'

// ─── Panel size constraints ────────────────────────────────────────────────
const SIDEBAR_MIN = 160, SIDEBAR_MAX = 440, SIDEBAR_DEFAULT = 260
const CHAT_MIN    = 280, CHAT_MAX    = 780, CHAT_DEFAULT    = 420
const COLLAPSED_W = 28   // width of a collapsed panel strip

type SidebarTab = 'documents' | 'projects'

// ─── Drag + toggle divider ─────────────────────────────────────────────────
function PanelDivider({
  onDragStart, canDrag, panelVisible, onToggle, collapseDir,
}: {
  onDragStart: (e: React.MouseEvent) => void
  canDrag: boolean
  panelVisible: boolean
  onToggle: () => void
  collapseDir: 'left' | 'right'
}) {
  const icon = panelVisible
    ? (collapseDir === 'left' ? '◀' : '▶')
    : (collapseDir === 'left' ? '▶' : '◀')

  return (
    <div
      onMouseDown={canDrag ? onDragStart : undefined}
      style={{
        width: '12px', flexShrink: 0, position: 'relative', zIndex: 1,
        cursor: canDrag ? 'col-resize' : 'default',
        background: '#dedede',
        display: 'flex', alignItems: 'center', justifyContent: 'center',
        transition: 'background 0.15s',
      }}
      onMouseEnter={e => canDrag && (e.currentTarget.style.background = '#90caf9')}
      onMouseLeave={e => (e.currentTarget.style.background = '#dedede')}
    >
      <button
        onClick={e => { e.stopPropagation(); onToggle() }}
        title={panelVisible ? 'Ẩn panel' : 'Hiện panel'}
        style={{
          position: 'absolute', top: '50%', left: '50%',
          transform: 'translate(-50%, -50%)',
          width: '20px', height: '36px', padding: 0,
          borderRadius: '10px', border: '1px solid #bbb',
          background: '#fff', cursor: 'pointer',
          fontSize: '9px', lineHeight: 1, color: '#555',
          display: 'flex', alignItems: 'center', justifyContent: 'center',
          boxShadow: '0 1px 4px rgba(0,0,0,0.15)', zIndex: 2,
        }}
      >
        {icon}
      </button>
    </div>
  )
}

// ─── Collapsed panel strip ─────────────────────────────────────────────────
function CollapsedStrip({ label, onExpand }: { label: string; onExpand: () => void }) {
  return (
    <div
      style={{
        width: `${COLLAPSED_W}px`, flexShrink: 0,
        display: 'flex', flexDirection: 'column',
        alignItems: 'center', justifyContent: 'center', gap: '8px',
        background: '#f5f5f5', borderRight: '1px solid #e0e0e0',
        cursor: 'pointer',
      }}
      onClick={onExpand}
      title={`Hiện ${label}`}
    >
      <span style={{
        writingMode: 'vertical-rl', textOrientation: 'mixed',
        fontSize: '11px', color: '#888', fontWeight: 500,
        transform: 'rotate(180deg)', userSelect: 'none',
      }}>
        {label}
      </span>
      <span style={{ fontSize: '10px', color: '#aaa' }}>▶</span>
    </div>
  )
}

// ─── Main component ────────────────────────────────────────────────────────
export function ChatWindow() {
  const [sidebarTab, setSidebarTab] = useState<SidebarTab>('documents')
  const [selectedDocId, setSelectedDocId] = useState('')
  const [selectedProjectId, setSelectedProjectId] = useState('')
  const [currentPage, setCurrentPage] = useState(1)

  // visibility
  const [sidebarVisible, setSidebarVisible] = useState(true)
  const [chatVisible, setChatVisible]       = useState(true)
  const [pdfVisible, setPdfVisible]         = useState(true)

  // widths (when visible)
  const [sidebarWidth, setSidebarWidth] = useState(SIDEBAR_DEFAULT)
  const [chatWidth, setChatWidth]       = useState(CHAT_DEFAULT)

  // project mode: which doc is currently previewed in PDF panel
  const [previewDocId, setPreviewDocId] = useState('')

  // context window settings
  const [maxHistoryTurns, setMaxHistoryTurns] = useState(0)

  // ── sidebar drag ──
  const sdDragging = useRef(false)
  const sdStartX   = useRef(0)
  const sdStartW   = useRef(0)
  const onSidebarDragStart = useCallback((e: React.MouseEvent) => {
    sdDragging.current = true
    sdStartX.current   = e.clientX
    sdStartW.current   = sidebarWidth
    document.body.style.cursor     = 'col-resize'
    document.body.style.userSelect = 'none'
    const onMove = (ev: MouseEvent) => {
      if (!sdDragging.current) return
      setSidebarWidth(Math.min(SIDEBAR_MAX, Math.max(SIDEBAR_MIN,
        sdStartW.current + ev.clientX - sdStartX.current)))
    }
    const onUp = () => {
      sdDragging.current = false
      document.body.style.cursor = document.body.style.userSelect = ''
      window.removeEventListener('mousemove', onMove)
      window.removeEventListener('mouseup', onUp)
    }
    window.addEventListener('mousemove', onMove)
    window.addEventListener('mouseup', onUp)
  }, [sidebarWidth])

  // ── chat drag ──
  const cdDragging = useRef(false)
  const cdStartX   = useRef(0)
  const cdStartW   = useRef(0)
  const onChatDragStart = useCallback((e: React.MouseEvent) => {
    cdDragging.current = true
    cdStartX.current   = e.clientX
    cdStartW.current   = chatWidth
    document.body.style.cursor     = 'col-resize'
    document.body.style.userSelect = 'none'
    const onMove = (ev: MouseEvent) => {
      if (!cdDragging.current) return
      setChatWidth(Math.min(CHAT_MAX, Math.max(CHAT_MIN,
        cdStartW.current + ev.clientX - cdStartX.current)))
    }
    const onUp = () => {
      cdDragging.current = false
      document.body.style.cursor = document.body.style.userSelect = ''
      window.removeEventListener('mousemove', onMove)
      window.removeEventListener('mouseup', onUp)
    }
    window.addEventListener('mousemove', onMove)
    window.addEventListener('mouseup', onUp)
  }, [chatWidth])

  // ── data hooks ──
  const { documents, uploading, error: uploadError, upload, remove } = useDocuments()
  const { projects, loading: projectsLoading, error: projectsError,
          createProject, deleteProject, addDocument, removeDocument } = useProjects()

  const chatOpts = selectedProjectId
    ? { projectId: selectedProjectId, maxHistoryTurns }
    : { documentId: selectedDocId, maxHistoryTurns }
  const { messages, streaming, error: chatError, lastHistory, sendMessage, stop, clear } = useChat(chatOpts)

  function handleSelectDoc(id: string) {
    setSelectedDocId(id); setSelectedProjectId(''); setCurrentPage(1); clear()
  }
  function handleSelectProject(id: string) {
    const proj = projects.find(p => p.id === id)
    setSelectedProjectId(id)
    setSelectedDocId('')
    setCurrentPage(1)
    // default preview to first ready doc in project
    const firstReadyId = proj?.document_ids.find(did => documents.find(d => d.id === did)?.status === 'ready') ?? ''
    setPreviewDocId(firstReadyId)
    clear()
  }

  const selectedDoc     = documents.find(d => d.id === selectedDocId)
  const selectedProject = projects.find(p => p.id === selectedProjectId)
  const pdfReady        = selectedDoc?.status === 'ready'
  const chatDisabled    = !selectedDocId && !selectedProjectId

  // PDF viewer: in project mode use previewDocId, in single-doc mode use selectedDocId
  const viewerDocId  = selectedProjectId ? previewDocId : selectedDocId
  const viewerDoc    = documents.find(d => d.id === viewerDocId)
  const viewerReady  = viewerDoc?.status === 'ready'

  // Docs available to preview in the current project
  const projectDocs  = selectedProject
    ? selectedProject.document_ids
        .map(id => documents.find(d => d.id === id))
        .filter((d): d is NonNullable<typeof d> => !!d && d.status === 'ready')
    : []

  function handleJumpToPage(page: number, documentId: string) {
    setCurrentPage(page)
    if (selectedProjectId && documentId) setPreviewDocId(documentId)
  }

  const headerLabel = selectedProject
    ? `Project: ${selectedProject.name}`
    : selectedDoc ? selectedDoc.filename : 'RAG PDF Chat'

  // ── render ──
  return (
    <div style={{ display: 'flex', height: '100vh', fontFamily: 'system-ui, sans-serif', overflow: 'hidden' }}>

      {/* ═══ LEFT SIDEBAR ══════════════════════════════════════════════════ */}
      {sidebarVisible ? (
        <div style={{
          width: `${sidebarWidth}px`, flexShrink: 0,
          display: 'flex', flexDirection: 'column', overflow: 'hidden',
        }}>
          {/* Tab bar */}
          <div style={{ display: 'flex', borderBottom: '1px solid #e0e0e0', flexShrink: 0 }}>
            {(['documents', 'projects'] as SidebarTab[]).map(tab => (
              <button key={tab} onClick={() => setSidebarTab(tab)} style={{
                flex: 1, padding: '9px 0', fontSize: '12px', border: 'none',
                fontWeight: sidebarTab === tab ? 600 : 400,
                background: sidebarTab === tab ? '#fff' : '#f5f5f5',
                borderBottom: sidebarTab === tab ? '2px solid #1a73e8' : '2px solid transparent',
                cursor: 'pointer',
                color: sidebarTab === tab ? '#1a73e8' : '#555',
                textTransform: 'capitalize',
              }}>
                {tab}
              </button>
            ))}
          </div>
          {/* Panel content */}
          <div style={{ padding: '12px', flex: 1, overflow: 'hidden', display: 'flex', flexDirection: 'column' }}>
            {sidebarTab === 'documents' ? (
              <FileUploadPanel
                documents={documents} uploading={uploading} error={uploadError}
                selectedId={selectedDocId} onUpload={upload}
                onSelect={handleSelectDoc} onDelete={remove}
              />
            ) : (
              <ProjectPanel
                projects={projects} documents={documents}
                selectedId={selectedProjectId} previewDocId={previewDocId}
                loading={projectsLoading} error={projectsError}
                onSelect={handleSelectProject} onCreate={createProject}
                onDelete={deleteProject} onAddDoc={addDocument} onRemoveDoc={removeDocument}
                onPreviewDoc={docId => {
                  setPreviewDocId(docId)
                  setCurrentPage(1)
                  setPdfVisible(true)
                }}
              />
            )}
          </div>
        </div>
      ) : (
        <CollapsedStrip
          label={sidebarTab === 'documents' ? 'Docs' : 'Projects'}
          onExpand={() => setSidebarVisible(true)}
        />
      )}

      {/* ═══ DIVIDER 1 (sidebar ↔ chat) ═══════════════════════════════════ */}
      <PanelDivider
        onDragStart={onSidebarDragStart}
        canDrag={sidebarVisible}
        panelVisible={sidebarVisible}
        onToggle={() => setSidebarVisible(v => !v)}
        collapseDir="left"
      />

      {/* ═══ CHAT PANEL ════════════════════════════════════════════════════ */}
      {chatVisible ? (
        <div style={{ flex: `0 0 ${chatWidth}px`, display: 'flex', flexDirection: 'column', overflow: 'hidden' }}>
          {/* Header */}
          <div style={{
            padding: '10px 12px', borderBottom: '1px solid #e0e0e0',
            display: 'flex', alignItems: 'center', gap: '8px', flexShrink: 0,
          }}>
            <h2 style={{ margin: 0, fontSize: '14px', fontWeight: 600, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap', flex: 1 }}>
              {headerLabel}
            </h2>
            {selectedProjectId && (
              <span style={{ fontSize: '11px', background: '#e8f0fe', color: '#1a73e8', padding: '2px 6px', borderRadius: '10px', flexShrink: 0 }}>
                {selectedProject?.document_ids.length ?? 0} docs
              </span>
            )}
            {!chatDisabled && (
              <button onClick={clear} style={{
                flexShrink: 0, padding: '3px 8px', borderRadius: '5px',
                border: '1px solid #ccc', background: '#fff', cursor: 'pointer', fontSize: '11px',
              }}>Clear</button>
            )}
            {/* Collapse chat */}
            <button
              onClick={() => setChatVisible(false)}
              title="Ẩn chat"
              style={{
                flexShrink: 0, padding: '3px 6px', borderRadius: '5px',
                border: '1px solid #ccc', background: '#fff', cursor: 'pointer',
                fontSize: '11px', color: '#888', lineHeight: 1,
              }}
            >▼</button>
          </div>

          {chatError && (
            <div style={{ padding: '6px 12px', background: '#fdecea', color: '#c62828', fontSize: '12px', flexShrink: 0 }}>
              Lỗi: {chatError}
            </div>
          )}

          <MessageList messages={messages} streaming={streaming} onJumpToPage={handleJumpToPage} />

          <ContextWindowPanel
            messages={messages}
            lastHistory={lastHistory}
            maxHistoryTurns={maxHistoryTurns}
            onMaxHistoryChange={setMaxHistoryTurns}
          />

          <MessageInput onSend={sendMessage} onStop={stop} disabled={chatDisabled} streaming={streaming} />
        </div>
      ) : (
        /* Collapsed chat strip */
        <div style={{
          width: `${COLLAPSED_W}px`, flexShrink: 0,
          display: 'flex', flexDirection: 'column',
          alignItems: 'center', justifyContent: 'center', gap: '8px',
          background: '#f0f4ff', cursor: 'pointer',
          borderLeft: '1px solid #c5d3f7', borderRight: '1px solid #c5d3f7',
        }}
          onClick={() => setChatVisible(true)}
          title="Hiện Chat"
        >
          <span style={{
            writingMode: 'vertical-rl', transform: 'rotate(180deg)',
            fontSize: '11px', color: '#1a73e8', fontWeight: 600, userSelect: 'none',
          }}>Chat</span>
          <span style={{ fontSize: '10px', color: '#aaa' }}>▶</span>
        </div>
      )}

      {/* ═══ DIVIDER 2 (chat ↔ pdf) ════════════════════════════════════════ */}
      <PanelDivider
        onDragStart={onChatDragStart}
        canDrag={chatVisible}
        panelVisible={pdfVisible}
        onToggle={() => setPdfVisible(v => !v)}
        collapseDir="right"
      />

      {/* ═══ PDF PANEL ═════════════════════════════════════════════════════ */}
      {pdfVisible ? (
        <div style={{ flex: 1, display: 'flex', flexDirection: 'column', overflow: 'hidden' }}>
          {/* Toolbar */}
          <div style={{
            padding: '7px 12px', borderBottom: '1px solid #e0e0e0',
            display: 'flex', alignItems: 'center', gap: '8px', flexShrink: 0, background: '#fff',
          }}>
            {/* Project mode: document selector */}
            {selectedProjectId ? (
              projectDocs.length > 0 ? (
                <select
                  value={previewDocId}
                  onChange={e => { setPreviewDocId(e.target.value); setCurrentPage(1) }}
                  style={{
                    flex: 1, fontSize: '12px', padding: '3px 6px',
                    border: '1px solid #ccc', borderRadius: '5px',
                    background: '#fff', cursor: 'pointer', maxWidth: '100%',
                  }}
                >
                  <option value="" disabled>Chọn tài liệu để xem…</option>
                  {projectDocs.map(doc => (
                    <option key={doc.id} value={doc.id}>{doc.filename}</option>
                  ))}
                </select>
              ) : (
                <span style={{ fontSize: '12px', color: '#aaa', flex: 1 }}>
                  Chưa có tài liệu ready trong project
                </span>
              )
            ) : (
              <span style={{ fontSize: '13px', color: '#555', flex: 1 }}>
                {pdfReady ? `Trang ${currentPage}` : selectedDocId ? 'Đang xử lý…' : 'Chưa chọn tài liệu'}
              </span>
            )}

            {/* Page indicator for project mode */}
            {selectedProjectId && viewerReady && (
              <span style={{ fontSize: '12px', color: '#888', flexShrink: 0 }}>
                Trang {currentPage}
              </span>
            )}
          </div>

          <div style={{ flex: 1, overflow: 'hidden' }}>
            {viewerReady ? (
              <PdfViewerPanel documentId={viewerDocId} currentPage={currentPage} />
            ) : (
              <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', height: '100%', color: '#aaa', fontSize: '14px' }}>
                {selectedProjectId
                  ? projectDocs.length === 0
                    ? 'Thêm tài liệu vào project để xem PDF.'
                    : 'Chọn tài liệu ở trên để xem PDF.'
                  : selectedDocId
                    ? 'Đợi tài liệu sẵn sàng…'
                    : 'Chọn hoặc upload một tài liệu PDF để xem.'}
              </div>
            )}
          </div>
        </div>
      ) : (
        <CollapsedStrip label="PDF" onExpand={() => setPdfVisible(true)} />
      )}
    </div>
  )
}
