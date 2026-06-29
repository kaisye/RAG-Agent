import { useState } from 'react'
import type { DebugChunk, Message } from '../types'
import type { HistoryTurn } from '../hooks/useChat'

interface Props {
  messages: Message[]
  lastHistory: HistoryTurn[]
  maxHistoryTurns: number
  onMaxHistoryChange: (n: number) => void
}

const LAYER_COLORS = {
  system:    { bg: '#fce4ec', border: '#f48fb1', label: '#c62828' },
  context:   { bg: '#e8f5e9', border: '#a5d6a7', label: '#2e7d32' },
  history:   { bg: '#e3f2fd', border: '#90caf9', label: '#1565c0' },
  query:     { bg: '#fff3e0', border: '#ffcc80', label: '#e65100' },
}

// Rough token estimate: ~4 chars per token
function estTokens(text: string) {
  return Math.ceil(text.length / 4)
}

function truncate(s: string, n = 120) {
  return s.length > n ? s.slice(0, n) + '…' : s
}

function LayerSection({
  color, icon, title, badge, children, defaultOpen = true,
}: {
  color: typeof LAYER_COLORS.system
  icon: string
  title: string
  badge: string
  children: React.ReactNode
  defaultOpen?: boolean
}) {
  const [open, setOpen] = useState(defaultOpen)
  return (
    <div style={{ border: `1px solid ${color.border}`, borderRadius: '6px', marginBottom: '6px', overflow: 'hidden' }}>
      <div
        onClick={() => setOpen(v => !v)}
        style={{
          padding: '5px 10px', background: color.bg, cursor: 'pointer',
          display: 'flex', alignItems: 'center', gap: '6px', userSelect: 'none',
        }}
      >
        <span style={{ fontSize: '13px' }}>{icon}</span>
        <span style={{ fontSize: '12px', fontWeight: 600, color: color.label, flex: 1 }}>{title}</span>
        <span style={{ fontSize: '11px', color: '#888' }}>{badge}</span>
        <span style={{ fontSize: '9px', color: '#aaa' }}>{open ? '▲' : '▼'}</span>
      </div>
      {open && <div style={{ background: '#fff', padding: '6px 10px' }}>{children}</div>}
    </div>
  )
}

export function ContextWindowPanel({ messages, lastHistory, maxHistoryTurns, onMaxHistoryChange }: Props) {
  const [open, setOpen] = useState(false)

  // Get retrieved chunks from the last assistant message's debug info
  const lastAssistant = [...messages].reverse().find(m => m.role === 'assistant' && m.debug)
  const retrievedChunks: DebugChunk[] = lastAssistant?.debug?.reranked ?? []

  // Estimate tokens
  const SYSTEM_PROMPT = 'You are a helpful assistant that answers questions based on the provided document context. Always base your answers strictly on the context below. If the context does not contain enough information to answer, say so clearly. Cite the page number when referencing specific information.'
  const systemTokens  = estTokens(SYSTEM_PROMPT)
  const contextTokens = retrievedChunks.reduce((s, c) => s + estTokens(c.snippet), 0)
  const historyTokens = lastHistory.reduce((s, t) => s + estTokens(t.content), 0)
  const totalTokens   = systemTokens + contextTokens + historyTokens

  // Active history turns to display (each pair = user + assistant)
  const turnPairs: { user: string; assistant: string }[] = []
  for (let i = 0; i < lastHistory.length - 1; i += 2) {
    if (lastHistory[i]?.role === 'user' && lastHistory[i + 1]?.role === 'assistant') {
      turnPairs.push({ user: lastHistory[i].content, assistant: lastHistory[i + 1].content })
    }
  }

  return (
    <div style={{ borderTop: '1px solid #e0e0e0', flexShrink: 0 }}>
      {/* Toggle bar */}
      <div
        onClick={() => setOpen(v => !v)}
        style={{
          padding: '5px 12px', display: 'flex', alignItems: 'center', gap: '8px',
          cursor: 'pointer', background: open ? '#f8f9fa' : '#fff',
          borderBottom: open ? '1px solid #e0e0e0' : 'none',
          userSelect: 'none',
        }}
      >
        <span style={{ fontSize: '13px' }}>🧠</span>
        <span style={{ fontSize: '12px', fontWeight: 600, color: '#444', flex: 1 }}>
          Context Window
        </span>
        <span style={{ fontSize: '11px', color: '#aaa' }}>
          ~{totalTokens.toLocaleString()} tokens
          &nbsp;·&nbsp;{lastHistory.length} history msgs
          &nbsp;·&nbsp;{retrievedChunks.length} chunks
        </span>
        <span style={{ fontSize: '9px', color: '#bbb' }}>{open ? '▲' : '▼'}</span>
      </div>

      {open && (
        <div style={{ padding: '8px 10px', maxHeight: '340px', overflowY: 'auto', background: '#fafafa' }}>

          {/* Layer ① — System Prompt */}
          <LayerSection
            color={LAYER_COLORS.system}
            icon="⚙️" title="① System Prompt"
            badge={`~${systemTokens} tokens`}
            defaultOpen={false}
          >
            <p style={{ margin: 0, fontSize: '11px', color: '#555', lineHeight: 1.5, fontStyle: 'italic' }}>
              {SYSTEM_PROMPT}
            </p>
          </LayerSection>

          {/* Layer ② — Retrieved Context (Working Memory) */}
          <LayerSection
            color={LAYER_COLORS.context}
            icon="📄" title="② Retrieved Context"
            badge={`${retrievedChunks.length} chunks · ~${contextTokens} tokens`}
          >
            {retrievedChunks.length === 0 ? (
              <p style={{ margin: 0, fontSize: '11px', color: '#aaa' }}>
                Chưa có câu hỏi nào — chunks sẽ hiển thị sau lần chat đầu tiên.
              </p>
            ) : (
              retrievedChunks.map((c, i) => (
                <div key={i} style={{
                  padding: '4px 6px', marginBottom: '3px',
                  background: '#f9fbe7', borderRadius: '4px',
                  borderLeft: '3px solid #aed581', fontSize: '11px',
                }}>
                  <span style={{ color: '#558b2f', fontWeight: 600 }}>Trang {c.page + 1}</span>
                  <span style={{ color: '#888', marginLeft: '6px', fontFamily: 'monospace' }}>
                    score: {c.score.toFixed(3)}
                  </span>
                  <div style={{ color: '#444', marginTop: '2px', lineHeight: 1.4 }}>
                    {truncate(c.snippet)}
                  </div>
                </div>
              ))
            )}
          </LayerSection>

          {/* Layer ③ — Conversation History (Short-term Memory) */}
          <LayerSection
            color={LAYER_COLORS.history}
            icon="💬" title="③ Conversation History"
            badge={`${turnPairs.length} lượt · ~${historyTokens} tokens`}
          >
            {/* Max history control */}
            <div style={{ display: 'flex', alignItems: 'center', gap: '8px', marginBottom: '8px', padding: '4px 0' }}>
              <span style={{ fontSize: '11px', color: '#555', flexShrink: 0 }}>Giữ tối đa:</span>
              <input
                type="range" min={0} max={20} step={1}
                value={maxHistoryTurns}
                onChange={e => onMaxHistoryChange(Number(e.target.value))}
                style={{ flex: 1, accentColor: '#1a73e8' }}
              />
              <span style={{
                fontSize: '11px', fontWeight: 600, color: '#1a73e8',
                minWidth: '48px', textAlign: 'right',
              }}>
                {maxHistoryTurns === 0 ? 'All' : `${maxHistoryTurns} lượt`}
              </span>
            </div>

            {turnPairs.length === 0 ? (
              <p style={{ margin: 0, fontSize: '11px', color: '#aaa' }}>
                Chưa có lịch sử — history sẽ được gửi kể từ tin thứ 2 trở đi.
              </p>
            ) : (
              turnPairs.map((pair, i) => (
                <div key={i} style={{ marginBottom: '6px' }}>
                  <div style={{ fontSize: '11px', color: '#1565c0', marginBottom: '1px' }}>
                    <strong>User:</strong> {truncate(pair.user, 80)}
                  </div>
                  <div style={{ fontSize: '11px', color: '#555', paddingLeft: '8px' }}>
                    <strong>Asst:</strong> {truncate(pair.assistant, 80)}
                  </div>
                </div>
              ))
            )}
          </LayerSection>

          {/* Layer ④ — Current Query (placeholder) */}
          <LayerSection
            color={LAYER_COLORS.query}
            icon="❓" title="④ Current Query"
            badge="tin nhắn mới nhất"
            defaultOpen={false}
          >
            {(() => {
              const lastUser = [...messages].reverse().find(m => m.role === 'user')
              return (
                <p style={{ margin: 0, fontSize: '11px', color: '#555', fontStyle: 'italic' }}>
                  {lastUser ? `"${truncate(lastUser.content, 160)}"` : 'Chưa có câu hỏi nào.'}
                </p>
              )
            })()}
          </LayerSection>

          {/* Token summary */}
          <div style={{
            padding: '6px 8px', background: '#f5f5f5', borderRadius: '5px',
            fontSize: '11px', color: '#666', display: 'flex', gap: '12px', flexWrap: 'wrap',
          }}>
            <span>⚙️ System: ~{systemTokens}</span>
            <span>📄 Context: ~{contextTokens}</span>
            <span>💬 History: ~{historyTokens}</span>
            <span style={{ fontWeight: 600, color: '#333' }}>
              Total: ~{totalTokens.toLocaleString()} tokens
            </span>
          </div>
        </div>
      )}
    </div>
  )
}
