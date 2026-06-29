import { useEffect, useRef } from 'react'
import ReactMarkdown from 'react-markdown'
import remarkGfm from 'remark-gfm'
import type { Message } from '../types'
import { CitationChip } from './CitationChip'
import { DebugPanel } from './DebugPanel'

interface Props {
  messages: Message[]
  streaming: boolean
  onJumpToPage: (page: number, documentId: string) => void
}

const THINKING_STYLE = `
  @keyframes thinking-bounce {
    0%, 80%, 100% { transform: translateY(0); opacity: 0.35; }
    40%            { transform: translateY(-5px); opacity: 1; }
  }
  .thinking-dot {
    display: inline-block;
    width: 7px; height: 7px;
    border-radius: 50%;
    background: #90a4ae;
    margin: 0 2px;
    animation: thinking-bounce 1.3s ease-in-out infinite;
  }
  .thinking-dot:nth-child(2) { animation-delay: 0.18s; }
  .thinking-dot:nth-child(3) { animation-delay: 0.36s; }
`

function ThinkingIndicator() {
  return (
    <>
      <style>{THINKING_STYLE}</style>
      <span style={{ display: 'inline-flex', alignItems: 'center', gap: '6px', padding: '2px 0' }}>
        <span style={{ fontSize: '12px', color: '#90a4ae', fontStyle: 'italic', letterSpacing: '0.02em' }}>
          Đang suy nghĩ
        </span>
        <span style={{ display: 'inline-flex', alignItems: 'center' }}>
          <span className="thinking-dot" />
          <span className="thinking-dot" />
          <span className="thinking-dot" />
        </span>
      </span>
    </>
  )
}

export function MessageList({ messages, streaming, onJumpToPage }: Props) {
  const bottomRef = useRef<HTMLDivElement>(null)

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages])

  return (
    <div style={{ flex: 1, overflowY: 'auto', padding: '16px', display: 'flex', flexDirection: 'column', gap: '12px' }}>
      {messages.length === 0 && (
        <div style={{ color: '#aaa', textAlign: 'center', marginTop: '40px', fontSize: '14px' }}>
          Chọn tài liệu và đặt câu hỏi.
        </div>
      )}

      {messages.map(msg => (
        <div
          key={msg.id}
          style={{
            alignSelf: msg.role === 'user' ? 'flex-end' : 'flex-start',
            maxWidth: '88%',
          }}
        >
          {/* Bubble */}
          <div
            style={{
              padding: '10px 14px',
              borderRadius: msg.role === 'user' ? '16px 16px 4px 16px' : '16px 16px 16px 4px',
              background: msg.role === 'user' ? '#1976d2' : '#f1f3f4',
              color: msg.role === 'user' ? '#fff' : '#202124',
              fontSize: '14px',
              lineHeight: '1.6',
              wordBreak: 'break-word',
            }}
          >
            {msg.role === 'user' ? (
              <span style={{ whiteSpace: 'pre-wrap' }}>{msg.content}</span>
            ) : (
              <div className="md-body">
                {streaming && msg === messages[messages.length - 1] && !msg.content ? (
                  <ThinkingIndicator />
                ) : (
                  <>
                    <ReactMarkdown remarkPlugins={[remarkGfm]}>{msg.content}</ReactMarkdown>
                    {streaming && msg === messages[messages.length - 1] && (
                      <span style={{ opacity: 0.5 }}>▌</span>
                    )}
                  </>
                )}
              </div>
            )}
          </div>

          {/* Citations */}
          {msg.role === 'assistant' && msg.citations && msg.citations.length > 0 && (
            <div style={{ display: 'flex', flexWrap: 'wrap', gap: '6px', marginTop: '6px', paddingLeft: '4px' }}>
              <span style={{ fontSize: '11px', color: '#888', alignSelf: 'center' }}>Nguồn:</span>
              {msg.citations.map((c, i) => (
                <CitationChip key={i} citation={c} onJumpToPage={onJumpToPage} />
              ))}
            </div>
          )}

          {/* Debug panel */}
          {msg.role === 'assistant' && msg.debug && (
            <DebugPanel debug={msg.debug} />
          )}
        </div>
      ))}

      <div ref={bottomRef} />
    </div>
  )
}
