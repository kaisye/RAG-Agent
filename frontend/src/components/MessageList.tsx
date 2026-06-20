import { useEffect, useRef } from 'react'
import type { Message } from '../types'

interface Props {
  messages: Message[]
  streaming: boolean
}

export function MessageList({ messages, streaming }: Props) {
  const bottomRef = useRef<HTMLDivElement>(null)

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages])

  return (
    <div style={{ flex: 1, overflowY: 'auto', padding: '16px', display: 'flex', flexDirection: 'column', gap: '12px' }}>
      {messages.length === 0 && (
        <div style={{ color: '#aaa', textAlign: 'center', marginTop: '40px', fontSize: '14px' }}>
          Select a document and ask a question.
        </div>
      )}
      {messages.map(msg => (
        <div
          key={msg.id}
          style={{
            alignSelf: msg.role === 'user' ? 'flex-end' : 'flex-start',
            maxWidth: '75%',
          }}
        >
          <div
            style={{
              padding: '10px 14px',
              borderRadius: msg.role === 'user' ? '16px 16px 4px 16px' : '16px 16px 16px 4px',
              background: msg.role === 'user' ? '#1976d2' : '#f1f3f4',
              color: msg.role === 'user' ? '#fff' : '#202124',
              fontSize: '14px',
              lineHeight: '1.5',
              whiteSpace: 'pre-wrap',
              wordBreak: 'break-word',
            }}
          >
            {msg.content}
            {streaming && msg.role === 'assistant' && msg === messages[messages.length - 1] && (
              <span style={{ opacity: 0.6, animation: 'blink 1s step-end infinite' }}>▌</span>
            )}
          </div>
        </div>
      ))}
      <div ref={bottomRef} />
    </div>
  )
}
