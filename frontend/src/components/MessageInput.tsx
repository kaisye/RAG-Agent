import { useState, useRef } from 'react'

interface Props {
  onSend: (text: string) => void
  onStop: () => void
  disabled: boolean
  streaming: boolean
}

export function MessageInput({ onSend, onStop, disabled, streaming }: Props) {
  const [value, setValue] = useState('')
  const textareaRef = useRef<HTMLTextAreaElement>(null)

  function handleSubmit(e: React.FormEvent) {
    e.preventDefault()
    const text = value.trim()
    if (!text) return
    onSend(text)
    setValue('')
    textareaRef.current?.focus()
  }

  function handleKeyDown(e: React.KeyboardEvent<HTMLTextAreaElement>) {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      if (!streaming && !disabled) handleSubmit(e as unknown as React.FormEvent)
    }
  }

  return (
    <form
      onSubmit={handleSubmit}
      style={{ display: 'flex', gap: '8px', padding: '12px 16px', borderTop: '1px solid #e0e0e0', background: '#fff' }}
    >
      <textarea
        ref={textareaRef}
        value={value}
        onChange={e => setValue(e.target.value)}
        onKeyDown={handleKeyDown}
        placeholder={disabled ? 'Select a ready document first…' : 'Ask a question… (Enter to send, Shift+Enter for newline)'}
        disabled={disabled || streaming}
        rows={2}
        style={{
          flex: 1,
          resize: 'none',
          padding: '8px 12px',
          borderRadius: '8px',
          border: '1px solid #ccc',
          fontSize: '14px',
          fontFamily: 'inherit',
          outline: 'none',
        }}
      />
      {streaming ? (
        <button
          type="button"
          onClick={onStop}
          style={{ padding: '8px 16px', borderRadius: '8px', border: 'none', background: '#f44336', color: '#fff', cursor: 'pointer', fontWeight: 600 }}
        >
          Stop
        </button>
      ) : (
        <button
          type="submit"
          disabled={disabled || !value.trim()}
          style={{ padding: '8px 16px', borderRadius: '8px', border: 'none', background: '#1976d2', color: '#fff', cursor: 'pointer', fontWeight: 600, opacity: disabled || !value.trim() ? 0.5 : 1 }}
        >
          Send
        </button>
      )}
    </form>
  )
}
