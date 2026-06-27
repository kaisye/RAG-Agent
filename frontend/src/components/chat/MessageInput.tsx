import { useState, useRef, useCallback } from "react"

interface Props {
  onSend:     (text: string) => void
  disabled?:  boolean
  placeholder?: string
}

export function MessageInput({ onSend, disabled, placeholder }: Props) {
  const [value, setValue] = useState("")
  const textareaRef       = useRef<HTMLTextAreaElement>(null)

  const submit = useCallback(() => {
    const text = value.trim()
    if (!text || disabled) return
    onSend(text)
    setValue("")
    // Reset textarea height
    if (textareaRef.current) textareaRef.current.style.height = "auto"
  }, [value, disabled, onSend])

  const handleKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault()
      submit()
    }
  }

  // Auto-grow textarea up to 5 lines
  const handleChange = (e: React.ChangeEvent<HTMLTextAreaElement>) => {
    setValue(e.target.value)
    const el = e.target
    el.style.height = "auto"
    el.style.height = Math.min(el.scrollHeight, 120) + "px"
  }

  return (
    <div style={{
      borderTop:  "1px solid #e5e7eb",
      padding:    "12px 16px",
      display:    "flex",
      gap:        8,
      background: "#fff",
      alignItems: "flex-end",
    }}>
      <textarea
        ref={textareaRef}
        value={value}
        onChange={handleChange}
        onKeyDown={handleKeyDown}
        disabled={disabled}
        rows={1}
        placeholder={placeholder ?? "Đặt câu hỏi về tài liệu… (Enter gửi, Shift+Enter xuống dòng)"}
        style={{
          flex:       1,
          resize:     "none",
          border:     "1px solid #d1d5db",
          borderRadius: 12,
          padding:    "9px 12px",
          fontSize:   14,
          lineHeight: 1.5,
          outline:    "none",
          background: disabled ? "#f9fafb" : "#fff",
          color:      "#111827",
          fontFamily: "inherit",
          transition: "border-color 0.15s",
        }}
        onFocus={e => { e.target.style.borderColor = "#2563eb" }}
        onBlur={e  => { e.target.style.borderColor = "#d1d5db" }}
      />

      <button
        onClick={submit}
        disabled={disabled || !value.trim()}
        title="Gửi (Enter)"
        style={{
          width:        40,
          height:       40,
          borderRadius: "50%",
          border:       "none",
          background:   disabled || !value.trim() ? "#e5e7eb" : "#2563eb",
          color:        disabled || !value.trim() ? "#9ca3af" : "#fff",
          cursor:       disabled || !value.trim() ? "not-allowed" : "pointer",
          display:      "flex",
          alignItems:   "center",
          justifyContent: "center",
          flexShrink:   0,
          transition:   "background 0.15s",
        }}
      >
        {disabled ? (
          /* Spinner khi đang stream */
          <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth={2}>
            <path d="M12 2v4M12 18v4M4.93 4.93l2.83 2.83M16.24 16.24l2.83 2.83M2 12h4M18 12h4M4.93 19.07l2.83-2.83M16.24 7.76l2.83-2.83"
              strokeLinecap="round">
              <animateTransform attributeName="transform" type="rotate"
                from="0 12 12" to="360 12 12" dur="1s" repeatCount="indefinite" />
            </path>
          </svg>
        ) : (
          /* Send icon */
          <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth={2}>
            <path d="M22 2L11 13M22 2L15 22l-4-9-9-4 20-7z" strokeLinecap="round" strokeLinejoin="round" />
          </svg>
        )}
      </button>
    </div>
  )
}
