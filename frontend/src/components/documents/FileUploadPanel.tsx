import { useRef, useState, useCallback } from "react"

interface Props {
  onUpload:  (file: File) => Promise<unknown>
  uploading: boolean
  error?:    string | null
  onClearError?: () => void
}

export function FileUploadPanel({ onUpload, uploading, error, onClearError }: Props) {
  const inputRef    = useRef<HTMLInputElement>(null)
  const [dragging, setDragging] = useState(false)

  const handleFiles = useCallback((files: FileList | null) => {
    if (!files || files.length === 0) return
    const file = files[0]
    onUpload(file)
  }, [onUpload])

  const onDragOver  = (e: React.DragEvent) => { e.preventDefault(); setDragging(true) }
  const onDragLeave = ()                     => setDragging(false)
  const onDrop      = (e: React.DragEvent)  => {
    e.preventDefault()
    setDragging(false)
    handleFiles(e.dataTransfer.files)
  }
  const onInputChange = (e: React.ChangeEvent<HTMLInputElement>) => handleFiles(e.target.files)

  const borderColor = dragging ? "#2563eb" : error ? "#dc2626" : "#d1d5db"
  const bgColor     = dragging ? "#eff6ff" : error  ? "#fef2f2" : "#fafafa"

  return (
    <div>
      {/* Drop zone */}
      <div
        onDragOver={onDragOver}
        onDragLeave={onDragLeave}
        onDrop={onDrop}
        onClick={() => !uploading && inputRef.current?.click()}
        style={{
          border:       `2px dashed ${borderColor}`,
          borderRadius: 10,
          padding:      "20px 12px",
          background:   bgColor,
          cursor:       uploading ? "not-allowed" : "pointer",
          textAlign:    "center",
          transition:   "border-color 0.2s, background 0.2s",
          userSelect:   "none",
        }}
      >
        {uploading ? (
          <div style={{ display: "flex", flexDirection: "column", alignItems: "center", gap: 8 }}>
            {/* Spinner */}
            <svg width="28" height="28" viewBox="0 0 24 24" fill="none"
              stroke="#2563eb" strokeWidth={2} strokeLinecap="round">
              <path d="M12 2v4M12 18v4M4.93 4.93l2.83 2.83M16.24 16.24l2.83 2.83M2 12h4M18 12h4M4.93 19.07l2.83-2.83M16.24 7.76l2.83-2.83">
                <animateTransform attributeName="transform" type="rotate"
                  from="0 12 12" to="360 12 12" dur="1s" repeatCount="indefinite" />
              </path>
            </svg>
            <p style={{ margin: 0, fontSize: 13, color: "#2563eb" }}>Đang tải lên…</p>
          </div>
        ) : (
          <div style={{ display: "flex", flexDirection: "column", alignItems: "center", gap: 6 }}>
            <svg width="28" height="28" viewBox="0 0 24 24" fill="none"
              stroke={dragging ? "#2563eb" : "#9ca3af"} strokeWidth={1.5}>
              <path d="M12 16V4m0 0-4 4m4-4 4 4" strokeLinecap="round" strokeLinejoin="round" />
              <path d="M20 16.7A5 5 0 0018 7h-1.26A8 8 0 104 15.25" strokeLinecap="round" />
            </svg>
            <p style={{ margin: 0, fontSize: 13, color: dragging ? "#2563eb" : "#6b7280" }}>
              {dragging ? "Thả file vào đây" : "Kéo thả PDF hoặc click để chọn"}
            </p>
            <p style={{ margin: 0, fontSize: 11, color: "#9ca3af" }}>Tối đa 50 MB</p>
          </div>
        )}
      </div>

      {/* Hidden file input — PDF only */}
      <input
        ref={inputRef}
        type="file"
        accept=".pdf,application/pdf"
        style={{ display: "none" }}
        onChange={onInputChange}
        disabled={uploading}
      />

      {/* Error message */}
      {error && (
        <div style={{
          marginTop: 8, padding: "6px 10px", borderRadius: 6,
          background: "#fef2f2", border: "1px solid #fecaca",
          display: "flex", alignItems: "center", justifyContent: "space-between", gap: 6,
        }}>
          <span style={{ fontSize: 12, color: "#dc2626" }}>⚠ {error}</span>
          {onClearError && (
            <button onClick={onClearError} style={{
              border: "none", background: "none", cursor: "pointer",
              color: "#dc2626", fontSize: 14, lineHeight: 1, padding: 2,
            }}>×</button>
          )}
        </div>
      )}
    </div>
  )
}
