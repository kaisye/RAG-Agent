import { useState, useRef } from "react"

interface TestsetRecord {
  user_input:   string
  reference?:   string
  synthesizer_name?: string
}

interface GenStatus {
  running:      boolean
  done:         boolean
  num_records?: number
  records?:     TestsetRecord[]
  error?:       string
}

export function TestsetGenPanel() {
  const [status, setStatus]     = useState<GenStatus>({ running: false, done: false })
  const [uploading, setUploading] = useState(false)
  const [uploadMsg, setUploadMsg] = useState<string | null>(null)
  const inputRef = useRef<HTMLInputElement>(null)
  const pollRef  = useRef<ReturnType<typeof setInterval> | null>(null)

  // ── Upload sample PDF ───────────────────────────────────────────────────

  const handleUpload = async (file: File) => {
    if (!file.name.toLowerCase().endsWith(".pdf")) {
      setUploadMsg("⚠ Chỉ chấp nhận PDF.")
      return
    }
    setUploading(true)
    setUploadMsg(null)
    try {
      const form = new FormData()
      form.append("file", file)
      const res = await fetch("/api/documents", { method: "POST", body: form })
      if (!res.ok) throw new Error((await res.json()).detail ?? res.statusText)
      const doc = await res.json()
      setUploadMsg(`✓ Đã tải lên: ${doc.filename ?? file.name}`)
    } catch (err) {
      setUploadMsg(`⚠ ${err instanceof Error ? err.message : String(err)}`)
    } finally {
      setUploading(false)
    }
  }

  // ── Generate testset ────────────────────────────────────────────────────

  const startGenerate = async () => {
    setStatus({ running: true, done: false })
    try {
      const res = await fetch("/api/evaluation/generate-testset", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ size: 50 }),
      })
      if (!res.ok) throw new Error((await res.json()).detail ?? res.statusText)
    } catch (err) {
      setStatus({ running: false, done: false, error: String(err) })
      return
    }
    // Poll status every 3s
    pollRef.current = setInterval(async () => {
      try {
        const res = await fetch("/api/evaluation/generate-testset/status")
        if (!res.ok) return
        const data = await res.json()

        if (!data.running && (data.done || data.error)) {
          clearInterval(pollRef.current!)
          setStatus({
            running:    false,
            done:       Boolean(data.done),
            num_records: data.num_records,
            records:    data.preview ?? [],
            error:      data.error,
          })
        } else {
          setStatus(s => ({ ...s, running: true }))
        }
      } catch { /* network blip */ }
    }, 3000)
  }

  // ── Render ──────────────────────────────────────────────────────────────

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 12 }}>

      {/* Upload sample PDF */}
      <div style={{
        border: "1px dashed #d1d5db", borderRadius: 8, padding: "12px 14px",
        background: "#fafafa",
      }}>
        <p style={{ margin: "0 0 8px", fontSize: 12, fontWeight: 600, color: "#374151" }}>
          Tài liệu mẫu (để sinh câu hỏi)
        </p>
        <div style={{ display: "flex", gap: 8, alignItems: "center" }}>
          <button
            onClick={() => inputRef.current?.click()}
            disabled={uploading}
            style={secondaryBtn(uploading)}
          >
            {uploading ? "Đang tải…" : "Chọn PDF"}
          </button>
          <input
            ref={inputRef} type="file" accept=".pdf" style={{ display: "none" }}
            onChange={e => e.target.files?.[0] && handleUpload(e.target.files[0])}
          />
          {uploadMsg && (
            <span style={{ fontSize: 12, color: uploadMsg.startsWith("⚠") ? "#dc2626" : "#059669" }}>
              {uploadMsg}
            </span>
          )}
        </div>
      </div>

      {/* Generate button */}
      <div style={{ display: "flex", alignItems: "center", gap: 12 }}>
        <button
          onClick={startGenerate}
          disabled={status.running}
          style={primaryBtn(status.running)}
        >
          {status.running ? (
            <span style={{ display: "flex", alignItems: "center", gap: 6 }}>
              <SpinnerSVG /> Đang sinh testset…
            </span>
          ) : "Sinh testset (50 câu)"}
        </button>
        {status.done && status.num_records !== undefined && (
          <span style={{ fontSize: 12, color: "#059669", fontWeight: 600 }}>
            ✓ {status.num_records} câu hỏi
          </span>
        )}
        {status.error && (
          <span style={{ fontSize: 12, color: "#dc2626" }}>⚠ {status.error}</span>
        )}
      </div>

      {/* Preview */}
      {status.done && status.records && status.records.length > 0 && (
        <div>
          <p style={{ margin: "0 0 6px", fontSize: 11, fontWeight: 600, color: "#6b7280",
            textTransform: "uppercase", letterSpacing: "0.05em" }}>
            Preview ({Math.min(3, status.records.length)} / {status.num_records} câu)
          </p>
          <div style={{ display: "flex", flexDirection: "column", gap: 6 }}>
            {status.records.slice(0, 3).map((r, i) => (
              <div key={i} style={{
                border: "1px solid #e5e7eb", borderRadius: 7,
                padding: "8px 10px", background: "#f9fafb",
              }}>
                <p style={{ margin: "0 0 2px", fontSize: 12, fontWeight: 600, color: "#111827" }}>
                  Q{i + 1}: {r.user_input}
                </p>
                {r.synthesizer_name && (
                  <span style={{
                    fontSize: 10, color: "#6b7280", background: "#e5e7eb",
                    borderRadius: 4, padding: "1px 5px",
                  }}>
                    {r.synthesizer_name}
                  </span>
                )}
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  )
}

// ── Helpers ───────────────────────────────────────────────────────────────

function primaryBtn(disabled: boolean): React.CSSProperties {
  return {
    padding: "8px 16px", fontSize: 13, fontWeight: 600, borderRadius: 8,
    border: "none", cursor: disabled ? "not-allowed" : "pointer",
    background: disabled ? "#93c5fd" : "#2563eb", color: "#fff",
  }
}

function secondaryBtn(disabled: boolean): React.CSSProperties {
  return {
    padding: "5px 12px", fontSize: 12, borderRadius: 7,
    border: "1px solid #d1d5db", cursor: disabled ? "not-allowed" : "pointer",
    background: disabled ? "#f3f4f6" : "#fff", color: disabled ? "#9ca3af" : "#374151",
  }
}

function SpinnerSVG() {
  return (
    <svg width="14" height="14" viewBox="0 0 24 24" fill="none"
      stroke="currentColor" strokeWidth={2.5} strokeLinecap="round">
      <path d="M12 2v4M12 18v4M4.93 4.93l2.83 2.83M16.24 16.24l2.83 2.83M2 12h4M18 12h4M4.93 19.07l2.83-2.83M16.24 7.76l2.83-2.83">
        <animateTransform attributeName="transform" type="rotate"
          from="0 12 12" to="360 12 12" dur="1s" repeatCount="indefinite" />
      </path>
    </svg>
  )
}
