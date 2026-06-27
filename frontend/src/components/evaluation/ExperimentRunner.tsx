import { useState, useRef } from "react"
import type { AblationStatus } from "../../types/evaluation"

interface Props {
  onResults: (results: AblationStatus["results"]) => void
}

const TOTAL_CONFIGS = 8

const CONFIG_LABELS = [
  "Baseline (Recursive+Vector)",
  "Semantic Chunking",
  "Hybrid+RRF",
  "HyDE",
  "Decomposition",
  "Cross-Encoder",
  "MMR",
  "Full Pipeline ★",
]

export function ExperimentRunner({ onResults }: Props) {
  const [status, setStatus] = useState<AblationStatus>({
    running: false, done: false, current_config: 0,
    total_configs: TOTAL_CONFIGS, current_label: "",
  })
  const [docId, setDocId]   = useState("")
  const pollRef = useRef<ReturnType<typeof setInterval> | null>(null)

  const stopPoll = () => {
    if (pollRef.current) { clearInterval(pollRef.current); pollRef.current = null }
  }

  const startAblation = async () => {
    if (!docId.trim()) {
      setStatus(s => ({ ...s, error: "Nhập Document ID trước." }))
      return
    }
    setStatus({ running: true, done: false, current_config: 0,
      total_configs: TOTAL_CONFIGS, current_label: CONFIG_LABELS[0] })

    try {
      const res = await fetch("/api/evaluation/ablation", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ document_id: docId.trim(),
          testset_path: "evaluation/testset.json" }),
      })
      if (!res.ok) throw new Error((await res.json()).detail ?? res.statusText)
    } catch (err) {
      setStatus(s => ({ ...s, running: false, error: String(err) }))
      return
    }

    // Poll every 5s
    pollRef.current = setInterval(async () => {
      try {
        const res = await fetch("/api/evaluation/ablation/status")
        if (!res.ok) return
        const data: AblationStatus = await res.json()
        setStatus(data)

        if (!data.running && (data.done || data.error)) {
          stopPoll()
          if (data.results) onResults(data.results)
        }
      } catch { /* network blip */ }
    }, 5000)
  }

  const pct = status.total_configs > 0
    ? Math.round((status.current_config / status.total_configs) * 100)
    : 0

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 12 }}>

      {/* Document ID input */}
      <div style={{ display: "flex", gap: 8, alignItems: "center" }}>
        <input
          value={docId}
          onChange={e => setDocId(e.target.value)}
          placeholder="Document ID (từ trang Chat)"
          disabled={status.running}
          style={{
            flex: 1, fontSize: 12, border: "1px solid #d1d5db", borderRadius: 7,
            padding: "6px 10px", outline: "none",
            background: status.running ? "#f9fafb" : "#fff",
          }}
        />
        <button
          onClick={startAblation}
          disabled={status.running}
          style={{
            padding: "6px 16px", fontSize: 13, fontWeight: 600, borderRadius: 8,
            border: "none", cursor: status.running ? "not-allowed" : "pointer",
            background: status.running ? "#93c5fd" : "#2563eb", color: "#fff",
            whiteSpace: "nowrap",
          }}
        >
          {status.running ? (
            <span style={{ display: "flex", alignItems: "center", gap: 6 }}>
              <SpinnerSVG /> Đang chạy…
            </span>
          ) : "Chạy Ablation (8 configs)"}
        </button>
      </div>

      {/* Error */}
      {status.error && (
        <div style={{
          padding: "6px 10px", borderRadius: 6,
          background: "#fef2f2", border: "1px solid #fecaca",
          fontSize: 12, color: "#dc2626",
        }}>
          ⚠ {status.error}
        </div>
      )}

      {/* Progress */}
      {(status.running || status.done) && (
        <div>
          {/* Bar */}
          <div style={{
            height: 8, borderRadius: 4, background: "#e5e7eb", overflow: "hidden",
          }}>
            <div style={{
              height: "100%", borderRadius: 4,
              background: status.done ? "#16a34a" : "#2563eb",
              width: `${status.done ? 100 : pct}%`,
              transition: "width 0.4s ease",
            }} />
          </div>

          {/* Status text */}
          <div style={{
            marginTop: 6, display: "flex", justifyContent: "space-between",
            fontSize: 11, color: "#6b7280",
          }}>
            <span>
              {status.done
                ? "✓ Hoàn thành tất cả 8 configs"
                : `Config ${status.current_config} / ${status.total_configs}: ${status.current_label ?? ""}`}
            </span>
            <span>{status.done ? "100%" : `${pct}%`}</span>
          </div>

          {/* Config checklist */}
          <div style={{ marginTop: 8, display: "flex", flexDirection: "column", gap: 3 }}>
            {CONFIG_LABELS.map((label, i) => {
              const done    = i < (status.current_config ?? 0)
              const current = !status.done && i === (status.current_config ?? 0)
              return (
                <div key={i} style={{
                  display: "flex", alignItems: "center", gap: 6,
                  fontSize: 11,
                  color:  done ? "#059669" : current ? "#2563eb" : "#9ca3af",
                  fontWeight: current ? 600 : 400,
                }}>
                  <span style={{ width: 14, textAlign: "center" }}>
                    {done ? "✓" : current ? "▶" : "○"}
                  </span>
                  {label}
                </div>
              )
            })}
          </div>
        </div>
      )}

      {status.done && (
        <div style={{
          padding: "8px 12px", borderRadius: 7,
          background: "#f0fdf4", border: "1px solid #bbf7d0",
          fontSize: 12, color: "#15803d", fontWeight: 600,
        }}>
          ✓ Ablation study hoàn thành! Kết quả đã được lưu và hiển thị bên dưới.
        </div>
      )}
    </div>
  )
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
