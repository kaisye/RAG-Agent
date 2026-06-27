import type { AblationResult } from "../../types/evaluation"

interface Props { results: AblationResult[] }

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

const METRICS: { key: keyof AblationResult["scores"]; label: string }[] = [
  { key: "faithfulness",      label: "Faithfulness" },
  { key: "answer_relevancy",  label: "Answer Relevancy" },
  { key: "context_precision", label: "Context Precision" },
  { key: "context_recall",    label: "Context Recall" },
]

function Delta({ val }: { val: number }) {
  if (Math.abs(val) < 0.005) {
    return <span style={{ fontSize: 10, color: "#9ca3af" }}>±0.00</span>
  }
  const pos = val > 0
  return (
    <span style={{
      fontSize:   10,
      fontWeight: 600,
      color:      pos ? "#16a34a" : "#dc2626",
    }}>
      {pos ? "+" : ""}{val.toFixed(2)}
    </span>
  )
}

export function RagasScoreTable({ results }: Props) {
  if (results.length === 0) {
    return (
      <div style={{ padding: "24px 0", textAlign: "center", color: "#9ca3af", fontSize: 13 }}>
        Chưa có kết quả ablation. Chạy thực nghiệm để xem bảng so sánh.
      </div>
    )
  }

  const baseline = results[0].scores

  return (
    <div style={{ overflowX: "auto" }}>
      <table style={{ width: "100%", borderCollapse: "collapse", fontSize: 12 }}>
        <thead>
          <tr style={{ background: "#f3f4f6" }}>
            <th style={th("left", 160)}>Cấu hình</th>
            {METRICS.map(m => (
              <th key={m.key} style={th("center")}>
                <div>{m.label}</div>
                <div style={{ fontSize: 9, fontWeight: 400, color: "#9ca3af" }}>/ Δ vs baseline</div>
              </th>
            ))}
            <th style={th("center", 60)}>Samples</th>
          </tr>
        </thead>
        <tbody>
          {results.map((r, i) => {
            const isBase = i === 0
            return (
              <tr
                key={i}
                style={{
                  background:  isBase ? "#eff6ff" : i % 2 === 0 ? "#fafafa" : "#fff",
                  borderBottom: "1px solid #e5e7eb",
                  fontWeight:  isBase ? 600 : 400,
                }}
              >
                {/* Config label */}
                <td style={{ padding: "8px 10px", color: "#111827", whiteSpace: "nowrap" }}>
                  {isBase && (
                    <span style={{
                      fontSize: 9, background: "#dbeafe", color: "#1d4ed8",
                      borderRadius: 4, padding: "1px 5px", marginRight: 5, fontWeight: 700,
                    }}>
                      BASE
                    </span>
                  )}
                  {CONFIG_LABELS[i] ?? `Config ${i + 1}`}
                </td>

                {/* Metric cells */}
                {METRICS.map(m => {
                  const score = r.scores[m.key]
                  const delta = score - baseline[m.key]
                  return (
                    <td key={m.key} style={{ padding: "8px 10px", textAlign: "center" }}>
                      <div style={{ fontFamily: "monospace", fontSize: 13 }}>
                        {isNaN(score) ? "—" : score.toFixed(3)}
                      </div>
                      {!isBase && !isNaN(score) && (
                        <Delta val={parseFloat(delta.toFixed(3))} />
                      )}
                    </td>
                  )
                })}

                {/* Samples */}
                <td style={{ padding: "8px 10px", textAlign: "center", color: "#6b7280" }}>
                  {r.num_samples}
                </td>
              </tr>
            )
          })}
        </tbody>
      </table>
    </div>
  )
}

function th(align: "left" | "center", minWidth?: number): React.CSSProperties {
  return {
    padding:     "8px 10px",
    textAlign:   align,
    fontWeight:  600,
    fontSize:    11,
    color:       "#374151",
    borderBottom: "2px solid #d1d5db",
    whiteSpace:  "nowrap",
    minWidth:    minWidth,
  }
}
