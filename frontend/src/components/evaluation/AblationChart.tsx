import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid,
  Tooltip, Legend, ResponsiveContainer, ReferenceLine,
} from "recharts"
import type { AblationResult } from "../../types/evaluation"

interface Props { results: AblationResult[] }

const CONFIG_LABELS_SHORT = [
  "Baseline", "Semantic", "Hybrid+RRF", "HyDE",
  "Decomp", "Cross-Enc", "MMR", "Full ★",
]

const METRICS: { key: keyof AblationResult["scores"]; label: string; color: string }[] = [
  { key: "faithfulness",      label: "Faithfulness",       color: "#3b82f6" },
  { key: "answer_relevancy",  label: "Answer Relevancy",   color: "#10b981" },
  { key: "context_precision", label: "Context Precision",  color: "#f59e0b" },
  { key: "context_recall",    label: "Context Recall",     color: "#ef4444" },
]

const CustomTooltip = ({ active, payload, label }: any) => {
  if (!active || !payload?.length) return null
  return (
    <div style={{
      background: "#fff", border: "1px solid #e5e7eb",
      borderRadius: 8, padding: "10px 14px", fontSize: 12, boxShadow: "0 4px 12px rgba(0,0,0,.1)",
    }}>
      <p style={{ margin: "0 0 6px", fontWeight: 700, color: "#111827" }}>{label}</p>
      {payload.map((p: any) => (
        <p key={p.dataKey} style={{ margin: "2px 0", color: p.color }}>
          {p.name}: <strong>{typeof p.value === "number" ? p.value.toFixed(3) : "—"}</strong>
        </p>
      ))}
    </div>
  )
}

export function AblationChart({ results }: Props) {
  if (results.length === 0) {
    return (
      <div style={{ padding: "32px 0", textAlign: "center", color: "#9ca3af", fontSize: 13 }}>
        Chưa có dữ liệu để vẽ biểu đồ.
      </div>
    )
  }

  const data = results.map((r, i) => ({
    name:              CONFIG_LABELS_SHORT[i] ?? `C${i + 1}`,
    Faithfulness:      r.scores.faithfulness,
    "Answer Relevancy":  r.scores.answer_relevancy,
    "Context Precision": r.scores.context_precision,
    "Context Recall":    r.scores.context_recall,
  }))

  return (
    <ResponsiveContainer width="100%" height={300}>
      <BarChart
        data={data}
        margin={{ top: 8, right: 16, bottom: 8, left: 0 }}
        barCategoryGap="20%"
        barGap={2}
      >
        <CartesianGrid strokeDasharray="3 3" stroke="#f0f0f0" vertical={false} />
        <XAxis
          dataKey="name"
          tick={{ fontSize: 10, fill: "#6b7280" }}
          axisLine={false}
          tickLine={false}
        />
        <YAxis
          domain={[0, 1]}
          tick={{ fontSize: 10, fill: "#6b7280" }}
          axisLine={false}
          tickLine={false}
          tickFormatter={v => v.toFixed(1)}
        />
        <Tooltip content={<CustomTooltip />} />
        <Legend
          wrapperStyle={{ fontSize: 11, paddingTop: 8 }}
          iconType="square"
          iconSize={10}
        />
        {/* Baseline reference line */}
        <ReferenceLine y={results[0]?.scores.faithfulness} stroke="#3b82f6"
          strokeDasharray="4 2" strokeOpacity={0.4} />

        {METRICS.map(m => (
          <Bar
            key={m.key}
            dataKey={m.label}
            fill={m.color}
            radius={[3, 3, 0, 0]}
            maxBarSize={18}
          />
        ))}
      </BarChart>
    </ResponsiveContainer>
  )
}
