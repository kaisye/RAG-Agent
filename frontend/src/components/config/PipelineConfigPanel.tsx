import type { PipelineConfig } from "../../types/pipeline"

interface Props {
  config:    PipelineConfig
  onChange:  (config: PipelineConfig) => void
  disabled?: boolean
}

// ── Option definitions with RAGAS experimental numbers ────────────────────

const CHUNKING_OPTIONS: { value: string; label: string; note: string }[] = [
  { value: "recursive", label: "Recursive",       note: "Baseline — Faith=0.73, CP=0.81" },
  { value: "semantic",  label: "Semantic ★",      note: "Faith↑0.08 vs baseline" },
]

const RETRIEVAL_OPTIONS: { value: string; label: string; note: string }[] = [
  { value: "vector",               label: "Vector Search",   note: "Baseline — CP=0.81, CR=0.67" },
  { value: "bm25",                 label: "BM25 Sparse",     note: "⚠ Faith=0.44 — chỉ để so sánh" },
  { value: "hybrid_interleaving",  label: "Hybrid Xen kẽ",  note: "CR↑0.06 nhưng CP↓0.24" },
  { value: "hybrid_rrf",           label: "Hybrid RRF ★",   note: "Tốt nhất — CP=0.80, CR=0.80" },
]

const TRANSFORM_OPTIONS: { value: string; label: string; note: string }[] = [
  { value: "none",          label: "Không biến đổi",  note: "Baseline" },
  { value: "hyde",          label: "HyDE",             note: "Faith↑0.08, AR↓0.03" },
  { value: "decomposition", label: "Decomposition ★",  note: "Faith↑0.20 — cao nhất!" },
]

const RERANK_OPTIONS: { value: string; label: string; note: string }[] = [
  { value: "none",          label: "Không rerank",   note: "Baseline" },
  { value: "cross_encoder", label: "Cross-Encoder",  note: "AR↑0.08, cải thiện đều" },
  { value: "mmr",           label: "MMR (đa dạng)",  note: "Tốt cho câu hỏi tổng quan" },
]

// ── Sub-component: one labeled dropdown with tooltip notes in options ─────

interface SelectFieldProps {
  label:    string
  value:    string
  options:  { value: string; label: string; note: string }[]
  onChange: (val: string) => void
  disabled: boolean
}

function SelectField({ label, value, options, onChange, disabled }: SelectFieldProps) {
  return (
    <div style={{ marginBottom: 10 }}>
      <label style={{
        display: "block", fontSize: 11,
        fontWeight: 600, color: "#6b7280",
        marginBottom: 4, textTransform: "uppercase", letterSpacing: "0.05em",
      }}>
        {label}
      </label>

      <div style={{ position: "relative" }}>
        <select
          value={value}
          onChange={e => onChange(e.target.value)}
          disabled={disabled}
          style={{
            width:        "100%",
            fontSize:     12,
            border:       `1px solid ${disabled ? "#e5e7eb" : "#d1d5db"}`,
            borderRadius: 7,
            padding:      "5px 28px 5px 8px",
            background:   disabled ? "#f9fafb" : "#fff",
            color:        disabled ? "#9ca3af" : "#111827",
            cursor:       disabled ? "not-allowed" : "pointer",
            appearance:   "none",
            outline:      "none",
          }}
        >
          {options.map(opt => (
            <option key={opt.value} value={opt.value} title={opt.note}>
              {opt.label}{opt.note ? ` — ${opt.note}` : ""}
            </option>
          ))}
        </select>

        {/* Custom chevron */}
        <span style={{
          position:      "absolute",
          right:         8,
          top:           "50%",
          transform:     "translateY(-50%)",
          pointerEvents: "none",
          color:         disabled ? "#d1d5db" : "#6b7280",
          fontSize:      10,
        }}>
          ▼
        </span>
      </div>

      {/* Inline note for selected option */}
      {(() => {
        const sel = options.find(o => o.value === value)
        return sel?.note ? (
          <p style={{
            margin: "3px 0 0",
            fontSize: 10,
            color: sel.note.startsWith("⚠") ? "#dc2626" : "#059669",
            lineHeight: 1.3,
          }}>
            {sel.note}
          </p>
        ) : null
      })()}
    </div>
  )
}

// ── Main component ────────────────────────────────────────────────────────

export function PipelineConfigPanel({ config, onChange, disabled = false }: Props) {
  const set = (key: keyof PipelineConfig) => (val: string) =>
    onChange({ ...config, [key]: val })

  return (
    <div style={{
      borderTop:   "1px solid #e5e7eb",
      padding:     "12px",
      background:  "#fafafa",
    }}>
      {/* Header */}
      <div style={{
        display:      "flex",
        alignItems:   "center",
        justifyContent: "space-between",
        marginBottom: 10,
      }}>
        <p style={{ margin: 0, fontSize: 11, fontWeight: 700, color: "#374151",
          textTransform: "uppercase", letterSpacing: "0.06em" }}>
          Pipeline Config
        </p>
        {disabled && (
          <span style={{
            fontSize: 10, color: "#f59e0b", background: "#fef3c7",
            padding: "1px 6px", borderRadius: 8, fontWeight: 600,
          }}>
            Đang stream…
          </span>
        )}
      </div>

      <SelectField
        label="Chunking"
        value={config.chunking_strategy}
        options={CHUNKING_OPTIONS}
        onChange={set("chunking_strategy")}
        disabled={disabled}
      />
      <SelectField
        label="Retrieval"
        value={config.retrieval_strategy}
        options={RETRIEVAL_OPTIONS}
        onChange={set("retrieval_strategy")}
        disabled={disabled}
      />
      <SelectField
        label="Query Transform"
        value={config.query_transform}
        options={TRANSFORM_OPTIONS}
        onChange={set("query_transform")}
        disabled={disabled}
      />
      <SelectField
        label="Reranking"
        value={config.rerank_strategy}
        options={RERANK_OPTIONS}
        onChange={set("rerank_strategy")}
        disabled={disabled}
      />

      {/* Current config summary badge */}
      <div style={{
        marginTop:  8,
        padding:    "5px 8px",
        borderRadius: 6,
        background: "#eff6ff",
        border:     "1px solid #bfdbfe",
      }}>
        <p style={{ margin: 0, fontSize: 10, color: "#1d4ed8", lineHeight: 1.5 }}>
          <strong>Full Pipeline ★:</strong> Semantic + Hybrid RRF + Decomposition + Cross-Encoder
        </p>
      </div>
    </div>
  )
}
