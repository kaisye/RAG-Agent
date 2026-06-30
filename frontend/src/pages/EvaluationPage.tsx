import { useState } from "react"
import { TestsetGenPanel } from "../components/evaluation/TestsetGenPanel"
import { ExperimentRunner } from "../components/evaluation/ExperimentRunner"
import { RagasScoreTable } from "../components/evaluation/RagasScoreTable"
import { AblationChart } from "../components/evaluation/AblationChart"
import type { AblationResult, AblationStatus } from "../types/evaluation"

// ── Section card wrapper ──────────────────────────────────────────────────

function Section({
  title, badge, children,
}: { title: string; badge?: string; children: React.ReactNode }) {
  return (
    <div style={{
      background:   "#fff",
      border:       "1px solid #e5e7eb",
      borderRadius: 12,
      overflow:     "hidden",
    }}>
      <div style={{
        padding:      "12px 16px",
        borderBottom: "1px solid #e5e7eb",
        display:      "flex",
        alignItems:   "center",
        gap:          8,
        background:   "#fafafa",
      }}>
        <h2 style={{ margin: 0, fontSize: 14, fontWeight: 700, color: "#111827" }}>
          {title}
        </h2>
        {badge && (
          <span style={{
            fontSize: 10, fontWeight: 700, padding: "2px 7px", borderRadius: 8,
            background: "#dbeafe", color: "#1d4ed8",
          }}>
            {badge}
          </span>
        )}
      </div>
      <div style={{ padding: "16px" }}>
        {children}
      </div>
    </div>
  )
}

// ── Page ─────────────────────────────────────────────────────────────────

export function EvaluationPage() {
  const [results, setResults] = useState<AblationResult[]>([])

  const handleResults = (raw: AblationStatus["results"]) => {
    if (raw) setResults(raw)
  }

  return (
    <div style={{
      height:     "100%",
      overflowY:  "auto",
      padding:    "24px",
      background: "#f3f4f6",
    }}>
      {/* Page header */}
      <div style={{ marginBottom: 20 }}>
        <h1 style={{ margin: 0, fontSize: 20, fontWeight: 800, color: "#111827" }}>
          Evaluation & Ablation Study
        </h1>
        <p style={{ margin: "4px 0 0", fontSize: 13, color: "#6b7280" }}>
          Đánh giá RAG pipeline theo 4 chỉ số RAGAS trên 8 cấu hình thực nghiệm
        </p>
      </div>

      <div style={{ display: "flex", flexDirection: "column", gap: 16, maxWidth: 1100 }}>

        {/* ── Step 1: Testset generation ── */}
        <Section title="Bước 1 — Sinh Testset" badge="RAGAS 0.1.x">
          <TestsetGenPanel />
        </Section>

        {/* ── Step 2: Ablation runner ── */}
        <Section title="Bước 2 — Chạy Ablation Study" badge="8 Configs">
          <ExperimentRunner onResults={handleResults} />
        </Section>

        {/* ── Step 3: Results table ── */}
        <Section title="Bước 3 — Kết Quả So Sánh" badge="Δ vs Baseline">
          {results.length === 0 ? (
            <div style={{ padding: "8px 0", textAlign: "center" }}>
              <p style={{ margin: 0, fontSize: 13, color: "#9ca3af" }}>
                Chạy ablation study để xem bảng kết quả.
              </p>
              <button
                onClick={() => setResults(MOCK_RESULTS)}
                style={{
                  marginTop: 10, fontSize: 11, padding: "4px 12px",
                  border: "1px solid #d1d5db", borderRadius: 6,
                  background: "#fff", cursor: "pointer", color: "#6b7280",
                }}
              >
                Xem demo (dữ liệu mẫu)
              </button>
            </div>
          ) : (
            <>
              <div style={{ marginBottom: 8, display: "flex", justifyContent: "flex-end" }}>
                <button
                  onClick={() => setResults([])}
                  style={{
                    fontSize: 11, padding: "3px 10px",
                    border: "1px solid #e5e7eb", borderRadius: 6,
                    background: "#fff", cursor: "pointer", color: "#9ca3af",
                  }}
                >
                  Xóa kết quả
                </button>
              </div>
              <RagasScoreTable results={results} />
            </>
          )}
        </Section>

        {/* ── Step 4: Chart ── */}
        <Section title="Bước 4 — Biểu Đồ So Sánh" badge="BarChart">
          <AblationChart results={results} />
        </Section>

      </div>
    </div>
  )
}

// ── Mock results (thực nghiệm từ tài liệu AIO2026) ───────────────────────

const MOCK_RESULTS: AblationResult[] = [
  { config: {} as any, num_samples: 7, scores: { faithfulness: 0.73, answer_relevancy: 0.53, context_precision: 0.81, context_recall: 0.67 } },
  { config: {} as any, num_samples: 7, scores: { faithfulness: 0.81, answer_relevancy: 0.55, context_precision: 0.79, context_recall: 0.71 } },
  { config: {} as any, num_samples: 7, scores: { faithfulness: 0.86, answer_relevancy: 0.57, context_precision: 0.80, context_recall: 0.80 } },
  { config: {} as any, num_samples: 7, scores: { faithfulness: 0.81, answer_relevancy: 0.54, context_precision: 0.82, context_recall: 0.75 } },
  { config: {} as any, num_samples: 7, scores: { faithfulness: 0.93, answer_relevancy: 0.61, context_precision: 0.85, context_recall: 0.82 } },
  { config: {} as any, num_samples: 7, scores: { faithfulness: 0.88, answer_relevancy: 0.65, context_precision: 0.83, context_recall: 0.79 } },
  { config: {} as any, num_samples: 7, scores: { faithfulness: 0.85, answer_relevancy: 0.60, context_precision: 0.78, context_recall: 0.84 } },
  { config: {} as any, num_samples: 7, scores: { faithfulness: 0.95, answer_relevancy: 0.68, context_precision: 0.88, context_recall: 0.86 } },
]
