import { useState } from "react"

// ── Types ─────────────────────────────────────────────────────────────────

interface QuizQuestion {
  question:      string
  options:       string[]          // ["A. ...", "B. ...", "C. ...", "D. ..."]
  correct_index: number            // 0-3
  explanation:   string
  source_page:   number
}

interface Props {
  documentId:     string           // required — must be "ready"
  onJumpToPage?:  (page: number) => void
}

type QuizState = "idle" | "loading" | "done" | "error"

// ── Colours ───────────────────────────────────────────────────────────────

const OPTION_LETTERS = ["A", "B", "C", "D"]

function optionBg(
  optIdx:   number,
  selected: number | null,
  correct:  number,
  revealed: boolean,
): React.CSSProperties {
  if (!revealed) {
    return {
      background:   selected === optIdx ? "#eff6ff" : "#fff",
      borderColor:  selected === optIdx ? "#2563eb" : "#e5e7eb",
      color:        "#111827",
    }
  }
  if (optIdx === correct) {
    return { background: "#f0fdf4", borderColor: "#16a34a", color: "#15803d" }
  }
  if (optIdx === selected && selected !== correct) {
    return { background: "#fef2f2", borderColor: "#dc2626", color: "#dc2626" }
  }
  return { background: "#fafafa", borderColor: "#e5e7eb", color: "#6b7280" }
}

// ── Single question card ──────────────────────────────────────────────────

function QuestionCard({
  q, idx, onJumpToPage,
}: {
  q: QuizQuestion; idx: number; onJumpToPage?: (p: number) => void
}) {
  const [selected, setSelected] = useState<number | null>(null)
  const revealed = selected !== null
  const correct  = selected === q.correct_index

  return (
    <div style={{
      border: "1px solid #e5e7eb", borderRadius: 10,
      overflow: "hidden", background: "#fff",
    }}>
      {/* Question header */}
      <div style={{
        padding: "10px 14px", background: "#fafafa",
        borderBottom: "1px solid #e5e7eb",
        display: "flex", justifyContent: "space-between", alignItems: "flex-start",
      }}>
        <p style={{ margin: 0, fontSize: 13, fontWeight: 600, color: "#111827", flex: 1 }}>
          <span style={{
            display: "inline-block", minWidth: 22, height: 22,
            borderRadius: "50%", background: "#2563eb", color: "#fff",
            fontSize: 11, fontWeight: 700, textAlign: "center", lineHeight: "22px",
            marginRight: 8, flexShrink: 0,
          }}>
            {idx + 1}
          </span>
          {q.question}
        </p>
        {q.source_page > 0 && (
          <button
            onClick={() => onJumpToPage?.(q.source_page)}
            title={`Xem trang ${q.source_page}`}
            style={{
              marginLeft: 8, flexShrink: 0, fontSize: 10, padding: "2px 7px",
              borderRadius: 5, border: "1px solid #e5e7eb",
              background: "#fff", cursor: "pointer", color: "#6b7280",
              whiteSpace: "nowrap",
            }}
          >
            📄 T.{q.source_page}
          </button>
        )}
      </div>

      {/* Options */}
      <div style={{ padding: "10px 14px", display: "flex", flexDirection: "column", gap: 6 }}>
        {q.options.map((opt, i) => {
          const style = optionBg(i, selected, q.correct_index, revealed)
          return (
            <button
              key={i}
              onClick={() => !revealed && setSelected(i)}
              disabled={revealed}
              style={{
                display:     "flex",
                alignItems:  "center",
                gap:         8,
                padding:     "7px 10px",
                border:      `1px solid ${style.borderColor}`,
                borderRadius: 7,
                background:  style.background,
                color:       style.color,
                cursor:      revealed ? "default" : "pointer",
                textAlign:   "left",
                fontSize:    13,
                transition:  "background 0.15s, border-color 0.15s",
              }}
            >
              {/* Letter badge */}
              <span style={{
                width: 20, height: 20, borderRadius: "50%",
                background: revealed && i === q.correct_index ? "#16a34a"
                  : revealed && i === selected ? "#dc2626"
                  : selected === i ? "#2563eb" : "#e5e7eb",
                color: (revealed || selected === i) ? "#fff" : "#6b7280",
                fontSize: 11, fontWeight: 700,
                display: "flex", alignItems: "center", justifyContent: "center",
                flexShrink: 0,
              }}>
                {OPTION_LETTERS[i]}
              </span>
              {opt.replace(/^[A-D]\.\s*/, "")}
              {/* Tick / cross */}
              {revealed && i === q.correct_index && (
                <span style={{ marginLeft: "auto", fontWeight: 700 }}>✓</span>
              )}
              {revealed && i === selected && selected !== q.correct_index && (
                <span style={{ marginLeft: "auto", fontWeight: 700 }}>✗</span>
              )}
            </button>
          )
        })}
      </div>

      {/* Explanation */}
      {revealed && (
        <div style={{
          margin: "0 14px 12px",
          padding: "8px 10px",
          borderRadius: 7,
          background: correct ? "#f0fdf4" : "#fef2f2",
          border: `1px solid ${correct ? "#bbf7d0" : "#fecaca"}`,
          fontSize: 12,
          color: correct ? "#15803d" : "#dc2626",
        }}>
          <span style={{ fontWeight: 700 }}>
            {correct ? "✓ Đúng! " : "✗ Sai. "}
          </span>
          {q.explanation}
        </div>
      )}
    </div>
  )
}

// ── Score summary ─────────────────────────────────────────────────────────

function ScoreSummary({ questions }: { questions: QuizQuestion[] }) {
  const [answers, _] = useState<null>(null)  // placeholder — actual answers tracked per card
  void answers
  return (
    <div style={{
      padding: "10px 14px", background: "#f0fdf4",
      border: "1px solid #bbf7d0", borderRadius: 10,
      fontSize: 13, color: "#15803d", fontWeight: 600,
      textAlign: "center",
    }}>
      {questions.length} câu hỏi — chọn đáp án để xem giải thích
    </div>
  )
}

// ── Main panel ───────────────────────────────────────────────────────────

export function QuizPanel({ documentId, onJumpToPage }: Props) {
  const [topic, setTopic]             = useState("")
  const [numQ, setNumQ]               = useState(5)
  const [state, setState]             = useState<QuizState>("idle")
  const [questions, setQuestions]     = useState<QuizQuestion[]>([])
  const [errorMsg, setErrorMsg]       = useState<string | null>(null)

  const generate = async () => {
    if (!documentId) return
    setState("loading")
    setErrorMsg(null)
    setQuestions([])

    try {
      const res = await fetch(`/api/documents/${documentId}/quiz`, {
        method:  "POST",
        headers: { "Content-Type": "application/json" },
        body:    JSON.stringify({ topic: topic.trim() || undefined, num_questions: numQ }),
      })
      if (!res.ok) {
        const detail = await res.json().catch(() => ({ detail: res.statusText }))
        throw new Error(detail.detail ?? res.statusText)
      }
      const data = await res.json()
      setQuestions(data.questions)
      setState("done")
    } catch (err) {
      setErrorMsg(err instanceof Error ? err.message : String(err))
      setState("error")
    }
  }

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 14 }}>

      {/* Controls */}
      <div style={{
        display: "flex", gap: 8, alignItems: "flex-end", flexWrap: "wrap",
      }}>
        <div style={{ flex: 1, minWidth: 160 }}>
          <label style={{ display: "block", fontSize: 11, color: "#6b7280",
            fontWeight: 600, marginBottom: 4 }}>
            CHỦ ĐỀ (tuỳ chọn)
          </label>
          <input
            value={topic}
            onChange={e => setTopic(e.target.value)}
            onKeyDown={e => e.key === "Enter" && generate()}
            disabled={state === "loading"}
            placeholder="Ví dụ: RAG pipeline, HNSW..."
            style={{
              width: "100%", fontSize: 12, border: "1px solid #d1d5db",
              borderRadius: 7, padding: "6px 10px", outline: "none",
              boxSizing: "border-box",
            }}
          />
        </div>

        <div style={{ minWidth: 80 }}>
          <label style={{ display: "block", fontSize: 11, color: "#6b7280",
            fontWeight: 600, marginBottom: 4 }}>
            SỐ CÂU
          </label>
          <select
            value={numQ}
            onChange={e => setNumQ(Number(e.target.value))}
            disabled={state === "loading"}
            style={{
              fontSize: 12, border: "1px solid #d1d5db", borderRadius: 7,
              padding: "6px 8px", background: "#fff", width: "100%",
            }}
          >
            {[3, 5, 8, 10].map(n => (
              <option key={n} value={n}>{n} câu</option>
            ))}
          </select>
        </div>

        <button
          onClick={generate}
          disabled={state === "loading" || !documentId}
          style={{
            padding: "7px 18px", fontSize: 13, fontWeight: 600, borderRadius: 8,
            border: "none", cursor: state === "loading" || !documentId ? "not-allowed" : "pointer",
            background: state === "loading" || !documentId ? "#93c5fd" : "#2563eb",
            color: "#fff", whiteSpace: "nowrap", alignSelf: "flex-end",
          }}
        >
          {state === "loading" ? (
            <span style={{ display: "flex", alignItems: "center", gap: 6 }}>
              <SpinnerSVG /> Đang tạo…
            </span>
          ) : "Tạo Quiz"}
        </button>

        {questions.length > 0 && (
          <button
            onClick={() => { setQuestions([]); setState("idle") }}
            style={{
              padding: "7px 12px", fontSize: 12, borderRadius: 8,
              border: "1px solid #e5e7eb", background: "#fff",
              cursor: "pointer", color: "#6b7280", alignSelf: "flex-end",
            }}
          >
            Reset
          </button>
        )}
      </div>

      {/* Error */}
      {state === "error" && errorMsg && (
        <div style={{
          padding: "8px 12px", borderRadius: 7,
          background: "#fef2f2", border: "1px solid #fecaca",
          fontSize: 12, color: "#dc2626",
        }}>
          ⚠ {errorMsg}
        </div>
      )}

      {/* Empty state */}
      {state === "idle" && questions.length === 0 && (
        <div style={{
          padding: "24px", textAlign: "center",
          color: "#9ca3af", fontSize: 13,
        }}>
          {documentId
            ? "Nhập chủ đề (tuỳ chọn) và nhấn \"Tạo Quiz\"."
            : "Chọn tài liệu ở trang Chat trước."}
        </div>
      )}

      {/* Questions */}
      {questions.length > 0 && (
        <>
          <ScoreSummary questions={questions} />
          {questions.map((q, i) => (
            <QuestionCard
              key={i} idx={i} q={q}
              onJumpToPage={onJumpToPage}
            />
          ))}
        </>
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
