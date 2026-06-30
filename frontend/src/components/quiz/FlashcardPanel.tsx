import { useState, useCallback } from "react"

// ── Types ─────────────────────────────────────────────────────────────────

interface Flashcard {
  front:       string
  back:        string
  tag:         "concept" | "formula" | "example" | "comparison" | "warning"
  source_page: number
}

interface Props {
  documentId:    string
  onJumpToPage?: (page: number) => void
}

// ── Tag config ────────────────────────────────────────────────────────────

const TAG_CONFIG: Record<Flashcard["tag"], { label: string; bg: string; color: string }> = {
  concept:    { label: "Khái niệm",   bg: "#dbeafe", color: "#1d4ed8" },
  formula:    { label: "Công thức",   bg: "#fef9c3", color: "#854d0e" },
  example:    { label: "Ví dụ",       bg: "#dcfce7", color: "#15803d" },
  comparison: { label: "So sánh",     bg: "#fae8ff", color: "#7e22ce" },
  warning:    { label: "Lưu ý ⚠",    bg: "#fee2e2", color: "#b91c1c" },
}

// ── Flip card ─────────────────────────────────────────────────────────────

function FlipCard({
  card, index, total, onJumpToPage,
}: {
  card:          Flashcard
  index:         number
  total:         number
  onJumpToPage?: (p: number) => void
}) {
  const [flipped, setFlipped] = useState(false)
  const tag = TAG_CONFIG[card.tag] ?? TAG_CONFIG.concept

  return (
    <div style={{ perspective: 1000 }}>
      {/* Flip animation via CSS transform */}
      <style>{`
        .fc-inner {
          position: relative;
          width: 100%;
          height: 220px;
          transition: transform 0.45s cubic-bezier(.4,0,.2,1);
          transform-style: preserve-3d;
        }
        .fc-inner.flipped { transform: rotateY(180deg); }
        .fc-face {
          position: absolute;
          inset: 0;
          backface-visibility: hidden;
          -webkit-backface-visibility: hidden;
          border-radius: 14px;
          display: flex;
          flex-direction: column;
          justify-content: center;
          align-items: center;
          padding: 24px;
          cursor: pointer;
          user-select: none;
          box-shadow: 0 2px 12px rgba(0,0,0,.08);
        }
        .fc-front {
          background: #fff;
          border: 2px solid #e5e7eb;
        }
        .fc-back {
          background: #1e40af;
          transform: rotateY(180deg);
          border: 2px solid #1e40af;
          color: #fff;
          align-items: flex-start;
          justify-content: flex-start;
        }
        .fc-inner:not(.flipped) .fc-front:hover {
          border-color: #93c5fd;
          box-shadow: 0 4px 20px rgba(37,99,235,.12);
        }
      `}</style>

      {/* Counter + tag row */}
      <div style={{
        display: "flex", justifyContent: "space-between",
        alignItems: "center", marginBottom: 8,
      }}>
        <span style={{ fontSize: 11, color: "#9ca3af" }}>
          {index + 1} / {total}
        </span>
        <span style={{
          fontSize: 10, fontWeight: 700, padding: "2px 8px", borderRadius: 8,
          background: tag.bg, color: tag.color,
        }}>
          {tag.label}
        </span>
      </div>

      {/* Card */}
      <div
        className={`fc-inner${flipped ? " flipped" : ""}`}
        onClick={() => setFlipped(f => !f)}
      >
        {/* Front */}
        <div className="fc-face fc-front">
          <p style={{
            margin: 0, fontSize: 16, fontWeight: 600, color: "#111827",
            textAlign: "center", lineHeight: 1.5,
          }}>
            {card.front}
          </p>
          <p style={{
            position: "absolute", bottom: 12, fontSize: 11,
            color: "#9ca3af",
          }}>
            Nhấn để xem đáp án ↩
          </p>
        </div>

        {/* Back */}
        <div className="fc-face fc-back" onClick={e => e.stopPropagation()}>
          <p style={{
            margin: "0 0 10px", fontSize: 12, fontWeight: 700,
            color: "#93c5fd", textTransform: "uppercase", letterSpacing: "0.05em",
          }}>
            Đáp án
          </p>
          <p style={{
            margin: 0, fontSize: 14, lineHeight: 1.6, whiteSpace: "pre-wrap",
          }}>
            {card.back}
          </p>

          {/* Source + flip back */}
          <div style={{
            position: "absolute", bottom: 12, left: 24, right: 24,
            display: "flex", justifyContent: "space-between", alignItems: "center",
          }}>
            {card.source_page > 0 ? (
              <button
                onClick={e => { e.stopPropagation(); onJumpToPage?.(card.source_page) }}
                style={{
                  fontSize: 11, padding: "2px 8px", borderRadius: 5,
                  border: "1px solid rgba(255,255,255,.3)", background: "transparent",
                  color: "#bfdbfe", cursor: "pointer",
                }}
              >
                📄 Trang {card.source_page}
              </button>
            ) : <span />}
            <button
              onClick={e => { e.stopPropagation(); setFlipped(false) }}
              style={{
                fontSize: 11, padding: "2px 8px", borderRadius: 5,
                border: "1px solid rgba(255,255,255,.3)", background: "transparent",
                color: "#bfdbfe", cursor: "pointer",
              }}
            >
              ↩ Lật lại
            </button>
          </div>
        </div>
      </div>
    </div>
  )
}

// ── Deck navigator (prev / next / shuffle) ────────────────────────────────

function DeckNav({
  current, total, onPrev, onNext, onShuffle,
}: {
  current:   number
  total:     number
  onPrev:    () => void
  onNext:    () => void
  onShuffle: () => void
}) {
  return (
    <div style={{
      display: "flex", alignItems: "center", justifyContent: "center",
      gap: 12,
    }}>
      <button onClick={onPrev} disabled={current === 0} style={navBtn(current === 0)}>
        ‹ Trước
      </button>

      {/* Progress dots (max 10 shown) */}
      <div style={{ display: "flex", gap: 4, alignItems: "center" }}>
        {Array.from({ length: Math.min(total, 10) }).map((_, i) => {
          const idx = total <= 10 ? i : Math.round((i / 9) * (total - 1))
          const active = total <= 10 ? i === current : idx === current
          return (
            <span key={i} style={{
              width:        active ? 10 : 6,
              height:       active ? 10 : 6,
              borderRadius: "50%",
              background:   active ? "#2563eb" : "#d1d5db",
              transition:   "all 0.2s",
              display:      "inline-block",
            }} />
          )
        })}
      </div>

      <button onClick={onNext} disabled={current === total - 1} style={navBtn(current === total - 1)}>
        Sau ›
      </button>

      <button onClick={onShuffle} style={navBtn(false)} title="Xáo bộ bài">
        🔀
      </button>
    </div>
  )
}

function navBtn(disabled: boolean): React.CSSProperties {
  return {
    padding: "5px 14px", fontSize: 13, borderRadius: 7,
    border: "1px solid #e5e7eb",
    background: disabled ? "#f9fafb" : "#fff",
    color: disabled ? "#d1d5db" : "#374151",
    cursor: disabled ? "not-allowed" : "pointer",
  }
}

// ── Main panel ───────────────────────────────────────────────────────────

type PanelState = "idle" | "loading" | "done" | "error"

export function FlashcardPanel({ documentId, onJumpToPage }: Props) {
  const [topic, setTopic]           = useState("")
  const [numCards, setNumCards]     = useState(10)
  const [panelState, setPanelState] = useState<PanelState>("idle")
  const [cards, setCards]           = useState<Flashcard[]>([])
  const [current, setCurrent]       = useState(0)
  const [errorMsg, setErrorMsg]     = useState<string | null>(null)

  // Reset card flip when navigating
  const goTo = useCallback((idx: number) => {
    setCurrent(idx)
  }, [])

  const shuffle = useCallback(() => {
    setCards(prev => {
      const arr = [...prev]
      for (let i = arr.length - 1; i > 0; i--) {
        const j = Math.floor(Math.random() * (i + 1))
        ;[arr[i], arr[j]] = [arr[j], arr[i]]
      }
      return arr
    })
    setCurrent(0)
  }, [])

  const generate = async () => {
    if (!documentId) return
    setPanelState("loading")
    setErrorMsg(null)
    setCards([])
    setCurrent(0)

    try {
      const res = await fetch(`/api/documents/${documentId}/flashcards`, {
        method:  "POST",
        headers: { "Content-Type": "application/json" },
        body:    JSON.stringify({ topic: topic.trim() || undefined, num_cards: numCards }),
      })
      if (!res.ok) {
        const d = await res.json().catch(() => ({ detail: res.statusText }))
        throw new Error(d.detail ?? res.statusText)
      }
      const data = await res.json()
      setCards(data.flashcards)
      setPanelState("done")
    } catch (err) {
      setErrorMsg(err instanceof Error ? err.message : String(err))
      setPanelState("error")
    }
  }

  // ── Tag filter ────────────────────────────────────────────────────────
  const [tagFilter, setTagFilter] = useState<Flashcard["tag"] | "all">("all")
  const visible = tagFilter === "all" ? cards : cards.filter(c => c.tag === tagFilter)
  const safeIdx = Math.min(current, Math.max(0, visible.length - 1))

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 16 }}>

      {/* Controls */}
      <div style={{ display: "flex", gap: 8, flexWrap: "wrap", alignItems: "flex-end" }}>
        <div style={{ flex: 1, minWidth: 160 }}>
          <label style={{ display: "block", fontSize: 11, fontWeight: 600,
            color: "#6b7280", marginBottom: 4, textTransform: "uppercase",
            letterSpacing: "0.05em" }}>
            Chủ đề (tuỳ chọn)
          </label>
          <input
            value={topic}
            onChange={e => setTopic(e.target.value)}
            onKeyDown={e => e.key === "Enter" && generate()}
            disabled={panelState === "loading"}
            placeholder="Ví dụ: hybrid retrieval, MMR..."
            style={{
              width: "100%", fontSize: 12, border: "1px solid #d1d5db",
              borderRadius: 7, padding: "6px 10px", outline: "none",
              boxSizing: "border-box",
            }}
          />
        </div>

        <div style={{ minWidth: 80 }}>
          <label style={{ display: "block", fontSize: 11, fontWeight: 600,
            color: "#6b7280", marginBottom: 4, textTransform: "uppercase",
            letterSpacing: "0.05em" }}>
            Số thẻ
          </label>
          <select
            value={numCards}
            onChange={e => setNumCards(Number(e.target.value))}
            disabled={panelState === "loading"}
            style={{
              fontSize: 12, border: "1px solid #d1d5db", borderRadius: 7,
              padding: "6px 8px", background: "#fff", width: "100%",
            }}
          >
            {[5, 10, 15, 20].map(n => (
              <option key={n} value={n}>{n} thẻ</option>
            ))}
          </select>
        </div>

        <button
          onClick={generate}
          disabled={panelState === "loading" || !documentId}
          style={{
            padding: "7px 18px", fontSize: 13, fontWeight: 600,
            borderRadius: 8, border: "none",
            cursor: panelState === "loading" || !documentId ? "not-allowed" : "pointer",
            background: panelState === "loading" || !documentId ? "#93c5fd" : "#2563eb",
            color: "#fff", whiteSpace: "nowrap", alignSelf: "flex-end",
          }}
        >
          {panelState === "loading" ? (
            <span style={{ display: "flex", alignItems: "center", gap: 6 }}>
              <SpinnerSVG /> Đang tạo…
            </span>
          ) : "Tạo Flashcard"}
        </button>

        {cards.length > 0 && (
          <button
            onClick={() => { setCards([]); setPanelState("idle"); setCurrent(0) }}
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
      {panelState === "error" && errorMsg && (
        <div style={{
          padding: "8px 12px", borderRadius: 7,
          background: "#fef2f2", border: "1px solid #fecaca",
          fontSize: 12, color: "#dc2626",
        }}>
          ⚠ {errorMsg}
        </div>
      )}

      {/* Empty state */}
      {panelState === "idle" && (
        <div style={{
          padding: "32px", textAlign: "center", color: "#9ca3af", fontSize: 13,
        }}>
          {documentId
            ? "Nhập chủ đề (tuỳ chọn) và nhấn \"Tạo Flashcard\"."
            : "Chọn tài liệu ở bên trái trước."}
        </div>
      )}

      {/* Deck */}
      {panelState === "done" && cards.length > 0 && (
        <>
          {/* Tag filter row */}
          <div style={{ display: "flex", gap: 6, flexWrap: "wrap", alignItems: "center" }}>
            <span style={{ fontSize: 11, color: "#6b7280" }}>Lọc:</span>
            {(["all", ...Object.keys(TAG_CONFIG)] as (Flashcard["tag"] | "all")[]).map(t => {
              const cfg = t === "all" ? null : TAG_CONFIG[t]
              const active = tagFilter === t
              return (
                <button
                  key={t}
                  onClick={() => { setTagFilter(t); setCurrent(0) }}
                  style={{
                    fontSize: 11, fontWeight: 600, padding: "2px 9px", borderRadius: 8,
                    border: "1px solid",
                    borderColor: active ? (cfg?.color ?? "#2563eb") : "#e5e7eb",
                    background:  active ? (cfg?.bg ?? "#eff6ff") : "#fff",
                    color:       active ? (cfg?.color ?? "#2563eb") : "#6b7280",
                    cursor:      "pointer",
                  }}
                >
                  {t === "all" ? `Tất cả (${cards.length})` : `${cfg!.label} (${cards.filter(c => c.tag === t).length})`}
                </button>
              )
            })}
          </div>

          {visible.length === 0 ? (
            <p style={{ fontSize: 13, color: "#9ca3af", textAlign: "center", padding: 16 }}>
              Không có thẻ nào với tag này.
            </p>
          ) : (
            <>
              <FlipCard
                key={`${safeIdx}-${visible[safeIdx]?.front}`}
                card={visible[safeIdx]}
                index={safeIdx}
                total={visible.length}
                onJumpToPage={onJumpToPage}
              />
              <DeckNav
                current={safeIdx}
                total={visible.length}
                onPrev={() => goTo(Math.max(0, safeIdx - 1))}
                onNext={() => goTo(Math.min(visible.length - 1, safeIdx + 1))}
                onShuffle={shuffle}
              />
            </>
          )}
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
