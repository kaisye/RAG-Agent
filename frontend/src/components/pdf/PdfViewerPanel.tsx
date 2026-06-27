import { useState, useEffect, useCallback } from "react"
import { Document, Page, pdfjs } from "react-pdf"
import "react-pdf/dist/Page/AnnotationLayer.css"
import "react-pdf/dist/Page/TextLayer.css"

// react-pdf v10 / pdfjs-dist 4.x — worker must be set once at module level
pdfjs.GlobalWorkerOptions.workerSrc = new URL(
  "pdfjs-dist/build/pdf.worker.min.mjs",
  import.meta.url,
).toString()

interface Props {
  fileUrl:     string       // e.g. "/static/uploads/<id>.pdf"
  currentPage: number       // controlled from parent; 1-based
  onPageChange?: (page: number) => void  // optional: let parent know when user navigates
}

export function PdfViewerPanel({ fileUrl, currentPage, onPageChange }: Props) {
  const [numPages, setNumPages]   = useState<number>(0)
  const [inputVal, setInputVal]   = useState<string>(String(currentPage))
  const [width, setWidth]         = useState<number>(340)
  const [loading, setLoading]     = useState(true)
  const [error, setError]         = useState<string | null>(null)

  // Sync input when parent drives currentPage (e.g. CitationChip click)
  useEffect(() => {
    setInputVal(String(currentPage))
  }, [currentPage])

  const goTo = useCallback((page: number) => {
    if (!numPages) return
    const clamped = Math.max(1, Math.min(page, numPages))
    onPageChange?.(clamped)
    setInputVal(String(clamped))
  }, [numPages, onPageChange])

  const handleInputCommit = () => {
    const parsed = parseInt(inputVal, 10)
    if (!isNaN(parsed)) goTo(parsed)
    else setInputVal(String(currentPage))
  }

  const onDocumentLoad = ({ numPages }: { numPages: number }) => {
    setNumPages(numPages)
    setLoading(false)
    setError(null)
    // Jump to the requested page immediately after load
    onPageChange?.(Math.min(currentPage, numPages))
  }

  const onDocumentError = (err: Error) => {
    setLoading(false)
    setError(err.message)
  }

  if (!fileUrl) {
    return (
      <div style={{ display: "flex", flexDirection: "column", alignItems: "center",
        justifyContent: "center", height: "100%", color: "#9ca3af", gap: 8 }}>
        <span style={{ fontSize: 36 }}>📄</span>
        <p style={{ margin: 0, fontSize: 13 }}>Chọn tài liệu để xem PDF</p>
      </div>
    )
  }

  return (
    <div style={{ display: "flex", flexDirection: "column", height: "100%", overflow: "hidden" }}>

      {/* ── Toolbar ── */}
      <div style={{
        display:       "flex",
        alignItems:    "center",
        gap:           6,
        padding:       "6px 10px",
        borderBottom:  "1px solid #e5e7eb",
        background:    "#f9fafb",
        flexShrink:    0,
      }}>
        <button
          onClick={() => goTo(currentPage - 1)}
          disabled={currentPage <= 1 || !numPages}
          title="Trang trước"
          style={btnStyle(currentPage <= 1 || !numPages)}
        >
          ‹
        </button>

        {/* Page input */}
        <input
          value={inputVal}
          onChange={e => setInputVal(e.target.value)}
          onBlur={handleInputCommit}
          onKeyDown={e => { if (e.key === "Enter") handleInputCommit() }}
          disabled={!numPages}
          style={{
            width: 36, textAlign: "center", fontSize: 12,
            border: "1px solid #d1d5db", borderRadius: 5,
            padding: "2px 4px", outline: "none",
          }}
        />
        <span style={{ fontSize: 12, color: "#6b7280", whiteSpace: "nowrap" }}>
          / {numPages || "–"}
        </span>

        <button
          onClick={() => goTo(currentPage + 1)}
          disabled={currentPage >= numPages || !numPages}
          title="Trang sau"
          style={btnStyle(currentPage >= numPages || !numPages)}
        >
          ›
        </button>

        {/* Width slider */}
        <div style={{ flex: 1 }} />
        <input
          type="range" min={200} max={600} value={width}
          onChange={e => setWidth(Number(e.target.value))}
          title="Độ rộng trang"
          style={{ width: 60, cursor: "pointer" }}
        />
      </div>

      {/* ── PDF canvas area ── */}
      <div style={{ flex: 1, overflowY: "auto", overflowX: "auto",
        display: "flex", justifyContent: "center", padding: "16px 8px", background: "#e5e7eb" }}>

        {error ? (
          <div style={{ color: "#dc2626", fontSize: 13, padding: 16, textAlign: "center" }}>
            <p>⚠ Không tải được PDF</p>
            <p style={{ fontSize: 11, opacity: 0.8 }}>{error}</p>
          </div>
        ) : (
          <Document
            file={fileUrl}
            onLoadSuccess={onDocumentLoad}
            onLoadError={onDocumentError}
            loading={
              <div style={{ color: "#6b7280", fontSize: 13, padding: 24, display: "flex",
                flexDirection: "column", alignItems: "center", gap: 10 }}>
                <Spinner />
                <span>Đang tải PDF…</span>
              </div>
            }
          >
            {!loading && (
              <Page
                pageNumber={currentPage}
                width={width}
                renderAnnotationLayer
                renderTextLayer
                loading={
                  <div style={{ width, minHeight: 400, display: "flex",
                    alignItems: "center", justifyContent: "center" }}>
                    <Spinner />
                  </div>
                }
              />
            )}
          </Document>
        )}
      </div>
    </div>
  )
}

// ── Helpers ──────────────────────────────────────────────────────────────────

function btnStyle(disabled: boolean): React.CSSProperties {
  return {
    width: 26, height: 26, border: "1px solid #d1d5db", borderRadius: 5,
    background: disabled ? "#f3f4f6" : "#fff",
    color: disabled ? "#d1d5db" : "#374151",
    cursor: disabled ? "not-allowed" : "pointer",
    fontSize: 16, lineHeight: 1, display: "flex",
    alignItems: "center", justifyContent: "center",
    flexShrink: 0,
  }
}

function Spinner() {
  return (
    <svg width="24" height="24" viewBox="0 0 24 24" fill="none"
      stroke="#2563eb" strokeWidth={2} strokeLinecap="round">
      <path d="M12 2v4M12 18v4M4.93 4.93l2.83 2.83M16.24 16.24l2.83 2.83M2 12h4M18 12h4M4.93 19.07l2.83-2.83M16.24 7.76l2.83-2.83">
        <animateTransform attributeName="transform" type="rotate"
          from="0 12 12" to="360 12 12" dur="1s" repeatCount="indefinite" />
      </path>
    </svg>
  )
}
