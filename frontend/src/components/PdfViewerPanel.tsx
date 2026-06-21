import { useEffect, useRef, useState } from 'react'
import { Document, Page, pdfjs } from 'react-pdf'
import 'react-pdf/dist/Page/AnnotationLayer.css'
import 'react-pdf/dist/Page/TextLayer.css'

// Configure PDF.js worker (pdfjs-dist 4.x bundled with react-pdf v9+)
pdfjs.GlobalWorkerOptions.workerSrc = new URL(
  'pdfjs-dist/build/pdf.worker.min.mjs',
  import.meta.url,
).toString()

interface Props {
  documentId: string
  currentPage: number   // 1-based (react-pdf convention)
}

export function PdfViewerPanel({ documentId, currentPage }: Props) {
  const [numPages, setNumPages] = useState<number>(0)
  const [containerWidth, setContainerWidth] = useState<number>(600)
  const containerRef = useRef<HTMLDivElement>(null)
  const pageRefs = useRef<Record<number, HTMLDivElement | null>>({})

  // Measure container width for responsive page rendering
  useEffect(() => {
    if (!containerRef.current) return
    const obs = new ResizeObserver(entries => {
      const w = entries[0]?.contentRect.width
      if (w) setContainerWidth(w - 24)  // 12px padding each side
    })
    obs.observe(containerRef.current)
    return () => obs.disconnect()
  }, [])

  // Scroll to the cited page whenever currentPage changes
  useEffect(() => {
    const el = pageRefs.current[currentPage]
    if (el) el.scrollIntoView({ behavior: 'smooth', block: 'start' })
  }, [currentPage])

  const fileUrl = `/api/documents/${documentId}/file`

  return (
    <div
      ref={containerRef}
      style={{
        width: '100%',
        height: '100%',
        overflowY: 'auto',
        background: '#525659',
        padding: '12px',
        boxSizing: 'border-box',
      }}
    >
      <Document
        file={fileUrl}
        onLoadSuccess={({ numPages }) => setNumPages(numPages)}
        loading={<div style={{ color: '#ccc', padding: '16px', textAlign: 'center' }}>Đang tải PDF…</div>}
        error={<div style={{ color: '#f88', padding: '16px', textAlign: 'center' }}>Không tải được PDF.</div>}
      >
        {Array.from({ length: numPages }, (_, i) => {
          const page = i + 1
          return (
            <div
              key={page}
              ref={el => { pageRefs.current[page] = el }}
              style={{
                marginBottom: '8px',
                outline: page === currentPage ? '3px solid #ffeb3b' : 'none',
                borderRadius: '2px',
              }}
            >
              <Page
                pageNumber={page}
                width={containerWidth}
                renderTextLayer
                renderAnnotationLayer
              />
            </div>
          )
        })}
      </Document>
    </div>
  )
}
