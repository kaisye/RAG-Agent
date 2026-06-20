import { useState } from 'react'
import type { Citation } from '../types'

interface Props {
  citation: Citation
  index: number
}

export function CitationChip({ citation, index }: Props) {
  const [imgError, setImgError] = useState(false)
  const [showPreview, setShowPreview] = useState(false)

  const label = `Trang ${citation.page + 1}`   // page is 0-indexed internally

  if (citation.type === 'image') {
    return (
      <div style={{ position: 'relative', display: 'inline-block' }}>
        <button
          onClick={() => setShowPreview(v => !v)}
          title={`Ảnh — ${label}`}
          style={{
            display: 'inline-flex',
            alignItems: 'center',
            gap: '4px',
            padding: '3px 8px',
            borderRadius: '12px',
            border: '1px solid #c5cae9',
            background: '#e8eaf6',
            color: '#3949ab',
            fontSize: '12px',
            cursor: 'pointer',
            fontFamily: 'inherit',
          }}
        >
          🖼 {label}
        </button>

        {showPreview && (
          <div
            style={{
              position: 'absolute',
              bottom: 'calc(100% + 6px)',
              left: 0,
              zIndex: 100,
              background: '#fff',
              border: '1px solid #ccc',
              borderRadius: '8px',
              boxShadow: '0 4px 16px rgba(0,0,0,0.15)',
              padding: '8px',
              minWidth: '180px',
            }}
          >
            <div style={{ fontSize: '11px', color: '#666', marginBottom: '6px' }}>
              {label} · {citation.document_id.slice(0, 8)}…
            </div>
            {imgError ? (
              <div style={{ color: '#999', fontSize: '12px' }}>Không tải được ảnh</div>
            ) : (
              <img
                src={citation.thumbnail_url}
                alt={label}
                onError={() => setImgError(true)}
                style={{ maxWidth: '220px', maxHeight: '160px', borderRadius: '4px', display: 'block' }}
              />
            )}
            <button
              onClick={() => setShowPreview(false)}
              style={{ marginTop: '6px', fontSize: '11px', color: '#999', background: 'none', border: 'none', cursor: 'pointer', padding: 0 }}
            >
              Đóng
            </button>
          </div>
        )}
      </div>
    )
  }

  // Text citation
  return (
    <span
      title={citation.snippet}
      style={{
        display: 'inline-flex',
        alignItems: 'center',
        gap: '4px',
        padding: '3px 8px',
        borderRadius: '12px',
        border: '1px solid #c8e6c9',
        background: '#e8f5e9',
        color: '#2e7d32',
        fontSize: '12px',
        cursor: 'default',
      }}
    >
      📄 {label}
    </span>
  )
}
