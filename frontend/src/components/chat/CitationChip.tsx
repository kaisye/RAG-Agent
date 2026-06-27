import type { Citation } from "../../types/document"

interface Props {
  citation:      Citation
  onJumpToPage?: (page: number) => void
}

export function CitationChip({ citation, onJumpToPage }: Props) {
  const handleClick = () => onJumpToPage?.(citation.page)

  const baseStyle: React.CSSProperties = {
    display:      "inline-flex",
    alignItems:   "center",
    gap:          4,
    padding:      "2px 8px",
    borderRadius: 12,
    fontSize:     12,
    cursor:       "pointer",
    border:       "1px solid #d1d5db",
    background:   "#f9fafb",
    color:        "#374151",
    verticalAlign: "middle",
    margin:       "2px 3px",
    transition:   "background 0.15s",
  }

  if (citation.type === "image" && citation.thumbnail_url) {
    return (
      <button onClick={handleClick} style={baseStyle} title={citation.snippet ?? `Trang ${citation.page}`}>
        <img
          src={citation.thumbnail_url}
          alt={`Trang ${citation.page}`}
          style={{ width: 20, height: 20, objectFit: "cover", borderRadius: 3 }}
        />
        <span>Trang {citation.page}</span>
      </button>
    )
  }

  return (
    <button onClick={handleClick} style={baseStyle} title={citation.snippet ?? `Trang ${citation.page}`}>
      <span style={{ fontSize: 10, opacity: 0.6 }}>📄</span>
      <span>Trang {citation.page}</span>
    </button>
  )
}
